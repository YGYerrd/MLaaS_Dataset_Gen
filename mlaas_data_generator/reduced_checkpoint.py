"""Versioned, lossy storage representations with separate source and reload hashes.

Training/evaluation state is never modified by encoding. New representation
implementations (for example INT8 with scale tensors) supply encode/decode methods;
artifact integrity, source identity and quality validation remain shared.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import safetensors

from .checkpoint_store import (
    CompressionSpec, IntegrityError, TensorEntry, TensorState, UnsupportedTensorStateError,
    _canonical_json_bytes, _compress, _decompress, _identity_manifest_path,
    _publish_bytes_once, _read_json, _sha256_file,
)

FORMAT = "mlaas-reduced-checkpoint-v1"
DTYPES = {"bool": "BOOL", "int8": "I8", "int16": "I16", "int32": "I32", "int64": "I64",
          "uint8": "U8", "uint16": "U16", "uint32": "U32", "uint64": "U64",
          "float16": "F16", "float32": "F32", "float64": "F64", "bfloat16": "BF16"}


class ReducedPrecisionRejected(ValueError):
    """The representation exceeded the requested performance tolerance."""


class FP16Representation:
    version = "fp16_safetensors_v1"

    def encode(self, state):
        tensors, layout = {}, []
        squared_error = squared_source = max_error = 0.0
        converted = retained = 0
        for entry in state.entries:
            if entry.dtype not in DTYPES:
                raise UnsupportedTensorStateError("unsupported_safetensors_dtype", entry.dtype)
            data, dtype = entry.data, entry.dtype
            if dtype in {"float32", "float64"}:
                values = entry.to_numpy()
                with np.errstate(over="ignore", invalid="ignore"):
                    half = values.astype("<f2")
                # Preserve exceptional tensors at their original precision.
                if np.isfinite(values).all() and np.isfinite(half).all():
                    data, dtype = half.tobytes(), "float16"
                    delta = half.astype(np.float64) - values
                    squared_error += float(np.sum(delta * delta))
                    squared_source += float(np.sum(values.astype(np.float64) ** 2))
                    max_error = max(max_error, float(np.max(np.abs(delta), initial=0)))
                    converted += 1
                else:
                    retained += 1
            tensors[entry.name] = {"dtype": dtype, "shape": list(entry.shape), "data": data}
            layout.append({"name": entry.name, "source_dtype": entry.dtype, "stored_dtype": dtype,
                           "shape": list(entry.shape)})
        return tensors, layout, {"converted_tensors": converted, "exceptional_tensors_retained": retained,
            "max_absolute_weight_error": max_error,
            "relative_l2_weight_error": (squared_error / squared_source) ** 0.5 if squared_source else 0.0}

    def decode(self, tensors, layout):
        entries = []
        if set(tensors) != {item["name"] for item in layout}:
            raise IntegrityError("reduced checkpoint tensor names differ from its layout")
        for item in layout:
            tensor = tensors[item["name"]]
            if tensor["dtype"] != DTYPES[item["stored_dtype"]] or list(tensor["shape"]) != item["shape"]:
                raise IntegrityError("reduced checkpoint tensor layout mismatch")
            data = bytes(tensor["data"])
            if item["stored_dtype"] != item["source_dtype"]:
                if item["stored_dtype"] != "float16" or item["source_dtype"] not in {"float32", "float64"}:
                    raise IntegrityError("unsupported FP16 dtype restoration")
                data = np.frombuffer(data, dtype="<f2").astype("<f4" if item["source_dtype"] == "float32" else "<f8").tobytes()
            entries.append(TensorEntry(item["name"], item["source_dtype"], tuple(item["shape"]), data))
        return TensorState(tuple(entries))


class INT8Representation:
    """Symmetric INT8 blocks with one positive FP32 scale per 256 values.

    Quantization follows canonical C-order, including across row boundaries.
    Runtime dtypes are restored on decode; integers, BF16 and exceptional
    floating tensors are retained verbatim. Work arrays are bounded to 1M values.
    """

    version = "int8_block256_safetensors_v1"
    block_size = 256
    chunk_size = 1024 * 1024
    float_dtypes = {"float16": "<f2", "float32": "<f4", "float64": "<f8"}

    def _quantize(self, entry):
        values = np.frombuffer(entry.data, dtype=self.float_dtypes[entry.dtype])
        quantized = np.empty(values.size, dtype=np.int8)
        scales = np.empty((values.size + self.block_size - 1) // self.block_size, dtype="<f4")
        squared_error = squared_source = max_error = 0.0
        for start in range(0, values.size, self.chunk_size):
            original = values[start:start + self.chunk_size]
            blocks = (original.size + self.block_size - 1) // self.block_size
            work = np.zeros(blocks * self.block_size, dtype=np.float64)
            work[:original.size] = original
            work = work.reshape(blocks, self.block_size)
            if not np.isfinite(work).all():
                return None
            maxima = np.max(np.abs(work), axis=1)
            with np.errstate(over="ignore", under="ignore"):
                scale = (maxima / 127).astype("<f4")
            scale[maxima == 0] = 1.0
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                return None
            q = np.clip(np.rint(work / scale.astype(np.float64)[:, None]), -127, 127).astype(np.int8)
            with np.errstate(over="ignore", under="ignore"):
                restored = (q.astype(np.float64) * scale[:, None]).ravel()[:original.size].astype(original.dtype)
            if not np.isfinite(restored).all():
                return None
            delta = restored.astype(np.float64) - original
            squared_error += float(np.sum(delta * delta))
            squared_source += float(np.sum(original.astype(np.float64) ** 2))
            max_error = max(max_error, float(np.max(np.abs(delta), initial=0)))
            quantized[start:start + original.size] = q.ravel()[:original.size]
            offset = start // self.block_size
            scales[offset:offset + blocks] = scale
        return quantized, scales, squared_error, squared_source, max_error

    def encode(self, state):
        tensors, layout = {}, []
        names = {entry.name for entry in state.entries}
        converted = retained = 0
        squared_error = squared_source = max_error = 0.0
        for index, entry in enumerate(state.entries):
            if entry.dtype not in DTYPES:
                raise UnsupportedTensorStateError("unsupported_safetensors_dtype", entry.dtype)
            item = {"name": entry.name, "source_dtype": entry.dtype,
                    "stored_dtype": entry.dtype, "shape": list(entry.shape)}
            data = entry.data
            if entry.dtype in self.float_dtypes:
                result = self._quantize(entry)
                if result is None:
                    retained += 1
                else:
                    quantized, scales, error, source_norm, maximum = result
                    scale_name = f"__mlaas_int8_scale_{index}"
                    while scale_name in names:
                        scale_name += "_"
                    names.add(scale_name)
                    tensors[scale_name] = {"dtype": "float32", "shape": [scales.size], "data": scales.tobytes()}
                    item.update(stored_dtype="int8", quantization={
                        "scheme": "symmetric_blockwise", "block_size": self.block_size, "scale_tensor": scale_name})
                    data = quantized.tobytes()
                    squared_error += error
                    squared_source += source_norm
                    max_error = max(max_error, maximum)
                    converted += 1
            tensors[entry.name] = {"dtype": item["stored_dtype"], "shape": item["shape"], "data": data}
            layout.append(item)
        return tensors, layout, {"converted_tensors": converted, "exceptional_tensors_retained": retained,
            "max_absolute_weight_error": max_error,
            "relative_l2_weight_error": (squared_error / squared_source) ** 0.5 if squared_source else 0.0}

    def decode(self, tensors, layout):
        expected_names = [item["name"] for item in layout]
        expected_names += [item["quantization"]["scale_tensor"] for item in layout if "quantization" in item]
        if len(set(expected_names)) != len(expected_names) or set(tensors) != set(expected_names):
            raise IntegrityError("INT8 checkpoint tensor names differ from its layout")
        entries = []
        for item in layout:
            tensor = tensors[item["name"]]
            if tensor["dtype"] != DTYPES.get(item["stored_dtype"]) or list(tensor["shape"]) != item["shape"]:
                raise IntegrityError("INT8 checkpoint tensor layout mismatch")
            # Validate raw length and shape before interpreting quantized bytes.
            entry = TensorEntry(item["name"], item["stored_dtype"], tuple(item["shape"]), bytes(tensor["data"]))
            quantization = item.get("quantization")
            if quantization is None:
                if item["stored_dtype"] != item["source_dtype"]:
                    raise IntegrityError("INT8 checkpoint is missing quantization metadata")
                entries.append(entry)
                continue
            if (item["stored_dtype"] != "int8" or item["source_dtype"] not in self.float_dtypes
                    or quantization.get("scheme") != "symmetric_blockwise"
                    or quantization.get("block_size") != self.block_size):
                raise IntegrityError("unsupported INT8 quantization layout")
            scale_tensor = tensors[quantization["scale_tensor"]]
            count = (entry.numel + self.block_size - 1) // self.block_size
            if (scale_tensor["dtype"] != "F32" or list(scale_tensor["shape"]) != [count]
                    or len(scale_tensor["data"]) != count * 4):
                raise IntegrityError("INT8 checkpoint scale layout mismatch")
            scales = np.frombuffer(scale_tensor["data"], dtype="<f4")
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise IntegrityError("INT8 checkpoint scales must be finite and positive")
            q = np.frombuffer(entry.data, dtype=np.int8)
            if np.any(q == -128):
                raise IntegrityError("INT8 checkpoint values exceed the symmetric range")
            values = np.empty(q.size, dtype=self.float_dtypes[item["source_dtype"]])
            for start in range(0, q.size, self.chunk_size):
                end = min(start + self.chunk_size, q.size)
                scale = np.repeat(scales[start // self.block_size:(end + self.block_size - 1) // self.block_size], self.block_size)
                with np.errstate(over="ignore", under="ignore"):
                    values[start:end] = q[start:end].astype(np.float64) * scale[:end - start]
                if not np.isfinite(values[start:end]).all():
                    raise IntegrityError("INT8 checkpoint dequantization overflow")
            entries.append(TensorEntry(entry.name, item["source_dtype"], entry.shape, values.tobytes()))
        return TensorState(tuple(entries))


REPRESENTATIONS = {"fp16": FP16Representation(), "int8": INT8Representation()}


@dataclass(frozen=True)
class ReducedCheckpointResult:
    manifest_path: Path
    payload_path: Path
    document: dict

    def artifact(self):
        return {"artifact_type": "model_final_reduced_checkpoint", "artifact_uri": str(self.manifest_path),
            "metadata": {"sha256": _sha256_file(self.manifest_path), "size_bytes": self.manifest_path.stat().st_size,
                "format": FORMAT, "role": "final", "source_state_sha256": self.document["source_state_sha256"],
                "reloaded_state_sha256": self.document["reloaded_state_sha256"],
                "weight_format": self.document["representation"], "lossy": True,
                "quality_validation": self.document["quality_validation"]}}


def persist_reduced_checkpoint(store_root, service_id, lineage_id, source, *, base_ref,
                               parameter_schema_hash, representation="int8", compression=CompressionSpec("none"),
                               validate=None, measure_full=False):
    codec = REPRESENTATIONS[representation]
    tensors, layout, error = codec.encode(source)
    payload = safetensors.serialize(tensors, metadata={"format": FORMAT, "representation": codec.version,
                                                    "source_state_sha256": source.sha256})
    # Verify the serialized representation before evaluating it.
    reloaded = codec.decode(dict(safetensors.deserialize(payload)), layout)
    quality = validate(reloaded) if validate else {"status": "not_run"}
    if validate and quality.get("status") != "passed":
        raise ReducedPrecisionRejected(f"reloaded checkpoint quality check failed: {quality}")
    encoded = _compress(payload, compression)
    full_bytes = None
    if measure_full:
        full = safetensors.serialize({e.name: {"dtype": e.dtype, "shape": list(e.shape), "data": e.data}
                                     for e in source.entries}, metadata={"source_state_sha256": source.sha256})
        full_bytes = len(_compress(full, compression))
    digest = hashlib.sha256(encoded).hexdigest()
    from .storage_overflow import choose_storage_root
    root = Path(store_root).resolve()
    payload_root = choose_storage_root(root, len(encoded))
    suffix = ".safetensors" if compression.algorithm == "none" else f".safetensors.{compression.algorithm}-l{compression.level}"
    path = payload_root / "objects" / "sha256" / digest[:2] / (digest + suffix)
    _publish_bytes_once(path, encoded)
    document = {"format": FORMAT, "service_id": service_id, "lineage_id": lineage_id,
        "representation": representation, "representation_version": codec.version, "lossy": True,
        "source_state_sha256": source.sha256, "reloaded_state_sha256": reloaded.sha256,
        "parameter_schema_hash": parameter_schema_hash, "layout": layout,
        "base_state": base_ref.link(root), "base_state_sha256": base_ref.content_sha256,
        "payload_path": str(path), "payload_sha256": digest, "payload_bytes": len(encoded),
        "uncompressed_bytes": len(payload), "source_tensor_bytes": source.raw_tensor_size_bytes,
        "full_comparison_payload_bytes": full_bytes,
        "compression": compression.as_dict(), "weight_error": error, "quality_validation": quality}
    # The on-disk read-back check is against the representation, never the FP32 source hash.
    if load_reduced_state(document).sha256 != reloaded.sha256:
        raise IntegrityError("persisted reduced representation changed on read-back")
    manifest = _identity_manifest_path(root, "services", str(service_id))
    _publish_bytes_once(manifest, _canonical_json_bytes(document))
    return ReducedCheckpointResult(manifest, path, document)


def load_reduced_state(document):
    if document.get("format") != FORMAT:
        raise IntegrityError("unsupported reduced checkpoint format")
    codec = REPRESENTATIONS.get(document.get("representation"))
    if codec is None or codec.version != document.get("representation_version"):
        raise IntegrityError("unsupported checkpoint representation version")
    payload = Path(document["payload_path"]).read_bytes()
    if len(payload) != document["payload_bytes"] or hashlib.sha256(payload).hexdigest() != document["payload_sha256"]:
        raise IntegrityError("reduced checkpoint payload failed integrity verification")
    compression = document["compression"]
    payload = _decompress(payload, CompressionSpec(compression["algorithm"], compression.get("level")))
    if len(payload) != document["uncompressed_bytes"]:
        raise IntegrityError("reduced checkpoint decoded size mismatch")
    state = codec.decode(dict(safetensors.deserialize(payload)), document["layout"])
    if state.sha256 != document["reloaded_state_sha256"]:
        raise IntegrityError("reduced checkpoint reload hash mismatch")
    return state


def source_state_hash(store_root, service_id, reloaded):
    """Verify reload evidence and return the unchanged source-model identity."""
    doc = _read_json(_identity_manifest_path(Path(store_root).resolve(), "services", str(service_id)))
    if doc.get("format") == FORMAT:
        if reloaded.sha256 != doc["reloaded_state_sha256"]:
            raise IntegrityError("reloaded state differs from its representation manifest")
        return doc["source_state_sha256"]
    return reloaded.sha256
