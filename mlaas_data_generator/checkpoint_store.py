"""Lossless shared-base checkpoint storage for generated services.

This module owns *tensor values*, not the parameter schema.  A caller may
associate a separately computed ``parameter_schema_hash`` with a lineage or
service manifest, but that hash never changes the canonical tensor-state hash
or the patch content hash.

Canonical tensor-state hashing specification (``mlaas-tensor-state-v1``)
==========================================================================

The SHA-256 tensor-state digest is computed over this exact byte stream::

    b"MLAAS-TENSOR-STATE-V1\\0"
    uint64_be(number of entries)
    repeated, in native model-state order:
        uint64_be(UTF-8 NFC tensor-name byte length)
        UTF-8 NFC tensor-name bytes
        uint64_be(ASCII canonical-dtype byte length)
        ASCII canonical-dtype bytes
        uint64_be(rank)
        uint64_be(dimension 0) ... uint64_be(dimension rank-1)
        uint64_be(raw tensor byte length)
        canonical raw tensor bytes

Raw tensor bytes are contiguous C-order bytes in little-endian order.  The
supported dtype tokens are ``bool``, ``int8/16/32/64``,
``uint8/16/32/64``, ``float16/32/64``, ``complex64/128``, and
``bfloat16``.  Scalar tensors have rank zero.  Empty tensors have an empty
payload.  Framework identity, parameter/buffer kind, trainability, aliases,
and the parameter-schema hash are deliberately excluded.

Patch semantics (``mlaas-tensor-patch-v1``) are likewise canonical.  They
contain only changed tensor indexes with XOR or exact REPLACE operations;
unchanged tensors are inherited from the direct base and omitted.  The
semantic patch bytes never contain a compression setting.  The persistence
layer may evaluate XOR and REPLACE candidates using the selected codec, but
records that decision in the service manifest.  Compression remains an
independent content-store envelope property rather than part of the patch
format.  Its exact byte stream is the patch magic, 32 raw base-hash bytes,
32 raw target-hash bytes, the uint64 target-entry count, the uint64 changed-op
count, then each changed operation in ascending index order: uint64 index,
one-byte opcode (1 XOR, 2 REPLACE), the same name/dtype/shape descriptor used
above, uint64 payload length, and payload bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import unicodedata
import zlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .checkpoint_profile import profiled, profile_call


TENSOR_STATE_FORMAT = "mlaas-tensor-state-v1"
TENSOR_PATCH_FORMAT = "mlaas-tensor-patch-v1"
CAS_FORMAT = "mlaas-content-object-v1"
LINEAGE_BASE_MANIFEST_FORMAT = "mlaas-lineage-base-v1"
SERVICE_PATCH_MANIFEST_FORMAT = "mlaas-service-patch-v1"

_STATE_MAGIC = b"MLAAS-TENSOR-STATE-V1\0"
_PATCH_MAGIC = b"MLAAS-TENSOR-PATCH-V1\0"
_XOR = 1
_REPLACE = 2
_OPERATION_NAMES = {_XOR: "xor", _REPLACE: "replace"}
_OPERATION_CODES = {value: key for key, value in _OPERATION_NAMES.items()}

CANONICAL_TENSOR_STATE_SPEC = {
    "format": TENSOR_STATE_FORMAT,
    "digest": "sha256",
    "magic": "MLAAS-TENSOR-STATE-V1\\0",
    "entry_order": "native model-state order",
    "name": "UTF-8 after Unicode NFC normalization",
    "integer_framing": "unsigned 64-bit big-endian",
    "tensor_bytes": "contiguous C order, little-endian",
    "dtype_tokens": [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "complex64", "complex128", "bfloat16",
    ],
    "hash_excludes": [
        "framework",
        "parameter_schema",
        "parameter_or_buffer_kind",
        "trainability",
        "compression",
    ],
}

_NUMPY_DTYPES: dict[str, str] = {
    "bool": "|b1",
    "int8": "|i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "uint8": "|u1",
    "uint16": "<u2",
    "uint32": "<u4",
    "uint64": "<u8",
    "float16": "<f2",
    "float32": "<f4",
    "float64": "<f8",
    "complex64": "<c8",
    "complex128": "<c16",
}
_DTYPE_ITEMSIZE = {name: np.dtype(value).itemsize for name, value in _NUMPY_DTYPES.items()}
_DTYPE_ITEMSIZE["bfloat16"] = 2


class CheckpointStoreError(RuntimeError):
    """Base exception for checkpoint-store failures."""


class IntegrityError(CheckpointStoreError):
    """An object or reconstruction failed an integrity check."""


class ManifestConflictError(CheckpointStoreError):
    """An immutable lineage/service manifest already has different content."""


class ParameterSchemaMismatchError(CheckpointStoreError):
    """A caller supplied a different, separately-owned parameter schema."""


class UnsupportedTensorStateError(CheckpointStoreError):
    """Signal that the native full-object fallback must be used."""

    def __init__(self, reason_code: str, detail: str, *, model: Any | None = None):
        super().__init__(detail)
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        self.model_class = (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}"
            if model is not None
            else None
        )

    def as_fallback_signal(self) -> dict[str, Any]:
        return {
            "required": True,
            "policy": "full_object_v1",
            "reason_code": self.reason_code,
            "detail": self.detail,
            "model_class": self.model_class,
        }


@dataclass(frozen=True)
class TensorEntry:
    """One ordered tensor value in canonical byte representation."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        name = unicodedata.normalize("NFC", str(self.name))
        dtype = str(self.dtype)
        shape = tuple(int(value) for value in self.shape)
        data = bytes(self.data)
        if not name:
            raise ValueError("tensor names must not be empty")
        if dtype not in _DTYPE_ITEMSIZE:
            raise ValueError(f"unsupported canonical dtype: {dtype}")
        if any(value < 0 for value in shape):
            raise ValueError(f"negative tensor dimension for {name!r}")
        elements = 1
        for value in shape:
            elements *= value
        expected = elements * _DTYPE_ITEMSIZE[dtype]
        if len(data) != expected:
            raise ValueError(
                f"tensor {name!r} has {len(data)} bytes; expected {expected} "
                f"for {dtype}{shape}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "data", data)

    @property
    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def descriptor(self) -> tuple[str, str, tuple[int, ...]]:
        return self.name, self.dtype, self.shape

    def to_numpy(self) -> np.ndarray:
        if self.dtype == "bfloat16":
            raise UnsupportedTensorStateError(
                "bfloat16_requires_framework",
                "bfloat16 tensors must be restored through a framework tensor",
            )
        array = np.frombuffer(self.data, dtype=np.dtype(_NUMPY_DTYPES[self.dtype]))
        return array.reshape(self.shape).copy()


@dataclass(frozen=True)
class TensorState:
    """Ordered tensor values; ``framework`` is excluded from its content hash."""

    entries: tuple[TensorEntry, ...]
    framework: str = "unknown"

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("canonical tensor-state names must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "framework", str(self.framework))

    @cached_property
    def sha256(self) -> str:
        return canonical_tensor_state_hash(self)

    @property
    def raw_tensor_size_bytes(self) -> int:
        return sum(len(entry.data) for entry in self.entries)


@dataclass(frozen=True)
class PatchOperation:
    index: int
    operation: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    payload: bytes = b""

    def __post_init__(self) -> None:
        index = int(self.index)
        if index < 0:
            raise ValueError("patch operation indexes must be non-negative")
        operation = str(self.operation)
        if operation not in _OPERATION_CODES:
            raise ValueError(f"unknown patch operation: {operation}")
        name = unicodedata.normalize("NFC", str(self.name))
        dtype = str(self.dtype)
        shape = tuple(int(value) for value in self.shape)
        if not name:
            raise ValueError("tensor names must not be empty")
        if dtype not in _DTYPE_ITEMSIZE:
            raise ValueError(f"unsupported canonical dtype: {dtype}")
        if any(value < 0 for value in shape):
            raise ValueError(f"negative tensor dimension for {name!r}")
        payload = bytes(self.payload)
        if len(payload) != _entry_nbytes(dtype, shape):
            raise ValueError(f"{operation.upper()} payload has the wrong length for {name!r}")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "payload", payload)

    def descriptor(self) -> tuple[str, str, tuple[int, ...]]:
        return self.name, self.dtype, self.shape


@dataclass(frozen=True)
class TensorPatch:
    base_state_sha256: str
    target_state_sha256: str
    target_entry_count: int
    operations: tuple[PatchOperation, ...]

    def __post_init__(self) -> None:
        for value in (self.base_state_sha256, self.target_state_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value)):
                raise ValueError("state hashes must be lowercase SHA-256 hex strings")
        target_entry_count = int(self.target_entry_count)
        operations = tuple(self.operations)
        indexes = [operation.index for operation in operations]
        if target_entry_count < 0:
            raise ValueError("target entry count must be non-negative")
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise ValueError("patch operations must have unique ascending indexes")
        if indexes and indexes[-1] >= target_entry_count:
            raise ValueError("patch operation index exceeds target entry count")
        object.__setattr__(self, "target_entry_count", target_entry_count)
        object.__setattr__(self, "operations", operations)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(serialize_tensor_patch(self)).hexdigest()

    @property
    def operation_counts(self) -> dict[str, int]:
        return {
            "inherit": self.target_entry_count - len(self.operations),
            "xor": sum(operation.operation == "xor" for operation in self.operations),
            "replace": sum(operation.operation == "replace" for operation in self.operations),
        }


@dataclass(frozen=True)
class CompressionSpec:
    algorithm: str = "zlib"
    level: int | None = 1

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm).lower()
        if algorithm not in {"none", "zlib", "zstd"}:
            raise ValueError(f"unsupported compression algorithm: {algorithm}")
        level = None if algorithm == "none" else int(1 if self.level is None else self.level)
        if algorithm == "zlib" and not 0 <= level <= 9:
            raise ValueError("zlib compression level must be between 0 and 9")
        if algorithm == "zstd" and not -5 <= level <= 22:
            raise ValueError("zstd compression level must be between -5 and 22")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "level", level)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"algorithm": self.algorithm}
        if self.level is not None:
            result["level"] = self.level
        return result


@dataclass(frozen=True)
class ContentObjectRef:
    object_kind: str
    semantic_format: str
    content_sha256: str
    compression: CompressionSpec
    logical_size_bytes: int
    stored_payload_size_bytes: int
    payload_sha256: str
    manifest_path: Path
    payload_path: Path

    @property
    def manifest_size_bytes(self) -> int:
        return self.manifest_path.stat().st_size

    @property
    def physical_size_bytes(self) -> int:
        """Actual file lengths; filesystem allocation-unit padding is excluded."""
        return self.manifest_path.stat().st_size + self.payload_path.stat().st_size

    def link(self, store_root: str | Path) -> dict[str, Any]:
        root = Path(store_root).resolve()
        return {
            "object_kind": self.object_kind,
            "semantic_format": self.semantic_format,
            "content_sha256": self.content_sha256,
            "object_manifest": self.manifest_path.resolve().relative_to(root).as_posix(),
        }


@dataclass(frozen=True)
class LineageBaseResult:
    lineage_id: str
    manifest_path: Path
    base: ContentObjectRef
    tensor_state_sha256: str
    parameter_schema_hash: str | None
    created: bool
    restored_existing_base: bool

    def artifact(self) -> dict[str, Any]:
        """Return a ``service_artifacts``-compatible initial-checkpoint row."""
        return {
            "artifact_type": "model_initial_checkpoint",
            "artifact_uri": str(self.manifest_path),
            "metadata": {
                "sha256": _sha256_file(self.manifest_path),
                "size_bytes": self.manifest_path.stat().st_size,
                "format": LINEAGE_BASE_MANIFEST_FORMAT,
                "role": "initial",
                "tensor_state_sha256": self.tensor_state_sha256,
                "base_object_sha256": self.base.content_sha256,
                "parameter_schema_hash": self.parameter_schema_hash,
                "compression": self.base.compression.as_dict(),
                "logical_size_bytes": self.base.logical_size_bytes,
                "stored_payload_size_bytes": self.base.stored_payload_size_bytes,
                "physical_size_bytes": self.base.physical_size_bytes,
            },
        }


@dataclass(frozen=True)
class ServicePatchResult:
    service_id: str
    lineage_id: str
    manifest_path: Path
    base: ContentObjectRef
    patch: ContentObjectRef | None
    target_state_sha256: str
    parameter_schema_hash: str | None
    measurements: Mapping[str, Any]
    operation_counts: Mapping[str, int]
    operation_selection: Mapping[str, Any]

    def artifact(self) -> dict[str, Any]:
        metadata = {
            "sha256": _sha256_file(self.manifest_path),
            "size_bytes": self.manifest_path.stat().st_size,
            "format": SERVICE_PATCH_MANIFEST_FORMAT,
            "role": "final",
            "base_state_sha256": self.base.content_sha256,
            "target_state_sha256": self.target_state_sha256,
            "patch_sha256": self.patch.content_sha256 if self.patch else None,
            "patch_mode": "exact_patch" if self.patch else "base_reference_only",
            "parameter_schema_hash": self.parameter_schema_hash,
            "compression": dict(self.measurements.get("compression", {})),
            "operation_counts": dict(self.operation_counts),
            "operation_selection": dict(self.operation_selection),
            "storage_measurements": dict(self.measurements),
        }
        return {
            "artifact_type": "model_final_tensor_patch",
            "artifact_uri": str(self.manifest_path),
            "metadata": metadata,
        }


def _u64(value: int) -> bytes:
    return struct.pack(">Q", int(value))


def _read_u64(reader: "_Reader") -> int:
    return struct.unpack(">Q", reader.read(8))[0]


def _encode_text(value: str, *, ascii_only: bool = False) -> bytes:
    value = unicodedata.normalize("NFC", str(value))
    encoded = value.encode("ascii" if ascii_only else "utf-8")
    return _u64(len(encoded)) + encoded


def _decode_text(reader: "_Reader", *, ascii_only: bool = False) -> str:
    return reader.read(_read_u64(reader)).decode("ascii" if ascii_only else "utf-8")


def _encode_descriptor(name: str, dtype: str, shape: Sequence[int]) -> bytes:
    result = bytearray(_encode_text(name))
    result.extend(_encode_text(dtype, ascii_only=True))
    result.extend(_u64(len(shape)))
    for value in shape:
        result.extend(_u64(value))
    return bytes(result)


def _decode_descriptor(reader: "_Reader") -> tuple[str, str, tuple[int, ...]]:
    name = unicodedata.normalize("NFC", _decode_text(reader))
    dtype = _decode_text(reader, ascii_only=True)
    shape = tuple(_read_u64(reader) for _ in range(_read_u64(reader)))
    return name, dtype, shape


class _Reader:
    def __init__(self, payload: bytes):
        self.payload = memoryview(payload)
        self.offset = 0

    def read(self, size: int) -> bytes:
        size = int(size)
        end = self.offset + size
        if size < 0 or end > len(self.payload):
            raise IntegrityError("truncated canonical checkpoint payload")
        result = self.payload[self.offset:end].tobytes()
        self.offset = end
        return result

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise IntegrityError("trailing bytes in canonical checkpoint payload")


@profiled("state_serialize")
def serialize_tensor_state(state: TensorState) -> bytes:
    payload = bytearray(_STATE_MAGIC)
    payload.extend(_u64(len(state.entries)))
    for entry in state.entries:
        payload.extend(_encode_descriptor(entry.name, entry.dtype, entry.shape))
        payload.extend(_u64(len(entry.data)))
        payload.extend(entry.data)
    return bytes(payload)


@profiled("state_deserialize")
def deserialize_tensor_state(payload: bytes, *, framework: str = "unknown") -> TensorState:
    reader = _Reader(payload)
    if reader.read(len(_STATE_MAGIC)) != _STATE_MAGIC:
        raise IntegrityError("not an mlaas-tensor-state-v1 payload")
    entries = []
    for _ in range(_read_u64(reader)):
        name, dtype, shape = _decode_descriptor(reader)
        entries.append(TensorEntry(name, dtype, shape, reader.read(_read_u64(reader))))
    reader.finish()
    return TensorState(tuple(entries), framework=framework)


@profiled("state_hash")
def canonical_tensor_state_hash(state: TensorState) -> str:
    # Identical framing to serialize_tensor_state, without allocating a second
    # full-model byte stream. TensorEntry normalizes its payload to immutable bytes.
    digest = hashlib.sha256()
    digest.update(_STATE_MAGIC)
    digest.update(_u64(len(state.entries)))
    for entry in state.entries:
        digest.update(_encode_descriptor(entry.name, entry.dtype, entry.shape))
        digest.update(_u64(len(entry.data)))
        digest.update(entry.data)
    return digest.hexdigest()


@profiled("patch_serialize")
def serialize_tensor_patch(patch: TensorPatch) -> bytes:
    payload = bytearray(_PATCH_MAGIC)
    payload.extend(bytes.fromhex(patch.base_state_sha256))
    payload.extend(bytes.fromhex(patch.target_state_sha256))
    payload.extend(_u64(patch.target_entry_count))
    payload.extend(_u64(len(patch.operations)))
    for operation in patch.operations:
        payload.extend(_u64(operation.index))
        payload.append(_OPERATION_CODES[operation.operation])
        payload.extend(_encode_descriptor(operation.name, operation.dtype, operation.shape))
        payload.extend(_u64(len(operation.payload)))
        payload.extend(operation.payload)
    return bytes(payload)


@profiled("patch_deserialize")
def deserialize_tensor_patch(payload: bytes) -> TensorPatch:
    reader = _Reader(payload)
    if reader.read(len(_PATCH_MAGIC)) != _PATCH_MAGIC:
        raise IntegrityError("not an mlaas-tensor-patch-v1 payload")
    base_hash = reader.read(32).hex()
    target_hash = reader.read(32).hex()
    target_entry_count = _read_u64(reader)
    operations = []
    for _ in range(_read_u64(reader)):
        index = _read_u64(reader)
        code = reader.read(1)[0]
        if code not in _OPERATION_NAMES:
            raise IntegrityError(f"unknown tensor-patch opcode: {code}")
        name, dtype, shape = _decode_descriptor(reader)
        operations.append(
            PatchOperation(index, _OPERATION_NAMES[code], name, dtype, shape, reader.read(_read_u64(reader)))
        )
    reader.finish()
    return TensorPatch(base_hash, target_hash, target_entry_count, tuple(operations))


@profiled("patch_build")
def build_tensor_patch(
    base: TensorState,
    target: TensorState,
    *,
    replacement_indexes: Iterable[int] = (),
) -> TensorPatch:
    """Build a direct lossless patch; no compression decision is made here."""
    base_descriptors = [entry.descriptor() for entry in base.entries]
    target_descriptors = [entry.descriptor() for entry in target.entries]
    if base_descriptors != target_descriptors:
        raise UnsupportedTensorStateError(
            "state_layout_mismatch",
            "shared-base patches require the same ordered state names, dtypes, and shapes",
        )
    replacements = {int(value) for value in replacement_indexes}
    if any(value < 0 or value >= len(target.entries) for value in replacements):
        raise ValueError("replacement index exceeds target state")
    operations: list[PatchOperation] = []
    for index, target_entry in enumerate(target.entries):
        base_entry = base.entries[index]
        if base_entry.data == target_entry.data:
            continue
        if index in replacements:
            operations.append(
                PatchOperation(index, "replace", target_entry.name, target_entry.dtype, target_entry.shape, target_entry.data)
            )
            continue
        xor = np.bitwise_xor(
            np.frombuffer(base_entry.data, dtype=np.uint8),
            np.frombuffer(target_entry.data, dtype=np.uint8),
        ).tobytes()
        operations.append(
            PatchOperation(index, "xor", target_entry.name, target_entry.dtype, target_entry.shape, xor)
        )
    return TensorPatch(base.sha256, target.sha256, len(target.entries), tuple(operations))


_EXACT_PATCH_SELECTION_MAX_BYTES = 64 * 1024 * 1024


@profiled("patch_selection")
def _select_tensor_patch_for_compression(
    base: TensorState,
    target: TensorState,
    *,
    compression: CompressionSpec | str = CompressionSpec(),
    operation_policy: str = "adaptive",
) -> tuple[TensorPatch, dict[str, Any], bytes | None]:
    """Select exact XOR/REPLACE semantics under the requested storage policy.

    Compression does not enter the serialized patch format.  This storage-layer
    helper measures complete canonical patch candidates after applying the
    requested codec for adaptive selection. Fixed policies skip comparisons.
    Decision metadata is returned separately for the service manifest.
    """
    codec = compression if isinstance(compression, CompressionSpec) else CompressionSpec(str(compression))
    if operation_policy not in {"adaptive", "fixed_xor", "fixed_replace"}:
        raise ValueError(f"unsupported checkpoint operation policy: {operation_policy}")
    if operation_policy != "adaptive":
        selected = "xor" if operation_policy == "fixed_xor" else "replace"
        patch = build_tensor_patch(
            base, target,
            replacement_indexes=range(len(target.entries)) if selected == "replace" else (),
        )
        return patch, {
            "selection_version": 3,
            "strategy": "fixed_lossless_operation_v1",
            "policy": operation_policy,
            "selected": selected,
            "candidate_comparison_performed": False,
        }, None
    xor_patch = build_tensor_patch(base, target)
    changed = [operation.index for operation in xor_patch.operations]
    if not changed:
        payload = serialize_tensor_patch(xor_patch)
        encoded = _compress(payload, codec)
        return xor_patch, {
            "selection_version": 1,
            "strategy": "no_changes",
            "evaluated_compression": codec.as_dict(),
            "candidate_stored_payload_size_bytes": {"xor": len(encoded)},
            "selected": "xor",
        }, encoded

    replacement_patch = build_tensor_patch(base, target, replacement_indexes=changed)
    hybrid_replacements: set[int] = set()
    per_tensor_sizes = {"xor": 0, "replace": 0, "per_tensor_hybrid": 0}
    for operation in xor_patch.operations:
        target_payload = target.entries[operation.index].data
        xor_size = len(profile_call("candidate_xor_compression", _compress, operation.payload, codec))
        replace_size = len(profile_call("candidate_replace_compression", _compress, target_payload, codec))
        per_tensor_sizes["xor"] += xor_size
        per_tensor_sizes["replace"] += replace_size
        per_tensor_sizes["per_tensor_hybrid"] += min(xor_size, replace_size)
        if replace_size < xor_size:
            hybrid_replacements.add(operation.index)
    hybrid_patch = build_tensor_patch(base, target, replacement_indexes=hybrid_replacements)
    candidates = {
        "xor": xor_patch,
        "replace": replacement_patch,
        "per_tensor_hybrid": hybrid_patch,
    }
    # Ties prefer XOR, the simplest representation of a true delta.
    priority = {"xor": 0, "per_tensor_hybrid": 1, "replace": 2}
    changed_payload_bytes = sum(len(operation.payload) for operation in xor_patch.operations)
    if changed_payload_bytes <= _EXACT_PATCH_SELECTION_MAX_BYTES:
        encoded_candidates = {
            name: profile_call(f"candidate_{name}_compression", _compress, serialize_tensor_patch(candidate), codec)
            for name, candidate in candidates.items()
        }
        measured = {name: len(payload) for name, payload in encoded_candidates.items()}
        selected_name = min(candidates, key=lambda name: (measured[name], priority[name]))
        return candidates[selected_name], {
            "selection_version": 1,
            "strategy": "minimum_complete_patch_encoded_size",
            "measurement_scope": "complete_patch_encoded_size",
            "evaluated_compression": codec.as_dict(),
            "candidate_stored_payload_size_bytes": measured,
            "selected": selected_name,
            "hybrid_replacement_indexes": sorted(hybrid_replacements),
        }, encoded_candidates[selected_name]

    estimated = dict(per_tensor_sizes)
    selected_name = min(candidates, key=lambda name: (estimated[name], priority[name]))
    return candidates[selected_name], {
        "selection_version": 2,
        "strategy": "minimum_independently_encoded_tensor_payload_size",
        "measurement_scope": "sum_of_independently_encoded_tensor_payloads",
        "evaluated_compression": codec.as_dict(),
        "candidate_estimated_stored_payload_size_bytes": estimated,
        "selected": selected_name,
        "hybrid_replacement_indexes": sorted(hybrid_replacements),
        "exact_complete_candidate_measurement_skipped": True,
        "exact_measurement_limit_bytes": _EXACT_PATCH_SELECTION_MAX_BYTES,
        "changed_tensor_payload_bytes": changed_payload_bytes,
    }, None


def select_tensor_patch_for_compression(
    base: TensorState,
    target: TensorState,
    *,
    compression: CompressionSpec | str = CompressionSpec(),
    operation_policy: str = "adaptive",
) -> tuple[TensorPatch, dict[str, Any]]:
    """Select exact patch operations while bounding large-checkpoint work.

    Small patches retain complete-candidate measurements. Transformer-sized
    patches compare independently encoded tensor payloads so selection does
    not repeatedly compress several complete model states. Fixed policies skip
    candidate comparison and retain exact XOR or replacement for changed entries.
    """
    patch, selection, _ = _select_tensor_patch_for_compression(
        base, target, compression=compression, operation_policy=operation_policy
    )
    return patch, selection


@profiled("patch_apply_verify")
def apply_tensor_patch(base: TensorState, patch: TensorPatch) -> TensorState:
    """Apply and strictly verify a patch against its direct immutable base."""
    if base.sha256 != patch.base_state_sha256:
        raise IntegrityError(
            f"patch expects base {patch.base_state_sha256}, received {base.sha256}"
        )
    if patch.target_entry_count != len(base.entries):
        raise IntegrityError("patch target entry count differs from its base")
    entries = list(base.entries)
    for operation in patch.operations:
        base_entry = base.entries[operation.index]
        if base_entry.descriptor() != operation.descriptor():
            raise IntegrityError(f"patch/base descriptor mismatch at index {operation.index}")
        if operation.operation == "xor":
            data = np.bitwise_xor(
                np.frombuffer(base_entry.data, dtype=np.uint8),
                np.frombuffer(operation.payload, dtype=np.uint8),
            ).tobytes()
        else:
            data = operation.payload
        entries[operation.index] = TensorEntry(
            operation.name, operation.dtype, operation.shape, data
        )
    result = TensorState(tuple(entries), framework=base.framework)
    if result.sha256 != patch.target_state_sha256:
        raise IntegrityError(
            f"reconstructed state hash {result.sha256} does not match {patch.target_state_sha256}"
        )
    return result


def _entry_nbytes(dtype: str, shape: Sequence[int]) -> int:
    if dtype not in _DTYPE_ITEMSIZE:
        raise ValueError(f"unsupported canonical dtype: {dtype}")
    result = _DTYPE_ITEMSIZE[dtype]
    for value in shape:
        result *= int(value)
    return result


def _numpy_entry(name: str, value: Any) -> TensorEntry:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise UnsupportedTensorStateError(
            "non_array_state", f"state entry {name!r} cannot be represented as an array: {exc}"
        ) from exc
    if array.dtype.fields or array.dtype.hasobject or array.dtype.subdtype:
        raise UnsupportedTensorStateError(
            "unsupported_dtype", f"state entry {name!r} has unsupported dtype {array.dtype}"
        )
    key = (array.dtype.kind, array.dtype.itemsize)
    token = {
        ("b", 1): "bool",
        ("i", 1): "int8", ("i", 2): "int16", ("i", 4): "int32", ("i", 8): "int64",
        ("u", 1): "uint8", ("u", 2): "uint16", ("u", 4): "uint32", ("u", 8): "uint64",
        ("f", 2): "float16", ("f", 4): "float32", ("f", 8): "float64",
        ("c", 8): "complex64", ("c", 16): "complex128",
    }.get(key)
    if token is None:
        raise UnsupportedTensorStateError(
            "unsupported_dtype", f"state entry {name!r} has unsupported dtype {array.dtype}"
        )
    canonical_dtype = np.dtype(_NUMPY_DTYPES[token])
    shape = tuple(int(value) for value in array.shape)
    # np.ascontiguousarray promotes a rank-zero array to shape (1,), so restore
    # the original shape before framing it. Scalar rank is hash-significant.
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False)).reshape(shape)
    return TensorEntry(str(name), token, shape, canonical.tobytes(order="C"))


def _torch_entry(name: str, value: Any) -> TensorEntry:
    if bool(getattr(value, "is_meta", False)) or str(getattr(value, "device", "")) == "meta":
        raise UnsupportedTensorStateError("meta_tensor", f"state entry {name!r} is a meta tensor")
    if bool(getattr(value, "is_quantized", False)):
        raise UnsupportedTensorStateError(
            "quantized_tensor", f"state entry {name!r} is quantized and needs a native full-object checkpoint"
        )
    layout = str(getattr(value, "layout", "torch.strided"))
    if layout not in {"torch.strided", "strided"}:
        raise UnsupportedTensorStateError(
            "non_strided_tensor", f"state entry {name!r} uses unsupported layout {layout}"
        )
    try:
        detached = value.detach().cpu().contiguous()
    except Exception as exc:
        raise UnsupportedTensorStateError(
            "tensor_materialization_failed", f"could not materialize state entry {name!r}: {exc}"
        ) from exc
    dtype_name = str(getattr(detached, "dtype", "")).removeprefix("torch.")
    if dtype_name == "bfloat16":
        try:
            import torch

            raw = detached.view(torch.int16).numpy().view(np.uint16)
            if sys.byteorder != "little":
                raw = raw.byteswap().view(raw.dtype.newbyteorder("<"))
            return TensorEntry(
                str(name), "bfloat16", tuple(int(value) for value in detached.shape), raw.tobytes(order="C")
            )
        except Exception as exc:
            raise UnsupportedTensorStateError(
                "bfloat16_materialization_failed", f"could not materialize bfloat16 entry {name!r}: {exc}"
            ) from exc
    try:
        return _numpy_entry(name, detached.numpy())
    except UnsupportedTensorStateError:
        raise
    except Exception as exc:
        raise UnsupportedTensorStateError(
            "tensor_materialization_failed", f"could not convert state entry {name!r}: {exc}"
        ) from exc


def _model_candidates(model: Any) -> list[Any]:
    candidates: list[Any] = []
    seen: set[int] = set()
    for candidate in (
        model,
        getattr(model, "core", None),
        getattr(getattr(model, "core", None), "model", None),
        getattr(model, "model", None),
    ):
        if candidate is not None and id(candidate) not in seen:
            candidates.append(candidate)
            seen.add(id(candidate))
    return candidates


@profiled("state_capture")
def capture_model_tensor_state(model: Any) -> TensorState:
    """Capture an ordered PyTorch/HF or Keras-style model state.

    Raises :class:`UnsupportedTensorStateError` with a serializable full-object
    fallback signal when exact raw tensor capture is unsafe.
    """
    for candidate in _model_candidates(model):
        state_dict = getattr(candidate, "state_dict", None)
        if not callable(state_dict):
            continue
        try:
            values = state_dict()
        except Exception as exc:
            raise UnsupportedTensorStateError(
                "state_dict_failed", f"model state_dict() failed: {exc}", model=candidate
            ) from exc
        if not isinstance(values, Mapping):
            raise UnsupportedTensorStateError(
                "state_dict_not_mapping", "model state_dict() did not return a mapping", model=candidate
            )
        entries = []
        for name, value in values.items():
            if callable(getattr(value, "detach", None)):
                entries.append(_torch_entry(str(name), value))
            else:
                entries.append(_numpy_entry(str(name), value))
        return TensorState(tuple(entries), framework="pytorch")

    for candidate in _model_candidates(model):
        get_weights = getattr(candidate, "get_weights", None)
        if not callable(get_weights):
            continue
        try:
            values = list(get_weights())
        except Exception as exc:
            raise UnsupportedTensorStateError(
                "get_weights_failed", f"model get_weights() failed: {exc}", model=candidate
            ) from exc
        variables = getattr(candidate, "weights", None)
        try:
            variables = list(variables) if variables is not None else []
        except Exception:
            variables = []
        names = [
            str(
                getattr(value, "path", None)
                or getattr(value, "name", None)
                or f"weight_{index}"
            )
            for index, value in enumerate(variables)
        ]
        if len(names) != len(values):
            names = [f"weight_{index}" for index in range(len(values))]
        return TensorState(
            tuple(_numpy_entry(names[index], value) for index, value in enumerate(values)),
            framework="keras",
        )
    raise UnsupportedTensorStateError(
        "no_tensor_state_interface",
        "model exposes neither a state_dict() mapping nor get_weights()",
        model=model,
    )


def capture_model_tensor_state_or_fallback(model: Any) -> tuple[TensorState | None, dict[str, Any] | None]:
    try:
        return capture_model_tensor_state(model), None
    except UnsupportedTensorStateError as exc:
        return None, exc.as_fallback_signal()


def _torch_tensor_from_entry(entry: TensorEntry, template: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise UnsupportedTensorStateError(
            "torch_unavailable", "PyTorch is required to restore this model state"
        ) from exc
    if entry.dtype == "bfloat16":
        words = np.frombuffer(entry.data, dtype=np.dtype("<u2")).reshape(entry.shape).copy()
        tensor = torch.from_numpy(words.view(np.int16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(entry.to_numpy())
    expected_dtype = str(getattr(template, "dtype", "")).removeprefix("torch.")
    if expected_dtype and expected_dtype != entry.dtype:
        raise IntegrityError(
            f"model tensor {entry.name!r} has dtype {expected_dtype}; checkpoint has {entry.dtype}"
        )
    return tensor


@profiled("model_restore_verify")
def restore_model_tensor_state(model: Any, state: TensorState, *, verify: bool = True) -> str:
    """Strictly restore a state and optionally recapture it for bitwise verification."""
    for candidate in _model_candidates(model):
        state_dict = getattr(candidate, "state_dict", None)
        load_state_dict = getattr(candidate, "load_state_dict", None)
        if not callable(state_dict) or not callable(load_state_dict):
            continue
        current = state_dict()
        if not isinstance(current, Mapping):
            continue
        if list(current.keys()) != [entry.name for entry in state.entries]:
            raise IntegrityError("model state keys/order do not match the checkpoint state")
        restored: dict[str, Any] = {}
        for entry in state.entries:
            template = current[entry.name]
            if tuple(int(value) for value in getattr(template, "shape", ())) != entry.shape:
                raise IntegrityError(f"model tensor {entry.name!r} shape does not match checkpoint")
            if callable(getattr(template, "detach", None)):
                restored[entry.name] = _torch_tensor_from_entry(entry, template)
            else:
                restored[entry.name] = entry.to_numpy()
        try:
            result = load_state_dict(restored, strict=True)
        except TypeError:
            result = load_state_dict(restored)
        except Exception as exc:
            raise IntegrityError(f"strict state restore failed: {exc}") from exc
        missing = list(getattr(result, "missing_keys", ()) or ())
        unexpected = list(getattr(result, "unexpected_keys", ()) or ())
        if missing or unexpected:
            raise IntegrityError(f"state restore was not strict: missing={missing}, unexpected={unexpected}")
        if verify:
            actual = capture_model_tensor_state(candidate)
            if actual.sha256 != state.sha256:
                raise IntegrityError(
                    f"model round-trip hash {actual.sha256} does not match checkpoint {state.sha256}"
                )
        return state.sha256

    for candidate in _model_candidates(model):
        set_weights = getattr(candidate, "set_weights", None)
        get_weights = getattr(candidate, "get_weights", None)
        if not callable(set_weights) or not callable(get_weights):
            continue
        current = capture_model_tensor_state(candidate)
        if [entry.descriptor() for entry in current.entries] != [entry.descriptor() for entry in state.entries]:
            raise IntegrityError("Keras variable names, dtypes, or shapes do not match the checkpoint")
        if any(entry.dtype == "bfloat16" for entry in state.entries):
            raise UnsupportedTensorStateError(
                "keras_bfloat16_restore_unsupported",
                "Keras bfloat16 restoration requires a framework-native full object",
                model=candidate,
            )
        set_weights([entry.to_numpy() for entry in state.entries])
        if verify:
            actual = capture_model_tensor_state(candidate)
            if actual.sha256 != state.sha256:
                raise IntegrityError(
                    f"model round-trip hash {actual.sha256} does not match checkpoint {state.sha256}"
                )
        return state.sha256
    raise UnsupportedTensorStateError(
        "state_restore_unsupported",
        "model has no strict load_state_dict() or set_weights() restoration interface",
        model=model,
    )


@profiled("content_hash")
def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@profiled("file_hash")
def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@profiled("compression")
def _compress(payload: bytes, compression: CompressionSpec) -> bytes:
    if compression.algorithm == "none":
        return payload
    if compression.algorithm == "zstd":
        import zstandard
        return zstandard.ZstdCompressor(level=int(compression.level)).compress(payload)
    return zlib.compress(payload, level=int(compression.level))


@profiled("decompression")
def _decompress(payload: bytes, compression: CompressionSpec) -> bytes:
    if compression.algorithm == "none":
        return payload
    if compression.algorithm == "zstd":
        import zstandard
        try:
            return zstandard.ZstdDecompressor().decompress(payload)
        except zstandard.ZstdError as exc:
            raise IntegrityError(f"invalid zstd content object: {exc}") from exc
    try:
        return zlib.decompress(payload)
    except zlib.error as exc:
        raise IntegrityError(f"invalid zlib content object: {exc}") from exc


@profiled("publish")
def _publish_bytes_once(path: Path, payload: bytes) -> bool:
    """Publish immutable bytes without exposing a partially written target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ManifestConflictError(f"immutable path already contains different bytes: {path}")
        return False
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
            created = True
        except FileExistsError:
            created = False
        except OSError:
            try:
                exclusive = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                created = False
            else:
                with os.fdopen(exclusive, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                created = True
        if path.read_bytes() != payload:
            raise ManifestConflictError(f"immutable path publication conflict: {path}")
        return created
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


class ContentAddressedTensorStore:
    """Immutable SHA-256 store for canonical state and patch byte streams."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.object_root = self.root / "objects" / "sha256"

    @staticmethod
    def _compression(value: CompressionSpec | str) -> CompressionSpec:
        return value if isinstance(value, CompressionSpec) else CompressionSpec(str(value))

    @profiled("cas_candidate")
    def _candidate(
        self,
        payload: bytes,
        *,
        object_kind: str,
        semantic_format: str,
        compression: CompressionSpec,
        encoded_payload: bytes | None = None,
    ) -> tuple[dict[str, Any], bytes, str, Path, Path]:
        digest = _hash_bytes(payload)
        encoded = (
            _compress(payload, compression)
            if encoded_payload is None
            else encoded_payload
        )
        suffix = (
            "none"
            if compression.algorithm == "none"
            else f"{compression.algorithm}-l{compression.level}"
        )
        folder = self.object_root / digest[:2]
        payload_path = folder / f"{digest}.{suffix}.bin"
        manifest_path = folder / f"{digest}.{suffix}.json"
        manifest = {
            "cas_format": CAS_FORMAT,
            "object_kind": object_kind,
            "semantic_format": semantic_format,
            "content_sha256": digest,
            "logical_size_bytes": len(payload),
            "encoding": {"compression": compression.as_dict()},
            "stored_payload": payload_path.name,
            "stored_payload_sha256": _hash_bytes(encoded),
            "stored_payload_size_bytes": len(encoded),
        }
        return manifest, encoded, digest, manifest_path, payload_path

    @profiled("full_baseline")
    def measure_candidate(
        self,
        payload: bytes,
        *,
        object_kind: str,
        semantic_format: str,
        compression: CompressionSpec | str = CompressionSpec(),
    ) -> dict[str, Any]:
        codec = self._compression(compression)
        manifest, encoded, digest, _, _ = self._candidate(
            payload, object_kind=object_kind, semantic_format=semantic_format, compression=codec
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        return {
            "content_sha256": digest,
            "logical_size_bytes": len(payload),
            "stored_payload_size_bytes": len(encoded),
            "manifest_size_bytes": len(manifest_bytes),
            "physical_file_size_bytes": len(encoded) + len(manifest_bytes),
            "compression": codec.as_dict(),
        }

    def put(
        self, payload: bytes, *, object_kind: str, semantic_format: str,
        compression: CompressionSpec | str = CompressionSpec(),
        _encoded_payload: bytes | None = None,
    ) -> ContentObjectRef:
        return self.put_verified(payload, object_kind=object_kind,
            semantic_format=semantic_format, compression=compression,
            _encoded_payload=_encoded_payload)[0]

    @profiled("cas_put")
    def put_verified(
        self,
        payload: bytes,
        *,
        object_kind: str,
        semantic_format: str,
        compression: CompressionSpec | str = CompressionSpec(),
        _encoded_payload: bytes | None = None,
    ) -> tuple[ContentObjectRef, bytes]:
        codec = self._compression(compression)
        manifest, encoded, digest, manifest_path, payload_path = self._candidate(
            payload,
            object_kind=object_kind,
            semantic_format=semantic_format,
            compression=codec,
            encoded_payload=_encoded_payload,
        )
        from .storage_overflow import choose_storage_root

        # Existing objects retain their original paths. Route the payload and
        # its manifest together so relative references remain valid.
        if not (payload_path.exists() and manifest_path.exists()):
            selected_root = choose_storage_root(
                self.root, len(encoded) + len(_canonical_json_bytes(manifest))
            )
            folder = selected_root / "objects" / "sha256" / digest[:2]
            payload_path = folder / payload_path.name
            manifest_path = folder / manifest_path.name
        _publish_bytes_once(payload_path, encoded)
        _publish_bytes_once(manifest_path, _canonical_json_bytes(manifest))
        ref = ContentObjectRef(
            object_kind=object_kind,
            semantic_format=semantic_format,
            content_sha256=digest,
            compression=codec,
            logical_size_bytes=len(payload),
            stored_payload_size_bytes=len(encoded),
            payload_sha256=manifest["stored_payload_sha256"],
            manifest_path=manifest_path,
            payload_path=payload_path,
        )
        return ref, self.read(ref)

    def put_state(
        self,
        state: TensorState,
        *,
        compression: CompressionSpec | str = CompressionSpec(),
    ) -> ContentObjectRef:
        return self.put(
            serialize_tensor_state(state),
            object_kind="tensor_state",
            semantic_format=TENSOR_STATE_FORMAT,
            compression=compression,
        )

    @profiled("patch_write")
    def put_patch(
        self,
        patch: TensorPatch,
        *,
        compression: CompressionSpec | str = CompressionSpec(),
        _encoded_payload: bytes | None = None,
    ) -> ContentObjectRef:
        return self.put(
            serialize_tensor_patch(patch),
            object_kind="tensor_patch",
            semantic_format=TENSOR_PATCH_FORMAT,
            compression=compression,
            _encoded_payload=_encoded_payload,
        )

    @profiled("cas_read_verify")
    def read(self, ref: ContentObjectRef) -> bytes:
        document = json.loads(ref.manifest_path.read_text(encoding="utf-8"))
        if document.get("cas_format") != CAS_FORMAT:
            raise IntegrityError(f"unsupported content-object manifest: {ref.manifest_path}")
        encoded = ref.payload_path.read_bytes()
        if _hash_bytes(encoded) != document.get("stored_payload_sha256"):
            raise IntegrityError(f"stored payload hash mismatch: {ref.payload_path}")
        if len(encoded) != int(document.get("stored_payload_size_bytes", -1)):
            raise IntegrityError(f"stored payload size mismatch: {ref.payload_path}")
        payload = _decompress(encoded, ref.compression)
        if len(payload) != int(document.get("logical_size_bytes", -1)):
            raise IntegrityError(f"logical payload size mismatch: {ref.payload_path}")
        if _hash_bytes(payload) != ref.content_sha256:
            raise IntegrityError(f"content hash mismatch: {ref.payload_path}")
        return payload

    def ref_from_link(self, link: Mapping[str, Any]) -> ContentObjectRef:
        return self.read_link(link)[0]

    @profiled("cas_resolve_verify")
    def read_link(self, link: Mapping[str, Any]) -> tuple[ContentObjectRef, bytes]:
        """Resolve and verify once, returning the already decoded payload."""
        ref = self.inspect_link(link)
        return ref, self.read(ref)

    def inspect_link(self, link: Mapping[str, Any]) -> ContentObjectRef:
        """Validate reference metadata without reading or decoding its payload.

        Callers must separately verify payload integrity; this is not a read-back
        check of the canonical tensor state.
        """
        relative = Path(str(link["object_manifest"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise IntegrityError("content-object link escapes its store")
        manifest_path = (self.root / relative).resolve()
        if self.root not in manifest_path.parents:
            raise IntegrityError("content-object link escapes its store")
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("cas_format") != CAS_FORMAT:
            raise IntegrityError(f"unsupported content-object manifest: {manifest_path}")
        compression_data = document.get("encoding", {}).get("compression", {})
        compression = CompressionSpec(
            compression_data.get("algorithm", "none"), compression_data.get("level")
        )
        payload_path = manifest_path.parent / str(document["stored_payload"])
        ref = ContentObjectRef(
            object_kind=str(document["object_kind"]),
            semantic_format=str(document["semantic_format"]),
            content_sha256=str(document["content_sha256"]),
            compression=compression,
            logical_size_bytes=int(document["logical_size_bytes"]),
            stored_payload_size_bytes=int(document["stored_payload_size_bytes"]),
            payload_sha256=str(document["stored_payload_sha256"]),
            manifest_path=manifest_path,
            payload_path=payload_path,
        )
        if link.get("content_sha256") != ref.content_sha256:
            raise IntegrityError("content-object link hash does not match its manifest")
        if link.get("object_kind") != ref.object_kind or link.get("semantic_format") != ref.semantic_format:
            raise IntegrityError("content-object link type does not match its manifest")
        return ref


def _identity_manifest_path(root: Path, namespace: str, identity: str) -> Path:
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
    return root / "references" / namespace / digest[:2] / f"{digest}.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"manifest is not a JSON object: {path}")
    return value


def _validate_schema_hash(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("parameter_schema_hash must be a SHA-256 hex string or None")
    return value


@profiled("base_load_verify")
def _load_base_document(
    root: Path, lineage_id: str
) -> tuple[dict[str, Any], Path, ContentAddressedTensorStore, ContentObjectRef, TensorState]:
    manifest_path = _identity_manifest_path(root, "lineages", lineage_id)
    document = _read_json(manifest_path)
    if document.get("format") != LINEAGE_BASE_MANIFEST_FORMAT or document.get("lineage_id") != str(lineage_id):
        raise IntegrityError(f"invalid lineage base manifest: {manifest_path}")
    store = ContentAddressedTensorStore(root)
    ref, payload = store.read_link(document["base_state"])
    if ref.object_kind != "tensor_state" or ref.semantic_format != TENSOR_STATE_FORMAT:
        raise IntegrityError("lineage base does not reference a canonical tensor state")
    state = deserialize_tensor_state(payload, framework=str(document.get("framework", "unknown")))
    if state.sha256 != document.get("tensor_state_sha256"):
        raise IntegrityError("lineage base tensor-state hash mismatch")
    return document, manifest_path, store, ref, state


def ensure_lineage_base(
    store_root: str | Path,
    lineage_id: str,
    model: Any,
    *,
    parameter_schema_hash: str | None,
    compression: CompressionSpec | str = CompressionSpec(),
    restore_if_existing: bool = True,
) -> LineageBaseResult:
    """Publish a lineage's sole canonical full state, or restore the winner.

    Publication uses an atomic create-if-absent manifest.  Concurrent creators
    can write harmless unreferenced CAS candidates, but exactly one immutable
    lineage manifest wins; losing processes restore that winner before
    returning when ``restore_if_existing`` is true.
    """
    root = Path(store_root).resolve()
    lineage_id = str(lineage_id)
    schema_hash = _validate_schema_hash(parameter_schema_hash)
    manifest_path = _identity_manifest_path(root, "lineages", lineage_id)
    if manifest_path.exists():
        document, _, _, ref, state = _load_base_document(root, lineage_id)
        if document.get("parameter_schema_hash") != schema_hash:
            raise ParameterSchemaMismatchError(
                "lineage base parameter schema differs from the constructed model schema"
            )
        restored = False
        if restore_if_existing:
            restore_model_tensor_state(model, state, verify=True)
            restored = True
        return LineageBaseResult(
            lineage_id, manifest_path, ref, state.sha256, schema_hash, False, restored
        )

    state = capture_model_tensor_state(model)
    # Capture alone is insufficient: a get_weights-only estimator cannot be
    # reconstructed from the canonical archive and must use full_object_v1.
    restore_model_tensor_state(model, state, verify=True)
    store = ContentAddressedTensorStore(root)
    ref = store.put_state(state, compression=compression)
    document = {
        "format": LINEAGE_BASE_MANIFEST_FORMAT,
        "lineage_id": lineage_id,
        "tensor_state_format": TENSOR_STATE_FORMAT,
        "tensor_state_sha256": state.sha256,
        "framework": state.framework,
        "parameter_schema": {
            "hash": schema_hash,
            "ownership": "separate_generator_provenance",
        },
        "parameter_schema_hash": schema_hash,
        "base_state": ref.link(root),
    }
    try:
        created = _publish_bytes_once(manifest_path, _canonical_json_bytes(document))
    except ManifestConflictError:
        # A concurrent creator can legitimately win with a different random
        # initialization. Its published lineage base is authoritative.
        if not manifest_path.exists():
            raise
        created = False
    if created:
        return LineageBaseResult(lineage_id, manifest_path, ref, state.sha256, schema_hash, True, False)

    winner, _, _, winner_ref, winner_state = _load_base_document(root, lineage_id)
    if winner.get("parameter_schema_hash") != schema_hash:
        raise ParameterSchemaMismatchError("concurrently published lineage uses another parameter schema")
    restored = False
    if restore_if_existing:
        restore_model_tensor_state(model, winner_state, verify=True)
        restored = True
    return LineageBaseResult(
        lineage_id, manifest_path, winner_ref, winner_state.sha256, schema_hash, False, restored
    )


def load_lineage_base(
    store_root: str | Path,
    lineage_id: str,
    *,
    expected_parameter_schema_hash: str | None = None,
) -> TensorState:
    root = Path(store_root).resolve()
    document, _, _, _, state = _load_base_document(root, str(lineage_id))
    expected = _validate_schema_hash(expected_parameter_schema_hash)
    if expected is not None and document.get("parameter_schema_hash") != expected:
        raise ParameterSchemaMismatchError("lineage base parameter schema hash does not match")
    return state


def restore_lineage_base(
    store_root: str | Path,
    lineage_id: str,
    model: Any,
    *,
    expected_parameter_schema_hash: str | None = None,
) -> str:
    state = load_lineage_base(
        store_root,
        lineage_id,
        expected_parameter_schema_hash=expected_parameter_schema_hash,
    )
    return restore_model_tensor_state(model, state, verify=True)


@profiled("state_compare")
def _same_tensor_state_bytes(first: TensorState, second: TensorState) -> bool:
    # Entries already contain immutable canonical descriptors and raw bytes.
    # Compare every byte (including NaN payloads and signed zero), without
    # allocating two additional serialized copies of the complete model.
    return first.entries == second.entries


def _storage_measurements(store, target, base_ref, patch_ref, codec, *, measure_full):
    # Canonical framing length is exact without allocating or compressing a full state.
    logical = len(_STATE_MAGIC) + 8 + sum(
        len(_encode_descriptor(e.name, e.dtype, e.shape)) + 8 + len(e.data)
        for e in target.entries
    )
    full = store.measure_candidate(serialize_tensor_state(target), object_kind="tensor_state",
        semantic_format=TENSOR_STATE_FORMAT, compression=codec) if measure_full else None
    patch_bytes = patch_ref.physical_size_bytes if patch_ref else 0
    full_bytes = int(full["physical_file_size_bytes"]) if full else None
    saved = full_bytes - patch_bytes if full_bytes is not None else None
    result = {
        "measurement_version": 1 if measure_full else 2,
        "scope": "incremental CAS payload and object-manifest file lengths; reference manifests excluded symmetrically",
        "full_baseline_kind": "canonical_full_tensor_state_same_cas_envelope" if measure_full else "not_measured",
        "compression": codec.as_dict(),
        "base_shared_physical_size_bytes": base_ref.physical_size_bytes,
        "patch_logical_size_bytes": patch_ref.logical_size_bytes if patch_ref else 0,
        "patch_stored_payload_size_bytes": patch_ref.stored_payload_size_bytes if patch_ref else 0,
        "patch_physical_size_bytes": patch_bytes,
        "full_state_baseline_logical_size_bytes": logical,
        "full_state_baseline_stored_payload_size_bytes": full["stored_payload_size_bytes"] if full else None,
        "full_state_baseline_physical_size_bytes": full_bytes,
        "incremental_physical_bytes_saved": saved,
        "incremental_savings_ratio": (saved / full_bytes if full_bytes else 0.0) if measure_full else None,
        "patch_saves_storage": saved > 0 if saved is not None else None,
        "shared_base_plus_patch_physical_size_bytes": base_ref.physical_size_bytes + patch_bytes,
        "base_plus_full_target_physical_size_bytes": base_ref.physical_size_bytes + full_bytes if full_bytes is not None else None,
    }
    return result


@profiled("patch_persist")
def persist_service_patch(
    store_root: str | Path,
    service_id: str,
    lineage_id: str,
    model: Any,
    *,
    parameter_schema_hash: str | None,
    compression: CompressionSpec | str = CompressionSpec(),
    operation_policy: str = "adaptive",
    measure_full: bool = True,
) -> ServicePatchResult:
    """Persist and verify a direct service patch against its lineage base.

    The returned measurement compares actual patch-object file lengths with a
    full target-state object encoded in memory by the exact same CAS envelope
    and compression codec.  Reference manifests are excluded symmetrically.
    """
    root = Path(store_root).resolve()
    service_id = str(service_id)
    lineage_id = str(lineage_id)
    schema_hash = _validate_schema_hash(parameter_schema_hash)
    base_document, _, store, base_ref, base_state = _load_base_document(root, lineage_id)
    if base_document.get("parameter_schema_hash") != schema_hash:
        raise ParameterSchemaMismatchError("service parameter schema differs from its lineage base")

    target = capture_model_tensor_state(model)
    codec = store._compression(compression)
    patch, operation_selection, selected_encoded_payload = _select_tensor_patch_for_compression(
        base_state, target, compression=codec, operation_policy=operation_policy
    )
    reconstructed = apply_tensor_patch(base_state, patch)
    if not _same_tensor_state_bytes(reconstructed, target):
        raise IntegrityError("patch reconstruction is not byte-identical to the target state")
    restore_model_tensor_state(model, reconstructed, verify=True)

    patch_ref: ContentObjectRef | None = None
    if patch.operations:
        patch_ref, loaded_payload = store.put_verified(
            serialize_tensor_patch(patch),
            object_kind="tensor_patch", semantic_format=TENSOR_PATCH_FORMAT,
            compression=codec,
            _encoded_payload=selected_encoded_payload,
        )
        loaded_patch = deserialize_tensor_patch(loaded_payload)
        loaded_reconstruction = apply_tensor_patch(base_state, loaded_patch)
        if loaded_reconstruction.sha256 != target.sha256:
            raise IntegrityError("persisted patch failed strict read-back verification")
    elif target.sha256 != base_state.sha256:
        raise IntegrityError("an empty patch cannot represent a changed target state")

    measurements = _storage_measurements(store, target, base_ref, patch_ref, codec, measure_full=measure_full)
    service_manifest = {
        "format": SERVICE_PATCH_MANIFEST_FORMAT,
        "policy": "shared_base_exact_patch_v1",
        "service_id": service_id,
        "lineage_id": lineage_id,
        "tensor_state_format": TENSOR_STATE_FORMAT,
        "patch_semantic_format": TENSOR_PATCH_FORMAT,
        "base_state_sha256": base_state.sha256,
        "target_state_sha256": target.sha256,
        "parameter_schema": {
            "hash": schema_hash,
            "ownership": "separate_generator_provenance",
        },
        "parameter_schema_hash": schema_hash,
        "base_state": base_ref.link(root),
        "patch": patch_ref.link(root) if patch_ref else None,
        "patch_mode": "exact_patch" if patch_ref else "base_reference_only",
        "storage_encoding": {"compression": codec.as_dict()},
        "operation_counts": patch.operation_counts,
        "operation_selection": operation_selection,
        "storage_measurements": measurements,
        "round_trip_verified": True,
    }
    manifest_path = _identity_manifest_path(root, "services", service_id)
    _publish_bytes_once(manifest_path, _canonical_json_bytes(service_manifest))
    return ServicePatchResult(
        service_id=service_id,
        lineage_id=lineage_id,
        manifest_path=manifest_path,
        base=base_ref,
        patch=patch_ref,
        target_state_sha256=target.sha256,
        parameter_schema_hash=schema_hash,
        measurements=measurements,
        operation_counts=patch.operation_counts,
        operation_selection=operation_selection,
    )


@profiled("patch_persist")
def persist_service_patch_from_base_state(
    store_root: str | Path,
    service_id: str,
    lineage_id: str,
    model: Any,
    *,
    base_state: TensorState,
    base_ref: ContentObjectRef,
    base_round_id: int,
    parameter_schema_hash: str | None,
    compression: CompressionSpec | str = CompressionSpec(),
    operation_policy: str = "adaptive",
    measure_full: bool = True,
) -> ServicePatchResult:
    """Persist a direct client-round patch against an immutable full global state."""
    root = Path(store_root).resolve()
    service_id = str(service_id)
    lineage_id = str(lineage_id)
    schema_hash = _validate_schema_hash(parameter_schema_hash)
    store = ContentAddressedTensorStore(root)
    if base_ref.object_kind != "tensor_state" or base_ref.content_sha256 != base_state.sha256:
        raise IntegrityError("FedAvg round base reference does not match its canonical state")

    target = capture_model_tensor_state(model)
    if [entry.descriptor() for entry in target.entries] != [
        entry.descriptor() for entry in base_state.entries
    ]:
        raise ParameterSchemaMismatchError("client-round tensor schema differs from its global parent")
    codec = store._compression(compression)
    patch, operation_selection, selected_encoded_payload = _select_tensor_patch_for_compression(
        base_state, target, compression=codec, operation_policy=operation_policy
    )
    reconstructed = apply_tensor_patch(base_state, patch)
    if not _same_tensor_state_bytes(reconstructed, target):
        raise IntegrityError("client-round patch is not byte-identical to its target state")
    restore_model_tensor_state(model, reconstructed, verify=True)

    patch_ref: ContentObjectRef | None = None
    if patch.operations:
        patch_ref, loaded_payload = store.put_verified(
            serialize_tensor_patch(patch),
            object_kind="tensor_patch", semantic_format=TENSOR_PATCH_FORMAT,
            compression=codec,
            _encoded_payload=selected_encoded_payload,
        )
        loaded_patch = deserialize_tensor_patch(loaded_payload)
        if apply_tensor_patch(base_state, loaded_patch).sha256 != target.sha256:
            raise IntegrityError("persisted client-round patch failed strict read-back verification")
    elif target.sha256 != base_state.sha256:
        raise IntegrityError("an empty client-round patch cannot represent a changed target")

    measurements = _storage_measurements(store, target, base_ref, patch_ref, codec, measure_full=measure_full)
    service_manifest = {
        "format": SERVICE_PATCH_MANIFEST_FORMAT,
        "policy": "fedavg_round_base_exact_patch_v1",
        "service_id": service_id,
        "lineage_id": lineage_id,
        "base_round_id": int(base_round_id),
        "tensor_state_format": TENSOR_STATE_FORMAT,
        "patch_semantic_format": TENSOR_PATCH_FORMAT,
        "base_state_sha256": base_state.sha256,
        "target_state_sha256": target.sha256,
        "parameter_schema": {"hash": schema_hash, "ownership": "separate_generator_provenance"},
        "parameter_schema_hash": schema_hash,
        "base_state": base_ref.link(root),
        "patch": patch_ref.link(root) if patch_ref else None,
        "patch_mode": "exact_patch" if patch_ref else "base_reference_only",
        "storage_encoding": {"compression": codec.as_dict()},
        "operation_counts": patch.operation_counts,
        "operation_selection": operation_selection,
        "storage_measurements": measurements,
        "round_trip_verified": True,
    }
    manifest_path = _identity_manifest_path(root, "services", service_id)
    _publish_bytes_once(manifest_path, _canonical_json_bytes(service_manifest))
    return ServicePatchResult(
        service_id=service_id,
        lineage_id=lineage_id,
        manifest_path=manifest_path,
        base=base_ref,
        patch=patch_ref,
        target_state_sha256=target.sha256,
        parameter_schema_hash=schema_hash,
        measurements=measurements,
        operation_counts=patch.operation_counts,
        operation_selection=operation_selection,
    )


def reconstruct_service_state(store_root: str | Path, service_id: str) -> TensorState:
    """Load a service manifest, verify both objects, and reconstruct its state."""
    root = Path(store_root).resolve()
    manifest_path = _identity_manifest_path(root, "services", str(service_id))
    document = _read_json(manifest_path)
    from .reduced_checkpoint import FORMAT as REDUCED_FORMAT, load_reduced_state
    if document.get("format") == REDUCED_FORMAT:
        if document.get("service_id") != str(service_id):
            raise IntegrityError("reduced checkpoint service identity mismatch")
        return load_reduced_state(document)
    if document.get("format") != SERVICE_PATCH_MANIFEST_FORMAT or document.get("service_id") != str(service_id):
        raise IntegrityError(f"invalid service patch manifest: {manifest_path}")
    store = ContentAddressedTensorStore(root)
    base_ref, base_payload = store.read_link(document["base_state"])
    base = deserialize_tensor_state(base_payload)
    if document.get("patch") is None:
        result = base
    else:
        patch_ref, patch_payload = store.read_link(document["patch"])
        patch = deserialize_tensor_patch(patch_payload)
        result = apply_tensor_patch(base, patch)
    if result.sha256 != document.get("target_state_sha256"):
        raise IntegrityError("service manifest target-state hash mismatch")
    return result


def reconstruct_service_base_state(
    store_root: str | Path, service_id: str
) -> TensorState:
    """Load and verify the immutable parent state named by a service manifest."""
    root = Path(store_root).resolve()
    manifest_path = _identity_manifest_path(root, "services", str(service_id))
    document = _read_json(manifest_path)
    from .reduced_checkpoint import FORMAT as REDUCED_FORMAT
    if (
        document.get("format") not in {SERVICE_PATCH_MANIFEST_FORMAT, REDUCED_FORMAT}
        or document.get("service_id") != str(service_id)
    ):
        raise IntegrityError(f"invalid service patch manifest: {manifest_path}")
    store = ContentAddressedTensorStore(root)
    base_ref, base_payload = store.read_link(document["base_state"])
    base = deserialize_tensor_state(base_payload)
    if base.sha256 != document.get("base_state_sha256"):
        raise IntegrityError("service manifest base-state hash mismatch")
    return base


def restore_service_state(store_root: str | Path, service_id: str, model: Any) -> str:
    return restore_model_tensor_state(
        model,
        reconstruct_service_state(store_root, service_id),
        verify=True,
    )


__all__ = [
    "CANONICAL_TENSOR_STATE_SPEC",
    "CAS_FORMAT",
    "LINEAGE_BASE_MANIFEST_FORMAT",
    "SERVICE_PATCH_MANIFEST_FORMAT",
    "TENSOR_PATCH_FORMAT",
    "TENSOR_STATE_FORMAT",
    "CheckpointStoreError",
    "CompressionSpec",
    "ContentAddressedTensorStore",
    "ContentObjectRef",
    "IntegrityError",
    "LineageBaseResult",
    "ManifestConflictError",
    "ParameterSchemaMismatchError",
    "PatchOperation",
    "ServicePatchResult",
    "TensorEntry",
    "TensorPatch",
    "TensorState",
    "UnsupportedTensorStateError",
    "apply_tensor_patch",
    "build_tensor_patch",
    "canonical_tensor_state_hash",
    "capture_model_tensor_state",
    "capture_model_tensor_state_or_fallback",
    "deserialize_tensor_patch",
    "deserialize_tensor_state",
    "ensure_lineage_base",
    "load_lineage_base",
    "persist_service_patch",
    "persist_service_patch_from_base_state",
    "reconstruct_service_base_state",
    "reconstruct_service_state",
    "restore_lineage_base",
    "restore_model_tensor_state",
    "restore_service_state",
    "select_tensor_patch_for_compression",
    "serialize_tensor_patch",
    "serialize_tensor_state",
]
