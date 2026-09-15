import hashlib
import json
import struct

import numpy as np
import pytest

from mlaas_data_generator import checkpoint_store as checkpoint_store_module
from mlaas_data_generator.checkpoint_store import (
    CompressionSpec,
    ContentAddressedTensorStore,
    IntegrityError,
    ParameterSchemaMismatchError,
    TensorEntry,
    TensorState,
    UnsupportedTensorStateError,
    apply_tensor_patch,
    build_tensor_patch,
    capture_model_tensor_state,
    capture_model_tensor_state_or_fallback,
    deserialize_tensor_patch,
    deserialize_tensor_state,
    ensure_lineage_base,
    load_lineage_base,
    persist_service_patch,
    reconstruct_service_state,
    restore_lineage_base,
    restore_service_state,
    select_tensor_patch_for_compression,
    serialize_tensor_patch,
    serialize_tensor_state,
)


SCHEMA_HASH = "a" * 64


class Variable:
    def __init__(self, name, path=None):
        self.name = name
        self.path = path


class ArrayModel:
    def __init__(self, values, names=None):
        self._values = [np.asarray(value).copy() for value in values]
        names = names or [f"weight_{index}" for index in range(len(values))]
        self.weights = [Variable(name) for name in names]

    def get_weights(self):
        return [value.copy() for value in self._values]

    def set_weights(self, values):
        if len(values) != len(self._values):
            raise ValueError("wrong number of weights")
        for current, value in zip(self._values, values, strict=True):
            value = np.asarray(value)
            if current.shape != value.shape or current.dtype != value.dtype:
                raise ValueError("weight layout mismatch")
        self._values = [np.asarray(value).copy() for value in values]


class ReadOnlyArrayModel:
    def get_weights(self):
        return [np.ones((2,), dtype=np.float32)]


def _state(*entries):
    return TensorState(tuple(entries), framework="test")


def test_canonical_tensor_state_has_a_golden_hash_and_round_trips():
    state = _state(
        TensorEntry("layer.weight", "float32", (2,), struct.pack("<ff", 1.0, -0.0)),
        TensorEntry("step", "int16", (), struct.pack("<h", -7)),
    )
    payload = serialize_tensor_state(state)

    assert payload.startswith(b"MLAAS-TENSOR-STATE-V1\0")
    assert hashlib.sha256(payload).hexdigest() == "4120ef4f7129a9ccc5beabead5dc29e3a775ef8c9ec8c5255cacf997bd6a3569"
    assert deserialize_tensor_state(payload).entries == state.entries


def test_state_hash_normalizes_byte_order_and_memory_layout():
    little = np.array([[1.25, -2.5], [3.75, 4.0]], dtype="<f4")
    big = little.astype(">f4")
    noncontiguous = np.asfortranarray(little)

    first = capture_model_tensor_state(ArrayModel([little], ["value"]))
    second = capture_model_tensor_state(ArrayModel([big], ["value"]))
    third = capture_model_tensor_state(ArrayModel([noncontiguous], ["value"]))

    assert first.sha256 == second.sha256 == third.sha256


def test_keras_variable_paths_disambiguate_repeated_leaf_names():
    model = ArrayModel(
        [np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)],
        ["kernel", "kernel"],
    )
    model.weights = [
        Variable("kernel", "encoder/kernel"),
        Variable("kernel", "head/kernel"),
    ]

    state = capture_model_tensor_state(model)

    assert [entry.name for entry in state.entries] == ["encoder/kernel", "head/kernel"]


def test_capture_preserves_rank_zero_tensor_shape():
    state = capture_model_tensor_state(ArrayModel([np.array(7, dtype=np.int64)], ["step"]))

    assert state.entries[0].shape == ()
    assert state.entries[0].to_numpy().shape == ()


def test_sparse_patch_omits_unchanged_tensors_and_round_trips_exact_bits():
    base = _state(
        TensorEntry("same", "uint8", (4,), b"same"),
        TensorEntry("changed", "uint8", (4,), b"base"),
    )
    target = _state(
        TensorEntry("same", "uint8", (4,), b"same"),
        TensorEntry("changed", "uint8", (4,), b"targ"),
    )

    patch = build_tensor_patch(base, target)

    assert len(patch.operations) == 1
    assert patch.operations[0].index == 1
    assert patch.operations[0].operation == "xor"
    assert patch.operation_counts == {"inherit": 1, "xor": 1, "replace": 0}
    assert serialize_tensor_state(apply_tensor_patch(base, patch)) == serialize_tensor_state(target)
    assert deserialize_tensor_patch(serialize_tensor_patch(patch)) == patch


def test_patch_compression_is_an_envelope_and_not_patch_semantics(tmp_path):
    base = _state(TensorEntry("weight", "uint8", (4096,), bytes(range(256)) * 16))
    target_bytes = bytearray(base.entries[0].data)
    target_bytes[100] ^= 1
    target = _state(TensorEntry("weight", "uint8", (4096,), bytes(target_bytes)))
    patch = build_tensor_patch(base, target)
    store = ContentAddressedTensorStore(tmp_path)

    plain = store.put_patch(patch, compression="none")
    compressed = store.put_patch(patch, compression="zlib")

    assert plain.content_sha256 == compressed.content_sha256 == patch.sha256
    assert plain.compression.algorithm == "none"
    assert compressed.compression.algorithm == "zlib"
    assert plain.payload_sha256 != compressed.payload_sha256
    assert deserialize_tensor_patch(store.read(plain)) == deserialize_tensor_patch(store.read(compressed))


def test_same_content_can_use_multiple_compression_levels(tmp_path):
    state = _state(TensorEntry("weight", "uint8", (4096,), b"a" * 4096))
    store = ContentAddressedTensorStore(tmp_path)

    fast = store.put_state(state, compression=CompressionSpec("zlib", 1))
    dense = store.put_state(state, compression=CompressionSpec("zlib", 9))

    assert fast.content_sha256 == dense.content_sha256 == state.sha256
    assert fast.payload_path != dense.payload_path
    assert fast.manifest_path != dense.manifest_path
    assert deserialize_tensor_state(store.read(fast)).sha256 == state.sha256
    assert deserialize_tensor_state(store.read(dense)).sha256 == state.sha256


def test_codec_aware_selection_records_decision_separately():
    rng = np.random.default_rng(91)
    base_bytes = rng.integers(0, 256, size=16384, dtype=np.uint8).tobytes()
    target_bytes = b"\0" * len(base_bytes)
    base = _state(TensorEntry("weight", "uint8", (len(base_bytes),), base_bytes))
    target = _state(TensorEntry("weight", "uint8", (len(target_bytes),), target_bytes))

    patch, selection = select_tensor_patch_for_compression(base, target, compression="zlib")

    assert patch.operations[0].operation == "replace"
    assert selection["selected"] in {"replace", "per_tensor_hybrid"}
    assert selection["evaluated_compression"] == {"algorithm": "zlib", "level": 1}
    assert b"zlib" not in serialize_tensor_patch(patch)
    assert apply_tensor_patch(base, patch).sha256 == target.sha256


@pytest.mark.parametrize("policy,operation", [("fixed_xor", "xor"), ("fixed_replace", "replace")])
@pytest.mark.parametrize("codec", ["none", "zlib"])
def test_fixed_policy_skips_candidate_compression_and_preserves_exact_bits(monkeypatch, policy, operation, codec):
    # Include distinct NaN payloads and signed zero: numeric equality is insufficient.
    first = np.asarray([0x7FC00001, 0x80000000, 0x3F800000], dtype="<u4").tobytes()
    second = np.asarray([0x7FC00002, 0, 0x3F800000], dtype="<u4").tobytes()
    base = _state(TensorEntry("bits", "float32", (3,), first))
    target = _state(TensorEntry("bits", "float32", (3,), second))

    def unexpected(*args, **kwargs):
        raise AssertionError("fixed policy evaluated a compression candidate")

    monkeypatch.setattr(checkpoint_store_module, "_compress", unexpected)
    patch, selection = select_tensor_patch_for_compression(base, target, compression=codec, operation_policy=policy)
    assert [item.operation for item in patch.operations] == [operation]
    assert selection["policy"] == policy
    assert selection["selection_version"] == 3
    assert selection["candidate_comparison_performed"] is False
    assert apply_tensor_patch(base, patch).entries == target.entries
    assert checkpoint_store_module._same_tensor_state_bytes(base, target) is False
    assert checkpoint_store_module._same_tensor_state_bytes(apply_tensor_patch(base, patch), target)


@pytest.mark.parametrize("policy", ["fixed_xor", "fixed_replace"])
@pytest.mark.parametrize("round_base", [False, True])
def test_fixed_persistence_keeps_verification_and_measured_storage(tmp_path, policy, round_base):
    model = ArrayModel([np.zeros(1024, dtype=np.float32)])
    lineage = ensure_lineage_base(tmp_path, "lineage", model, parameter_schema_hash=SCHEMA_HASH)
    base = capture_model_tensor_state(model)
    model._values[0][0] = 1.0
    kwargs = {"parameter_schema_hash": SCHEMA_HASH, "operation_policy": policy}
    function = persist_service_patch
    if round_base:
        function = checkpoint_store_module.persist_service_patch_from_base_state
        kwargs.update(base_state=base, base_ref=lineage.base, base_round_id=0)
    result = function(tmp_path, "fixed", "lineage", model, **kwargs)
    document = json.loads(result.manifest_path.read_text())
    assert document["operation_selection"]["policy"] == policy
    assert document["round_trip_verified"] is True
    assert reconstruct_service_state(tmp_path, "fixed").sha256 == capture_model_tensor_state(model).sha256
    assert result.measurements["full_state_baseline_physical_size_bytes"] > 0
    result.patch.payload_path.write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        reconstruct_service_state(tmp_path, "fixed")


@pytest.mark.parametrize("policy", ["fixed_xor", "fixed_replace"])
def test_fixed_noop_keeps_base_reference_only(tmp_path, policy):
    model = ArrayModel([np.ones(8, dtype=np.float32)])
    ensure_lineage_base(tmp_path, "lineage", model, parameter_schema_hash=SCHEMA_HASH)
    result = persist_service_patch(tmp_path, "noop", "lineage", model,
                                   parameter_schema_hash=SCHEMA_HASH, operation_policy=policy)
    assert result.patch is None
    assert result.operation_counts == {"inherit": 1, "xor": 0, "replace": 0}
    assert reconstruct_service_state(tmp_path, "noop").sha256 == result.target_state_sha256


def test_unknown_operation_policy_is_rejected():
    state = _state(TensorEntry("value", "uint8", (1,), b"x"))
    with pytest.raises(ValueError, match="operation policy"):
        select_tensor_patch_for_compression(state, state, operation_policy="typo")


def test_policy_change_cannot_overwrite_immutable_service_manifest(tmp_path):
    model = ArrayModel([np.zeros(16, dtype=np.float32)])
    ensure_lineage_base(tmp_path, "lineage", model, parameter_schema_hash=SCHEMA_HASH)
    model._values[0][0] = 1.0
    original = persist_service_patch(tmp_path, "service", "lineage", model,
                                     parameter_schema_hash=SCHEMA_HASH)
    manifest_bytes = original.manifest_path.read_bytes()
    with pytest.raises(checkpoint_store_module.ManifestConflictError):
        persist_service_patch(tmp_path, "service", "lineage", model,
                              parameter_schema_hash=SCHEMA_HASH, operation_policy="fixed_replace")
    assert original.manifest_path.read_bytes() == manifest_bytes
    assert reconstruct_service_state(tmp_path, "service").sha256 == original.target_state_sha256


def test_large_patch_selection_avoids_complete_candidate_recompression(monkeypatch):
    base = _state(
        TensorEntry("first", "uint8", (4096,), b"a" * 4096),
        TensorEntry("second", "uint8", (4096,), b"b" * 4096),
    )
    target = _state(
        TensorEntry("first", "uint8", (4096,), b"c" * 4096),
        TensorEntry("second", "uint8", (4096,), b"d" * 4096),
    )
    calls = []
    real_compress = checkpoint_store_module._compress

    def counted(payload, compression):
        calls.append(len(payload))
        return real_compress(payload, compression)

    monkeypatch.setattr(checkpoint_store_module, "_EXACT_PATCH_SELECTION_MAX_BYTES", 1)
    monkeypatch.setattr(checkpoint_store_module, "_compress", counted)

    patch, selection = select_tensor_patch_for_compression(base, target)

    assert apply_tensor_patch(base, patch).sha256 == target.sha256
    assert selection["selection_version"] == 2
    assert selection["exact_complete_candidate_measurement_skipped"] is True
    assert selection["strategy"] == "minimum_independently_encoded_tensor_payload_size"
    assert len(calls) == 4
    assert max(calls) == 4096


def test_lineage_base_is_immutable_shared_state_and_restores_existing_model(tmp_path):
    initial = np.arange(64, dtype=np.float32).reshape(8, 8)
    first_model = ArrayModel([initial], ["dense/kernel:0"])
    first = ensure_lineage_base(
        tmp_path, "lineage-1", first_model, parameter_schema_hash=SCHEMA_HASH
    )
    second_model = ArrayModel([np.zeros_like(initial)], ["dense/kernel:0"])
    second = ensure_lineage_base(
        tmp_path, "lineage-1", second_model, parameter_schema_hash=SCHEMA_HASH
    )

    assert first.created is True
    assert second.created is False
    assert second.restored_existing_base is True
    assert second.tensor_state_sha256 == first.tensor_state_sha256
    np.testing.assert_array_equal(second_model.get_weights()[0], initial)
    assert first.artifact()["artifact_type"] == "model_initial_checkpoint"
    assert first.artifact()["metadata"]["tensor_state_sha256"] == first.tensor_state_sha256


def test_lineage_base_rejects_capture_only_models(tmp_path):
    with pytest.raises(UnsupportedTensorStateError) as error:
        ensure_lineage_base(
            tmp_path, "read-only", ReadOnlyArrayModel(), parameter_schema_hash=SCHEMA_HASH
        )
    assert error.value.reason_code == "state_restore_unsupported"
    assert error.value.as_fallback_signal()["policy"] == "full_object_v1"


def test_parameter_schema_is_linked_but_separate_from_tensor_state(tmp_path):
    model = ArrayModel([np.ones((4,), dtype=np.float32)])
    result = ensure_lineage_base(tmp_path, "schema", model, parameter_schema_hash=SCHEMA_HASH)
    document = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert document["tensor_state_sha256"] == capture_model_tensor_state(model).sha256
    assert document["parameter_schema"]["hash"] == SCHEMA_HASH
    assert document["parameter_schema"]["ownership"] == "separate_generator_provenance"
    with pytest.raises(ParameterSchemaMismatchError):
        restore_lineage_base(
            tmp_path, "schema", model, expected_parameter_schema_hash="b" * 64
        )


def test_unchanged_service_uses_only_base_reference_and_saves_storage(tmp_path):
    model = ArrayModel([np.array([1.0], dtype=np.float32)], ["w"])
    ensure_lineage_base(tmp_path, "same", model, parameter_schema_hash=SCHEMA_HASH)

    result = persist_service_patch(
        tmp_path, "same-service", "same", model, parameter_schema_hash=SCHEMA_HASH
    )

    assert result.patch is None
    assert result.operation_counts == {"inherit": 1, "xor": 0, "replace": 0}
    assert result.measurements["patch_physical_size_bytes"] == 0
    assert result.measurements["patch_saves_storage"] is True
    assert result.artifact()["metadata"]["patch_mode"] == "base_reference_only"
    assert reconstruct_service_state(tmp_path, "same-service").sha256 == capture_model_tensor_state(model).sha256


def test_service_patch_measures_real_savings_and_restores(tmp_path):
    rng = np.random.default_rng(1234)
    initial = rng.integers(0, 256, size=128 * 1024, dtype=np.uint8)
    model = ArrayModel([initial], ["large"])
    ensure_lineage_base(tmp_path, "shared", model, parameter_schema_hash=SCHEMA_HASH)
    final = initial.copy()
    final[913] ^= 7
    model.set_weights([final])

    result = persist_service_patch(
        tmp_path, "client-1", "shared", model, parameter_schema_hash=SCHEMA_HASH
    )

    assert result.patch is not None
    assert result.measurements["patch_saves_storage"] is True
    assert result.measurements["incremental_physical_bytes_saved"] > 0
    assert result.measurements["patch_physical_size_bytes"] == (
        result.patch.manifest_path.stat().st_size + result.patch.payload_path.stat().st_size
    )
    rebuilt = reconstruct_service_state(tmp_path, "client-1")
    assert rebuilt.sha256 == capture_model_tensor_state(model).sha256
    blank = ArrayModel([np.zeros_like(final)], ["large"])
    restore_service_state(tmp_path, "client-1", blank)
    np.testing.assert_array_equal(blank.get_weights()[0], final)


def test_incompressible_patch_honestly_reports_when_it_does_not_save(tmp_path):
    rng = np.random.default_rng(77)
    base = rng.integers(0, 256, size=64, dtype=np.uint8)
    target = rng.integers(0, 256, size=64, dtype=np.uint8)
    model = ArrayModel([base], ["tiny"])
    ensure_lineage_base(tmp_path, "tiny", model, parameter_schema_hash=SCHEMA_HASH)
    model.set_weights([target])

    result = persist_service_patch(
        tmp_path, "tiny-service", "tiny", model, parameter_schema_hash=SCHEMA_HASH
    )

    assert result.patch is not None
    assert result.measurements["patch_saves_storage"] is False
    assert result.measurements["incremental_physical_bytes_saved"] <= 0
    assert reconstruct_service_state(tmp_path, "tiny-service").sha256 == capture_model_tensor_state(model).sha256


def test_wrong_base_and_unsupported_state_emit_strict_failures():
    base = _state(TensorEntry("w", "uint8", (4,), b"base"))
    target = _state(TensorEntry("w", "uint8", (4,), b"done"))
    patch = build_tensor_patch(base, target)
    other = _state(TensorEntry("w", "uint8", (4,), b"nope"))

    with pytest.raises(IntegrityError):
        apply_tensor_patch(other, patch)
    state, fallback = capture_model_tensor_state_or_fallback(object())
    assert state is None
    assert fallback["required"] is True
    assert fallback["reason_code"] == "no_tensor_state_interface"


def test_corrupt_persisted_base_fails_closed(tmp_path):
    model = ArrayModel([np.arange(8, dtype=np.float32)], ["weight"])
    result = ensure_lineage_base(
        tmp_path, "corrupt", model, parameter_schema_hash=SCHEMA_HASH
    )
    result.base.payload_path.write_bytes(b"corrupt")

    with pytest.raises(IntegrityError):
        load_lineage_base(
            tmp_path,
            "corrupt",
            expected_parameter_schema_hash=SCHEMA_HASH,
        )


def test_real_pytorch_state_dict_when_available(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    base = capture_model_tensor_state(model)
    assert base.framework == "pytorch"
    assert [entry.name for entry in base.entries] == list(model.state_dict().keys())
    ensure_lineage_base(tmp_path, "torch", model, parameter_schema_hash=SCHEMA_HASH)
    with torch.no_grad():
        model[0].weight[0, 0] += 0.125
    result = persist_service_patch(
        tmp_path, "torch-client", "torch", model, parameter_schema_hash=SCHEMA_HASH
    )
    assert result.patch is not None
    reconstructed = reconstruct_service_state(tmp_path, "torch-client")
    assert reconstructed.sha256 == capture_model_tensor_state(model).sha256
