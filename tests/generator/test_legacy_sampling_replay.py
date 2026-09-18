from copy import deepcopy

import pytest

from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.provenance import sha256_json


@pytest.fixture
def legacy_interface(monkeypatch):
    old_args = {"benchmark_sample_size": None, "seed": 42, "training_regime": "generic", "scaler": "standard"}
    dataset = {"dataset": "wine", "dataset_name": None, "dataset_config": None,
               "dataset_revision": None, "train_split": None, "test_split": None,
               "feature_schema": None, "loader_args": old_args}
    compatibility = {"architecture_id": "model", "parameter_schema_hash": "schema", "label_map_hash": "labels",
                     "preprocessing_contract_hash": "preprocessing", "logical_dataset_contract_hash": sha256_json(dataset)}
    doc = {"version": "raw_source_family_v3", "config": {"evaluation_seed": 42},
           "contracts": {"dataset": deepcopy(dataset)}, "compatibility": compatibility,
           "reservation": {"metric_source_ids": ["test:0", "test:1"]},
           "evaluation_data_hash": "prepared", "raw_source": {"source": "bytes"},
           "shared_fitted_state": {"scale": 1}}
    actual_dataset = {**dataset, "loader_args": {"seed": 42, "training_regime": "generic", "scaler": "standard",
                                                "evaluation_seed": 42, "_defer_evaluation_sampling": True}}
    actual = {"dataset": actual_dataset, "compatibility": {**compatibility,
              "logical_dataset_contract_hash": sha256_json(actual_dataset)}}
    monkeypatch.setattr(preparation.shared, "_contracts", lambda *a: actual)
    monkeypatch.setattr(preparation.shared, "_compatibility", lambda value: dict(value["compatibility"]))
    monkeypatch.setattr(preparation, "prepared_fingerprint", lambda *a: "prepared")
    monkeypatch.setattr(preparation, "_raw_contract", lambda *a: {"source": "bytes"})
    monkeypatch.setattr(preparation, "_fitted_state", lambda *a: {"scale": 1})
    prepared = (None, None, {}, "classification", object(), ["train:0"], ["test:0", "test:1"])
    return doc, actual, prepared, {"evaluation_seed": 42}


def test_legacy_null_cap_and_recorded_seed_recompute_original_logical_hash(legacy_interface):
    doc, actual, prepared, cfg = legacy_interface
    before = deepcopy(actual)
    preparation.verify_loaded(doc, prepared, cfg)
    assert actual == before


@pytest.mark.parametrize("change", ["seed", "configured_seed", "scaler", "cap", "defer", "schema", "architecture", "dataset", "current_version"])
def test_legacy_sampling_adapter_rejects_changed_semantics(legacy_interface, change):
    doc, actual, prepared, cfg = legacy_interface
    args = actual["dataset"]["loader_args"]
    if change == "seed":
        args["evaluation_seed"] = 43
    elif change == "configured_seed":
        cfg["evaluation_seed"] = 43
    elif change == "scaler":
        args["scaler"] = "minmax"
    elif change == "cap":
        args["benchmark_sample_size"] = 1024
    elif change == "defer":
        args["_defer_evaluation_sampling"] = False
    elif change == "schema":
        actual["compatibility"]["parameter_schema_hash"] = "different"
    elif change == "architecture":
        actual["compatibility"]["architecture_id"] = "different"
    elif change == "dataset":
        actual["dataset"]["dataset"] = "different"
    else:
        doc["version"] = preparation.VERSION
    with pytest.raises(ValueError, match="differs from frozen family"):
        preparation.verify_loaded(doc, prepared, cfg)


@pytest.mark.parametrize("change", ["prepared", "raw", "fitted", "ids", "overlap"])
def test_legacy_sampling_adapter_keeps_all_data_integrity_checks(legacy_interface, change, monkeypatch):
    doc, actual, prepared, cfg = legacy_interface
    if change == "prepared":
        monkeypatch.setattr(preparation, "prepared_fingerprint", lambda *a: "changed")
    elif change == "raw":
        monkeypatch.setattr(preparation, "_raw_contract", lambda *a: {"source": "changed"})
    elif change == "fitted":
        monkeypatch.setattr(preparation, "_fitted_state", lambda *a: {"scale": 2})
    elif change == "ids":
        prepared = (*prepared[:6], ["test:1", "test:0"])
    else:
        prepared = (*prepared[:5], ["test:0"], prepared[6])
    with pytest.raises(ValueError):
        preparation.verify_loaded(doc, prepared, cfg)
