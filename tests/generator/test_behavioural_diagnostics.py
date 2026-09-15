import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator import behavioural_diagnostics as diagnostics


def test_prediction_diversity_distinguishes_duplicates_constants_and_shared_errors():
    result = diagnostics.analyse_predictions([0, 1, 0, 1], {
        "a": [0, 1, 0, 1], "b": [0, 1, 0, 1], "c": [1, 1, 1, 1]})
    assert result["prediction_diversity"] == {"unique_prediction_vectors": 2, "constant_predictors": 1,
                                               "identical_prediction_groups": [["a", "b"]]}
    assert [p["pairwise_disagreement"] for p in result["pairs"]] == [0, 0.5, 0.5]
    assert result["prediction_profiles"][2]["predicted_class_counts"] == {"1": 4}
    assert result["services"][2]["accuracy"] == 0.5


def test_regression_diversity_reports_continuous_differences_and_constant_outputs():
    result = diagnostics.analyse_predictions([0., 1., 2.],
        {"a": [0., 1., 2.], "b": [1., 1., 1.], "c": [0.1, 1.1, 2.1]}, task_family="regression")
    assert result["prediction_diversity"]["constant_predictors"] == 1
    assert result["prediction_diversity"]["unique_prediction_vectors"] == 3
    assert result["pairs"][1]["prediction_mae"] == pytest.approx(0.1)
    assert result["pairs"][0]["residual_correlation"] is None
    assert result["prediction_profiles"][1]["prediction_std"] == 0
    assert diagnostics.diagnostic_targets([[0, 1, -100]], "token_classification").tolist() == [0, 1]


@pytest.mark.parametrize("overlap,bad_hash", [(False, False), (True, False), (False, True)])
def test_independent_diagnostics_use_only_catalogue_and_verify_member_hashes(tmp_path, monkeypatch, overlap, bad_hash):
    from mlaas_data_generator.cli.manifest import preparation
    from mlaas_data_generator import reduced_checkpoint
    db = tmp_path / "services.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE services(service_id TEXT,status TEXT,service_config_json TEXT);
        CREATE TABLE service_generation_provenance(service_id TEXT,family_id TEXT,family_document_ref TEXT,
            family_document_hash TEXT,final_model_hash TEXT,initial_state_hash TEXT,parameter_schema_hash TEXT,
            compatibility_contract_hash TEXT,composition_compatible INTEGER,generation_mode TEXT);
    """)
    for sid in ("a", "b", "c"):
        con.execute("INSERT INTO services VALUES (?,?,?)", (sid, "completed", '{"optimizer":"adam","sample_size":2}'))
        con.execute("INSERT INTO service_generation_provenance VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, "family", "doc", "hash", sid, "initial", "schema", "contract", 1, "independent"))
    con.commit(); con.close()
    doc = {"family_id": "family", "store_root": str(tmp_path), "config": {}, "initial_state_hash": "initial",
        "compatibility_contract_hash": "contract", "compatibility": {"parameter_schema_hash": "schema"},
        "contracts": {"labels": {"num_labels": 2, "labels": []}},
        "reservation": {"catalogue_positions": [0, 1], "catalogue_metric_ids": ["t0", "t1"],
                        "audit_ids": ["t0", "t3"] if overlap else ["t2", "t3"]}}
    model = SimpleNamespace(current=None)
    prepared = ((np.array([[0], [1]]), np.array([0, 1])),
        (np.array([[10], [11], [12], [13]]), np.array([0, 1, 0, 1])), {}, "classification", model,
        ["r0", "r1"], ["t0", "t1", "t2", "t3"])
    monkeypatch.setattr(preparation, "read_preparation", lambda *a: doc)
    monkeypatch.setattr(preparation, "load_prepared", lambda cfg, **kw: (cfg, prepared))
    monkeypatch.setattr(preparation, "verify_loaded", lambda *a: None)
    monkeypatch.setattr(diagnostics, "reconstruct_service_state", lambda root, sid: SimpleNamespace(sid=sid))
    monkeypatch.setattr(reduced_checkpoint, "source_state_hash", lambda root, sid, state: "wrong" if bad_hash and sid == "c" else sid)
    monkeypatch.setattr(diagnostics, "restore_model_tensor_state", lambda model, state, **kw: setattr(model, "current", state.sid))
    seen = []
    def predict(model, x, y, **kw):
        seen.append(x.flatten().tolist())
        assert x.flatten().tolist() == [10, 11]
        return np.array([1, 1] if model.current == "c" else [0, 1])
    monkeypatch.setattr(diagnostics, "predict_decisions", predict)
    out = tmp_path / "diagnostics"
    if overlap:
        with pytest.raises(ValueError, match="overlaps training/audit"):
            diagnostics.family_diagnostics(db=db, family_id="family", output_dir=out)
        assert not seen
        return
    result = diagnostics.family_diagnostics(db=db, family_id="family", output_dir=out)
    assert result["status"] == ("partial" if bad_hash else "completed")
    assert result["diagnosed_clients"] == (2 if bad_hash else 3)
    assert result["evaluation_subset"] == "catalogue_metric"
    assert result["prediction_diversity"]["identical_prediction_groups"] == [["a", "b"]]
    with np.load(out / "predictions.npz", allow_pickle=False) as arrays:
        assert arrays["targets"].tolist() == [0, 1]
        assert arrays["predictions"].shape == (result["diagnosed_clients"], 2)
    assert json.loads((out / "summary.json").read_text())["report_hash"] == result["report_hash"]
