import json
import sqlite3

import pytest

from mlaas_data_generator.config import CONFIG
from mlaas_data_generator.evaluation_protocol import PROTOCOL_VERSION, BENCHMARK_PROTOCOL_VERSION, resolve_evaluation_plan
from mlaas_data_generator.storage.writer import make_writer


def test_public_dispatch_labels_disabled_and_legacy_results():
    from mlaas_data_generator.services.perturbation import run_perturbation_stage
    assert run_perturbation_stage(None, [])["evaluation_protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    legacy = run_perturbation_stage(None, [], config={"evaluation_protocol_version": "legacy_perturbation_v1"})
    assert legacy["evaluation_protocol_version"] == "legacy_perturbation_v1"
    with pytest.raises(ValueError, match="unsupported evaluation_protocol_version"):
        run_perturbation_stage(None, [], config={"evaluation_protocol_version": "unknown"})


def test_manifest_pilot_has_distinct_service_identity_and_same_training_recipe():
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest
    options = dict(task_keys=["text_classification"], models_per_task=1, datasets_per_model=1,
        clients_per_family=1, training_regimes=["finetune_transfer"], seed=123)
    diagnostic = build_hf_manifest(**options, metric_evaluation_profile="diagnostic")
    pilot = build_hf_manifest(**options, metric_evaluation_profile="pilot")
    benchmark = build_hf_manifest(**options)
    assert set(benchmark.metric_evaluation_profile) == {"benchmark"}
    assert set(benchmark.evaluation_protocol_version) == {BENCHMARK_PROTOCOL_VERSION}
    assert not set(benchmark.service_id) & (set(diagnostic.service_id) | set(pilot.service_id))
    assert len(diagnostic) == len(pilot) > 0
    assert not set(diagnostic["service_id"]) & set(pilot["service_id"])
    for column in ("sample_seed", "training_seed", "learning_rate", "training_epochs", "sample_size"):
        assert diagnostic[column].tolist() == pilot[column].tolist()
    assert {json.loads(value)["metric_evaluation_profile"] for value in pilot["service_config"]} == {"pilot"}


def test_profile_budgets_ignore_inherited_legacy_limits():
    plan = resolve_evaluation_plan({**CONFIG, "metric_evaluation_profile": "pilot"})
    assert (plan["trust_sample_count"], plan["trust_trials"]) == (512, 12)
    assert (plan["explainability_sample_count"], plan["explainability_candidate_units"],
            plan["explainability_random_trials"]) == (128, 32, 16)
    assert plan["explainability_max_duration_s"] > CONFIG["perturbation_max_duration_s"]
    assert resolve_evaluation_plan(CONFIG)["profile"] == "benchmark"
    assert resolve_evaluation_plan({"metric_evaluation_profile": "production", "trust_trials": 24})["trust_trials"] == 24


def test_benchmark_defaults_and_stage_floor_overrides():
    plan = resolve_evaluation_plan(CONFIG)
    assert CONFIG["enable_perturbation_metrics"] is True
    assert (plan["trust_sample_count"], plan["trust_trials"], plan["explainability_sample_count"],
            plan["explainability_random_trials"]) == (32, 4, 16, 4)
    assert (plan["trust_min_independent_inputs"], plan["explainability_min_independent_inputs"]) == (16, 8)
    assert plan["min_coverage"] == .8
    assert plan["explanation_replacement_methods"] == ["empirical"]
    assert resolve_evaluation_plan({"metric_evaluation_profile": "pilot"})["explanation_replacement_methods"] == ["empirical", "mean"]
    assert resolve_evaluation_plan({"explanation_replacement_methods": ["mean"]})["explanation_replacement_methods"] == ["mean"]
    custom = resolve_evaluation_plan({"metric_evaluation_min_independent_inputs": 10, "trust_min_independent_inputs": 20})
    assert (custom["trust_min_independent_inputs"], custom["explainability_min_independent_inputs"]) == (20, 10)


def test_resume_preset_overrides_frozen_budgets_and_preserves_training_identity():
    import pandas as pd
    from mlaas_data_generator.cli.run_manifest import _resolve_execution_row
    row = pd.Series({"service_id": "existing", "sample_seed": 19, "training_seed": 20,
        "evaluation_seed": 21, "trust_trials": 99, "trust_min_independent_inputs": 999,
        "metric_evaluation_min_coverage": .95, "explanation_replacement_methods": ["empirical", "mean"],
        "_frozen_manifest_defaults": {**CONFIG, "metric_evaluation_profile": "diagnostic",
            "evaluation_protocol_version": PROTOCOL_VERSION, "enable_perturbation_metrics": False,
            "metric_evaluation_progress_path": "old_progress.json"}})
    ordinary = _resolve_execution_row(row, {})
    resumed = _resolve_execution_row(row, {}, "benchmark")
    assert ordinary["trust_trials"] == 99
    assert resumed["enable_perturbation_metrics"] is True
    assert "metric_evaluation_progress_path" not in resumed
    plan = resolve_evaluation_plan(resumed)
    assert plan["protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    assert plan["trust_trials"] == 4 and plan["trust_min_independent_inputs"] == 16
    assert plan["min_coverage"] == .8 and plan["explanation_replacement_methods"] == ["empirical"]
    for key in ("service_id", "sample_seed", "training_seed", "evaluation_seed"):
        assert resumed[key] == ordinary[key]
    row["smoke_test"] = True
    assert _resolve_execution_row(row, {}, "benchmark")["enable_perturbation_metrics"] is False


@pytest.mark.parametrize("before_protocol_fields_existed", [False, True])
def test_manifest_binding_is_disposable_and_does_not_restrict_later_invocations(tmp_path, before_protocol_fields_existed):
    import pandas as pd
    from mlaas_data_generator.cli.run_manifest import _resolve_execution_row
    from mlaas_data_generator.cli.manifest.preparation import bind_manifest, freeze_manifest_revisions, group_key
    old = {**CONFIG, "evaluation_protocol_version": PROTOCOL_VERSION,
        "metric_evaluation_profile": "diagnostic", "enable_perturbation_metrics": False}
    if before_protocol_fields_existed:
        old.pop("evaluation_protocol_version")
        old.pop("metric_evaluation_profile")
    row = pd.Series({"service_id": "prepared", "dataset": "iris", "model_type": "mlp",
        "task_type": "classification", "training_regime": "generic", "learning_rate": .001,
        "_frozen_manifest_defaults": old})
    path = tmp_path / "current.csv"
    original = _resolve_execution_row(row, {})
    bind_manifest(original, path)
    freeze_manifest_revisions(original)
    resumed = _resolve_execution_row(row, {}, "benchmark")
    resumed["row_index"] = 99
    bind_manifest(resumed, tmp_path / "renamed-and-reordered.csv")
    assert group_key(resumed) == group_key(original)
    assert resolve_evaluation_plan(resumed)["protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    changed = _resolve_execution_row(row, {}, "benchmark")
    changed["learning_rate"] = .01
    bind_manifest(changed, tmp_path / "another.csv")
    assert changed["learning_rate"] == .01
    assert not path.with_suffix(".resolved").exists()


@pytest.mark.parametrize("override", [{"trust_sample_count": 0}, {"trust_trials": 1.5},
    {"metric_evaluation_min_independent_inputs": 1}, {"metric_evaluation_max_ci_width": float("nan")},
    {"metric_evaluation_budget_fractions": [.1, .1]}, {"explanation_replacement_methods": []},
    {"explanation_replacement_methods": ["empirical", "empirical"]}, {"explanation_replacement_methods": ["unknown"]},
    {"metric_evaluation_min_coverage": 0}, {"metric_evaluation_min_coverage": 1.1},
    {"metric_evaluation_min_coverage": float("nan")}, {"metric_evaluation_min_coverage": True}])
def test_invalid_plans_fail_before_inference(override):
    with pytest.raises(ValueError):
        resolve_evaluation_plan(override)


def test_missing_new_measurement_clears_old_score_and_preserves_evidence(tmp_path):
    db = tmp_path / "metrics.db"
    writer = make_writer("sqlite", db_path=str(db))
    writer.start()
    writer.write_service({"service_id": "s", "status": "completed", "task_family": "classification",
        "task_type": "classification", "model_type": "mlp", "dataset": "iris"})
    writer.write_service_metrics("s", {"trust_score": .9, "explainability_score": .8})
    writer.write_service_metrics("s", {"trust_score": {"value": None}, "explainability_score": {"value": None},
        "evaluation_protocol_version": PROTOCOL_VERSION, "trust_status": "unsupported",
        "trust_evidence_sufficient": False, "trust_evidence": {"requested": 512, "successful": 0}})
    writer.finish()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value_num,value_json FROM service_metrics WHERE metric_name='trust_score'").fetchone() == (None, "null")
        assert json.loads(conn.execute("SELECT value_json FROM service_metrics WHERE metric_name='trust_evidence'").fetchone()[0])["successful"] == 0


def test_compressed_evaluation_evidence_is_content_addressed_and_checked(tmp_path):
    import gzip
    import hashlib
    from pathlib import Path
    from mlaas_data_generator.services.evaluation_artifacts import persist_evaluation_evidence
    values = {"evaluation_protocol_version": PROTOCOL_VERSION, "trust_score": None, "trust_status": "unsupported"}
    samples = [{"sample_index": 1, "trials": [{"status": "excluded"}] * 100}]
    artifact = persist_evaluation_evidence(tmp_path, values, samples)
    path = Path(artifact["artifact_uri"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["metadata"]["sha256"]
    decoded = json.loads(gzip.decompress(path.read_bytes()))
    assert decoded["evaluation"] == values and decoded["perturbation_samples"] == samples
    assert artifact["metadata"]["size_bytes"] < artifact["metadata"]["uncompressed_size_bytes"]
    assert persist_evaluation_evidence(tmp_path, values, samples) == artifact
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        persist_evaluation_evidence(tmp_path, values, samples)
