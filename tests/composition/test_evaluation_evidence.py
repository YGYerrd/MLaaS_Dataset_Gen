import json

import numpy as np
import pandas as pd
import pytest

from mlaas_composition.catalogue import load_services_from_db, normalise_service_rows
from mlaas_composition.pool import build_composition_instance
from mlaas_composition.requirements import generate_service_requests
from mlaas_composition.sampler import calculate_constraint_penalty, calculate_utility, evaluate_provisional_composition
from mlaas_data_generator.storage.writer import make_writer


def service(service_id="measured", **overrides):
    return {"service_id": service_id, "task_family": "classification", "metric_score": .8,
            "trust_score": .9, "explainability_score": .7,
            "trust_independent_inputs": 32, "explainability_independent_inputs": 16,
            "evaluation_protocol_version": "benchmark_evaluation_v1",
            "trust_status": "complete", "explainability_status": "complete",
            "trust_evidence_sufficient": True, "explainability_evidence_sufficient": True,
            **overrides}


def request(**overrides):
    return {"composition_type": "collaborative", "functional_requirements": {"task_family": "classification"},
            "hard_constraints": {}, "composition_size": 1, "preference_weights": {"trust": 1}, **overrides}


@pytest.mark.parametrize("missing", [None, "not available", float("nan"), float("inf"), -1, 2])
def test_missing_or_invalid_quality_scores_are_not_imputed(missing):
    row = normalise_service_rows(pd.DataFrame([service(trust_score=missing, explainability_score=missing)])).iloc[0]
    assert pd.isna(row.trust_score) and pd.isna(row.explainability_score)


@pytest.mark.parametrize("metric", ["trust", "explainability"])
@pytest.mark.parametrize("trigger", ["weight", "hard", "qos", "explicit"])
def test_each_requested_metric_requires_complete_current_evidence(metric, trigger):
    rows = [service(), service("absent", **{f"{metric}_score": None}),
            service("failed", **{f"{metric}_status": "failed"}),
            service("partial", **{f"{metric}_status": "insufficient_evidence"}),
            service("unsupported", **{f"{metric}_status": "unsupported"}),
            service("insufficient", **{f"{metric}_evidence_sufficient": False}),
            service("missing_flag", **{f"{metric}_evidence_sufficient": None}),
            service("legacy", evaluation_protocol_version=None),
            service("future", evaluation_protocol_version="production_evaluation_v2")]
    req = request(preference_weights={"accuracy": 1})
    if trigger == "weight":
        req["preference_weights"] = {metric: 1}
    elif trigger == "hard":
        req["hard_constraints"] = {f"min_{metric}_score": 0}
    elif trigger == "qos":
        req["qos_requirements"] = {f"min_{metric}": 0}
    else:
        req["evaluation_requirements"] = {metric: True}
    assert build_composition_instance(pd.DataFrame(rows), req).C_R.service_id.tolist() == ["measured"]


def test_requests_require_only_the_metrics_they_ask_for():
    rows = pd.DataFrame([service(explainability_status="unsupported", explainability_score=None)])
    assert len(build_composition_instance(rows, request()).C_R) == 1
    assert build_composition_instance(rows, request(preference_weights={"trust": 1, "explainability": 1})).C_R.empty


def test_explicit_legacy_replay_never_imputes_missing_scores_or_accepts_failed_production():
    rows = pd.DataFrame([service("legacy", evaluation_protocol_version=None),
                         service("tagged_legacy", evaluation_protocol_version="legacy_perturbation_v1"),
                         service("missing", evaluation_protocol_version=None, trust_score=None),
                         service("failed", trust_status="failed")])
    pool = build_composition_instance(rows, request(evaluation_requirements={"allow_legacy": True})).C_R
    assert pool.service_id.tolist() == ["legacy", "tagged_legacy"]
    assert pool.evaluation_protocol_version.tolist() == ["legacy_unversioned", "legacy_perturbation_v1"]


def test_missing_member_does_not_disappear_from_group_means_or_leak_into_unrequested_utility():
    selected = normalise_service_rows(pd.DataFrame([service(), service("missing", trust_score=None, explainability_score=None)]))
    scores = evaluate_provisional_composition(selected)
    assert np.isnan(scores["mean_trust"]) and np.isnan(scores["mean_explainability"])
    assert np.isfinite(calculate_utility(scores, {"accuracy": 1}))
    with pytest.raises(ValueError, match="lack measured"):
        calculate_utility(scores, {"trust": 1})
    with pytest.raises(ValueError, match="lacks a measured"):
        calculate_constraint_penalty(scores, {"min_explainability": .1}, {})


def test_trust_has_standalone_objective_and_qos_requirement():
    scores = evaluate_provisional_composition(normalise_service_rows(pd.DataFrame([service(trust_score=.4)])))
    assert calculate_utility(scores, {"trust": 1}) == pytest.approx(.4)
    assert calculate_constraint_penalty(scores, {"min_trust": .8}, {}) == pytest.approx(.25)
    assert build_composition_instance(pd.DataFrame([service(trust_score=.4)]), request(hard_constraints={"min_trust_score": .5})).C_R.empty


def test_generated_requests_record_only_supported_evaluation_dimensions():
    for measured in (True, False):
        rows = pd.DataFrame([service("a", **({} if measured else {"evaluation_protocol_version": None, "trust_score": None}))])
        req = generate_service_requests(rows, request_count=1).iloc[0].to_dict()
        weights = json.loads(req["preference_weights"])
        requirements = json.loads(req["evaluation_requirements"])
        assert requirements["trust"] is measured and requirements["explainability"] is measured
        assert ("trust" in weights) is measured and ("explainability" in weights) is measured
        assert len(build_composition_instance(rows, req).C_R) == 1
        assert "NaN" not in req["hard_constraints"] and "NaN" not in req["qos_requirements"]


def test_evaluation_metadata_roundtrips_through_normalized_catalogue(tmp_path):
    path = tmp_path / "catalogue.db"
    writer = make_writer("sqlite", db_path=str(path))
    writer.start()
    writer.write_service({"service_id": "measured", "status": "completed", "task_family": "classification"})
    writer.write_service_metrics("measured", {k: v for k, v in service().items() if k not in {"service_id", "task_family"}})
    writer.write_service({"service_id": "legacy", "status": "completed", "task_family": "classification"})
    writer.write_service_metrics("legacy", {"explainability_score": .8})
    writer.finish()
    rows = load_services_from_db(path).set_index("service_id")
    assert rows.loc["measured", "evaluation_protocol_version"] == "benchmark_evaluation_v1"
    assert rows.loc["measured", "trust_independent_inputs"] == 32
    assert rows.loc["measured", "explainability_independent_inputs"] == 16
    assert rows.loc["measured", "trust_evidence_sufficient"]
    assert rows.loc["legacy", "evaluation_protocol_version"] == "legacy_unversioned"
    assert rows.loc["legacy", "trust_status"] == "insufficient_evidence"
    assert pd.isna(rows.loc["legacy", "trust_score"])
    assert not rows.loc["legacy", "explainability_evidence_sufficient"]
    assert pd.isna(rows.loc["legacy", "trust_independent_inputs"])


def test_protocol_versions_are_explicitly_selected_and_never_mixed():
    rows = pd.DataFrame([service("benchmark"), service("production", evaluation_protocol_version="production_evaluation_v1")])
    assert build_composition_instance(rows, request()).C_R.service_id.tolist() == ["benchmark"]
    req = request(evaluation_requirements={"protocol_version": "production_evaluation_v1"})
    assert build_composition_instance(rows, req).C_R.service_id.tolist() == ["production"]


def test_latest_observation_cannot_fall_back_to_older_complete_evidence():
    provenance = {"fl_run_id": "run", "client_id": "client", "participated": 1, "fedavg_succeeded": 1}
    rows = pd.DataFrame([service("old", round_id=1, **provenance),
                         service("latest", round_id=2, trust_status="failed", **provenance)])
    req = request(request_schema_version="2.0.0", compatibility_mode="functional", compatibility_requirements={})
    assert build_composition_instance(rows, req).C_R.empty


def test_string_false_cannot_opt_into_legacy_evidence():
    with pytest.raises(ValueError, match="must be a boolean"):
        build_composition_instance(pd.DataFrame([service()]), request(evaluation_requirements={"allow_legacy": "false"}))
