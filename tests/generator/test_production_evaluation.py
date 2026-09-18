import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator.services import production_evaluation as evaluation


class FeatureModel:
    def __init__(self, feature=0, constant=False):
        self.feature = feature
        self.constant = constant
        self.calls = 0

    def get_weights(self):
        return [np.array([self.feature, self.constant], dtype=float)]

    def predict(self, x, **kwargs):
        self.calls += 1
        values = np.asarray(x)
        prob = np.zeros(len(values)) if self.constant else 1 / (1 + np.exp(-values[:, self.feature] * 6))
        return np.column_stack([1 - prob, prob])


def config(**changes):
    return {
        "enable_perturbation_metrics": True,
        "evaluation_protocol_version": "production_evaluation_v1",
        "trust_sample_count": 4, "trust_trials": 4,
        "explainability_sample_count": 4, "explainability_candidate_units": 4,
        "metric_evaluation_random_trials": 2, "metric_evaluation_budget_fractions": [.25, .5],
        "metric_evaluation_bootstrap_resamples": 100,
        "metric_evaluation_min_independent_inputs": 2, "metric_evaluation_max_ci_width": 1.,
        "trust_max_duration_s": 30, "explainability_max_duration_s": 30,
        **changes,
    }


@pytest.fixture
def dataset():
    x = np.random.default_rng(18).normal(size=(12, 4)).astype(np.float64)
    return x, (x[:, 0] > 0).astype(int)


def run(model, dataset, **changes):
    return evaluation.run_production_evaluation(model, *dataset, task_family="classification", config=config(**changes))


def test_terminal_progress_reports_scoring_stages_and_completed_inputs(dataset, capsys):
    run(FeatureModel(), dataset, trust_sample_count=2, explainability_sample_count=2)
    output = capsys.readouterr().out
    assert "[metric evaluation] trust starts | completed_inputs=0/2" in output
    assert "[metric evaluation] trust | completed_inputs=2/2" in output
    assert "[metric evaluation] explainability starts | completed_inputs=0/2" in output
    assert "[metric evaluation] explainability | completed_inputs=2/2" in output
    assert "[metric evaluation] explainability ends | completed_inputs=2/2" in output


def test_shared_samples_and_perturbations_exclude_service_identity(dataset):
    first = evaluation.run_production_evaluation(FeatureModel(), *dataset, task_family="classification", config=config(), service_id="alpha")
    second = evaluation.run_production_evaluation(FeatureModel(), *dataset, task_family="classification", config=config(), service_id="beta")
    assert first["metric_evaluation_provenance"] == second["metric_evaluation_provenance"]
    assert first["perturbation_samples"] == second["perturbation_samples"]
    assert first["trust_status"] == "complete"
    assert first["trust_evaluation"]["trial_counts"] == {"requested": 16, "attempted": 16, "successful": 16, "failed": 0, "excluded": 0}
    json.dumps(first, allow_nan=False)


def test_constant_wrong_predictor_retains_quality_and_has_no_faithfulness(dataset):
    x, _ = dataset
    result = run(FeatureModel(constant=True), (x, np.ones(len(x), dtype=int)))
    assert result["trust_score"] == 1
    assert result["trust_evaluation"]["clean_quality_mean"] == 0
    assert result["trust_evaluation"]["perturbed_quality_mean"] == 0
    assert result["explainability_score"] == 0
    assert result["metric_evaluation_provenance"]["explanation_method"] == "occlusion"


def test_deliberately_fragile_predictor_loses_stability(dataset):
    class Fragile(FeatureModel):
        def predict(self, x, **kwargs):
            prob = (np.sin(np.asarray(x)[:, 0] * 10000) > 0).astype(float)
            return np.column_stack([1 - prob, prob])
    result = run(Fragile(), dataset, trust_sample_count=12, trust_trials=12)
    stable = run(FeatureModel(constant=True), dataset, trust_sample_count=12, trust_trials=12)
    assert result["trust_score"] < .8
    assert stable["trust_score"] == 1


def test_known_feature_explanation_beats_random_and_reverse(dataset):
    kwargs = {"trust_sample_count": 12, "explainability_sample_count": 12, "metric_evaluation_random_trials": 4}
    faithful = run(FeatureModel(), dataset, **kwargs)
    random = run(FeatureModel(), dataset, metric_evaluation_explanation_method="random", **kwargs)
    reversed_result = run(FeatureModel(), dataset, metric_evaluation_explanation_method="reverse_occlusion", **kwargs)
    assert faithful["explainability_score"] > random["explainability_score"] + .05
    assert faithful["explainability_score"] > reversed_result["explainability_score"] + .05
    for sample in faithful["perturbation_samples"]:
        evidence = sample["explainability"]
        ranking_seeds = {entry["seed"] for entry in evidence["ranking_evidence"]}
        evaluation_seeds = {entry["seed"] for entry in evidence["trials"]}
        assert ranking_seeds.isdisjoint(evaluation_seeds)
        assert {entry["replacement_method"] for entry in evidence["trials"]} == {"empirical", "mean"}
        assert all(len(entry["effects"]) == 6 for entry in evidence["trials"])
        assert evidence["semantic_alignment"]["included_in_faithfulness"] is False


def test_model_randomization_changes_ranking_and_model_identity(dataset):
    first, second = run(FeatureModel(0), dataset), run(FeatureModel(1), dataset)
    assert first["metric_evaluation_provenance"]["model_identity"]["sha256"] != second["metric_evaluation_provenance"]["model_identity"]["sha256"]
    assert any(a["explainability"]["ranking"] != b["explainability"]["ranking"]
               for a, b in zip(first["perturbation_samples"], second["perturbation_samples"]))


def test_group_bootstrap_does_not_count_repeated_input_trials_as_independent():
    records = [{"sample_index": i, "group_id": i // 10, "score": float(i // 10)} for i in range(20)]
    result = evaluation.summarize_inputs(records, seed=1, bootstrap_resamples=100)
    repeated = evaluation.summarize_inputs(records * 100, seed=1, bootstrap_resamples=100)
    assert result == repeated
    assert result["independent_inputs"] == 2
    assert result["ci95"] == [0, 1]


def test_random_explanation_signed_gains_cancel_before_score_clipping():
    records = [{"sample_index": i, "score": 1. if i % 2 else -1.} for i in range(20)]
    summary = evaluation._metric_summary(records, "explainability", seed=1, bootstrap_resamples=100)
    assert summary["mean"] == 0
    assert summary["signed_control_gain"]["p10"] == -1


def test_unsupported_trust_does_not_prevent_explanations(dataset, monkeypatch):
    def unsupported(*args, **kwargs):
        raise evaluation.outputs.UnsupportedTaskOutput("no_validated_operating_operator")
    monkeypatch.setattr(evaluation.outputs, "perturb_input", unsupported)
    result = run(FeatureModel(), dataset)
    assert result["trust_status"] == "unsupported"
    assert result["trust_score"] is None
    assert result["trust_evaluation"]["sample_counts"]["excluded"] == 4
    assert result["trust_evaluation"]["trial_counts"]["excluded"] == 16
    assert all(row["trust"]["inference_attempted"] is False for row in result["perturbation_samples"])
    assert result["explainability_score"] is not None


def test_missing_text_transformations_skip_trust_clean_inference(monkeypatch):
    def unsupported(*args, **kwargs):
        raise evaluation.outputs.UnsupportedTaskOutput("no_validated_operating_operator")
    monkeypatch.setattr(evaluation.outputs, "candidate_units", unsupported)
    model = FeatureModel()
    x = np.asarray([f"held out sentence {index}" for index in range(12)], dtype=object)
    result = run(model, (x, np.arange(12) % 2))
    assert result["trust_status"] == "unsupported"
    assert result["trust_evaluation"]["exclusion_reasons"] == {
        "no_validated_meaning_preserving_text_transformations": 16}
    assert model.calls == 0


def test_explanation_ranking_uses_metric_evaluation_batches(dataset, monkeypatch):
    observed_batch_sizes = []
    original = evaluation.outputs.predict_task_outputs

    def tracked(model, samples, **kwargs):
        observed_batch_sizes.append(len(samples))
        return original(model, samples, **kwargs)

    monkeypatch.setattr(evaluation.outputs, "predict_task_outputs", tracked)
    result = run(FeatureModel(), dataset, trust_sample_count=2, explainability_sample_count=2,
                 metric_evaluation_batch_size=4)
    # The explanation reuses each trust baseline, then ranks four candidates.
    # Validation also respects the configured memory cap, so a six-condition
    # trial becomes one batch of four and one batch of two.
    assert observed_batch_sizes[:7] == [1, 4, 1, 4, 4, 4, 2]
    evidence = result["perturbation_samples"][0]["explainability"]["ranking_evidence"]
    assert [row["unit_index"] for row in evidence] == [0, 1, 2, 3]


def test_validation_batches_multiple_complete_trials_and_reuses_clean_baseline(dataset, monkeypatch):
    observed_batch_sizes = []
    original = evaluation.outputs.predict_task_outputs

    def tracked(model, samples, **kwargs):
        observed_batch_sizes.append(len(samples))
        return original(model, samples, **kwargs)

    monkeypatch.setattr(evaluation.outputs, "predict_task_outputs", tracked)
    result = run(FeatureModel(), dataset, trust_sample_count=2, explainability_sample_count=2,
                 metric_evaluation_batch_size=24)

    # Per trust input: one clean + four perturbations. Per explanation input:
    # the clean result is reused, four ranking interventions run, then eight
    # validation trials run in two 24-example batches.
    assert observed_batch_sizes == [1, 4, 1, 4, 4, 24, 24, 4, 24, 24]
    trials = result["perturbation_samples"][0]["explainability"]["trials"]
    assert [row["trial_index"] for row in trials] == list(range(8))
    assert all(row["status"] == "successful" and len(row["effects"]) == 6 for row in trials)


def test_frozen_predictions_rankings_scores_intervals_and_statuses_match_optimized_inference(dataset, monkeypatch):
    class RecordingModel(FeatureModel):
        def __init__(self):
            super().__init__()
            self.prediction_log = []

        def predict(self, x, **kwargs):
            predictions = super().predict(x, **kwargs)
            self.prediction_log.extend([
                (np.asarray(sample).tolist(), np.asarray(prediction).tolist())
                for sample, prediction in zip(np.asarray(x), predictions)
            ])
            return predictions

    settings = {"trust_sample_count": 4, "explainability_sample_count": 4, "evaluation_seed": 913}
    serial_model, batched_model = RecordingModel(), RecordingModel()

    original_explanation_stage = evaluation._explanation_stage
    def explanation_without_shared_baselines(*args, **kwargs):
        # Reproduce the former execution path: validation is one complete
        # six-condition trial per batch and explanation predicts clean inputs
        # again instead of receiving trust's run-local cache.
        positional = list(args)
        positional[-1] = {}
        return original_explanation_stage(*positional, **kwargs)

    monkeypatch.setattr(evaluation, "_explanation_stage", explanation_without_shared_baselines)
    serial = run(serial_model, dataset, metric_evaluation_batch_size=6, **settings)
    monkeypatch.setattr(evaluation, "_explanation_stage", original_explanation_stage)
    batched = run(batched_model, dataset, metric_evaluation_batch_size=24, **settings)

    # Frozen inputs and seeds produce exactly the same per-example model
    # outputs even though the reference repeats clean predictions and uses
    # different inference-call boundaries.
    def prediction_map(log):
        mapped = {}
        for sample, prediction in log:
            mapped.setdefault(json.dumps(sample), set()).add(json.dumps(prediction))
        return mapped

    assert batched_model.prediction_log != serial_model.prediction_log
    assert prediction_map(batched_model.prediction_log) == prediction_map(serial_model.prediction_log)
    assert all(len(outputs) == 1 for outputs in prediction_map(serial_model.prediction_log).values())
    assert batched["metric_evaluation_provenance"]["sample_indices"] == serial[
        "metric_evaluation_provenance"]["sample_indices"]
    assert batched["metric_evaluation_provenance"]["seed"] == serial[
        "metric_evaluation_provenance"]["seed"] == 913
    assert batched["perturbation_samples"] == serial["perturbation_samples"]
    for stage in ("trust", "explainability"):
        assert batched[f"{stage}_score"] == serial[f"{stage}_score"]
        assert batched[f"{stage}_status"] == serial[f"{stage}_status"]
        assert batched[f"{stage}_evaluation"]["ci95"] == serial[f"{stage}_evaluation"]["ci95"]
        assert batched[f"{stage}_evaluation"]["trial_counts"] == serial[f"{stage}_evaluation"]["trial_counts"]
    assert batched_model.calls < serial_model.calls


def test_batched_ranking_preserves_seeds_effects_and_order(dataset):
    serial_model, batched_model = FeatureModel(), FeatureModel()
    settings = {"trust_sample_count": 2, "explainability_sample_count": 2}
    serial = run(serial_model, dataset, metric_evaluation_batch_size=1, **settings)
    batched = run(batched_model, dataset, metric_evaluation_batch_size=4, **settings)
    serial_evidence = [row["explainability"]["ranking_evidence"] for row in serial["perturbation_samples"]]
    batched_evidence = [row["explainability"]["ranking_evidence"] for row in batched["perturbation_samples"]]
    assert batched_evidence == serial_evidence
    assert batched_model.calls < serial_model.calls


def test_explanation_failure_does_not_discard_trust(dataset, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("explanation failure")
    monkeypatch.setattr(evaluation.outputs, "candidate_units", broken)
    result = run(FeatureModel(), dataset)
    assert result["trust_status"] == "complete"
    assert result["explainability_status"] == "failed"
    assert result["explainability_score"] is None
    assert result["explainability_evaluation"]["sample_counts"]["failed"] == 4


def test_timeout_is_independent_and_resume_preserves_completed_explanations(dataset, tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    original_deadline = evaluation._deadline
    monkeypatch.setattr(evaluation, "_deadline", lambda plan, stage: -1 if stage == "trust" else original_deadline(plan, stage))
    first = run(FeatureModel(), dataset, metric_evaluation_progress_path=str(path))
    assert first["trust_status"] == "insufficient_evidence"
    assert first["trust_evaluation"]["time_budget_exhausted"]
    assert first["explainability_score"] is not None
    monkeypatch.setattr(evaluation, "_deadline", original_deadline)
    second = run(FeatureModel(), dataset, metric_evaluation_progress_path=str(path), trust_max_duration_s=60)
    assert second["metric_evaluation_resumed"]
    assert second["trust_status"] == "complete"
    assert [r["explainability"] for r in first["perturbation_samples"]] == [r["explainability"] for r in second["perturbation_samples"]]
    saved = json.loads(path.read_text())
    assert saved["provenance"]["evaluation_definitions"]


@pytest.mark.parametrize("change", ["weights", "definition", "source_ids", "coverage", "replacements"])
def test_resume_rejects_changed_live_weights_definitions_or_source_ids(dataset, tmp_path, change):
    path = tmp_path / "progress.json"
    kwargs = {"metric_evaluation_progress_path": str(path), "metric_evaluation_source_ids": [f"source{i}" for i in range(12)],
              "metric_evaluation_model_hash": "stale-declared-hash"}
    run(FeatureModel(), dataset, **kwargs)
    model = FeatureModel(1 if change == "weights" else 0)
    if change == "definition":
        kwargs["feature_scales"] = [10., 10., 10., 10.]
    elif change == "source_ids":
        kwargs["metric_evaluation_source_ids"] = [f"different{i}" for i in range(12)]
    elif change == "coverage":
        kwargs["metric_evaluation_min_coverage"] = .9
    elif change == "replacements":
        kwargs["explanation_replacement_methods"] = ["empirical"]
    with pytest.raises(ValueError, match="does not match frozen"):
        run(model, dataset, **kwargs)


def test_empty_model_state_never_counts_as_verified_weights():
    class Empty:
        def state_dict(self):
            return {}
    identity = evaluation._model_identity(Empty(), {"final_model_hash": "declared"}, {})
    assert identity["verified_weights"] is False
    assert identity["declared_model_hash"] == "declared"


def test_resume_rejects_changed_effective_generation_configuration(dataset, tmp_path, monkeypatch):
    model = FeatureModel()
    model.core = SimpleNamespace(task_spec=SimpleNamespace(supports_generation=True), generation_config={"max_new_tokens": 10, "do_sample": True})
    def predict(model, samples, **kwargs):
        return [evaluation.outputs.TaskOutput("classification", 0, np.array([1., 0.])) for _ in samples]
    monkeypatch.setattr(evaluation.outputs, "predict_task_outputs", predict)
    path = tmp_path / "progress.json"
    result = run(model, dataset, metric_evaluation_progress_path=str(path))
    assert result["metric_evaluation_provenance"]["runtime"]["generation_config"] == {"max_new_tokens": 10, "do_sample": False}
    model.core.generation_config["max_new_tokens"] = 20
    with pytest.raises(ValueError, match="does not match frozen"):
        run(model, dataset, metric_evaluation_progress_path=str(path))


def test_larger_requested_budget_and_group_requirement_remain_insufficient(dataset):
    result = run(FeatureModel(), dataset, trust_sample_count=16, metric_evaluation_group_ids=["document"] * 12)
    assert result["trust_evaluation"]["sample_counts"]["requested"] == 16
    assert result["trust_evaluation"]["sample_counts"]["successful"] == 12
    assert result["trust_evaluation"]["independent_inputs"] == 1
    assert result["trust_evaluation"]["ci95"] is None
    assert result["trust_status"] == "insufficient_evidence"
    assert not result["trust_evidence_sufficient"]


def test_duplicate_source_ids_cannot_inflate_independent_input_evidence(dataset):
    result = run(FeatureModel(), dataset, metric_evaluation_source_ids=["repeated-source"] * 12)
    assert result["trust_evaluation"]["independent_inputs"] == 1
    assert not result["trust_evidence_sufficient"]


@pytest.mark.parametrize("count,trust_complete,explanation_complete", [
    (7, False, False), (8, False, True), (15, False, True), (16, True, True), (18, True, True), (40, True, True)])
def test_benchmark_caps_targets_without_lowering_metric_specific_floors(count, trust_complete, explanation_complete):
    x = np.random.default_rng(6).normal(size=(count, 4))
    result = evaluation.run_production_evaluation(FeatureModel(constant=True), x, np.zeros(count, dtype=int),
        task_family="classification", config={"enable_perturbation_metrics": True,
            "metric_evaluation_bootstrap_resamples": 100})
    assert result["evaluation_protocol_version"] == "benchmark_evaluation_v1"
    assert result["trust_evidence_sufficient"] is trust_complete
    assert result["explainability_evidence_sufficient"] is explanation_complete
    for stage, configured, floor in (("trust", 32, 16), ("explainability", 16, 8)):
        evidence = result[f"{stage}_evaluation"]
        assert evidence["minimum_independent_inputs"] == floor
        counts = evidence["sample_counts"]
        assert counts["configured_target"] == configured
        assert counts["eligible_independent_inputs"] == count
        assert counts["effective_target"] == counts["successful"] == min(count, configured)
        assert result[f"{stage}_independent_inputs"] == min(count, configured)
    assert len({r["sample_index"] for r in result["perturbation_samples"]}) == min(count, 32)


def test_benchmark_selects_one_row_per_independent_group_and_source():
    x = np.random.default_rng(6).normal(size=(40, 4))
    result = evaluation.run_production_evaluation(FeatureModel(constant=True), x, np.zeros(40),
        task_family="classification", config={"enable_perturbation_metrics": True,
            "metric_evaluation_group_ids": [f"doc{i // 4}" for i in range(40)],
            "metric_evaluation_source_ids": [f"source{i // 2}" for i in range(40)],
            "metric_evaluation_bootstrap_resamples": 100})
    rows = result["perturbation_samples"]
    assert len(rows) == len({r["source_id"] for r in rows}) == len({r["trust"]["group_id"] for r in rows}) == 10
    assert result["trust_evaluation"]["sample_counts"]["effective_target"] == 10
    assert not result["trust_evidence_sufficient"]
    assert result["explainability_evidence_sufficient"]


def test_repeated_source_bridging_groups_is_one_independent_component():
    selected, eligible = evaluation.independent_sample_indices(4, 32, [{}] * 4, 42,
        ["a", "a", "b", "c"], ["x", "y", "y", "z"])
    assert eligible == len(selected) == 2
    assert len(set(selected) & {0, 1, 2}) == 1


def test_benchmark_failures_do_not_reduce_effective_target(dataset, monkeypatch):
    def unsupported(*args, **kwargs):
        raise evaluation.outputs.UnsupportedTaskOutput("no_validated_operating_operator")
    monkeypatch.setattr(evaluation.outputs, "perturb_input", unsupported)
    result = run(FeatureModel(), dataset, evaluation_protocol_version="benchmark_evaluation_v1",
        metric_evaluation_profile="benchmark", trust_sample_count=32)
    assert result["trust_status"] == "unsupported"
    assert result["trust_evaluation"]["sample_counts"]["effective_target"] == 12
    assert result["trust_evaluation"]["sample_counts"]["excluded"] == 12
    assert result["trust_score"] is None


@pytest.mark.parametrize("stage,requested,successful,trial_successes,wide,expected", [
    ("trust", 32, 31, 4, False, []),
    ("trust", 32, 28, 4, False, []),
    ("explainability", 16, 13, 4, False, []),
    ("trust", 20, 16, 4, False, []),  # exact 80% boundary
    ("trust", 32, 25, 4, False, ["sample_coverage_below_minimum", "trial_coverage_below_minimum"]),
    ("trust", 18, 12, 4, False, ["sample_coverage_below_minimum", "too_few_independent_inputs", "trial_coverage_below_minimum"]),
    ("trust", 15, 15, 4, False, ["too_few_independent_inputs"]),
    ("explainability", 7, 7, 4, False, ["too_few_independent_inputs"]),
    ("explainability", 16, 16, 3, False, ["trial_coverage_below_minimum"]),
    ("explainability", 16, 16, 4, True, ["precision_requirement_not_met"]),
])
def test_benchmark_gates_counts_coverage_and_precision_independently(stage, requested, successful, trial_successes, wide, expected):
    plan = evaluation.resolve_evaluation_plan({"metric_evaluation_bootstrap_resamples": 100})
    records = {}
    for i in range(requested):
        score = float(i % 2) if wide else .5
        trials = [{"status": "successful", "score": score} if i < successful and j < trial_successes else
                  {"status": "excluded", "reason": "invalid_intervention"} for j in range(4)]
        records[str(i)] = {"sample_index": i, "group_id": str(i), "strata": {}, "trials": trials,
                           "score": score if i < successful else None, "done": True, "attempted": True}
    evidence = evaluation._stage_summary({"samples": records, "finished": True}, requested, 4, plan,
        {"eligible_independent_inputs": requested, "model_identity": {"verified_weights": True}, "explicit_grouping": False}, stage)
    assert evidence["insufficiency_reasons"] == expected
    assert set(evidence["insufficiency_details"]) == set(expected)
    assert evidence["status"] == ("insufficient_evidence" if expected else "complete")
    assert evidence["sample_coverage"] == successful / requested
    assert evidence["trial_coverage"] == successful * trial_successes / (requested * 4)
    if successful < requested or trial_successes < 4:
        assert evidence["exclusion_reasons"]["invalid_intervention"] == requested * 4 - successful * trial_successes
    if wide:
        assert evidence["insufficiency_details"]["precision_requirement_not_met"]["cause"] == "confidence_interval_too_wide"


def test_benchmark_deadline_can_leave_sufficient_evidence_but_stage_errors_cannot():
    plan = evaluation.resolve_evaluation_plan({"metric_evaluation_bootstrap_resamples": 100})
    stage = {"finished": False, "truncated": True, "samples": {
        str(i): {"sample_index": i, "group_id": str(i), "strata": {}, "score": .9, "attempted": True,
                 "done": True, "trials": [{"status": "successful", "score": .9}] * 4} for i in range(26)}}
    provenance = {"eligible_independent_inputs": 100, "model_identity": {"verified_weights": True}, "explicit_grouping": False}
    result = evaluation._stage_summary(stage, 32, 4, plan, provenance, "trust")
    assert result["status"] == "complete" and result["time_budget_exhausted"]
    assert result["sample_counts"]["effective_target"] == 32
    stage["stage_error"] = "RuntimeError: broken adapter"
    result = evaluation._stage_summary(stage, 32, 4, plan, provenance, "trust")
    assert result["insufficiency_reasons"] == ["stage_error"]
    assert not result["evidence_sufficient"]


def test_benchmark_single_replacement_reuses_empirical_trials_and_halves_validation_work(dataset):
    settings = dict(evaluation_protocol_version="benchmark_evaluation_v1", metric_evaluation_profile="benchmark")
    single_model, dual_model = FeatureModel(), FeatureModel()
    single = run(single_model, dataset, **settings)
    dual = run(dual_model, dataset, explanation_replacement_methods=["empirical", "mean"], **settings)
    for a, b in zip(single["perturbation_samples"], dual["perturbation_samples"]):
        a, b = a["explainability"], b["explainability"]
        assert a["ranking"] == b["ranking"]
        assert a["trials"] == [t for t in b["trials"] if t["replacement_method"] == "empirical"]
    assert single_model.calls < dual_model.calls
    assert single["explainability_evaluation"]["trial_counts"]["requested"] * 2 == dual["explainability_evaluation"]["trial_counts"]["requested"]
    for stage in ("trust", "explainability"):
        assert single[f"{stage}_duration_s"] == single[f"{stage}_evaluation"]["duration_s"] >= 0
        assert single[f"{stage}_insufficiency_reasons"] == single[f"{stage}_evaluation"]["insufficiency_reasons"]


def test_empty_pool_is_nullable_and_keeps_requested_budgets():
    result = run(FeatureModel(), (np.empty((0, 4)), np.empty(0)))
    assert result["trust_score"] is None
    assert result["explainability_score"] is None
    assert result["trust_evaluation"]["sample_counts"]["requested"] == 4
    assert result["trust_evaluation"]["sample_counts"]["attempted"] == 0


def test_label_annotations_do_not_change_individual_rankings(dataset):
    kwargs = {"trust_sample_count": 12, "explainability_sample_count": 12}
    first = run(FeatureModel(), dataset, **kwargs)
    second = run(FeatureModel(), (dataset[0], 1 - dataset[1]), **kwargs)
    evidence = lambda result: {row["sample_index"]: row["explainability"]["ranking_evidence"] for row in result["perturbation_samples"]}
    assert evidence(first) == evidence(second)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_causal_preparation_excludes_bad_boundaries_without_exposing_targets(monkeypatch, all_invalid):
    model = FeatureModel()
    model.core = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), label_pad_value=-100)
    x = {"input_ids": np.array([[1, 2, 80, 81], [2, 3, 82, 83], [4, 5, 84, 85]]),
         "attention_mask": np.ones((3, 4), dtype=int)}
    y = x["input_ids"].copy()
    if not all_invalid:
        y[:2, :2] = -100
    observed = []
    def predictions(model, samples, **kwargs):
        for sample in samples:
            assert np.all(np.asarray(sample["input_ids"]) < 80), "target tokens reached inference"
            observed.append(sample)
        return [evaluation.outputs.TaskOutput("generation", (1, 2)) for _ in samples]
    def changed(sample, *args, **kwargs):
        altered = {key: value.copy() for key, value in sample.items()}
        altered["input_ids"][0] += 1
        return evaluation.outputs.PerturbedInput(altered, "validated_test_transform", .1, True)
    monkeypatch.setattr(evaluation.outputs, "predict_task_outputs", predictions)
    monkeypatch.setattr(evaluation.outputs, "perturb_input", changed)
    result = evaluation.run_production_evaluation(model, x, y, task_family="generation", hf_task="causal_lm_generation",
        config=config(trust_sample_count=3, explainability_sample_count=3))
    for key in ("trust_evaluation", "explainability_evaluation"):
        assert result[key]["sample_counts"]["excluded"] == (3 if all_invalid else 1)
        assert result[key]["sample_counts"]["successful"] == (0 if all_invalid else 2)
        assert result[key]["status"] == ("unsupported" if all_invalid else "insufficient_evidence")
    assert bool(observed) is not all_invalid
    assert result["metric_evaluation_provenance"]["input_preparation"]["target_tokens_visible_to_model"] is False
