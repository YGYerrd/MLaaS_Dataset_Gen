import pytest

from mlaas_data_generator.terminal import print_service_summary


@pytest.mark.parametrize("wrapped", [False, True])
def test_summary_shows_separate_scores_times_and_insufficiency(capsys, wrapped):
    metrics = {
        "trust_score": .9876, "trust_status": "complete", "trust_duration_s": 2.7,
        "trust_independent_inputs": 31, "trust_evaluation": {"sample_coverage": 31 / 32},
        "explainability_score": .2389, "explainability_status": "insufficient_evidence",
        "explainability_duration_s": 25.95, "explainability_independent_inputs": 16,
        "explainability_evaluation": {"sample_coverage": 1.,
            "insufficiency_reasons": ["precision_requirement_not_met"],
            "exclusion_reasons": {"valid_output_alignment_changed": 2}},
    }
    if wrapped:
        metrics = {key: {"value": value} for key, value in metrics.items()}
    print_service_summary({"service_id": "test"}, metrics)
    output = capsys.readouterr().out
    assert "trust_score=0.9876 | status=complete | time=2.70s | n=31 | coverage=96.9%" in output
    assert "explainability_score=0.2389 | status=insufficient_evidence | time=25.9s | n=16 | coverage=100.0%" in output
    assert "reasons=precision_requirement_not_met, valid_output_alignment_changed" in output


@pytest.mark.parametrize("status", ["unsupported", "failed", "disabled"])
def test_summary_keeps_missing_scores_and_times_explicit(capsys, status):
    print_service_summary({"service_id": "test"}, {
        "trust_score": {"value": None}, "explainability_score": None,
        "trust_status": status, "explainability_status": status,
        "metric_evaluation_disabled": status == "disabled",
    })
    output = capsys.readouterr().out
    for dimension in ("trust", "explainability"):
        assert f"{dimension}_score=n/a | status={status} | time=n/a" in output


def test_summary_can_read_historical_nested_times_and_labels_resumed_totals(capsys):
    print_service_summary({"service_id": "test"}, {
        "trust_evaluation": {"duration_s": 1.25},
        "explainability_evaluation": {"duration_s": 4.75},
        "metric_evaluation_resumed": True,
    })
    output = capsys.readouterr().out
    assert "time=1.25s" in output and "time=4.75s" in output
    assert "Evaluation times include work from earlier attempts." in output
