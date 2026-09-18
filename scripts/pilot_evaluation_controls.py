"""Run the full evaluation budgets against deterministic synthetic controls.

This validates directional properties, not scientific validity for every real
task family. Service generation defaults to the smaller benchmark profile.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlaas_data_generator.services.production_evaluation import run_production_evaluation
from mlaas_data_generator.evaluation_protocol import PROFILES, resolve_evaluation_plan


class ControlModel:
    def __init__(self, mode="feature", feature=0):
        self.mode, self.feature = mode, feature

    def get_weights(self):
        return [np.array([{"feature": 0, "constant": 1, "fragile": 2}[self.mode], self.feature, 6., 10000.])]

    def predict(self, values, **kwargs):
        x = np.asarray(values)
        if self.mode == "constant":
            p = np.zeros(len(x))
        elif self.mode == "fragile":
            p = (np.sin(x[:, 0] * 10000.) > 0).astype(float)
        else:
            p = 1. / (1. + np.exp(-x[:, self.feature] * 6))
        return np.column_stack([1. - p, p])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/metric_evaluation_sanity_20260916"))
    parser.add_argument("--evaluation-profile", choices=tuple(PROFILES), default="pilot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(20260916).normal(size=(512, 32))
    y = (x[:, 0] > 0).astype(int)
    cases = [("known_feature", "feature", 0, "occlusion"), ("random_explanation", "feature", 0, "random"),
             ("reversed_explanation", "feature", 0, "reverse_occlusion"), ("constant_predictor", "constant", 0, "occlusion"),
             ("fragile_predictor", "fragile", 0, "occlusion"), ("randomized_model", "feature", 1, "occlusion")]
    plan = resolve_evaluation_plan({"metric_evaluation_profile": args.evaluation_profile})
    report = {"protocol_version": plan["protocol_version"], "profile": args.evaluation_profile, "seed": 20260916,
              "input_count": 512, "feature_count": 32, "purpose": "synthetic_directional_sanity_checks",
              "scope": "Control results are not a validation claim for every real task family.", "cases": {}}
    start = time.monotonic()
    for name, mode, feature, explanation in cases:
        print(f"[controls] {name} starts", flush=True)
        result = run_production_evaluation(ControlModel(mode, feature), x, y, task_family="classification", config={
            "enable_perturbation_metrics": True, "metric_evaluation_profile": args.evaluation_profile, "evaluation_seed": 20260916,
            "metric_evaluation_explanation_method": explanation, "explainability_expected_feature_indices": [0],
            "dataset_name": "synthetic_known_feature_32", "metric_evaluation_source_ids": [f"synthetic:{i}" for i in range(len(x))],
            "perturbation_progress_logging": True,
        })
        evidence_path = args.output / f"{name}.evidence.json.gz"
        with gzip.open(evidence_path, "wt", encoding="utf-8") as handle:
            json.dump(result, handle, allow_nan=False)
        top_units = [record["explainability"]["ranking"][0] for record in result["perturbation_samples"]
                     if record.get("explainability") and record["explainability"].get("ranking")]
        report["cases"][name] = {"trust_score": result["trust_score"], "explainability_score": result["explainability_score"],
            "trust_status": result["trust_status"], "explainability_status": result["explainability_status"],
            "trust_evaluation": result["trust_evaluation"], "explainability_evaluation": result["explainability_evaluation"],
            "provenance": result["metric_evaluation_provenance"], "duration_s": result["perturbation_duration_s"],
            "top_feature_0_fraction": float(np.mean([unit == ["feature", 0] for unit in top_units])) if top_units else None,
            "top_feature_1_fraction": float(np.mean([unit == ["feature", 1] for unit in top_units])) if top_units else None,
            "evidence_path": str(evidence_path)}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[controls] {name} trust={result['trust_score']} faithfulness={result['explainability_score']} duration={result['perturbation_duration_s']:.1f}s", flush=True)
    cases = report["cases"]
    report["checks"] = {
        "constant_is_stable": cases["constant_predictor"]["trust_score"] == 1.,
        "constant_quality_is_poor": cases["constant_predictor"]["trust_evaluation"]["clean_quality_mean"] < .6,
        "constant_has_zero_faithfulness": cases["constant_predictor"]["explainability_score"] == 0.,
        "known_feature_beats_random": cases["known_feature"]["explainability_score"] > cases["random_explanation"]["explainability_score"] + .05,
        "known_feature_beats_reverse": cases["known_feature"]["explainability_score"] > cases["reversed_explanation"]["explainability_score"] + .05,
        "fragile_predictor_loses_stability": cases["fragile_predictor"]["trust_score"] < .8,
        "model_randomization_changes_rankings": cases["randomized_model"]["top_feature_1_fraction"] > .8 and cases["known_feature"]["top_feature_0_fraction"] > .8,
        "all_requested_budgets_completed": all(
            case[f"{stage}_evaluation"]["sample_counts"]["completed"] == case[f"{stage}_evaluation"]["sample_counts"]["effective_target"]
            and case[f"{stage}_evaluation"]["trial_counts"]["successful"] == case[f"{stage}_evaluation"]["trial_counts"]["requested"]
            for case in cases.values() for stage in ("trust", "explainability")),
    }
    report["passed"] = all(report["checks"].values())
    report["duration_s"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "duration_s": report["duration_s"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
