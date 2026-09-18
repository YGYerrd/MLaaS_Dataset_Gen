"""Reassess frozen benchmark trials without inference or catalogue writes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlaas_data_generator.services.production_evaluation import _mean, _stage_summary


def reassess(report, dimension, methods=None):
    evaluation = report["evaluation"]
    original = evaluation[f"{dimension}_evaluation"]
    plan = {**report["plan"], "min_coverage": .8, "evidence_gate_version": "coverage_precision_v1"}
    records = [deepcopy(row[dimension]) for row in evaluation["perturbation_samples"] if row.get(dimension)]
    if methods is not None:
        plan["explanation_replacement_methods"] = methods
        for record in records:
            record["trials"] = [trial for trial in record["trials"] if trial["replacement_method"] in methods]
            record["score"] = _mean(trial.get("score") for trial in record["trials"])
    trials = (plan["trust_trials"] if dimension == "trust" else
              len(plan["explanation_replacement_methods"]) * len(plan["explainability_budget_fractions"]) * plan["explainability_random_trials"])
    stage = {"samples": {str(row["sample_index"]): row for row in records},
             "finished": all(row.get("done") for row in records), "truncated": original["time_budget_exhausted"],
             "stage_error": original.get("stage_error"),
             # Replacement-specific elapsed time was not measured in the pilot.
             "duration_s": original["duration_s"] if methods is None else None}
    result = _stage_summary(stage, plan[f"{dimension}_sample_count"], trials, plan,
                            evaluation["metric_evaluation_provenance"], dimension)
    return {key: result[key] for key in ("mean", "status", "independent_inputs", "ci_width", "sample_coverage",
        "trial_coverage", "insufficiency_reasons", "insufficiency_details", "exclusion_reasons", "duration_s")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/benchmark_evaluation_pilot_summary_20260916.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_evaluation_budget_review_20260916.json"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    rows = []
    for entry in source["reports"]:
        path = Path(entry["result_ref"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["result_sha256"]:
            raise ValueError(f"Pilot evidence hash mismatch: {path}")
        report = json.loads(raw)
        dual, single = reassess(report, "explainability"), reassess(report, "explainability", ["empirical"])
        rows.append({"profile": entry["profile_label"], "source_ref": str(path), "source_sha256": entry["result_sha256"],
            "trust": reassess(report, "trust"), "dual_replacement": dual, "empirical_only": single,
            "absolute_score_change": None if single["mean"] is None else abs(single["mean"] - dual["mean"]),
            "old_trust_status": report["evaluation"]["trust_status"]})
    supported = [row for row in rows if row["absolute_score_change"] is not None]
    result = {"source_ref": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "scope": "Post-hoc reaggregation of frozen pilot trials; no new inference or catalogue changes. Empirical trials and rankings are unchanged.",
        "decision": "Use empirical-only validation by default for benchmark; retain explicit dual replacement and larger-profile defaults.",
        "minimum_coverage": .8, "minimum_inputs": {"trust": 16, "explainability": 8}, "maximum_ci_width": .15,
        "maximum_absolute_score_change": max(row["absolute_score_change"] for row in supported),
        "status_changes_between_replacements": sum(row["dual_replacement"]["status"] != row["empirical_only"]["status"] for row in rows),
        "runtime_evidence": {"paired_validation_trials_per_input": {"dual": 24, "single": 12},
            "validation_predictions_per_input": {"dual": 144, "single": 72},
            "prediction_work_reduction_with_32_ranking_groups": 72 / (1 + 32 + 144),
            "limitation": "About 41% fewer prediction examples at 32 candidate groups, not measured wall-clock savings. Ranking, preparation and batching costs remain."},
        "profiles": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))
    for row in rows:
        print(row["profile"], "trust=" + row["trust"]["status"], "explanation=" + row["empirical_only"]["status"],
              "reasons=" + str(row["trust"]["insufficiency_reasons"] + row["empirical_only"]["insufficiency_reasons"]))


if __name__ == "__main__":
    main()
