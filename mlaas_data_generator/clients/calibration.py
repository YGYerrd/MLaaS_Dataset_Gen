"""Frozen historical recipe evidence. No database reads during generation.

Rebuild explicitly with ``python -m mlaas_data_generator.clients.calibration``.
Evidence pools stay separate by task and metric contract.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3
from statistics import median

from ..provenance import sha256_json

VERSION = "recipe_evidence_v3"
DEFAULT_EVIDENCE = Path(__file__).with_name("recipe_evidence.json")
MIXTURE = {"supported": 0.7, "challenging": 0.2, "stress": 0.1}
KNOBS = ("optimizer", "learning_rate", "batch_size", "gradient_accumulation_steps",
         "training_epochs", "weight_decay", "warmup_ratio")
MIN_OBSERVATIONS = 3
TASKS = {"sequence_classification": "text_classification", "image_classification": "image_classification",
         "token_classification": "token_classification", "sentence_similarity": "sentence_similarity"}


def outcome(accuracy, f1, distribution, *, average="weighted"):
    """Baseline-relative triage, not a prediction-level collapse diagnosis."""
    if not isinstance(distribution, dict) or len(distribution) < 2:
        return None
    counts = list(distribution.values())
    if any(not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in counts):
        return None
    n = sum(counts)
    if n <= 0 or not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in (accuracy, f1)):
        return None
    proportions = [c / n for c in counts]
    classes = sum(c > 0 for c in counts)
    def baseline_f1(p):
        return 2*p*p/(1+p) if average == "weighted" else 2*p/(1+p)/classes
    if any(abs(accuracy-p) < 1e-10 and abs(f1-baseline_f1(p)) < 1e-10 for p in proportions):
        return "collapsed_signature"
    p = max(proportions)
    if accuracy >= p + 0.05 and f1 >= baseline_f1(p) + 0.05:
        return "healthy"
    return "weak"


def token_distribution(distribution):
    """Only use token histograms when the stored top-k contains every label."""
    if not isinstance(distribution, dict):
        return None
    counts = distribution.get("top_supervised_token_ids")
    if (not isinstance(counts, dict) or len(counts) != distribution.get("unique_supervised_token_ids")
            or sum(counts.values()) != distribution.get("supervised_tokens")):
        return None
    return counts


def record_outcome(r, train, benchmark):
    task = TASKS[r["hf_task"]]
    if task == "sentence_similarity":
        # Old binary pair tasks wrote accuracy/F1 under correlation names. Do
        # not mine those aliases as continuous-score regression evidence.
        meta = json.loads(r.get("metadata_json") or "{}")
        if meta.get("loader_meta", {}).get("task_type") != "regression":
            raise ValueError("unverified sentence-pair regression contract")
        values = (r.get("pearson"), r.get("spearman"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and -1 <= v <= 1 for v in values):
            raise ValueError("invalid correlation metrics")
        # Zero correlation cannot prove constant outputs; keep it weak.
        label = "healthy" if min(values) >= 0.3 else "weak"
        return label, train, benchmark, "pearson_spearman_v1"
    if task == "token_classification":
        train, benchmark = token_distribution(train), token_distribution(benchmark)
    average = "macro" if task == "image_classification" else "weighted"
    f1 = r.get("macro_f1") if average == "macro" else r.get("f1")
    return outcome(r.get("accuracy"), f1, benchmark, average=average), train, benchmark, f"accuracy_{average}_f1_v1"


def build_evidence(db):
    """Read one consistent snapshot; preserve counts including weak outcomes.

    Imported duplicate observations are counted once. Single-class/very skewed
    training records inform family outcomes but not recipe viability estimates.
    Failures and other task/metric contracts are never counted as healthy runs.
    """
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN")
        total = con.execute("SELECT count(*) FROM services").fetchone()[0]
        columns = {r[1] for r in con.execute("PRAGMA table_info(services)")}
        meta_column = "metadata_json" if "metadata_json" in columns else "'{}' AS metadata_json"
        records = {r["service_id"]: dict(r) for r in con.execute(f"""
            SELECT service_id,model_id,dataset_name,dataset_config,service_config_json,hf_task,{meta_column}
            FROM services WHERE status='completed'
            AND hf_task IN ('sequence_classification','image_classification','token_classification','sentence_similarity')
            AND model_type='hf_finetune' AND training_regime='finetune_transfer'""")}
        keys = ("accuracy", "f1", "macro_f1", "pearson", "spearman", "loss", "train_distribution_json", "benchmark_distribution_json",
                "optimizer_step_count", "successful_optimizer_step_count", "train_stopped_early")
        for m in con.execute("SELECT service_id,metric_name,value_num,value_int,value_bool,value_text,value_json "
                             "FROM service_metrics WHERE metric_name IN (%s)" % ",".join("?" for _ in keys), keys):
            if m["service_id"] in records:
                records[m["service_id"]][m["metric_name"]] = next((v for v in tuple(m)[2:] if v is not None), None)
    finally:
        con.close()
    groups, family_outcomes, excluded, seen, observations = {}, defaultdict(Counter), Counter(), set(), []
    for sid, r in sorted(records.items()):
        try:
            cfg = json.loads(r["service_config_json"])
            train = json.loads(r.get("train_distribution_json") or "null")
            benchmark = json.loads(r.get("benchmark_distribution_json") or "null")
            task = TASKS[r["hf_task"]]
            label, train, benchmark, metric_contract = record_outcome(r, train, benchmark)
            if label is None or not isinstance(train, dict) or not train:
                raise ValueError("unusable or incomplete metric distributions")
            if not isinstance(r.get("loss"), (float, int)) or not math.isfinite(r["loss"]):
                raise ValueError("nonfinite or missing loss")
            knobs = {k: cfg[k] for k in KNOBS}
            if knobs["optimizer"] not in {"adam", "adamw", "sgd", "rmsprop"}:
                raise ValueError("unsupported optimizer")
            if any(not isinstance(knobs[k], (int, float)) or not math.isfinite(knobs[k]) for k in KNOBS[1:]):
                raise ValueError("nonfinite recipe")
            if min(knobs[k] for k in ("learning_rate", "batch_size", "gradient_accumulation_steps", "training_epochs")) <= 0:
                raise ValueError("nonpositive recipe")
            if knobs["weight_decay"] < 0 or not 0 <= knobs["warmup_ratio"] < 1:
                raise ValueError("invalid regularization")
            steps = r.get("successful_optimizer_step_count", r.get("optimizer_step_count"))
            if not isinstance(steps, (int, float)) or not math.isfinite(steps) or steps < 1:
                raise ValueError("no recorded optimizer steps")
            if cfg.get("generation_mode") in {"shared_parent", "fedavg"}:
                raise ValueError("different initialization regime")
            fingerprint = sha256_json({"model": r["model_id"], "dataset": r["dataset_name"], "subset": r["dataset_config"],
                "task": task, "metric_contract": metric_contract, "config": cfg, "train": train, "benchmark": benchmark,
                **{key: r.get(key) for key in ("accuracy", "f1", "macro_f1", "pearson", "spearman", "loss")}})
            if fingerprint in seen:
                excluded["duplicate_observation"] += 1
                continue
            seen.add(fingerprint)
            observations.append(fingerprint)
            family = (r["model_id"], r["dataset_name"], r["dataset_config"], task)
            family_outcomes[family][label] += 1
            counts = list(train.values())
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in counts) or sum(counts) <= 0:
                raise ValueError("invalid training distribution")
            min_fraction = 0.05 if task == "text_classification" else 0.1 / max(1, len(benchmark))
            if task != "sentence_similarity" and (len([v for v in counts if v > 0]) != len(benchmark) or min(counts) / sum(counts) < min_fraction):
                excluded["insufficient_training_class_coverage"] += 1
                continue
            if r.get("train_stopped_early"):
                excluded["training_time_censored"] += 1
                continue
            context = {"model_id": r["model_id"], "dataset_name": r["dataset_name"], "dataset_config": r["dataset_config"],
                       "task": task, "metric_contract": metric_contract,
                       "max_length": None if task == "image_classification" else cfg.get("max_length", 128)}
            key = sha256_json({**context, "knobs": knobs})
            group = groups.setdefault(key, {**context, "knobs": knobs, "outcomes": Counter(), "steps": [], "source_service_ids": []})
            group["outcomes"][label] += 1
            group["steps"].append(int(steps))
            group["source_service_ids"].append(sid)
        except (ValueError, TypeError, KeyError) as exc:
            excluded[str(exc)] += 1
    templates = []
    for key, group in sorted(groups.items()):
        steps = group.pop("steps")
        templates.append({"id": key[:16], **group, "observations": len(steps),
                          "minimum_steps": max(5, int(median(steps)))})
    doc = {"version": VERSION, "source": Path(db).name, "source_observation_hash": sha256_json(sorted(observations)),
           "scope": "independent_hf_composable_tasks", "source_records": total,
           "eligible_records": len(records), "eligible_classification_records": sum(r["hf_task"] == "sequence_classification" for r in records.values()),
           "eligible_by_task": dict(Counter(TASKS[r["hf_task"]] for r in records.values())), "unique_observations": len(observations),
           "exclusions": dict(excluded),
           "definitions": {"healthy": "classification: accuracy and task-specific F1 exceed majority baseline by >= 0.05; sentence regression: Pearson and Spearman >= 0.3",
               "collapsed_signature": "classification accuracy/task-specific F1 match a constant-class baseline; not prediction-level proof",
               "weak": "other finite outcomes; zero correlation does not prove regression collapse", "minimum_observations": MIN_OBSERVATIONS,
               "supported": "at least 3 observations, >= 50% healthy and <= 25% collapsed signatures",
               "limitations": "completed runs only; historical benchmarks informed calibration; no causal optimizer comparisons"},
           "families": [{"model_id": k[0], "dataset_name": k[1], "dataset_config": k[2], "task": k[3], "outcomes": dict(v)}
                        for k, v in sorted(family_outcomes.items(), key=lambda item: str(item[0]))], "templates": templates}
    doc["evidence_hash"] = sha256_json(doc)
    return doc


@lru_cache(maxsize=1)
def load_evidence():
    doc = json.loads(DEFAULT_EVIDENCE.read_text(encoding="utf-8"))
    if doc.get("version") != VERSION or doc.get("evidence_hash") != sha256_json({k: v for k, v in doc.items() if k != "evidence_hash"}):
        raise ValueError("recipe calibration evidence failed integrity verification")
    return doc


def attach_calibration(policy, config, *, evidence=None):
    """Prefer exact family, then same model/task, then task-wide evidence.

    Replay individual joint templates; never construct a Cartesian product of
    historically successful marginal ranges. Uncalibrated tasks stay explicit.
    """
    task = policy["task"]
    if config.get("mobilebert_head_normalization") == "layer_norm_v1":
        policy["calibration"] = {"status": "uncalibrated", "reason":
            "historical recipe evidence does not bind MobileBERT layer_norm_v1 adaptation"}
        return
    if config.get("task_contract_version") or config.get("task_contract"):
        policy["calibration"] = {"status": "uncalibrated", "reason": "historical evidence does not bind the new task semantics and evaluation protocol"}
        return
    if task not in TASKS.values() or config.get("parent_state_hash") or config.get("generation_mode") in {"shared_parent", "fedavg"}:
        policy["calibration"] = {"status": "uncalibrated", "reason": "no comparable audited metric/initialization evidence"}
        return
    if task == "sentence_similarity":
        from ..family_preflight import registered_dataset_spec
        if registered_dataset_spec({**config, "task": task}).get("task_type") != "regression":
            policy["calibration"] = {"status": "uncalibrated", "reason": "sentence-pair regression mode is not confirmed by the dataset contract"}
            return
    doc = load_evidence() if evidence is None else evidence
    model = config.get("hf_model_id") or config.get("model_id")
    length = None if task == "image_classification" else config.get("max_length", 128)
    candidates = [t for t in doc["templates"] if t["task"] == task and t["max_length"] == length]
    exact = [t for t in candidates if (t["model_id"], t["dataset_name"], t["dataset_config"]) ==
             (model, config.get("dataset_name"), config.get("dataset_config"))]
    same_model = [t for t in candidates if t["model_id"] == model]
    limits = policy["limits"]
    selected, scope = [], "none"
    for scope, pool in (("model_dataset_task", exact), ("model_task", same_model), ("task", candidates)):
        # Pool counts for the same complete recipe. Datasets are not scored
        # against each other: outcomes use each record's own baseline.
        merged = {}
        for t in pool:
            if (t["knobs"]["optimizer"] not in policy["optimizers"] or t["knobs"]["batch_size"] not in limits["batch_size"]
                    or t["knobs"]["batch_size"]*t["knobs"]["gradient_accumulation_steps"] > limits["max_effective_batch_size"]):
                continue
            key = sha256_json(t["knobs"])
            entry = merged.setdefault(key, {"id": key[:16], "knobs": t["knobs"], "outcomes": Counter(), "steps": [], "evidence_ids": []})
            entry["outcomes"].update(t["outcomes"])
            entry["steps"].extend([t["minimum_steps"]] * t["observations"])
            entry["evidence_ids"].append(t["id"])
        selected = []
        for entry in merged.values():
            counts = entry["outcomes"]
            n = sum(counts.values())
            if n >= MIN_OBSERVATIONS and counts["healthy"]/n >= 0.5 and counts["collapsed_signature"]/n <= 0.25:
                selected.append({k: v for k, v in entry.items() if k != "steps"} | {"minimum_steps": int(median(entry["steps"]))})
        if selected:
            break
    calibration = {"scope": scope if selected else "none", "templates": sorted(selected, key=lambda t: t["id"])}
    if selected:
        # Quantity remains independently sampled; extra epochs provide the
        # recorded update opportunity on a smaller subset, up to the frozen cap.
        limits["exploration_epochs"] = list(limits["training_epochs"])
        limits["training_epochs"] = sorted(set(limits["training_epochs"]) | set(range(1, int(policy.get("max_realized_training_epochs", 20)) + 1)))
        for template in selected:
            for key in ("weight_decay", "warmup_ratio", "gradient_accumulation_steps"):
                limits[key] = sorted(set(limits[key]) | {template["knobs"][key]})
            opt = template["knobs"]["optimizer"]
            limits["learning_rate_by_optimizer"][opt] = sorted(set(limits["learning_rate_by_optimizer"][opt]) | {template["knobs"]["learning_rate"]})
    policy["calibration"] = {"status": "available" if calibration["templates"] else "uncalibrated",
        "evidence_hash": doc["evidence_hash"], "source_observation_hash": doc["source_observation_hash"],
        "mixture": MIXTURE, **calibration,
        "class_coverage": {"supported": {"source_mix": 0.5, "minimum_classes": 2},
                           "challenging": {"source_mix": 0.1, "minimum_classes": 1},
                           "stress": {"source_mix": 0.0, "minimum_classes": 1}}}


def selected_template(recipe):
    if recipe.get("calibration_band") != "supported":
        return None
    from .training import calibration_limits
    templates = calibration_limits(recipe).get("templates", [])
    template = next((t for t in templates if t["id"] == recipe.get("calibration_template_id")), None)
    if template is None:
        raise ValueError("supported recipe has no matching frozen calibration template")
    return template


def sample_calibrated_knobs(policy, profiles, rows, *, seed):
    import numpy as np
    calibration = policy.get("calibration", {})
    if not profiles or not calibration.get("templates"):
        return
    indices = list(range(len(profiles)))
    rng = np.random.default_rng(int(sha256_json([seed, "calibration"])[:8], 16))
    weights = policy["calibration"]["mixture"]
    targets = {band: len(indices)*weight for band, weight in weights.items()}
    counts = {band: math.floor(value) for band, value in targets.items()}
    for band in sorted(weights, key=lambda b: -(targets[b]-counts[b]))[:len(indices)-sum(counts.values())]:
        counts[band] += 1
    bands = [band for band, count in counts.items() for _ in range(count)]
    rng.shuffle(bands)
    templates = list(calibration["templates"])
    rng.shuffle(templates)
    supported_index = 0
    for index, band in zip(indices, bands):
        rows[index]["calibration_band"] = band
        if band == "supported":
            template = templates[supported_index % len(templates)]
            supported_index += 1
            rows[index].update(template["knobs"])
            rows[index]["calibration_template_id"] = template["id"]
        elif band == "stress":
            limits = policy["limits"]
            # Explicitly short, aggressive recipes; all remain within the
            # frozen optimizer-specific policy. Outcomes are not guaranteed.
            rows[index]["training_epochs"] = min(limits.get("exploration_epochs", limits["training_epochs"]))
            rows[index]["learning_rate"] = max(limits["learning_rate_by_optimizer"][rows[index]["optimizer"]])
        rows[index]["effective_batch_size"] = rows[index]["batch_size"] * rows[index]["gradient_accumulation_steps"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Historical service database (opened read-only)")
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    doc = build_evidence(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("source_records", "eligible_by_task", "unique_observations", "exclusions", "evidence_hash")}, indent=2))


if __name__ == "__main__":
    main()
