"""Read-only historical evidence and automatic parent training selections."""
from __future__ import annotations

from .clients.resources import task_key

import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .provenance import canonical_json, sha256_json

SELECTION_VERSION = "parent_selection_v2"


def _parent_eligible(config):
    from .task_capabilities import resolve_capabilities
    decisions = resolve_capabilities(config)
    return all(decisions[key]["decision"] != "ineligible" for key in
               ("training", "checkpoint_restoration", "partitioning", "parameter_aggregation"))
ESSENTIAL_KNOBS = ("optimizer", "learning_rate", "batch_size", "training_epochs")
TRAINING_KNOBS = (*ESSENTIAL_KNOBS, "weight_decay", "momentum", "warmup_ratio",
                  "gradient_accumulation_steps", "mixed_precision", "precision_type")
FAMILY_FIELDS = (
    "model_type", "hf_model_id", "hf_model_revision", "dataset", "dataset_name",
    "dataset_config", "hf_dataset_revision", "task", "task_type", "hf_task",
    "input_schema", "output_schema", "max_length", "hidden_layers", "activation", "dropout",
    "column_mapping", "loader_template", "train_split", "test_split",
    "label_mapping", "label_format", "preprocessing_identity", "label_column", "text_column",
)




def quality_metric(config: Mapping[str, Any]) -> tuple[str, bool]:
    from .tasks import resolve_task_contract, TaskContractError
    try:
        contract = resolve_task_contract(config)
        return contract.evaluation["primary"], contract.evaluation["higher_is_better"]
    except TaskContractError:
        return "unsupported_metric_protocol", True


def family_identity(config: Mapping[str, Any]) -> str:
    return "family_" + sha256_json({k: config.get(k) for k in FAMILY_FIELDS})[:24]


def parent_identity(family_id: str, recipe_hash: str) -> str:
    return "parent_" + sha256_json([family_id, recipe_hash])[:24]


def _validate_training_config(config: Mapping[str, Any]) -> None:
    if any(config.get(k) is None for k in ESSENTIAL_KNOBS) or not _parent_eligible(config):
        raise ValueError("unsupported or incomplete parent configuration")
    optimizers = {"adam", "adamw", "sgd", "rmsprop"}
    if config.get("training_regime") == "generic":
        optimizers.add("adagrad")
        if int(config.get("gradient_accumulation_steps") or 1) != 1:
            raise ValueError("generic parent training requires gradient_accumulation_steps=1")
    if str(config.get("optimizer", "")).lower() not in optimizers:
        raise ValueError(f"unsupported parent optimizer: {config.get('optimizer')}")
    rate = float(config["learning_rate"])
    if not math.isfinite(rate) or rate <= 0 or min(int(config[k]) for k in ("batch_size", "training_epochs")) < 1:
        raise ValueError("invalid parent training settings")


def _object(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        decoded = json.loads(value or "{}")
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, TypeError):
        return {}


def historical_config(row: Mapping[str, Any]) -> tuple[dict, list[str]]:
    meta = _object(row.get("metadata_json"))
    loader = _object(meta.get("loader_meta"))
    stored = _object(row.get("service_config_json"))
    config = dict(_object(loader.get("dataset_args")))
    for key in (*FAMILY_FIELDS, "training_regime", "modality", "dataset"):
        value = row.get(key)
        if value is None:
            value = loader.get(key)
        if value is not None:
            config[key] = value
    config.update(stored)
    config["hf_model_id"] = config.get("hf_model_id") or row.get("model_id") or loader.get("hf_model_id")
    config["test_split"] = config.get("test_split") or row.get("benchmark_split") or loader.get("test_split")
    functional = _object(row.get("functional_attributes_json"))
    config.setdefault("label_mapping", loader.get("id2label", loader.get("label_mapping")))
    config.setdefault("label_format", loader.get("label_format", functional.get("label_format")))
    config.setdefault("preprocessing_identity", {
        key: loader.get(key) for key in ("schema", "input_shape", "max_length", "loader_template")
    })
    config["task"] = task_key(config)
    missing = [k for k in ESSENTIAL_KNOBS if config.get(k) is None]
    for key in ("hf_model_revision", "hf_dataset_revision"):
        if config.get("dataset") == "hf" and not config.get(key):
            missing.append(key)
    return config, missing


def select_parent_configs(history_db: str | Path, output: str | Path, *, limit: int = 5) -> dict:
    """Create evidence without opening the historical catalogue for writing."""
    if limit < 1:
        raise ValueError("parent selection limit must be positive")
    source = Path(history_db).resolve(strict=True)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        # One read transaction gives services and metrics a consistent snapshot.
        conn.execute("BEGIN")
        metrics: dict[str, dict] = defaultdict(dict)
        for row in conn.execute("SELECT service_id,metric_name,value_num,value_int,value_text FROM service_metrics"):
            metrics[row["service_id"]][row["metric_name"]] = (
                row["value_num"] if row["value_num"] is not None else
                row["value_int"] if row["value_int"] is not None else row["value_text"]
            )
        for raw in conn.execute("SELECT * FROM services ORDER BY service_id"):
            row = dict(raw)
            sid = row["service_id"]
            config, missing = historical_config(row)
            reason = None
            if row.get("status") != "completed":
                reason = "not_completed"
            elif not _parent_eligible(config):
                reason = "unsupported_family"
            elif any(k in missing for k in ESSENTIAL_KNOBS):
                reason = "missing_essential_training_settings"
            elif config.get("dataset") == "hf" and not config.get("hf_model_id"):
                reason = "missing_model_identity"
            if reason is None:
                try:
                    _validate_training_config(config)
                except (ValueError, TypeError, OverflowError):
                    reason = "unsupported_or_invalid_training_settings"
            name, higher = quality_metric(config)
            value = metrics[sid].get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                reason = reason or "missing_finite_task_metric"
            if reason:
                excluded.append({"service_id": sid, "reason": reason, "missing": missing})
                continue
            family = family_identity(config)
            functional = _object(row.get("functional_attributes_json"))
            loader = _object(_object(row.get("metadata_json")).get("loader_meta"))
            protocol = {
                "benchmark_split": config.get("test_split"),
                "benchmark_size": metrics[sid].get("benchmark_set_size"),
                "requested_benchmark_size": config.get("benchmark_sample_size"),
                "benchmark_seed": config.get("benchmark_sample_seed"),
                "evaluation_time_limit": config.get("max_eval_time_s"),
                "primary_metric": name,
                "output_semantics": functional.get("label_format"),
                "label_mapping": loader.get("id2label", loader.get("label_mapping")),
                "preprocessing": {k: loader.get(k) for k in ("schema", "input_shape", "max_length", "loader_template")},
            }
            # Training evidence retains local-data conditions rather than pooling
            # results from unlike budgets under one hyperparameter configuration.
            recipe = {k: config.get(k) for k in (*FAMILY_FIELDS, *TRAINING_KNOBS,
                      "sample_size", "max_samples", "split_strategy", "distribution_param", "max_train_time_s")}
            recipe_hash = sha256_json(recipe)
            key = (family, sha256_json(protocol), recipe_hash)
            groups[key].append({"service_id": sid, "quality": value, "config": config,
                                "missing": missing, "protocol": protocol, "higher": higher})
    families: dict[str, list[dict]] = defaultdict(list)
    evidence = []
    for (family, protocol_hash, recipe_hash), observations in sorted(groups.items()):
        values = [r["quality"] for r in observations]
        first = observations[0]
        candidate = {
            "recipe_id": "recipe_" + sha256_json([family, protocol_hash, recipe_hash])[:24],
            "recipe_hash": recipe_hash, "evaluation_protocol_hash": protocol_hash,
            "evaluation_protocol": first["protocol"], "config": first["config"],
            "quality_metric": first["protocol"]["primary_metric"], "higher_better": first["higher"],
            "median_quality": statistics.median(values), "observation_count": len(values),
            "quality_dispersion": statistics.pstdev(values),
            "historical_service_ids": [r["service_id"] for r in observations],
            "missing_provenance": sorted({k for r in observations for k in r["missing"]}),
            "historical_evaluation_exposure": "unknown",
        }
        families[family].append(candidate)
        evidence.append({"family_id": family, **{k: v for k, v in candidate.items()
                         if k not in {"config", "evaluation_protocol", "historical_service_ids"}},
                         "historical_service_ids": canonical_json(candidate["historical_service_ids"])})
    family_rows = []
    for family, candidates in sorted(families.items()):
        # Rank only within an evaluation protocol. Round-robin protocols avoids
        # asserting an ordering between scores measured on different benchmarks.
        protocols: dict[str, list[dict]] = defaultdict(list)
        for candidate in candidates:
            protocols[candidate["evaluation_protocol_hash"]].append(candidate)
        for values in protocols.values():
            values.sort(key=lambda c: ((-1 if c["higher_better"] else 1) * c["median_quality"],
                                      -c["observation_count"], c["quality_dispersion"], c["recipe_hash"]))
        ordered = [values[i] for i in range(max(map(len, protocols.values())))
                   for _, values in sorted(protocols.items()) if i < len(values)]
        # A recipe can have evidence under several protocols, but must train only
        # once. Keep its first occurrence in the existing protocol interleaving.
        selected = []
        seen = set()
        for candidate in ordered:
            if candidate["recipe_hash"] in seen:
                continue
            seen.add(candidate["recipe_hash"])
            selected.append({**candidate, "parent_rank": len(selected) + 1,
                             "parent_id": parent_identity(family, candidate["recipe_hash"])})
            if len(selected) == limit:
                break
        family_rows.append({"family_id": family, "selected_parents": selected,
                            "requested_parent_count": limit, "selected_parent_count": len(selected),
                            "eligible_recipe_count": len({c["recipe_hash"] for c in candidates}),
                            "evaluation_protocol_count": len(protocols)})
    document = {"version": SELECTION_VERSION, "history_db": str(source),
                "limit": limit, "family_count": len(family_rows),
                "parent_count": sum(f["selected_parent_count"] for f in family_rows),
                "evidence_hash": sha256_json(evidence), "families": family_rows, "excluded": excluded}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["family_id", "recipe_id", "recipe_hash", "evaluation_protocol_hash", "quality_metric",
               "higher_better", "median_quality", "observation_count", "quality_dispersion",
               "missing_provenance", "historical_evaluation_exposure", "historical_service_ids"]
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(evidence)
    return document


def selected_recipes(path: str | Path) -> list[dict]:
    """Validate and flatten automatically selected parents for execution."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("version") != SELECTION_VERSION:
        raise ValueError(f"expected a {SELECTION_VERSION} document; regenerate with select-parent-configs")
    result = []
    seen = set()
    for family in document["families"]:
        for selected in family["selected_parents"]:
            config = selected["config"]
            _validate_training_config(config)
            resolved_family = family_identity(config)
            if family["family_id"] != resolved_family:
                raise ValueError("selected parent configuration does not match its family")
            parent_id = parent_identity(resolved_family, selected["recipe_hash"])
            if selected["parent_id"] != parent_id:
                raise ValueError("selected parent identity does not match its recipe")
            if parent_id in seen:
                raise ValueError("duplicate selected recipe within a family")
            seen.add(parent_id)
            # Rank is presentation metadata; changing the limit must not change
            # a retained parent's execution identity.
            candidate = {k: v for k, v in selected.items() if k != "parent_rank"}
            result.append({**selected, "family_id": resolved_family,
                           "selection_hash": sha256_json({"candidate": candidate,
                                                          "evidence_hash": document["evidence_hash"]})})
    return result
