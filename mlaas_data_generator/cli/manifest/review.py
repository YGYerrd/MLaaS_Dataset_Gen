"""Compact manifest transport. Expands to the existing runner's row schema."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ...provenance import canonical_json, sha256_json

VERSION = "client_review_manifest_v1"
FROZEN_DEFAULTS = "_frozen_manifest_defaults"
REVIEW_COLUMNS = (
    "service_id", "family_ref", "enabled", "split_strategy", "sample_size",
    "sample_seed", "training_seed", "optimizer", "learning_rate", "batch_size",
    "training_epochs", "weight_decay", "gradient_accumulation_steps", "warmup_ratio",
    "calibration_band",
)
# Selection evidence and legacy row labels are preserved, but are not inputs a
# reviewer needs to edit alongside each client's training settings.
METADATA_COLUMNS = frozenset({
    "case_name", "notes", "dataset_variant", "split_variant", "knob_variant",
    "client_partition_index", "hf_service_meta_json", "fit_decision", "fit_reason",
    "shard_support_quantile",
    "calibration_template_id",
    "fit_quality_score", "realism_score",
    "domain_alignment", "dataset_hint", "service_source", "model_role",
    "hf_downloads", "hf_likes", "downloads", "likes", "model_size", "params_count",
    "pipeline_tag", "hf_pipeline_tag", "library_name", "license", "tags",
    "last_modified", "hf_author", "hf_url", "hf_dataset_id",
})
ALIASES = {"distribution_type": "split_strategy", "requested_quota": "sample_size",
           "knob_variant": "client_partition_index"}


def companion_path(path: Path) -> Path:
    return path.with_suffix(".families.json")


def _family_ref(defaults, definition):
    # A local review reference, not the runner's eventual composition family ID.
    return "F_" + sha256_json({"defaults": defaults, "definition": definition})[:16]


def compact_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from ..run_manifest import BASE_DEFAULTS, CONFIG, _coerce_by_column, _resolve_row
    from .preparation import contract_config

    if frame.columns.duplicated().any():
        raise ValueError("manifest contains duplicate columns")
    defaults = json.loads(canonical_json({**CONFIG, **BASE_DEFAULTS}))
    records = [{key: _coerce_by_column(key, value) for key, value in row.items()}
               for row in frame.to_dict("records")]
    # Do not propagate retired source caps when rewriting older manifests.
    for row in records:
        row.pop("max_samples", None)
        row.pop("fedavg_source_pool_cap", None)
        if isinstance(row.get("service_config"), dict):
            row["service_config"].pop("max_samples", None)
            row["service_config"].pop("fedavg_source_pool_cap", None)
    ids = [row.get("service_id") for row in records]
    if any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("review manifests require unique, nonempty service IDs")
    groups = {}
    for row in records:
        resolved = _resolve_row(pd.Series(row), {})
        # Review groups describe shared intended contracts. Legacy registry
        # annotations contain variant indexes but do not belong in this key.
        contract = {key: value for key, value in contract_config(resolved).items() if key not in METADATA_COLUMNS}
        groups.setdefault(sha256_json(contract), []).append(row)
    families, visible, extra_columns = {}, {}, set()
    for rows in groups.values():
        derived = {key: source for key, source in ALIASES.items()
                   if all(row.get(key) is not None and row.get(key) == row.get(source) for row in rows)}
        if all(row.get("effective_batch_size") is not None and row["effective_batch_size"] ==
               row.get("batch_size", 0) * row.get("gradient_accumulation_steps", 0) for row in rows):
            derived["effective_batch_size"] = "batch_size * gradient_accumulation_steps"
        keys = set().union(*(row.keys() for row in rows)) - {"service_config"} - set(derived)
        keys = {key for key in keys if any(row.get(key) is not None for row in rows)}
        shared = {key: rows[0].get(key) for key in keys - set(REVIEW_COLUMNS)
                  if len({canonical_json(row.get(key)) for row in rows}) == 1}
        varying = keys - set(shared) - set(REVIEW_COLUMNS)
        extras = varying - METADATA_COLUMNS
        extra_columns.update(extras)
        configs = [row.get("service_config") or {} for row in rows]
        config_keys = sorted(set().union(*(cfg.keys() for cfg in configs)))
        definition = {"settings": shared, "derived": derived,
                      "service_config_fields": config_keys,
                      "client_columns": sorted((set(REVIEW_COLUMNS) | extras) - {"family_ref"}),
                      "required_values": sorted(key for key in (set(REVIEW_COLUMNS) | extras) - {"family_ref"}
                                                if all(row.get(key) is not None for row in rows)),
                      "clients": {}}
        for row, config in zip(rows, configs):
            sid = row["service_id"]
            metadata = {key: row.get(key) for key in varying & METADATA_COLUMNS}
            # A field in service_config normally mirrors its top-level column.
            # Preserve exceptions and omitted keys exactly, without copying the
            # training policy into every service_config again.
            overrides = {key: value for key, value in config.items() if value != row.get(key)}
            client = {"metadata": metadata}
            if overrides:
                client["service_config_overrides"] = overrides
            missing = sorted(set(config_keys) - set(config))
            if missing:
                client["service_config_missing"] = missing
            definition["clients"][sid] = client
            visible[sid] = {key: row.get(key) for key in (*REVIEW_COLUMNS, *extras) if key != "family_ref"}
        reference = _family_ref(defaults, definition)
        families[reference] = definition
        for row in rows:
            visible[row["service_id"]]["family_ref"] = reference
    columns = [key for key in REVIEW_COLUMNS if key in {"service_id", "family_ref", "enabled"}
               or any(row.get(key) is not None for row in records)] + sorted(extra_columns)
    document = {"version": VERSION, "defaults": defaults, "families": families,
                "review_columns": columns}
    review = pd.DataFrame([visible[sid] for sid in ids], columns=columns)
    return review, document


def family_summary(document: dict) -> pd.DataFrame:
    columns = ("task", "hf_model_id", "model_type", "dataset_name", "dataset_config",
               "train_split", "test_split", "seed", "initialization_seed", "evaluation_seed",
               "max_length", "skew_axis", "heterogeneity_intensity")
    return pd.DataFrame([{"family_ref": ref, "clients": len(family["clients"]),
        **{key: family["settings"].get(key) for key in columns}}
        for ref, family in document["families"].items()], columns=["family_ref", "clients", *columns])


def save_review_manifest(frame: pd.DataFrame, path: Path, *, sheet_name="services") -> None:
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise ValueError("Output path must end with .csv or .xlsx")
    if sheet_name == "family_summary":
        raise ValueError("family_summary is reserved for the review workbook's reference sheet")
    review, document = compact_manifest(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    companion = companion_path(path)
    temporary = companion.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(companion)
    serialised = review.map(lambda value: canonical_json(value) if isinstance(value, (dict, list)) else value)
    if path.suffix.lower() == ".csv":
        serialised.to_csv(path, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            serialised.to_excel(writer, sheet_name=sheet_name, index=False)
            family_summary(document).to_excel(writer, sheet_name="family_summary", index=False)
            from openpyxl.utils import get_column_letter
            for name, worksheet in writer.sheets.items():
                worksheet.freeze_panes = "C2" if name == sheet_name else "B2"
                worksheet.auto_filter.ref = worksheet.dimensions
                for index, cells in enumerate(worksheet.columns, 1):
                    width = max((len(str(cell.value)) for cell in cells if cell.value is not None), default=10)
                    worksheet.column_dimensions[get_column_letter(index)].width = min(44, max(12, width + 2))
    print(f"Review table: {len(review)} clients, {len(review.columns)} columns. Shared definitions: {companion}")
    if document["families"]:
        print(family_summary(document)[["family_ref", "hf_model_id", "model_type", "dataset_name"]].to_string(index=False))


def expand_review_manifest(frame: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict]:
    from ..run_manifest import _coerce_by_column

    companion = companion_path(path)
    if not companion.is_file():
        raise ValueError(f"review manifest requires its shared definitions: {companion}")
    document = json.loads(companion.read_text(encoding="utf-8"))
    if document.get("version") != VERSION:
        raise ValueError("unsupported review manifest version")
    defaults, families = document["defaults"], document["families"]
    for reference, definition in families.items():
        if reference != _family_ref(defaults, definition):
            raise ValueError("review family definitions changed or are corrupt; regenerate the manifest")
    if frame.columns.duplicated().any():
        raise ValueError("review manifest contains duplicate columns")
    missing = set(document["review_columns"]) - set(frame.columns)
    if missing:
        raise ValueError("review manifest is missing columns: " + ", ".join(sorted(missing)))
    unknown = set(frame.columns) - set(document["review_columns"])
    if unknown:
        raise ValueError("unexpected review columns: " + ", ".join(sorted(unknown)))
    rows, seen = [], set()
    for raw in frame.to_dict("records"):
        values = {key: _coerce_by_column(key, value) for key, value in raw.items()}
        sid, ref = values.pop("service_id"), values.pop("family_ref")
        if not sid or sid in seen:
            raise ValueError("review manifest contains a missing or duplicate service ID")
        seen.add(sid)
        family = families.get(ref)
        if family is None or sid not in family["clients"]:
            raise ValueError(f"service {sid} does not belong to its review family; check the companion file")
        client = family["clients"][sid]
        other = [key for key, value in values.items() if key not in family["client_columns"] and value is not None]
        if other:
            raise ValueError(f"service {sid} has settings outside its review columns: {', '.join(other)}")
        row = {**family["settings"], **client["metadata"], "service_id": sid,
               **{key: value for key, value in values.items() if key in family["client_columns"]}}
        absent = [key for key in family["required_values"] if row.get(key) is None]
        if absent:
            raise ValueError(f"service {sid} is missing review values: {', '.join(absent)}")
        for key, source in family["derived"].items():
            if key == "effective_batch_size":
                row[key] = row["batch_size"] * row["gradient_accumulation_steps"]
            else:
                row[key] = row[source]
        missing_config = client.get("service_config_missing", [])
        payload = {key: row.get(key) for key in family["service_config_fields"] if key not in missing_config}
        payload.update(client.get("service_config_overrides", {}))
        row["service_config"] = json.dumps(payload, sort_keys=True)
        rows.append(row)
    expanded = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["service_id", "enabled"])
    return expanded, {FROZEN_DEFAULTS: defaults}
