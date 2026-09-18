"""Frozen, balanced local recipes derived from completed task parents."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .clients.resources import task_key
from .provenance import canonical_json, sha256_json
from .shared_parent import DEFAULT_COHORT_DB, PARENTS_VERSION, verify_parent
from .clients import recipes
from .clients.recipes import PROFILES
from .clients.training import (SAMPLER_VERSION, build_training_policy,
                                     validate_training_recipe, training_coverage)

def build_cohort_manifest(parents: str | Path, *, file: str | Path | None = None,
                          clients_per_parent: int = 100, seed: int = 42,
                          heterogeneity_intensity: float = 0.5, average_samples: int = 600, db: str = DEFAULT_COHORT_DB,
                          verify: bool = True, reuse_existing: bool = False,
                          regression_support_policy: str = "general", min_target_bin_fraction: float = 0.6) -> pd.DataFrame:
    if clients_per_parent < 1 or average_samples < 1:
        raise ValueError("client count and quantities must be positive")
    if not 0 <= heterogeneity_intensity <= 1:
        raise ValueError("heterogeneity intensity must be in [0,1]")
    document = json.loads(Path(parents).read_text(encoding="utf-8"))
    if document.get("version") != PARENTS_VERSION or not document.get("parents"):
        raise ValueError("expected a nonempty parent_checkpoints_v1 document")
    rows = []
    coverage = []
    for parent in document["parents"]:
        if verify:
            verify_parent(parent)
        policy = build_training_policy(parent)
        policy_hash = sha256_json(policy)
        settings = {"version": "client_recipes_v4", "recipe_sampler_version": SAMPLER_VERSION,
                    "partition_generator_version": recipes.PARTITION_VERSION,
                    "training_policy_hash": policy_hash, "parent": parent["parent_state_hash"],
                    "parent_document_hash": parent["document_hash"], "count": clients_per_parent,
                    "seed": seed, "intensity": heterogeneity_intensity,
                    "regression_support_policy": regression_support_policy, "min_target_bin_fraction": min_target_bin_fraction,
                    "average_samples": average_samples, "sample_quantity_spread": recipes.SAMPLE_QUANTITY_SPREAD}
        cohort_id = "cohort_" + sha256_json(settings)[:24]
        base = dict(parent["config"])
        for key in ("dataset_args", "service_id", "db_path", "generation_mode", "sample_size", "max_samples",
                    "sample_seed", "training_seed", "distribution_param", "distribution_type",
                    "resource_tier", "model_resource_tier", "dataset_resource_tier", "service_config"):
            base.pop(key, None)
        if base.get("hf_service_meta_json"):
            metadata = base["hf_service_meta_json"]
            metadata = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
            for key in ("resource_tier", "model_resource_tier", "dataset_resource_tier"):
                metadata.pop(key, None)
            base["hf_service_meta_json"] = canonical_json(metadata)
        task = task_key(base)
        family_rows = []
        sampled = recipes.sample_client_recipes(policy, count=clients_per_parent, seed=cohort_id,
            heterogeneity_intensity=heterogeneity_intensity, average_samples=average_samples,
            regression_support_policy=regression_support_policy, min_target_bin_fraction=min_target_bin_fraction,
            quantity_seed=sha256_json([seed, parent["parent_state_hash"], "quantities"]))
        for client in sampled:
            recipe = {**base, **client, "generation_mode": "shared_parent", "cohort_id": cohort_id,
                "cohort_size": clients_per_parent, "family_id": parent["family_id"],
                "parent_checkpoint_id": parent["parent_checkpoint_id"], "lineage_id": parent["lineage_id"],
                "parent_state_hash": parent["parent_state_hash"], "parent_document_ref": parent["document_ref"],
                "parent_document_hash": parent["document_hash"],
                "compatibility_contract_hash": parent["compatibility_contract_hash"], "task": task,
                "max_eval_time_s": None, "timeout_s": None}
            validate_training_recipe(recipe)
            recipe_hash = sha256_json(recipe)
            family_rows.append({**recipe, "service_id": "client_" + recipe_hash[:24], "enabled": True,
                                "db_path": str(Path(db).resolve()), "client_recipe_hash": recipe_hash,
                                "client_recipe": recipe})
        manifest_hash = sha256_json(sorted(row["client_recipe_hash"] for row in family_rows))
        for row in family_rows:
            row["cohort_manifest_hash"] = manifest_hash
        rows.extend(family_rows)
        coverage.append(training_coverage(family_rows, policy))
    from .config import CONFIG
    options = {key: value for key, value in CONFIG.items() if key.startswith("checkpoint_")}
    # Storage options are separate from the frozen training recipe and service identity.
    for row in rows:
        row["checkpoint_storage_options"] = dict(options)
    frame = pd.DataFrame(rows)
    if file:
        target = Path(file)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialised = frame.map(lambda value: canonical_json(value) if isinstance(value, (dict, list)) else value)
        if reuse_existing and target.exists():
            # A retry may add newly completed parents, but existing parents must
            # retain their complete frozen cohorts and unchanged recipe values.
            read_options = {"dtype": str, "keep_default_na": False}
            if target.suffix.lower() == ".csv":
                expected = pd.read_csv(io.StringIO(serialised.to_csv(index=False)), **read_options)
                existing = pd.read_csv(target, **read_options)
            elif target.suffix.lower() == ".xlsx":
                buffer = io.BytesIO()
                serialised.to_excel(buffer, sheet_name="services", index=False)
                buffer.seek(0)
                expected = pd.read_excel(buffer, sheet_name="services", **read_options)
                existing = pd.read_excel(target, sheet_name="services", **read_options)
            else:
                raise ValueError("cohort manifest must be CSV or XLSX")
            columns = sorted(set(existing.columns) | set(expected.columns))
            previous_parents = set(existing["parent_checkpoint_id"]) if "parent_checkpoint_id" in existing else set()
            retained = expected[expected["parent_checkpoint_id"].isin(previous_parents)]
            if (not previous_parents or not previous_parents.issubset(set(expected["parent_checkpoint_id"]))
                    or not existing.reindex(columns=columns, fill_value="").equals(
                        retained.reset_index(drop=True).reindex(columns=columns, fill_value=""))):
                raise ValueError(f"existing client manifest differs from requested recipes: {target}; "
                                 "use a new --client-manifest path, or run-manifest to execute the saved manifest")
            if len(existing) == len(expected):
                return frame
        if target.suffix.lower() == ".csv":
            serialised.to_csv(target, index=False)
        elif target.suffix.lower() == ".xlsx":
            serialised.to_excel(target, sheet_name="services", index=False)
        else:
            raise ValueError("cohort manifest must be CSV or XLSX")
        target.with_suffix(".coverage.json").write_text(json.dumps({"cohorts": coverage}, indent=2, allow_nan=False), encoding="utf-8")
    return frame


def validate_client_recipe(row: Mapping[str, Any]) -> None:
    recipe = row.get("client_recipe")
    if isinstance(recipe, str):
        recipe = json.loads(recipe)
    if not isinstance(recipe, Mapping) or sha256_json(recipe) != row.get("client_recipe_hash"):
        raise ValueError("client recipe failed integrity verification; regenerate the cohort manifest")
    if row.get("service_id") != "client_" + str(row["client_recipe_hash"])[:24]:
        raise ValueError("client service ID does not match its frozen recipe")
    for key, expected in recipe.items():
        actual = row.get(key)
        if expected is None:
            # CSV nulls are restored from the authoritative embedded recipe.
            continue
        if isinstance(expected, (dict, list)) and isinstance(actual, str):
            actual = json.loads(actual)
        if actual != expected:
            raise ValueError(f"client recipe field changed: {key}")
    if row.get("fl_run_id") or row.get("round_id"):
        raise ValueError("shared-parent clients must not declare federated rounds")
    validate_training_recipe(recipe)
    from .clients.partitioning import validate_partition_config
    validate_partition_config(recipe)


def validate_cohort_entries(entries) -> None:
    """Check complete frozen membership without loading data or model weights."""
    if not entries:
        raise ValueError("cohort is empty")
    first = entries[0].resolved
    for entry in entries:
        validate_client_recipe(entry.resolved)
        for key in ("cohort_id", "cohort_size", "cohort_manifest_hash", "parent_document_ref", "parent_document_hash", "db_path", "recipe_sampler_version", "training_policy_hash"):
            if entry.resolved.get(key) != first.get(key):
                raise ValueError(f"cohort rows disagree on {key}")
    count = int(first["cohort_size"])
    if len(entries) != count or len({e.resolved["client_id"] for e in entries}) != count:
        raise ValueError("execute the complete frozen cohort manifest; completed clients are resumed automatically")
    if sha256_json(sorted(e.resolved["client_recipe_hash"] for e in entries)) != first["cohort_manifest_hash"]:
        raise ValueError("cohort membership differs from its frozen manifest")
