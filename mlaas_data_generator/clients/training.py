"""Versioned, conditional training spaces for all client generators."""
from __future__ import annotations

from collections import Counter
from itertools import product
import math
from typing import Any, Mapping

import numpy as np

from .resources import task_key
from ..provenance import sha256_json
from .profiles import PROFILES, current_profiles

LEGACY_SAMPLER_VERSION = "conditional_balancing_v1"
PREVIOUS_SAMPLER_VERSION = "conditional_balancing_v2"
CALIBRATED_SAMPLER_VERSION = "conditional_balancing_v3"
TIER_FREE_SAMPLER_VERSION = "conditional_balancing_v4"
SAMPLER_VERSION = "conditional_balancing_v5"
TIER_FREE_SAMPLER_VERSIONS = frozenset({TIER_FREE_SAMPLER_VERSION, SAMPLER_VERSION})
CALIBRATED_SAMPLER_VERSIONS = frozenset({CALIBRATED_SAMPLER_VERSION, *TIER_FREE_SAMPLER_VERSIONS})
OPPORTUNITY_SAMPLER_VERSIONS = frozenset({PREVIOUS_SAMPLER_VERSION, *CALIBRATED_SAMPLER_VERSIONS})
SUPPORTED_SAMPLER_VERSIONS = frozenset({LEGACY_SAMPLER_VERSION, *OPPORTUNITY_SAMPLER_VERSIONS})
KNOBS = ("optimizer", "learning_rate", "batch_size", "training_epochs", "weight_decay",
         "gradient_accumulation_steps", "warmup_ratio")


def _seed(*values) -> int:
    return int(sha256_json(values)[:8], 16)


def build_training_policy(parent: Mapping[str, Any], *, task: str | None = None) -> dict:
    from .resources import (
        _batch_sizes_for, _epochs_for,
        _weight_decays_for, _gradient_accumulation_steps_for, _warmup_ratios_for,
    )
    from .policies import (POLICY_VERSION, LEARNING_RATES, policy_key,
                           uses_small_random_initialization, RANDOM_INITIALIZATION_EPOCHS,
                           RANDOM_INITIALIZATION_SAMPLE_MULTIPLIER)

    base = dict(parent["config"])
    generic = base.get("training_regime") == "generic"
    non_gradient = generic and base.get("model_type") in {"randomforest", "kmeans"}
    tabular_regression = generic and base.get("task_type") == "regression" and not non_gradient
    small_random = uses_small_random_initialization(parent)
    task = task or task_key(base)
    if task == "sentence_similarity" and base.get("task_type") == "classification":
        task = "text_classification"
    resource_task = "image_classification" if task == "keras_image_classification" else task
    entries = parent.get("contracts", {}).get("parameter_schema", {}).get("entries", [])
    parameter_count = sum(int(entry["numel"]) for entry in entries) if entries else None
    if parameter_count is not None:
        base["estimated_params_m"] = parameter_count / 1_000_000
    name = policy_key(base, task=task)
    rates = LEARNING_RATES[name]
    train_budget = base.get("max_train_time_s")
    if train_budget is None:
        train_budget = 120
    if isinstance(train_budget, bool) or not isinstance(train_budget, int) or train_budget < 1:
        raise ValueError("max_train_time_s must be a positive integer")
    policy = {"version": SAMPLER_VERSION, "task": task, "backend": "generic" if generic else "hf",
              "policy_version": POLICY_VERSION, "policy_key": name,
              "parameter_count": parameter_count, "optimizers": list(rates),
              "minimum_planned_optimizer_steps": 0 if non_gradient else 5,
              "sample_quantity_multiplier": RANDOM_INITIALIZATION_SAMPLE_MULTIPLIER if small_random else 1,
              "quality_gate": "none"}
    from .capacity import CAPACITY_POLICY, REALIZED_TRAINING_POLICY, known_training_capacity
    policy["partition_profiles"] = list(PROFILES)
    if base.get("task_contract"):
        policy["partition_profiles"] = list(current_profiles(base["task_contract"]["sampling"]["profiles"]))
    elif task in {"image_segmentation", "text2text_generation", "seq2seq_generation"}:
        policy["partition_profiles"] = ["iid"]
    axis = base.get("skew_axis")
    if not axis and task == "fill_mask":
        axis = "raw_text_length_bucket"
    if not axis and task in {"text_classification", "image_classification", "keras_image_classification", "tabular_classification"}:
        axis = "class_label"
    if not axis and task in {"regression", "tabular_regression", "sentence_similarity"}:
        axis = "score_bin"
    policy["skew_axis"] = axis
    if not axis:
        policy["partition_profiles"] = ["iid"]
    policy.update(quantity_capacity_policy=CAPACITY_POLICY,
                  known_training_capacity=known_training_capacity(base),
                  realized_training_policy=REALIZED_TRAINING_POLICY,
                  max_realized_training_epochs=200)
    if tabular_regression:
        policy["regression_convergence_policy"] = "tabular_regression_v1"
    if base.get("mobilebert_head_normalization") == "layer_norm_v1":
        policy["runtime_settings"] = {
            "mobilebert_head_normalization": "layer_norm_v1",
            "max_grad_norm": base.get("max_grad_norm") or 1.0,
            "mixed_precision": False, "precision_type": "fp32",
        }

    batches = list(_batch_sizes_for(resource_task, base))
    if resource_task == "image_segmentation" and str(base.get("hf_model_id", "")).lower().startswith("openmmlab/upernet-"):
        batches = [batch for batch in batches if batch >= 2]
        if not batches:
            raise ValueError("UperNet client training requires a batch size of at least two")
    if tabular_regression:
        epoch_options = (40, 80)
    elif small_random:
        epoch_options = RANDOM_INITIALIZATION_EPOCHS
    else:
        epoch_options = _epochs_for(resource_task)
    epochs = list(epoch_options)
    if len(epochs) == 1 and not non_gradient:
        epochs.append(epochs[0] + 1)  # The wall-time limit still bounds actual work.
    accumulation = [1] if generic else list(_gradient_accumulation_steps_for(resource_task, base))
    policy["limits"] = {
        "batch_size": batches, "training_epochs": epochs,
        "learning_rate_by_optimizer": {optimizer: list(values) for optimizer, values in rates.items()},
        "gradient_accumulation_steps": accumulation,
        "max_effective_batch_size": int(max(batches) * (1 if generic else 2)),
        "weight_decay": [0.0, 1e-5, 1e-4] if tabular_regression else list(_weight_decays_for(resource_task)),
        "warmup_ratio": [0.0] if generic else list(_warmup_ratios_for(resource_task)),
        "max_train_time_s": train_budget,
    }
    if non_gradient:
        policy["limits"].update(training_epochs=[1], weight_decay=[0.0], warmup_ratio=[0.0])
    if base.get("model_type") == "kmeans":
        policy["limits"]["model_settings"] = {
            "clustering_k": [2, 3, 4, 5, 6], "clustering_n_init": [5, 10, 20],
            "clustering_max_iter": [100, 200, 300, 500], "clustering_tol": [1e-3, 1e-4, 1e-5]}
    from .calibration import attach_calibration
    # A trained task parent is not comparable to independent base-model runs.
    attach_calibration(policy, {**base, "parent_state_hash": parent.get("parent_state_hash") or base.get("parent_state_hash")})
    from .reviewed_opportunity import attach_reviewed_opportunity
    attach_reviewed_opportunity(policy, {**base, "parent_state_hash": parent.get("parent_state_hash") or base.get("parent_state_hash")})
    return policy


def recipe_limits(recipe: Mapping[str, Any]) -> dict | None:
    """Read current limits or the frozen tier of a pre-v4 manifest."""
    policy = recipe["training_policy"]
    if recipe.get("recipe_sampler_version") in TIER_FREE_SAMPLER_VERSIONS:
        return policy.get("limits")
    return policy.get("tiers", {}).get(recipe.get("resource_tier"))


def calibration_limits(recipe: Mapping[str, Any]) -> dict:
    calibration = recipe["training_policy"].get("calibration", {})
    if recipe.get("recipe_sampler_version") in TIER_FREE_SAMPLER_VERSIONS:
        return calibration
    return calibration.get("tiers", {}).get(recipe.get("resource_tier"), {})


def optimizer_steps_per_epoch(recipe: Mapping[str, Any], sample_size: int) -> int:
    batches = math.ceil(sample_size / recipe["batch_size"])
    if (recipe.get("training_policy") or {}).get("optimizer_step_accounting") == "flush_partial_accumulation_v1":
        return math.ceil(batches / recipe["gradient_accumulation_steps"])
    return batches // recipe["gradient_accumulation_steps"]


def planned_optimizer_steps(recipe: Mapping[str, Any], sample_size: int) -> int:
    """Frozen counting convention, before skipped updates or time/resource stops."""
    return int(recipe["training_epochs"] * optimizer_steps_per_epoch(recipe, sample_size))


def realized_training_recipe(recipe: Mapping[str, Any], sample_size: int) -> tuple[dict, dict | None]:
    """Derive effective work from actual data without mutating the frozen recipe."""
    from .capacity import REALIZED_TRAINING_POLICY
    policy = recipe.get("training_policy") or {}
    effective = dict(recipe)
    if policy.get("realized_training_policy") != REALIZED_TRAINING_POLICY or recipe.get("optimizer") == "none":
        return effective, None
    if sample_size < 1:
        raise ValueError("no usable training examples are available")
    requested = int(recipe.get("sample_size") or sample_size)
    if sample_size >= requested:
        return effective, None
    target_steps = max(int(policy["minimum_planned_optimizer_steps"]), planned_optimizer_steps(recipe, requested))
    from .calibration import selected_template
    template = selected_template(recipe)
    if template:
        target_steps = max(target_steps, int(template["minimum_steps"]))
    batches = math.ceil(sample_size / recipe["batch_size"])
    accumulation = min(int(recipe["gradient_accumulation_steps"]), batches)
    per_epoch = optimizer_steps_per_epoch({**recipe, "gradient_accumulation_steps": accumulation}, sample_size)
    epochs = max(int(recipe["training_epochs"]), math.ceil(target_steps / per_epoch))
    from .reviewed_opportunity import requirements
    required = requirements(recipe)
    if required:
        epochs = max(epochs, math.ceil(required["minimum_sample_presentations"] / sample_size),
                     math.ceil(required["minimum_planned_optimizer_steps"] / per_epoch))
    if epochs > int(policy["max_realized_training_epochs"]):
        raise ValueError(f"realised sample count {sample_size} requires {epochs} epochs, beyond the frozen runtime limit")
    effective.update(sample_size=sample_size, training_epochs=epochs, gradient_accumulation_steps=accumulation,
                     effective_batch_size=int(recipe["batch_size"]) * accumulation)
    validate_training_opportunity(effective, sample_size=sample_size)
    keys = ("sample_size", "batch_size", "training_epochs", "gradient_accumulation_steps", "effective_batch_size", "max_train_time_s")
    evidence = {"version": REALIZED_TRAINING_POLICY, "reason": "realised_quantity_below_requested",
                "requested": {key: recipe.get(key) for key in keys},
                "effective": {key: effective.get(key) for key in keys},
                "target_optimizer_steps": target_steps,
                "effective_planned_optimizer_steps": planned_optimizer_steps(effective, sample_size),
                "joint_calibration_knobs_preserved": accumulation == recipe["gradient_accumulation_steps"],
                "wall_time_budget_unchanged": True}
    return effective, evidence


def validate_training_opportunity(recipe: Mapping[str, Any], *, sample_size=None) -> None:
    if recipe.get("recipe_sampler_version") not in OPPORTUNITY_SAMPLER_VERSIONS or recipe.get("optimizer") == "none":
        return
    sample_size = recipe.get("sample_size") if sample_size is None else sample_size
    if sample_size is None:
        return  # Knob-only calls have not assigned a quantity yet.
    minimum = recipe["training_policy"]["minimum_planned_optimizer_steps"]
    from .reviewed_opportunity import requirements
    required = requirements(recipe)
    if required:
        minimum = max(minimum, required["minimum_planned_optimizer_steps"])
        if (sample_size < required["minimum_classes"] or
                sample_size * recipe["training_epochs"] < required["minimum_sample_presentations"]):
            raise ValueError("supported recipe has insufficient training quantity/exposure for its frozen review reference")
    if recipe.get("recipe_sampler_version") in CALIBRATED_SAMPLER_VERSIONS:
        from .calibration import selected_template
        template = selected_template(recipe)
        if template:
            minimum = max(minimum, template["minimum_steps"])
    steps = planned_optimizer_steps(recipe, sample_size)
    if sample_size < 1 or steps < minimum or round(steps * recipe["warmup_ratio"]) >= steps:
        raise ValueError(f"client has insufficient training opportunity: at least {minimum} planned optimizer steps with post-warmup training required")


def constrain_training_opportunity(recipe: dict, *, seed: str) -> None:
    """Keep the quantity draw; change only infeasible training knobs within policy."""
    if recipe.get("recipe_sampler_version") in CALIBRATED_SAMPLER_VERSIONS and recipe.get("calibration_band") == "supported":
        from .calibration import selected_template
        template = selected_template(recipe)
        limits = recipe_limits(recipe)
        for epochs in limits["training_epochs"]:
            if epochs < template["knobs"]["training_epochs"]:
                continue
            candidate = {**recipe, "training_epochs": epochs}
            try:
                validate_training_opportunity(candidate)
            except ValueError:
                continue
            recipe["training_epochs"] = epochs
            return
        raise ValueError("sample quantity cannot support the calibrated recipe within its epoch limit")
    try:
        validate_training_opportunity(recipe)
        return
    except ValueError:
        pass
    limits = recipe_limits(recipe)
    keys = ("batch_size", "training_epochs", "gradient_accumulation_steps")
    feasible = []
    for values in product(*(limits[key] for key in keys)):
        candidate = {**recipe, **dict(zip(keys, values))}
        candidate["effective_batch_size"] = candidate["batch_size"] * candidate["gradient_accumulation_steps"]
        if candidate["effective_batch_size"] > limits["max_effective_batch_size"]:
            continue
        try:
            validate_training_opportunity(candidate)
        except ValueError:
            continue
        feasible.append(candidate)
    if not feasible:
        raise ValueError("sample quantity cannot provide minimum training opportunity within this resource policy")
    changes = [sum(row[key] != recipe[key] for key in keys) for row in feasible]
    choices = [row for row, count in zip(feasible, changes) if count == min(changes)]
    selected = choices[int(np.random.default_rng(_seed(seed, "training_opportunity")).integers(len(choices)))]
    recipe.update({key: selected[key] for key in (*keys, "effective_batch_size")})


def _assign_balanced(options, indices, rows, key, *, seed, strata) -> None:
    """Exact marginal balance, with stratum coverage breaking equal-count ties."""
    if not indices:
        return
    options = list(dict.fromkeys(options))
    if not options:
        raise ValueError(f"no feasible values for {key}")
    rng = np.random.default_rng(seed)
    order = list(indices)
    rng.shuffle(order)
    counts, cross_counts = Counter(), Counter()
    for index in order:
        tie_order = rng.permutation(len(options))
        value = min((options[i] for i in tie_order),
                    key=lambda option: (counts[option], cross_counts[(strata[index], option)]))
        rows[index][key] = value
        counts[value] += 1
        cross_counts[(strata[index], value)] += 1


def sample_training_knobs(policy: dict, profiles: list[str], *, seed: str) -> dict[int, dict]:
    rows = {index: {} for index in range(len(profiles))}
    strata = dict(enumerate(profiles))
    limits = policy["limits"]
    indices = list(rows)
    for key, options in (("optimizer", policy["optimizers"]), ("batch_size", limits["batch_size"]),
                         ("training_epochs", limits.get("exploration_epochs", limits["training_epochs"])), ("weight_decay", limits["weight_decay"]),
                         ("warmup_ratio", limits["warmup_ratio"])):
        _assign_balanced(options, indices, rows, key, seed=_seed(seed, key), strata=strata)
    for optimizer in policy["optimizers"]:
        selected = [i for i in indices if rows[i]["optimizer"] == optimizer]
        _assign_balanced(limits["learning_rate_by_optimizer"][optimizer], selected, rows, "learning_rate",
                         seed=_seed(seed, optimizer, "learning_rate"), strata=strata)
    for batch in limits["batch_size"]:
        selected = [i for i in indices if rows[i]["batch_size"] == batch]
        feasible = [steps for steps in limits["gradient_accumulation_steps"]
                    if batch * steps <= limits["max_effective_batch_size"]]
        _assign_balanced(feasible, selected, rows, "gradient_accumulation_steps",
                         seed=_seed(seed, batch, "accumulation"), strata=strata)
    for index in indices:
        rows[index].update(policy.get("runtime_settings", {}))
        rows[index]["max_train_time_s"] = limits["max_train_time_s"]
        rows[index]["effective_batch_size"] = rows[index]["batch_size"] * rows[index]["gradient_accumulation_steps"]
    for key, options in limits.get("model_settings", {}).items():
        _assign_balanced(options, indices, rows, key, seed=_seed(seed, key), strata=strata)
    if policy["version"] in CALIBRATED_SAMPLER_VERSIONS:
        from .calibration import sample_calibrated_knobs
        sample_calibrated_knobs(policy, profiles, rows, seed=seed)
    return rows


def validate_training_recipe(recipe: Mapping[str, Any]) -> None:
    version = recipe.get("recipe_sampler_version")
    if version is None:
        return  # Existing v1 manifests retain their frozen interpretation.
    if version not in SUPPORTED_SAMPLER_VERSIONS:
        raise ValueError(f"unsupported client recipe sampler: {version}")
    policy = recipe.get("training_policy")
    if not isinstance(policy, Mapping) or policy.get("version") != version or sha256_json(policy) != recipe.get("training_policy_hash"):
        raise ValueError("client training policy failed integrity verification")
    for key, value in policy.get("runtime_settings", {}).items():
        if recipe.get(key) != value:
            raise ValueError(f"client {key} differs from its frozen runtime settings")
    limits = recipe_limits(recipe)
    optimizer = recipe.get("optimizer")
    if limits is None or optimizer not in policy["optimizers"]:
        raise ValueError("client optimizer or limits is outside the training policy")
    for key in ("batch_size", "training_epochs", "gradient_accumulation_steps", "weight_decay", "warmup_ratio"):
        if recipe.get(key) not in limits[key]:
            raise ValueError(f"client {key} is outside its training policy")
    if recipe.get("learning_rate") not in limits["learning_rate_by_optimizer"][optimizer]:
        raise ValueError("client learning_rate is outside its optimizer-specific training policy")
    effective = recipe["batch_size"] * recipe["gradient_accumulation_steps"]
    if effective > limits["max_effective_batch_size"] or recipe.get("effective_batch_size") != effective:
        raise ValueError("client effective batch size violates its effective batch constraint")
    if recipe.get("max_train_time_s") != limits["max_train_time_s"]:
        raise ValueError("client training time limit differs from its training policy")
    for key, options in limits.get("model_settings", {}).items():
        if recipe.get(key) not in options:
            raise ValueError(f"client {key} is outside its model-specific training policy")
    if version in CALIBRATED_SAMPLER_VERSIONS:
        calibration = policy.get("calibration", {})
        calibrated = bool(calibration_limits(recipe).get("templates"))
        if calibrated:
            if recipe.get("calibration_band") not in calibration["mixture"]:
                raise ValueError("calibrated recipe requires a valid calibration band")
            from .calibration import selected_template
            template = selected_template(recipe)
            if template:
                for key, value in template["knobs"].items():
                    if (recipe[key] < value if key == "training_epochs" else recipe[key] != value):
                        raise ValueError(f"supported recipe {key} differs from its frozen joint template")
            elif recipe.get("calibration_template_id"):
                raise ValueError("only supported recipes may reference a calibration template")
        elif recipe.get("calibration_band") or recipe.get("calibration_template_id"):
            raise ValueError("recipe has calibration metadata without supporting evidence")
    validate_training_opportunity(recipe)


def training_coverage(rows: list[dict], policy: dict) -> dict:
    def counts(group, key, options):
        observed = Counter(row[key] for row in group)
        return {str(value): observed[value] for value in options}

    result = {"version": "cohort_training_coverage_v2", "cohort_id": rows[0]["cohort_id"],
              "cohort_manifest_hash": rows[0]["cohort_manifest_hash"], "scheduled_clients": len(rows),
              "training_policy_hash": sha256_json(policy), "training_policy": policy}
    result["calibration_band_counts"] = dict(Counter(row.get("calibration_band", "uncalibrated") for row in rows))
    limits = policy["limits"]
    group = rows
    profiles = list(dict.fromkeys([*PROFILES, *(row["split_strategy"] for row in group)]))
    result["coverage"] = {
        "client_count": len(group),
        "profile_counts": counts(group, "split_strategy", profiles),
        "knob_counts": {key: counts(group, key, policy["optimizers"] if key == "optimizer" else limits[key])
                        for key in ("optimizer", "batch_size", "training_epochs", "weight_decay", "warmup_ratio")},
        "learning_rate_counts_by_optimizer": {optimizer: counts([row for row in group if row["optimizer"] == optimizer],
            "learning_rate", rates) for optimizer, rates in limits["learning_rate_by_optimizer"].items()},
        "accumulation_counts_by_batch_size": {str(batch): counts([row for row in group if row["batch_size"] == batch],
            "gradient_accumulation_steps", [steps for steps in limits["gradient_accumulation_steps"]
             if batch * steps <= limits["max_effective_batch_size"]]) for batch in limits["batch_size"]},
        "batch_epoch_counts": {str(batch): counts([row for row in group if row["batch_size"] == batch],
            "training_epochs", limits["training_epochs"]) for batch in limits["batch_size"]},
    }
    return result
