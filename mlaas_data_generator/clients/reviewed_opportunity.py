"""Engineering budgets informed by the bounded September 2026 family review.

These reference workloads are not quality thresholds or clearance for new
quantities/recipes. Existing frozen policies do not acquire them retroactively.
"""
from copy import deepcopy
from .legacy_profiles import UNIFORM_PROFILE as LEGACY_UNIFORM_PROFILE


VIT_SVHN_REFERENCE = {
    "version": "vit_svhn_reviewed_opportunity_v1",
    "reference_sample_count": 730,
    "reference_epochs": 6,
    "reference_batch_size": 8,
    "reference_accumulation": 1,
    "minimum_sample_presentations": 4380,
    "minimum_planned_optimizer_steps": 552,
    "minimum_successful_optimizer_steps": 550,
    "minimum_classes": 10,
    "source_services": [f"review_78256bd138d4__followup_{i}" for i in (1, 2, 3)],
    "source_report": "outputs/family_review_20260913/report.md",
    "source_results_sha256": "926149b228e56119377baf739dcc5f46266d85ea4bf4440ea6fbe86c0e019ba1",
    "interpretation": "planning reference; changed quantities and joint recipes are extrapolations, not family clearance",
}


def attach_reviewed_opportunity(policy, config):
    if (config.get("hf_model_id") != "google/vit-base-patch16-224"
            or config.get("dataset_name") != "ufldl-stanford/svhn"
            or config.get("dataset_config") != "cropped_digits"
            or policy.get("task") != "image_classification"
            or config.get("parent_state_hash")
            or config.get("generation_mode") in {"shared_parent", "fedavg"}):
        return
    policy["reviewed_training_opportunity"] = deepcopy(VIT_SVHN_REFERENCE)
    policy["optimizer_step_accounting"] = "flush_partial_accumulation_v1"
    calibration = policy.get("calibration", {})
    if calibration.get("templates"):
        calibration["class_coverage"]["supported"].update(
            minimum_classes=10, require_source_classes=True)


def requirements(recipe):
    if recipe.get("calibration_band") != "supported":
        return {}
    return (recipe.get("training_policy") or {}).get("reviewed_training_opportunity", {})


def validate_realized_coverage(recipe, targets):
    if recipe.get("split_strategy", "iid") in {"iid", LEGACY_UNIFORM_PROFILE}:
        # Uniform sampling does not promise class coverage. Missing a rare
        # source class must not invalidate an otherwise valid IID client.
        return
    required = requirements(recipe)
    if required:
        import numpy as np
        values = np.asarray(targets)
        if values.ndim != 1 or np.unique(values).size < required["minimum_classes"]:
            raise ValueError("supported ViT/SVHN training requires all ten source classes")


def runtime_evidence(recipe, metrics, sample_size):
    required = requirements(recipe)
    if not required:
        return {}
    from .training import planned_optimizer_steps
    successful = metrics.get("successful_optimizer_step_count")
    presentations = metrics.get("train_samples")
    measured = successful is not None and presentations is not None
    met = bool(measured and successful >= required["minimum_successful_optimizer_steps"]
               and presentations >= required["minimum_sample_presentations"])
    return {
        "training_opportunity_policy": required["version"],
        "training_opportunity_planned_updates": planned_optimizer_steps(recipe, sample_size),
        "training_opportunity_required_successful_updates": required["minimum_successful_optimizer_steps"],
        "training_opportunity_required_presentations": required["minimum_sample_presentations"],
        "training_opportunity_met": met,
        "training_opportunity_status": "met" if met else "unmeasured" if not measured else
            "resource_limited" if metrics.get("train_stopped_early") else "shortfall",
    }
