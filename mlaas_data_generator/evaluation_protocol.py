"""Frozen, versioned budgets for behavioural evaluation (not training)."""
from __future__ import annotations

import math

PROTOCOL_VERSION = "production_evaluation_v1"
BENCHMARK_PROTOCOL_VERSION = "benchmark_evaluation_v1"
CURRENT_PROTOCOL_VERSIONS = frozenset({PROTOCOL_VERSION, BENCHMARK_PROTOCOL_VERSION})
DEFAULT_PROFILE = "benchmark"
LEGACY_PROTOCOL_VERSION = "legacy_perturbation_v1"
PROFILES = {
    "benchmark": (32, 4, 16, 32, 4, 600, 1800, 16),
    "diagnostic": (8, 4, 4, 8, 4, 30, 30, 32),
    "pilot": (512, 12, 128, 32, 16, 3600, 7200, 128),
    "production": (512, 12, 128, 32, 16, 3600, 7200, 128),
}
PLAN_CONFIG_KEYS = (
    "evaluation_protocol_version", "metric_evaluation_profile", "trust_sample_count", "trust_trials",
    "explainability_sample_count", "explainability_candidate_units", "metric_evaluation_random_trials",
    "metric_evaluation_budget_fractions", "explanation_replacement_methods", "trust_max_duration_s",
    "explainability_max_duration_s", "metric_evaluation_min_independent_inputs",
    "trust_min_independent_inputs", "explainability_min_independent_inputs",
    "metric_evaluation_min_coverage",
    "metric_evaluation_max_ci_width", "metric_evaluation_bootstrap_resamples",
    "metric_evaluation_progress_path", "metric_evaluation_batch_size",
)


def resolve_evaluation_plan(config=None):
    config = config or {}
    profile = config.get("metric_evaluation_profile") or (
        "diagnostic" if config.get("evaluation_protocol_version") == PROTOCOL_VERSION else DEFAULT_PROFILE)
    if profile not in PROFILES:
        raise ValueError(f"unknown metric_evaluation_profile: {profile}")
    values = PROFILES[profile]
    plan = dict(zip(("trust_sample_count", "trust_trials", "explainability_sample_count",
        "explainability_candidate_units", "explainability_random_trials", "trust_max_duration_s",
        "explainability_max_duration_s", "min_independent_inputs"), values))
    version = config.get("evaluation_protocol_version") or (
        BENCHMARK_PROTOCOL_VERSION if profile == "benchmark" else PROTOCOL_VERSION)
    if version not in CURRENT_PROTOCOL_VERSIONS:
        raise ValueError(f"unsupported evaluation_protocol_version: {version}")
    plan.update(profile=profile, protocol_version=version, max_ci_width=.15,
        bootstrap_resamples=1000, seed=42, explainability_budget_fractions=[.1, .25, .5],
        explanation_replacement_methods=["empirical"] if profile == "benchmark" else ["empirical", "mean"],
        min_coverage=.8, batch_size=32)
    if version == BENCHMARK_PROTOCOL_VERSION:
        plan["evidence_gate_version"] = "coverage_precision_v1"
    aliases = {"explainability_random_trials": "metric_evaluation_random_trials",
        "explainability_budget_fractions": "metric_evaluation_budget_fractions",
        "min_independent_inputs": "metric_evaluation_min_independent_inputs",
        "min_coverage": "metric_evaluation_min_coverage",
        "max_ci_width": "metric_evaluation_max_ci_width", "bootstrap_resamples": "metric_evaluation_bootstrap_resamples",
        "seed": "evaluation_seed", "batch_size": "metric_evaluation_batch_size"}
    for key in plan:
        if key in {"profile", "protocol_version", "evidence_gate_version"}:
            continue
        configured = config.get(aliases.get(key, key))
        if configured is not None:
            plan[key] = configured
    # Shared legacy overrides apply to both metrics unless a stage overrides it.
    # A smaller eligible pool never lowers an evidence floor.
    for stage in ("trust", "explainability"):
        key = f"{stage}_min_independent_inputs"
        plan[key] = config.get(key)
        if plan[key] is None:
            plan[key] = (8 if stage == "explainability" and profile == "benchmark"
                         and config.get("metric_evaluation_min_independent_inputs") is None
                         else plan["min_independent_inputs"])
    integers = ("trust_sample_count", "trust_trials", "explainability_sample_count", "explainability_candidate_units",
        "explainability_random_trials", "min_independent_inputs", "trust_min_independent_inputs",
        "explainability_min_independent_inputs", "bootstrap_resamples", "seed", "batch_size")
    for key in integers:
        value = plan[key]
        minimum = 0 if key == "seed" else 2 if key.endswith("min_independent_inputs") else 1
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value) or int(value) < minimum:
            raise ValueError(f"{aliases.get(key, key)} must be an integer >= {minimum}")
        plan[key] = int(value)
    for key in ("trust_max_duration_s", "explainability_max_duration_s", "max_ci_width", "min_coverage"):
        value = float(plan[key])
        if isinstance(plan[key], bool) or not math.isfinite(value) or value <= 0 or (key in {"max_ci_width", "min_coverage"} and value > 1):
            raise ValueError(f"invalid {key}")
        plan[key] = value
    fractions = plan["explainability_budget_fractions"]
    if not isinstance(fractions, (list, tuple)) or len(fractions) < 2 or any(
        isinstance(v, bool) or not 0 < float(v) < 1 for v in fractions):
        raise ValueError("metric_evaluation_budget_fractions needs at least two fractions in (0,1)")
    plan["explainability_budget_fractions"] = sorted(set(float(v) for v in fractions))
    if len(plan["explainability_budget_fractions"]) < 2:
        raise ValueError("masking fractions must be distinct")
    methods = plan["explanation_replacement_methods"]
    if (not isinstance(methods, (list, tuple)) or not methods
            or any(method not in ("empirical", "mean") for method in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("explanation_replacement_methods must contain distinct empirical and/or mean methods")
    plan["explanation_replacement_methods"] = list(methods)
    return plan


def manifest_evaluation_defaults(profile=DEFAULT_PROFILE):
    """Use the same profile resolver as Python callers; do not bake in probe budgets."""
    plan = resolve_evaluation_plan({"metric_evaluation_profile": profile})
    return {"evaluation_protocol_version": plan["protocol_version"], "metric_evaluation_profile": profile}


def apply_evaluation_profile(config, profile):
    """Explicit preset replacement for resumed manifests and reconstructed runs.

    Preserve training settings, seeds and service IDs. Discard previous metric
    overrides and progress paths so an old frozen plan cannot leak into this one.
    """
    result = {key: value for key, value in config.items() if key not in PLAN_CONFIG_KEYS}
    result.update(manifest_evaluation_defaults(profile), enable_perturbation_metrics=True)
    return result
