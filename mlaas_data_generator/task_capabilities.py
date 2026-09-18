"""Offline capability declarations and stage-specific instance decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .tasks import TaskContractError, resolve_task_contract
from .hf_tasks import CANONICAL_HF_TASKS, normalize_hf_task

CAPABILITIES = ("training", "checkpoint_restoration", "partitioning", "parameter_aggregation",
                "diagnostics", "measured_composition_validation", "recipe_persistence",
                "materialization", "reconstruction_validation")
VERSION = "task_capabilities_v1"
HF_BACKENDS = {"hf_finetune", "hf_train", "transformers_finetune", "hf", "hf_text", "transformers"}
NEURAL_BACKENDS = HF_BACKENDS | {"cnn", "mlp", "logreg", "mobilenetv2"}
RUNTIME_CHECKS = {"training": ("supervision", "training_completed"), "partitioning": ("partition_support",),
                  "diagnostics": ("checkpoint_present", "checkpoint_integrity", "diagnostic_support"),
                  "checkpoint_restoration": ("checkpoint_present", "checkpoint_integrity"),
                  "parameter_aggregation": ("checkpoint_present", "checkpoint_integrity", "parent", "parameter_schema", "target_semantics"),
                  "measured_composition_validation": ("checkpoint_present", "checkpoint_integrity", "parent", "parameter_schema", "target_semantics", "evaluation_protocol", "audit_support")}
REASONS = {"checkpoint_present": "checkpoint_missing", "checkpoint_integrity": "checkpoint_integrity_failed",
           "materialized_state": "materialized_state_unverified", "family_integrity": "family_integrity_failed",
           "recipe_integrity": "recipe_integrity_failed", "replay_supported": "recipe_replay_unsupported",
           "parent": "parent_mismatch", "parameter_schema": "parameter_schema_mismatch",
           "target_semantics": "target_semantics_mismatch", "evaluation_protocol": "evaluation_protocol_mismatch",
           "audit_support": "insufficient_audit_support", "supervision": "insufficient_supervision",
           "training_completed": "training_not_completed", "partition_support": "partition_axis_unavailable",
           "diagnostic_support": "insufficient_diagnostic_support"}


def _reconstruction_capability(capability, config, contract, error, checks, stage, blocked):
    """Reconstruction does not depend on composition or a client checkpoint."""
    from .family_preflight import independent_family_applies
    from .debug_mode import weights_disabled
    from .services.taxonomy import canonical_task_family
    hf_task = normalize_hf_task(config.get("hf_task"), default="unknown")
    family = canonical_task_family(config.get("task_type", "classification"), hf_task)
    code, support = None, "supported"
    if not independent_family_applies(config) or config.get("fl_run_id"):
        code, support = "independent_family_required", "unsupported"
    elif weights_disabled(config):
        code, support = "diagnostic_no_save_weights", "unsupported"
    elif error and error.code != "unknown_task_contract":
        code, support = error.code, "unsupported"
    elif family == "unknown" or hf_task == "multimodal":
        code, support = "unknown_task_contract", "unsupported"
    elif str(config.get("model_type", "")).lower() not in HF_BACKENDS and family not in {"classification", "regression"}:
        code, support = "backend_not_implemented", "unsupported"
    elif hf_task not in CANONICAL_HF_TASKS | {"unknown", "classification", "regression"}:
        code, support = "unknown_task_contract", "unsupported"
    elif capability != "recipe_persistence" and not (
        contract is not None or hf_task == "causal_lm_generation"
    ):
        code, support = "reconstruction_profile_unavailable", "unsupported"
    elif capability == "recipe_persistence" and contract is None and hf_task != "causal_lm_generation":
        support = "experimental"
    if blocked:
        code = "known_bad_runtime_pair"
    required = ("family_integrity", "recipe_integrity")
    if capability != "recipe_persistence":
        required += ("replay_supported",)
    if capability == "reconstruction_validation":
        required += ("reconstruction_fidelity",)
    decision = "ineligible" if code else "eligible"
    for key in required:
        if code:
            break
        if checks.get(key) is False:
            code, decision = f"{key}_failed", "ineligible"
        elif checks.get(key) is not True:
            decision = "pending"
    evidence = tuple(checks.get("reconstruction_evidence", ()))
    return asdict(CapabilityDecision(capability, support, decision, stage, code,
        blocked or ("Requires an implemented independent-family reconstruction profile." if code else
        "Implementation availability and measured fidelity are separate; a passing report applies only to its recorded family, recipe and runtime."),
        "Supply valid family/recipe evidence and a scoped reconstruction comparison." if code else None,
        {"client_checkpoint_required": False, "validation_scope": "family_recipe_runtime",
         "family_id": config.get("family_id"), "device": config.get("device")}, evidence))


@dataclass(frozen=True)
class CapabilityDecision:
    capability: str
    support: str
    decision: str
    stage: str
    reason_code: str | None
    explanation: str
    remediation: str | None
    constraints: dict
    evidence: tuple = ()
    version: str = VERSION


def resolve_capabilities(config: Mapping, meta=None, *, stage="preflight", checks=None):
    checks = dict(checks or {})
    backend = str(config.get("model_type") or "").lower()
    regime = str(config.get("training_regime") or "").lower()
    contract, error = None, None
    try:
        contract = resolve_task_contract(config, meta)
    except TaskContractError as exc:
        error = exc
    from .compatibility import known_bad_path_reason
    blocked = known_bad_path_reason(task_key=config.get("task"), hf_task=config.get("hf_task"),
        task_tag=config.get("task_tag"), model_id=config.get("hf_model_id"),
        dataset_name=config.get("dataset_name"), dataset_config=config.get("dataset_config"))
    result = {}
    for capability in CAPABILITIES:
        if capability in {"recipe_persistence", "materialization", "reconstruction_validation"}:
            result[capability] = _reconstruction_capability(capability, config, contract, error, checks, stage, blocked)
            continue
        support = "supported"
        code, message = None, "Declared implementation is available; runtime evidence is checked separately."
        if error:
            support, code, message = "unsupported", error.code, str(error)
            # Adapter availability and measured-protocol availability are different capabilities.
            known_task = normalize_hf_task(config.get("hf_task"), default="unknown")
            if error.code == "unknown_task_contract" and backend in HF_BACKENDS and known_task in CANONICAL_HF_TASKS - {"multimodal"} and capability in {"training", "checkpoint_restoration"}:
                support, code, message = "experimental", None, "Existing HF adapter; this task's shared evaluation contract is not implemented."
        elif backend not in NEURAL_BACKENDS:
            support, code, message = "unsupported", "backend_not_implemented", f"{backend or 'unknown backend'} has no {capability} implementation"
        elif backend not in HF_BACKENDS and contract.task not in {"classification", "regression"}:
            support, code, message = "unsupported", "backend_not_implemented", f"{backend} does not implement {contract.task}"
        elif contract.task in {"image_segmentation", "seq2seq_generation"} or (contract.task == "token_classification" and capability == "measured_composition_validation"):
            support = "experimental"
        elif contract.task == "fill_mask" and capability in {"diagnostics", "parameter_aggregation", "measured_composition_validation"}:
            support, code, message = "unsupported", "validation_not_implemented", "Masked-LM reconstruction does not enable downstream composition."
        if backend in {"randomforest", "kmeans"}:
            if capability == "training" and (config.get("task_type") in {"classification", "regression", "clustering"}
                    and normalize_hf_task(config.get("hf_task"), default="unknown") not in {"token_classification", "image_segmentation", "seq2seq_generation", "causal_lm_generation", "image_detection"}):
                support, code, message = "supported", None, "Existing non-gradient estimator training; no aligned parameter-delta operator is declared."
            elif capability in {"parameter_aggregation", "measured_composition_validation"}:
                support, code, message = "unsupported", "parameter_operator_unavailable", "Trees or cluster identities do not implement the shared aligned parameter-delta operator."
        if capability == "training" and (regime in {"inference", "inference_only"} or backend in {"hf", "hf_text", "transformers"}):
            support, code, message = "unsupported", "inference_only", "The selected backend/regime does not train client updates."
        if capability == "partitioning" and contract and config.get("split_strategy", "iid") not in contract.sampling["profiles"]:
            code, message = "partition_axis_unavailable", f"{config.get('split_strategy')} is unavailable for {contract.task}"
        if blocked:
            code, message = "known_bad_runtime_pair", blocked
        decision = "ineligible" if code else "eligible"
        required_checks = RUNTIME_CHECKS.get(capability, ())
        if config.get("weight_retention") == "recipe_only" and capability in {"parameter_aggregation", "measured_composition_validation"}:
            required_checks = ("family_integrity", "recipe_integrity", "replay_supported", "materialized_state") + tuple(
                key for key in required_checks if key not in {"checkpoint_present", "checkpoint_integrity"})
        for check in required_checks:
            if code:
                break
            if checks.get(check) is False:
                code, message = REASONS[check], f"Required {check} verification failed at {stage}."
                decision = "ineligible"
                break
            if checks.get(check) is not True:
                decision = "pending"
        if not code and stage == "preflight" and capability in {"training", "partitioning"}:
            decision = "pending"
        result[capability] = asdict(CapabilityDecision(capability, support, decision, stage, code, message,
            "Correct the declared contract or supply the required verified artifact; regenerate changed recipes." if code else None,
            {"profiles": contract.sampling["profiles"] if contract else [], "aggregation_weight_unit": "training_examples"},
            ("tests/generator/test_task_capabilities.py", "tests/smoke/run_task_extensions.py (opt-in; inspect scoped run evidence)")))
    return result


def capability_matrix():
    from .hf_tasks import CANONICAL_HF_TASKS
    for backend in sorted(NEURAL_BACKENDS | {"randomforest", "kmeans"}):
        for task in sorted(CANONICAL_HF_TASKS | {"classification", "regression"}):
            config = {"model_type": backend, "hf_task": task, "task_type": task,
                      "task_tag": "summarization" if task == "seq2seq_generation" else None}
            yield {"task": task, "backend": backend, "capabilities": resolve_capabilities(config)}


def reconstruction_report_capabilities(path):
    """Inspect an immutable comparison as scoped evidence, without a new registry."""
    import json
    from pathlib import Path
    from .provenance import sha256_file
    from .client_reconstruction import fidelity_gate
    path = Path(path)
    if sha256_file(path) != path.stem:
        raise ValueError("reconstruction report failed content-address verification")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("version") != "client_reconstruction_comparison_v3":
        raise ValueError("report lacks the current reconstruction evidence; rerun the comparison")
    replay = report["replay_evidence"]
    passed = (report["gate"] == fidelity_gate(report, report["gate"]["tolerances"])
              and report["gate"]["passed"] and replay["fresh_process"]
              and replay["client_checkpoint_reads"] == 0
              and replay["selected_rows_match"] and replay["effective_settings_match"])
    cfg = {**report["scope"]["config"], "family_id": report["family_id"], "device": report["scope"]["device"]}
    result = resolve_capabilities(cfg, stage="reconstruction_comparison", checks={
        "family_integrity": True, "recipe_integrity": True, "replay_supported": True,
        "reconstruction_fidelity": bool(passed), "reconstruction_evidence": [str(path.resolve())]})
    return {"family_id": report["family_id"], "recipe_hash": report["recipe_hash"],
            "scope": report["scope"], "runtime": report["runtime"],
            "capabilities": {k: result[k] for k in ("recipe_persistence", "materialization", "reconstruction_validation")}}
