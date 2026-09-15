"""Offline capability declarations and stage-specific instance decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .tasks import TaskContractError, resolve_task_contract
from .hf_tasks import CANONICAL_HF_TASKS, normalize_hf_task

CAPABILITIES = ("training", "checkpoint_restoration", "partitioning", "parameter_aggregation",
                "diagnostics", "measured_composition_validation")
VERSION = "task_capabilities_v1"
HF_BACKENDS = {"hf_finetune", "hf_train", "transformers_finetune", "hf", "hf_text", "transformers"}
NEURAL_BACKENDS = HF_BACKENDS | {"cnn", "mlp", "logreg", "mobilenetv2"}
RUNTIME_CHECKS = {"training": ("supervision", "training_completed"), "partitioning": ("partition_support",),
                  "diagnostics": ("checkpoint_present", "checkpoint_integrity", "diagnostic_support"),
                  "checkpoint_restoration": ("checkpoint_present", "checkpoint_integrity"),
                  "parameter_aggregation": ("checkpoint_present", "checkpoint_integrity", "parent", "parameter_schema", "target_semantics"),
                  "measured_composition_validation": ("checkpoint_present", "checkpoint_integrity", "parent", "parameter_schema", "target_semantics", "evaluation_protocol", "audit_support")}
REASONS = {"checkpoint_present": "checkpoint_missing", "checkpoint_integrity": "checkpoint_integrity_failed",
           "parent": "parent_mismatch", "parameter_schema": "parameter_schema_mismatch",
           "target_semantics": "target_semantics_mismatch", "evaluation_protocol": "evaluation_protocol_mismatch",
           "audit_support": "insufficient_audit_support", "supervision": "insufficient_supervision",
           "training_completed": "training_not_completed", "partition_support": "partition_axis_unavailable",
           "diagnostic_support": "insufficient_diagnostic_support"}


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
        for check in RUNTIME_CHECKS.get(capability, ()):
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
