"""Terminal diagnostics policy; result and warning output remains unconditional."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import math

from .config import CONFIG

_MODE = ContextVar("terminal_log_mode", default=None)
_SETUP_FIELDS = {
    "task_family", "task_type", "training_regime", "split_strategy",
    "split_strategy_requested", "split_strategy_effective", "split_skew_axis", "split_skew_axis_effective",
    "data_distribution", "dataset_distribution_json",
    "split_distribution_json", "split_provenance_json", "split_bucket_spec_json",
    "train_distribution_json", "benchmark_distribution_json", "distribution_json",
    "batch_size", "learning_rate", "epochs", "device", "mixed_precision", "precision_type",
    "optimizer", "training_epochs", "hf_task", "mixed_precision_requested", "mixed_precision_effective",
    "precision_type_requested", "precision_type_effective",
    "downloads", "likes", "pipeline_tag", "library_name", "license", "tags", "last_modified",
}
_DEBUG_FIELDS = {
    "service_id", "case_name", "run_partition_seed", "partition_hash", "source_fields",
    "bucket_spec", "l1_allocation_deviation", "checkpoint_compression",
    "dataset_load_s", "split_selection_s", "model_build_s", "model_reset_s",
    "finetune_s", "signature_s", "checkpoint_s", "db_write_s",
    "tokenizer_load_s", "model_load_s", "train_timeout_s", "perturbation_duration_s",
    "initial_state_s", "local_preparation_s", "setup_s", "pretrain_snapshot_s",
    "resource_tracker_start_s", "resource_tracker_stop_s", "hardware_snapshot_s",
    "final_state_capture_s", "post_checkpoint_s", "timing_scope",
    "update_signature_path", "update_signature_id",
}


def _value(metric):
    return metric.get("value", metric) if isinstance(metric, dict) else metric


def _fields(values):
    for key, value in values.items():
        rendered = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value
        print(f"  {key}: {rendered if rendered is not None else 'n/a'}", flush=True)


def print_fedavg_summary(config, *, train_pool, benchmark_size, device, precision=None):
    print("========== SERVICE RUN SUMMARY ==========")
    _fields({
        "fl_run_id": config.get("fl_run_id"),
        "model": config.get("hf_model_id") or config.get("model_type"),
        "dataset": config.get("dataset_name") or config.get("dataset"),
        "task": config.get("hf_task") or config.get("task_type"),
        "training regime": config.get("training_regime"),
        "resource tier": config.get("resource_tier"),
        "dataset variant": config.get("dataset_variant"),
        "split variant": config.get("split_variant"),
        "clients": config.get("fedavg_clients_per_run"),
        "rounds": config.get("fedavg_rounds"),
        "participation": config.get("fedavg_client_participation_rate"),
        "train pool": train_pool,
        "benchmark size": benchmark_size,
        "heterogeneity intensity": config.get("heterogeneity_intensity"),
        "partition generator version": config.get("partition_generator_version"),
        "device": device,
        "mixed precision": config.get("mixed_precision", False),
        "precision": precision or (config.get("precision_type", "fp16") if str(config.get("mixed_precision", False)).lower() in {"true", "1"} else "full"),
        "explainability": config.get("enable_perturbation_metrics", config.get("explainability_enabled", False)),
    })


def _number(value, spec=".3f"):
    value = _value(value)
    try:
        return format(float(value), spec) if math.isfinite(float(value)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def print_service_summary(config, metrics, checkpoint=None, *, status="success", error=None,
                          restore_verify_s=None):
    """Selected independent-service results; persistence keeps the full metrics."""
    checkpoint = checkpoint or {}
    def metric(*keys):
        return next((_value(metrics[key]) for key in keys
                     if key in metrics and _value(metrics[key]) is not None), None)
    def scaled(value, scale):
        try:
            return float(value) * scale
        except (TypeError, ValueError):
            return None
    def seconds(value):
        rendered = _number(value, ".1f" if value is not None and abs(float(value)) >= 10 else ".2f")
        return rendered + "s" if rendered != "n/a" else rendered

    family = config.get("family_id")
    prefix = f"Family {str(family)[:23]}: " if family else ""
    ordinal = config.get("_family_ordinal", config.get("_manifest_ordinal", "?"))
    total = config.get("_family_total", config.get("_manifest_total", "?"))
    print(f"\n{prefix}[Service {ordinal}/{total}] {config.get('service_id')} | {status.upper()}", flush=True)
    names = [metric("primary_metric_name"), metric("auxiliary_metric_name")]
    if not any(names):
        names = [key for key in ("accuracy", "f1", "rmse", "mae", "pearson", "spearman") if key in metrics]
    names = list(dict.fromkeys([name for name in names if name] + ["loss"]))
    print("\nPerformance\n  " + " | ".join(f"{name}={_number(metric(name), '.4f')}" for name in names), flush=True)
    params = metric("model_params_count", "params_count", "model_size")
    param_text = _number(params, ",.0f")
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if params is not None and float(params) >= threshold:
            param_text = _number(float(params) / threshold, ".1f") + suffix
            break
    print(f"  train={_number(metric('train_set_size', 'train_sequence_count'), ',.0f')} "
          f"| eval={_number(metric('benchmark_set_size', 'eval_sequence_count'), ',.0f')} | params={param_text}", flush=True)
    runtime = metric("runtime_total_s", "service_runtime_s")
    save_time = metric("checkpoint_s")
    print(f"\nRuntime\n  total={seconds(runtime)} | train={seconds(metric('training_runtime_s', 'train_time_s'))} "
          f"| eval={seconds(metric('evaluation_runtime_s', 'eval_runtime_s'))} | checkpoint={seconds(save_time)}", flush=True)
    if runtime is not None and float(runtime) > 0 and save_time is not None:
        print(f"  checkpoint={100 * float(save_time) / float(runtime):.1f}% of runtime", flush=True)
    latency = scaled(metric("inference_latency_s", "eval_latency_s_mean", "latency"), 1000)
    p95 = scaled(metric("inference_latency_s_p95", "eval_latency_s_p95", "tail_latency"), 1000)
    print(f"\nInference\n  latency={_number(latency, '.1f')}ms | p95={_number(p95, '.1f')}ms "
          f"| throughput={_number(metric('throughput_samples_s', 'throughput'), '.1f')} samples/s", flush=True)
    metadata = checkpoint.get("metadata_json") or {}
    selected = checkpoint.get("selected_physical_bytes")
    logical = checkpoint.get("target_logical_bytes")
    ratio = float(logical) / float(selected) if logical is not None and selected and float(selected) > 0 else None
    selected_format = metadata.get("representation") or checkpoint.get("patch_semantics_version") or checkpoint.get("storage_mode") or "n/a"
    from .debug_mode import weights_disabled
    if weights_disabled(config):
        selected_format = "disabled (debug: no weight files)"
    print(f"\nCheckpoint\n  format={selected_format} | size={_number(scaled(selected, 1e-6), '.2f')} MB "
          f"| compression={_number(ratio, '.2f')}x", flush=True)
    if restore_verify_s is None:
        restore_verify_s = metric("checkpoint_model_restore_verify_s", "checkpoint_native_restore_verify_s")
    print(f"  save={seconds(save_time)} | restore_verify={seconds(restore_verify_s)}", flush=True)
    warnings = []
    reason = metric("precision_fallback_reason")
    if reason:
        precision = str(metric("precision_type_requested") or config.get("precision_type") or "precision").upper()
        warnings.append(f"{precision} fallback -> {reason}")
    for warning in (metric("precision_fallback_error"), metric("gpu_fallback_warning"),
                    metric("label_space_warning"), error):
        if warning and str(warning) not in warnings:
            warnings.append(str(warning))
    if metric("train_stopped_early"):
        warnings.append("Training stopped early: timeout")
    if warnings:
        print("\nWARNING\n  " + "\n  ".join(warnings), flush=True)


def compact_metrics(metrics):
    """Task-aware primary/secondary metrics for the concise terminal view."""
    names = [_value(metrics.get(key)) for key in ("primary_metric_name", "auxiliary_metric_name")]
    if not any(names):
        names = [name for name in ("accuracy", "f1", "rmse", "mae", "pearson", "spearman") if name in metrics]
    parts = []
    for name in dict.fromkeys(name for name in names if name):
        label = {"accuracy": "acc", "f1": "F1"}.get(name, name)
        parts.append(f"{label}={_number(metrics.get(name))}")
    return " | ".join(parts) or "metrics=n/a"


def _compact_client(config, partition, metrics, *, position, participants):
    strategy = partition.get("effective_strategy") or config.get("split_strategy") or "n/a"
    params = partition.get("resolved_partition_params") or config.get("requested_partition_params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    if strategy == "dirichlet":
        strategy += f" α={_number(params.get('alpha', config.get('distribution_param')), '.3g')}"
    elif strategy == "quantity_skew":
        strategy += f" ×{_number(params.get('quota_multiplier', config.get('distribution_param')), '.3g')}"
    elif strategy == "shard":
        strategy += f" k={params.get('selected_bucket_count', partition.get('represented_bucket_count', 'n/a'))}"
    histogram = partition.get("realised_bucket_counts") or partition.get("bucket_histogram") or {}
    def bucket_order(key):
        try:
            return (0, float(key))
        except (TypeError, ValueError):
            return (1, str(key))
    counts = "/".join(str(histogram[key]) for key in sorted(histogram, key=bucket_order)) or "n/a"
    majority = _number(partition.get("majority_bucket_share"), '.1%')
    optimizer = str(config.get("optimizer", CONFIG.get("optimizer")))
    optimizer = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}.get(optimizer.lower(), optimizer)
    runtime_key = next((key for key in ("training_runtime_s", "train_time_s", "train_runtime_s", "compute_time_s") if key in metrics), None)
    runtime = _number(metrics.get(runtime_key), '.1f')
    runtime = runtime + "s" if runtime != "n/a" else runtime
    runtime_label = "compute" if runtime_key == "compute_time_s" else "train"
    first = (f"[Round {config.get('round_id', '?')}/{config.get('fedavg_rounds', '?')}] "
             f"[{position}/{participants}] {config.get('client_id')} | {strategy} "
             f"| n={partition.get('sample_count', partition.get('effective_quota', 'n/a'))} "
             f"| classes={counts} | majority={majority} "
             f"| H={_number(partition.get('normalised_bucket_entropy'))}")
    second = (f"{optimizer} | bs={config.get('batch_size', CONFIG.get('batch_size'))} "
              f"| lr={_number(config.get('learning_rate', CONFIG.get('learning_rate')), '.6g')} "
              f"| epochs={config.get('training_epochs', CONFIG.get('training_epochs'))} "
              f"| {compact_metrics(metrics)} | {runtime_label}={runtime}")
    total_key = next((key for key in ("client_runtime_s", "runtime_total_s") if key in metrics), None)
    if total_key:
        second += f" | total={_number(metrics[total_key], '.1f')}s"
    print(f"\n{first}\n{second}", flush=True)


def print_client_completion(config, partition, metrics, *, status, error=None, position=1, participants=1):
    """One verified client-round outcome, with readable partition details."""
    if not debug_enabled():
        _compact_client(config, partition, metrics, position=position, participants=participants)
        print_partition_notices(partition)
        if status != "success" or error:
            print(f"  Abnormal: status={status}" + (f" | {error}" if error else ""), flush=True)
        for key in ("train_stopped_early", "timeout_hit", "timed_out", "precision_fallback_reason"):
            if _value(metrics.get(key)):
                label = "training timeout" if key == "train_stopped_early" else key
                print(f"  Abnormal: {label}={_value(metrics[key])}", flush=True)
        return
    print(f"[Client complete] round={config.get('round_id')} client={config.get('client_id')} status={status}", flush=True)
    strategy = partition.get("effective_strategy") or config.get("split_strategy")
    params = partition.get("resolved_partition_params") or config.get("requested_partition_params") or {}
    if isinstance(params, str):
        params = json.loads(params)
    important = {
        "dirichlet": ("alpha", params.get("alpha", config.get("distribution_param"))),
        "quantity_skew": ("quota multiplier", params.get("quota_multiplier", config.get("distribution_param"))),
        "shard": ("selected buckets", params.get("selected_bucket_count", partition.get("represented_bucket_count"))),
    }.get(strategy, ("strategy parameter", config.get("distribution_param")))
    histogram = partition.get("realised_bucket_counts") or partition.get("bucket_histogram") or {}
    _fields({"partition strategy": strategy})
    if important[1] is not None:
        _fields({important[0]: important[1]})
    _fields({
        "sample count": partition.get("sample_count", partition.get("effective_quota")),
        "class/bucket distribution": histogram,
        "majority share": partition.get("majority_bucket_share"),
        "normalised entropy": partition.get("normalised_bucket_entropy"),
        "optimizer": config.get("optimizer", CONFIG.get("optimizer")),
        "batch size": config.get("batch_size", CONFIG.get("batch_size")),
        "learning rate": config.get("learning_rate", CONFIG.get("learning_rate")),
        "epochs": config.get("training_epochs", CONFIG.get("training_epochs")),
    })
    print_partition_notices(partition)
    if error:
        print(f"  Abnormal: {error}", flush=True)
    if metrics:
        print_results("  Results:", metrics)
    if debug_enabled():
        _fields({"service_id": config.get("service_id"), "case_name": config.get("case_name"),
                 "run_partition_seed": config.get("run_partition_seed"), "raw provenance JSON": partition})


def print_partition_notices(partition):
    requested = partition.get("requested_strategy")
    effective = partition.get("effective_strategy")
    if requested and effective and requested != effective:
        print(f"  Abnormal: requested strategy={requested}; effective strategy={effective}", flush=True)
    for key in ("quota_shortfall", "target_deviation_count"):
        if partition.get(key):
            print(f"  Abnormal: {key}={partition[key]}", flush=True)
    fallback = partition.get("fallback_reason")
    if fallback:
        print(f"  Abnormal: partition fallback={fallback}", flush=True)
    histogram = partition.get("realised_bucket_counts") or partition.get("bucket_histogram") or {}
    represented = partition.get("represented_bucket_count")
    if represented is None and histogram:
        represented = sum(float(count) > 0 for count in histogram.values())
    if represented == 1:
        print("  Abnormal: single-class/bucket partition", flush=True)


def log_mode(config=None):
    value = (config or {}).get("log_mode") or _MODE.get() or CONFIG.get("log_mode", "concise")
    value = str(value).strip().lower()
    if value not in {"concise", "debug"}:
        raise ValueError("log_mode must be concise or debug")
    return value


def debug_enabled(config=None):
    return log_mode(config) == "debug"


@contextmanager
def terminal_mode(config=None):
    token = _MODE.set(log_mode(config))
    try:
        yield
    finally:
        _MODE.reset(token)


def terminal_scope(config_for_call):
    """Carry the policy to nested loaders/adapters without changing their APIs."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with terminal_mode(config_for_call(*args, **kwargs)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def debug_print(*args, **kwargs):
    if debug_enabled():
        print(*args, **kwargs)


def print_results(label, metrics):
    """Print every result value without truncating nested metric payloads."""
    print(label, flush=True)
    for name, metric in metrics.items():
        # Unknown metrics may be classified as metadata (e.g. precision/recall).
        # Hide only known setup/provenance fields, never an entire metric domain.
        setup_field = name in _SETUP_FIELDS or name.startswith(("hf_model_", "hf_dataset_"))
        diagnostic = name in _DEBUG_FIELDS or name.startswith("preprocess_") or name.endswith("_hash")
        if (setup_field or diagnostic) and not debug_enabled():
            continue
        value = metric.get("value", metric) if isinstance(metric, dict) else metric
        if name in {"train_stopped_early", "timeout_hit", "timed_out", "quota_shortfall", "target_deviation_count", "precision_fallback_reason"} and not debug_enabled():
            if value:
                label = "training timeout" if name == "train_stopped_early" else name
                print(f"  Abnormal: {label}={value}", flush=True)
            continue
        unit = metric.get("unit") if isinstance(metric, dict) else None
        suffix = f" {unit}" if unit else ""
        print(f"  {name}: {json.dumps(value, default=str, ensure_ascii=False)}{suffix}", flush=True)
