# orchestrator.py
from __future__ import annotations
import os, uuid, json
import time
from datetime import datetime, timezone
import numpy as np
from numbers import Number

from ..config import CONFIG
from ..data.master_loader import load_dataset
from ..data.accounting import finalize_accounting
from ..data.splitters import split_data
from ..data.distributions import (
    get_data_distribution,
    get_mlm_masked_token_stats,
    get_retrieval_pair_stats,
    get_token_label_stats,
    get_vqa_answer_stats,
)
from ..data.sources.hf_meta import fetch_hf_model_meta
from ..storage.writer import make_writer
from .strategies.factory import make_task_strategy
from .strategies.base import canonical_task_family, canonical_label_format, canonical_metric_names, normalize_hf_task, metric_availability
from .system_metrics import capture_hardware_snapshot, summarize_round_usage
from .dynamics import (
    DEFAULT_TOLERANCE,
    carry_forward_metrics,
    client_update_metrics,
    global_update_metrics,
    repeated_round_metrics,
    snapshot_model_weights,
)
from .model_params import write_final_model_manifest, write_final_model_parameters
from .update_signature import compute_and_store_update_signature
from ..models.label_schema import infer_ignore_index, infer_label_format, infer_num_labels
from ..runtime_compat import is_rocm_miopen_runtime_error


class _RunSkipped(RuntimeError):
    pass


class _StageTimer:
    """Small helper for consistent stage timing in seconds."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed_s(self):
        return float(time.perf_counter() - self._start)


def _first_sample_shape(x_train):
    """Best-effort shape inference for metadata compatibility."""
    if isinstance(x_train, dict):
        if "pixel_values" in x_train and len(x_train["pixel_values"]) > 0:
            sample = x_train["pixel_values"][0]
            return tuple(getattr(sample, "shape", ()))
        if "input_ids" in x_train and len(x_train["input_ids"]) > 0:
            sample = x_train["input_ids"][0]
            seq_len = int(len(sample))
            return (seq_len,)
        return tuple()

    if hasattr(x_train, "shape"):
        shape = tuple(getattr(x_train, "shape", ()))
        if len(shape) > 1:
            return tuple(shape[1:])
        return shape

    if isinstance(x_train, (list, tuple)) and len(x_train) > 0:
        sample = x_train[0]
        if hasattr(sample, "shape"):
            return tuple(getattr(sample, "shape", ()))
        arr = np.asarray(sample)
        return tuple(arr.shape)

    return tuple()


def _sample_count(x, y=None):
    """Best-effort sample count for arrays, lists, and dict-backed HF batches."""
    source = y if y is not None else x
    if source is None:
        return 0

    if isinstance(source, dict):
        preferred_keys = ("pixel_values", "input_ids", "attention_mask", "features")
        for key in preferred_keys:
            value = source.get(key)
            if value is not None:
                try:
                    return int(len(value))
                except Exception:
                    pass
        for value in source.values():
            try:
                return int(len(value))
            except Exception:
                continue
        return 0

    try:
        return int(len(source))
    except Exception:
        return 0


def _count_model_params(model) -> int:
    """Best-effort trainable parameter count across adapter shapes."""
    if model is None:
        return 0

    candidates = []
    candidate_ids = set()
    for candidate in (model, getattr(model, "core", None)):
        if candidate is not None and id(candidate) not in candidate_ids:
            candidates.append(candidate)
            candidate_ids.add(id(candidate))

    for candidate in candidates:
        count_params = getattr(candidate, "count_params", None)
        if callable(count_params):
            try:
                count = int(count_params())
                if count > 0:
                    return count
            except Exception:
                pass

    nested_models = []
    nested_model_ids = set()
    for candidate in candidates:
        for nested in (candidate, getattr(candidate, "model", None)):
            if nested is not None and id(nested) not in nested_model_ids:
                nested_models.append(nested)
                nested_model_ids.add(id(nested))

    for candidate in nested_models:
        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            try:
                return int(sum(p.numel() for p in parameters()))
            except Exception:
                pass

    return 0


def _resolve_database_run_id(config: dict, fallback_run_id: str | None = None) -> str:
    external_run_id = config.get("external_run_id")
    if external_run_id is not None:
        external_run_id = str(external_run_id).strip()
        if external_run_id and external_run_id.lower() not in {"na", "nan", "null", "none"}:
            return external_run_id

    return fallback_run_id or str(uuid.uuid4())


def _per_client_total_cap(value, num_clients):
    """Convert a per-client sample cap into the total rows needed for a split."""
    if value in (None, ""):
        return None
    try:
        cap = int(value)
    except (TypeError, ValueError):
        return None
    try:
        clients = max(1, int(num_clients))
    except (TypeError, ValueError):
        clients = 1
    return max(0, cap) * clients


def _format_summary_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, list):
        return json.dumps(value, default=str)
    return value


def _bool_config(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _can_compute_numeric_range(values) -> bool:
    """Return True when values can be safely reduced via numeric min/max."""
    if values is None:
        return False
    try:
        arr = np.asarray(values)
    except Exception:
        return False
    if arr.size == 0:
        return False
    if arr.dtype.kind in {"b", "i", "u", "f"}:
        return True
    try:
        flat = arr.reshape(-1)
    except Exception:
        return False
    for item in flat:
        if item is None:
            continue
        if isinstance(item, Number):
            continue
        try:
            float(item)
        except Exception:
            return False
    return True


class FederatedDataGenerator:
    """Generate MLaaS client records using a simple federated-learning loop."""
    def __init__(
        self,
        config: dict | None = None,
        dataset: str | None = None,
        task_type: str | None = None,
        model_type: str | None = None,
        dataset_args: dict | None = None,
    ):
        config_init_timer = _StageTimer()
        self._run_stage_measurements = {}

        self.config = CONFIG.copy()
        if config:
            self.config.update(config)

        self.dataset = dataset or self.config.get("dataset", "fashion_mnist")
        self.model_type = model_type or self.config.get("model_type", "CNN")
        self.config["dataset"] = self.dataset
        self.config["model_type"] = self.model_type

        # dataset args
        self.dataset_args = {}
        config_dataset_args = self.config.get("dataset_args") or {}
        if isinstance(config_dataset_args, dict):
            self.dataset_args.update(config_dataset_args)
        if dataset_args:
            self.dataset_args.update(dataset_args)
        if self.dataset_args:
            self.config["dataset_args"] = dict(self.dataset_args)

        self._run_stage_measurements["stage_config_init_s"] = config_init_timer.elapsed_s()

        # load data
        dataset_load_timer = _StageTimer()
        loader_dataset_args = dict(self.dataset_args)
        if "inference_only" not in loader_dataset_args:
            loader_dataset_args["inference_only"] = (self.model_type or "").lower() in {"hf", "hf_text", "transformers"}
        total_loader_max_samples = _per_client_total_cap(
            loader_dataset_args.get("max_samples"),
            self.config.get("num_clients", 1),
        )
        if total_loader_max_samples is not None:
            loader_dataset_args["max_samples"] = total_loader_max_samples
            loader_dataset_args["requested_max_samples_per_client"] = self.dataset_args.get("max_samples")
        self.loader_dataset_args = loader_dataset_args
        train, test, meta = load_dataset(self.dataset, **loader_dataset_args)
        self._run_stage_measurements["stage_dataset_load_s"] = dataset_load_timer.elapsed_s()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = train, test
        self.meta = meta
        resolved_input_shape = meta.get("input_shape")
        if resolved_input_shape is None:
            resolved_input_shape = _first_sample_shape(self.x_train)
            self.meta["input_shape"] = tuple(resolved_input_shape)
        self.input_shape = tuple(resolved_input_shape)
        self.num_classes = meta.get("num_classes")

        # task type resolution
        requested_task = task_type or self.config.get("task_type")
        meta_task = meta.get("task_type", "classification")
        self.task_type = requested_task or meta_task
        if requested_task != meta_task:
            print(f"Warning: overriding dataset task type '{meta_task}' with requested '{self.task_type}'.")

        self.target_scaler = meta.get("target_scaler")
        self.save_weights = _bool_config(self.config.get("save_weights"), default=False)
        self.save_final_model_params = _bool_config(
            self.config.get("save_final_model_params"),
            default=self.save_weights,
        )
        self.final_model_params_dir = self.config.get(
            "final_model_params_dir",
            os.path.join("outputs", "final_model_params"),
        )
        self.distribution_bins = int(self.config.get("distribution_bins", 10) or 10)

        # Regression: set value range for distribution summaries
        if self.task_type == "regression" or self.num_classes is None:
            if _can_compute_numeric_range(self.y_train):
                y_min = float(np.min(self.y_train))
                y_max = float(np.max(self.y_train))
                if y_min == y_max:
                    y_min -= 0.5
                    y_max += 0.5
                self.distribution_range = (y_min, y_max)
            elif self.task_type == "regression":
                self.distribution_range = (0.0, 1.0)
            else:
                self.distribution_range = None
        else:
            self.distribution_range = None

        # knobs
        hidden_layers = self.config.get("hidden_layers", [self.config.get("reduced_neurons", 64)])
        if hidden_layers is None:
            hidden_layers = [self.config.get("reduced_neurons", 64)]
        self.hidden_layers = list(hidden_layers)

        self.knobs = {
            "num_clients": int(self.config["num_clients"]),
            "num_rounds": int(self.config["num_rounds"]),
            "local_epochs": int(self.config["local_epochs"]),
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
            "hidden_layers": self.hidden_layers,
            "activation": self.config.get("activation", "relu"),
            "dropout": float(self.config.get("dropout", 0.0) or 0.0),
            "weight_decay": float(self.config.get("weight_decay", 0.0) or 0.0),
            "optimizer": self.config.get("optimizer", "adam"),
            "distribution_type": self.config.get("distribution_type", "iid"),
            "distribution_param": self.config.get("distribution_param", None),
            "skew_axis": self.config.get("skew_axis"),
            "skew_axis_config": self.config.get("skew_axis_config"),
            "custom_distributions": self.config.get("custom_distributions", None),
            "sample_size": self.config.get("sample_size", None),
            "sample_frac": self.config.get("sample_frac", None),
            "distribution_bins": self.distribution_bins,
            "early_stopping_patience": self.config.get("early_stopping_patience"),
        }

        self.rng = np.random.default_rng(self.config.get("seed", 42))
        self.hf_task = normalize_hf_task(
            self.dataset_args.get("hf_task") or self.config.get("hf_task") or self.meta.get("hf_task")
        )
        self.task_family = canonical_task_family(self.task_type, self.hf_task)

        # metric keys: resolve from canonical task-family/HF-task mapping.
        task_tag = str(
            self.config.get("task_tag")
            or self.dataset_args.get("task_tag")
            or ""
        ).strip().lower() or None
        metric_primary_name, _metric_secondary_name = canonical_metric_names(
            self.task_family,
            "metric",
            hf_task=self.hf_task,
            task_tag=task_tag,
        )
        self.metric_key = metric_primary_name

        metric_label_overrides = {
            "rmse": "RMSE",
            "f1": "F1",
            "iou": "IoU",
            "map": "mAP",
        }
        self.metric_label = metric_label_overrides.get(
            self.metric_key,
            self.metric_key.replace("_", " ").title(),
        )

        # strategy encapsulates build/train/eval details
        self.strategy = make_task_strategy(
            task_type=self.task_type,
            meta=self.meta,
            knobs=self.knobs,
            config=self.config,
            x_test=self.x_test,
            y_test=self.y_test,
            metric_key=self.metric_key,
            save_weights=self.save_weights,
        )

        # Disable multi-round training for non-federated models
        if self.task_type == "clustering" or (self.model_type or "").lower() == "randomforest":
            print(f"Non-federated model detected ({self.model_type}); forcing single-round training.")
            self.knobs["num_rounds"] = 1
    
    def _early_stopping_patience(self):
        patience_cfg = self.config.get("early_stopping_patience")
        if patience_cfg in (None, "", False):
            return None
        try:
            p = int(patience_cfg)
        except (TypeError, ValueError):
            return None
        if p <= 0:
            return None
        return p

    def _resolve_execution_device(self, model):
        """Best-effort device string for run summaries (CPU/CUDA/DirectML/etc)."""
        candidates = [
            model,
            getattr(model, "core", None),
            getattr(model, "model", None),
        ]

        for obj in candidates:
            if obj is None:
                continue
            device = getattr(obj, "device", None)
            if device is not None:
                return str(device)

        torch_model = getattr(model, "model", None)
        if torch_model is not None:
            try:
                return str(next(torch_model.parameters()).device)
            except Exception:
                pass

        return "unknown"
    

    def _canonical_run_metadata(self):
        self.meta = finalize_accounting(self.meta, batch_size=self.knobs.get("batch_size"))
        hf_task = normalize_hf_task(getattr(self.strategy, "hf_task", self.hf_task))
        task_family = canonical_task_family(self.task_type, hf_task)
        label_format = infer_label_format(self.meta, task_type=self.task_type) or canonical_label_format(task_family)
        accounting = self.meta.get("accounting", {}) if isinstance(self.meta, dict) else {}
        config_dataset_args = self.config.get("dataset_args") or {}
        task_tag = str(self.config.get("task_tag") or config_dataset_args.get("task_tag") or "").strip().lower() or None
        metric_primary_name, metric_secondary_name = canonical_metric_names(
            task_family,
            self.metric_key,
            hf_task=hf_task,
            task_tag=task_tag,
        )
        has_labels = self.y_test is not None
        availability = metric_availability(task_family, task_tag=task_tag, has_labels=has_labels, hf_task=hf_task)
        if task_family == "generation":
            eval_metrics = availability.get("eval", tuple())
            if eval_metrics:
                metric_primary_name = eval_metrics[0]
                metric_secondary_name = eval_metrics[1] if len(eval_metrics) > 1 else None
        return {
            "task_family": task_family,
            "task_tag": task_tag,
            "label_format": label_format,
            "metric_primary_name": metric_primary_name,
            "metric_secondary_name": metric_secondary_name,
            "train_metric_names": list(availability.get("train", tuple())),
            "eval_metric_names": list(availability.get("eval", tuple())),
            "num_labels": infer_num_labels(self.meta, fallback=self.num_classes),
            "train_set_size": int(len(self.y_train)),
            "eval_set_size": int(len(self.y_test)),
            "raw_record_count": accounting.get("raw_record_count"),
            "post_filter_record_count": accounting.get("post_filter_record_count"),
            "tokenized_record_count": accounting.get("tokenized_record_count"),
            "sequence_count": accounting.get("sequence_count"),
            "supervised_token_count": accounting.get("supervised_token_count"),
            "batch_count": accounting.get("batch_count"),
            "metric_instance_count": accounting.get("metric_instance_count"),
        }

    def _extract_dynamic_metrics(self, outcome):
        extras = getattr(outcome, "extras", {}) if outcome is not None else {}

        def _pick(keys, default=None):
            for k in keys:
                if isinstance(extras, dict) and extras.get(k) is not None:
                    return extras.get(k)
            return default

        fail_reason = getattr(outcome, "fail_reason", "") or ""

        has_nan = any(
            isinstance(v, float) and np.isnan(v)
            for v in [
                getattr(outcome, "loss", np.nan),
                getattr(outcome, "metric_value", np.nan),
                getattr(outcome, "metric_score", np.nan),
                getattr(outcome, "extra_metric", np.nan),
            ]
        ) or ("nan" in fail_reason.lower())

        return {
            "effective_batch_size": int(_pick(["effective_batch_size", "batch_size", "train_batch_size"], self.knobs.get("batch_size") or 0) or 0),
            "tokens_in": int(_pick(["tokens_in", "input_tokens", "prompt_tokens", "train_tokens_in"], 0) or 0),
            "tokens_out": int(_pick(["tokens_out", "output_tokens", "completion_tokens", "train_tokens_out"], 0) or 0),
            "avg_seq_len": float(_pick(["avg_seq_len", "avg_sequence_length", "mean_seq_len"], 0.0) or 0.0),
            "truncation_rate": float(_pick(["truncation_rate", "truncated_fraction", "trunc_rate"], 0.0) or 0.0),
            "oom_count": int(_pick(["oom_count"], 0) or 0) + int("out of memory" in fail_reason.lower() or "cuda oom" in fail_reason.lower()),
            "nan_count": int(_pick(["nan_count"], 0) or 0) + int(has_nan),
            "fail_reason_category": self._categorize_fail_reason(fail_reason),
        }

    def _categorize_fail_reason(self, fail_reason: str):
        text = (fail_reason or "").strip().lower()
        if not text:
            return "none"
        if "dropout" in text:
            return "dropout"
        if "out of memory" in text or "cuda oom" in text or "oom" in text:
            return "oom"
        if "nan" in text:
            return "nan"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        return "runtime_error"
    
    def _safe_number(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, Number) and not isinstance(value, bool):
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return None
            return value
        return None

    def _safe_metric_value(self, value):
        if value is None:
            return None
        if isinstance(value, (bool, int, str, dict, list)):
            return value
        parsed_number = self._safe_number(value)
        if parsed_number is not None:
            return parsed_number
        if isinstance(value, np.bool_):
            return bool(value)
        return None

    def _is_final_round_idx(self, round_idx):
        try:
            return int(round_idx) == int(self.knobs.get("num_rounds", 0) or 0)
        except Exception:
            return False

    def _is_final_round_only_client_metric(self, metric_name):
        name = (metric_name or "").strip().lower()
        return name.startswith(("perturbation_", "explainability_", "trust_", "robustness_"))

    def _drop_non_final_trust_metrics(self, values, round_idx):
        if not _bool_config(self.config.get("perturbation_final_round_only"), default=True):
            return values
        if self._is_final_round_idx(round_idx):
            return values
        return {
            key: value
            for key, value in (values or {}).items()
            if not self._is_final_round_only_client_metric(key)
        }

    def _normalize_outcome_extras(self, outcome):
        extras = getattr(outcome, "extras", {}) if outcome is not None else {}
        if not isinstance(extras, dict):
            return {}

        canonical = {}

        aliases = {
            "train_step_latency_ms_mean": ("seconds_per_step", 1.0 / 1000.0),
            "train_step_latency_ms_p95": ("seconds_per_step_p95", 1.0 / 1000.0),
            "eval_latency_ms_mean": ("inference_latency_s", 1.0 / 1000.0),
            "eval_latency_ms_p95": ("inference_latency_s_p95", 1.0 / 1000.0),
            "inference_latency_ms_mean": ("inference_latency_s", 1.0 / 1000.0),
            "inference_latency_ms_p95": ("inference_latency_s_p95", 1.0 / 1000.0),
            "throughput_eps": ("examples_per_second", 1.0),
            "train_throughput_eps": ("examples_per_second", 1.0),
            "eval_throughput_eps": ("examples_per_second", 1.0),
            "throughput_tps": ("tokens_per_second", 1.0),
            "tokens_per_second": ("tokens_per_second", 1.0),
            "train_tokens_per_second": ("tokens_per_second", 1.0),
            "eval_tokens_per_second": ("tokens_per_second", 1.0),
        }

        for key, value in extras.items():
            parsed = self._safe_metric_value(value)
            if parsed is None:
                continue

            if key in aliases:
                canonical_name, multiplier = aliases[key]
                numeric = self._safe_number(parsed)
                if numeric is not None:
                    canonical[canonical_name] = float(numeric) * multiplier
                continue

            canonical[key] = parsed

        train_time_s = self._safe_number(extras.get("train_time_s"))
        if train_time_s is not None:
            canonical["train_time_s"] = float(train_time_s)
            epochs = self._safe_number(extras.get("epochs"))
            if epochs is None:
                epochs = self._safe_number(extras.get("train_epochs"))
            if epochs is None:
                epochs = self._safe_number(self.knobs.get("local_epochs"))
            if epochs and epochs > 0:
                canonical["seconds_per_epoch"] = float(train_time_s) / float(epochs)

        tokens_total = self._safe_number(extras.get("tokens_total"))
        if tokens_total is None:
            tokens_total = self._safe_number(extras.get("train_tokens_total"))
        if tokens_total is None:
            tokens_total = self._safe_number(extras.get("eval_tokens_total"))
        if tokens_total is None:
            token_in = self._safe_number(extras.get("tokens_in"))
            token_out = self._safe_number(extras.get("tokens_out"))
            if token_in is not None and token_out is not None:
                tokens_total = token_in + token_out
        if tokens_total is not None:
            canonical["tokens_total"] = int(tokens_total)

        return canonical

    def _round_qos_rollups(self, records):
        qos_metrics = [
            "seconds_per_step",
            "seconds_per_epoch",
            "examples_per_second",
            "tokens_per_second",
            "inference_latency_s",
        ]
        rollups = {}
        for metric_name in qos_metrics:
            vals = [float(r[metric_name]) for r in records if metric_name in r and self._safe_number(r[metric_name]) is not None]
            if not vals:
                continue
            rollups[f"round_{metric_name}_mean"] = float(np.mean(vals))
            rollups[f"round_{metric_name}_p95"] = float(np.percentile(vals, 95))
        return rollups

    def _round_expects_weight_update(self, client_payloads):
        if bool(getattr(self.strategy, "inference_only", False)):
            return False
        if self.task_type == "clustering" or (self.model_type or "").lower() == "randomforest":
            return False
        try:
            if int(self.knobs.get("local_epochs", 1)) <= 0:
                return False
        except Exception:
            pass
        return bool(client_payloads)

    def _update_signature_output_dir(self, db_path):
        configured = self.config.get("update_signature_dir")
        if configured:
            return configured
        db_folder = os.path.dirname(str(db_path))
        if db_folder:
            return os.path.join(db_folder, "update_signatures")
        return os.path.join("outputs", "update_signatures")

    def _update_signature_config(self, db_path):
        enabled = _bool_config(self.config.get("update_signature_enabled"), default=True)
        if bool(getattr(self.strategy, "inference_only", False)):
            enabled = False
        if self.task_type == "clustering" or (self.model_type or "").lower() == "randomforest":
            enabled = False

        try:
            dim = int(self.config.get("update_signature_dim", 256) or 256)
        except Exception:
            dim = 256
        dim = max(1, dim)

        max_source_elements = self.config.get("update_signature_max_source_elements")
        if max_source_elements in (None, ""):
            max_source_elements = None
        else:
            try:
                max_source_elements = max(1, int(max_source_elements))
            except Exception:
                max_source_elements = None

        return {
            "enabled": bool(enabled),
            "dim": dim,
            "dir": self._update_signature_output_dir(db_path),
            "max_source_elements": max_source_elements,
        }

    def _save_final_model_parameters(self, *, run_id, round_idx, global_model, client_outcomes):
        files = []
        metadata = {
            "dataset": self.dataset,
            "task_type": self.task_type,
            "model_type": self.model_type,
            "hf_task": self.hf_task,
            "hf_model_id": (
                self.dataset_args.get("hf_model_id")
                if isinstance(self.dataset_args, dict)
                else self.config.get("hf_model_id")
            ) or self.config.get("hf_model_id"),
            "external_run_id": self.config.get("external_run_id"),
            "case_name": self.config.get("case_name"),
        }

        save_global = self.task_type != "clustering" and (self.model_type or "").lower() != "randomforest"
        if save_global:
            global_path = write_final_model_parameters(
                output_dir=self.final_model_params_dir,
                run_id=run_id,
                model_role="global",
                model_id="global",
                round_idx=round_idx,
                model_type=self.model_type,
                task_type=self.task_type,
                model=global_model,
                config=self.config,
                metadata=metadata,
            )
            if global_path:
                files.append({
                    "role": "global",
                    "model_id": "global",
                    "path": global_path,
                    "artifact_type": "directory" if os.path.isdir(global_path) else "json",
                })

        client_adapters = getattr(self.strategy, "_client_adapters", {}) or {}
        for outcome in client_outcomes or []:
            if not getattr(outcome, "participated", False):
                continue
            client_id = getattr(outcome, "client_id", None)
            if not client_id:
                continue

            client_metadata = {
                **metadata,
                "client_id": client_id,
                "samples_count": getattr(outcome, "samples_count", None),
                "aggregation_weight_unit": getattr(outcome, "aggregation_weight_unit", None),
                "aggregation_weight_value": getattr(outcome, "aggregation_weight_value", None),
            }
            pre_extracted = getattr(outcome, "model_params", None)
            payload = getattr(outcome, "payload", None)
            if pre_extracted is None and payload is None:
                continue

            path = write_final_model_parameters(
                output_dir=self.final_model_params_dir,
                run_id=run_id,
                model_role="client",
                model_id=client_id,
                round_idx=round_idx,
                model_type=self.model_type,
                task_type=self.task_type,
                model=client_adapters.get(client_id),
                payload=payload,
                pre_extracted=pre_extracted,
                config=self.config,
                metadata=client_metadata,
            )
            if path:
                files.append({
                    "role": "client",
                    "model_id": client_id,
                    "path": path,
                    "artifact_type": "directory" if os.path.isdir(path) else "json",
                })

        manifest_path = write_final_model_manifest(
            output_dir=self.final_model_params_dir,
            run_id=run_id,
            files=files,
        )
        return manifest_path
    
    def run(self):
        run_start_epoch = time.time()
        run_start_ts = datetime.now(timezone.utc).isoformat()

        os.makedirs("weights", exist_ok=True)

        verbose_progress = bool(self.config.get("verbose_progress", True))
        phase_label = "inference" if bool(getattr(self.strategy, "inference_only", False)) else "training"

        early_stopping_patience = self._early_stopping_patience()

        run_stage_measurements = dict(getattr(self, "_run_stage_measurements", {}))

        inference_only = bool(getattr(self.strategy, "inference_only", False))
        split_source_name = "eval" if inference_only else "train"
        split_x = self.x_test if inference_only else self.x_train
        split_y = self.y_test if inference_only else self.y_train

        split_timer = _StageTimer()
        clients, split_info = split_data(
            split_x,
            split_y,
            self.knobs["num_clients"],
            strategy=self.knobs["distribution_type"],
            distribution_param=self.knobs["distribution_param"],
            custom_distributions=self.knobs["custom_distributions"],
            sample_size=self.knobs["sample_size"],
            sample_frac=self.knobs["sample_frac"],
            rng=self.rng,
            meta=self.meta,
            task_family=self.task_family,
            hf_task=self.hf_task,
            skew_axis=self.knobs.get("skew_axis"),
            skew_axis_config=self.knobs.get("skew_axis_config"),
        )
        run_stage_measurements["stage_split_s"] = split_timer.elapsed_s()
        
        model_build_timer = _StageTimer()
        global_model = self.strategy.build_model()
        run_stage_measurements["stage_global_model_build_s"] = model_build_timer.elapsed_s()
        execution_device = self._resolve_execution_device(global_model)

        loaded_train_samples = _sample_count(self.x_train, self.y_train)
        loaded_test_samples = _sample_count(self.x_test, self.y_test)
        loaded_split_source_samples = _sample_count(split_x, split_y)
        split_partition_samples = int(sum(_sample_count(data.get("x"), data.get("y")) for data in clients.values()))
        client_sample_counts = [int(_sample_count(data.get("x"), data.get("y"))) for data in clients.values()]
        requested_partition_samples = None
        if self.knobs.get("sample_size") is not None:
            requested_partition_samples = max(0, int(self.knobs.get("sample_size"))) * int(self.knobs["num_clients"])
        resolved_split_strategy = (split_info or {}).get("strategy", self.knobs.get("distribution_type"))
        resolved_split_param = (split_info or {}).get("distribution_param", self.knobs.get("distribution_param"))
        requested_split_strategy = self.knobs.get("distribution_type")
        requested_split_param = self.knobs.get("distribution_param")

        print("\n========== RUN SUMMARY ==========")

        # Universal info (runner-ish)
        base = [
            ("external_run_id", self.config.get("external_run_id")),
            ("case_name", self.config.get("case_name")),
            ("run_group_id", self.config.get("run_group_id")),
            ("dataset_source", self.dataset),
            ("task_type", self.task_type),
            ("hf_task", self.hf_task if self.hf_task != "unknown" else None),
            ("model_type", self.model_type),
            ("num_clients", self.knobs["num_clients"]),
            ("num_rounds", self.knobs["num_rounds"]),
            ("client_dropout_rate", self.config.get("client_dropout_rate", 0.0)),
            ("seed", self.config.get("seed", 42)),
            ("save_weights", self.save_weights),
            ("save_final_model_params", self.save_final_model_params),
            ("final_model_params_dir", self.final_model_params_dir if self.save_final_model_params else None),
            ("input_shape", self.input_shape),
            ("num_classes", self.num_classes),
            ("loaded_train_samples", loaded_train_samples),
            ("loaded_test_samples", loaded_test_samples),
            ("timeout_s", self.config.get("timeout_s")),
            ("requested_device", self.config.get("device")),
            ("execution_device", execution_device),
        ]

        # Splitter info (always relevant). Requested values are kept distinct
        # from resolved/effective values because split_data can shrink samples
        # or fall back to iid for structured labels.
        splitter = [
            ("split.source", split_source_name),
            ("split.loaded_source_samples", loaded_split_source_samples),
            ("split.requested_strategy", requested_split_strategy),
            ("split.resolved_strategy", resolved_split_strategy),
            ("split.requested_axis", self.knobs.get("skew_axis")),
            ("split.resolved_axis", (split_info or {}).get("skew_axis")),
            ("split.fallback_reason", (split_info or {}).get("fallback_reason")),
            ("split.requested_param", requested_split_param),
            ("split.resolved_param", resolved_split_param),
            ("split.bucket_spec", (split_info or {}).get("bucket_spec")),
            ("split.bucket_distribution", (split_info or {}).get("bucket_distribution")),
            ("split.requested_samples_per_client", self.knobs.get("sample_size")),
            ("split.requested_partition_samples_total", requested_partition_samples),
            ("split.effective_sample_size_total", (split_info or {}).get("effective_sample_size_total")),
            (
                "split.resampled_with_replacement",
                (
                    requested_partition_samples > loaded_split_source_samples
                    if requested_partition_samples is not None
                    else None
                ),
            ),
            ("split.requested_sample_frac", self.knobs.get("sample_frac")),
            (f"split.effective_{split_source_name}_samples", split_partition_samples),
            ("split.client_samples_min", min(client_sample_counts) if client_sample_counts else None),
            ("split.client_samples_max", max(client_sample_counts) if client_sample_counts else None),
            ("split.distribution_bins", self.knobs.get("distribution_bins")),
        ]

        def _print_kv(items, width=26):
            for k, v in items:
                if v is None:
                    continue
                print(f"{k:>{width}} : {_format_summary_value(v)}")

        _print_kv(base)
        print("------------------------------------------------")
        _print_kv(splitter)

        # Strategy-specific (adapter/dataset/etc) — the important part
        lines = self.strategy.summary_lines()
        if lines:
            print("------------------------------------------------")
            for k, v in lines:
                if k.startswith("[") and v == "":
                    print(k)
                    continue
                if v is None:
                    continue
                print(f"{k:>26} : {_format_summary_value(v)}")

        print("================================================\n")

        print(f"Execution mode: federated {phase_label}")
        if verbose_progress:
            print("Verbose progress logging is enabled.")
            print("Per-client lifecycle logs: start -> strategy call -> completion/failure.")
        print()

        distribution_heading = (
            "Client evaluation data distributions before inference:"
            if inference_only
            else "Client data distributions before training:"
        )
        print(distribution_heading)
        client_distributions = {}
        is_fill_mask = self.task_family == "fill_mask"
        is_generation = self.task_family == "generation"
        is_retrieval = self.task_family == "retrieval"
        is_vqa = self.task_family == "vqa"
        ignore_index = infer_ignore_index(self.meta if isinstance(self.meta, dict) else None, default=-100)
        pad_token_id = self.meta.get("pad_token_id")
        for client_id, data in clients.items():
            if is_fill_mask:
                dist = get_mlm_masked_token_stats(
                    data["y"],
                    ignore_index=ignore_index,
                )
            elif is_generation:
                dist = get_token_label_stats(
                    data["y"],
                    ignore_index=ignore_index,
                    pad_token_id=pad_token_id,
                )
            elif is_retrieval:
                dist = get_retrieval_pair_stats(data.get("x"))
            elif is_vqa:
                dist = get_vqa_answer_stats(
                    data.get("y"),
                    ignore_index=ignore_index,
                )
            else:
                dist = get_data_distribution(
                    data["y"],
                    self.num_classes,
                    bins=self.knobs.get("distribution_bins"),
                    value_range=self.distribution_range,
                    label_pad_value=ignore_index,
                )
            client_distributions[client_id] = dist
            if is_retrieval or is_vqa:
                print(f"{client_id}: {json.dumps(dist, sort_keys=True)}")
            else:
                print(f"{client_id}: {dist}")

        # build global model via strategy

        hardware_snapshot = capture_hardware_snapshot()

        params_count = _count_model_params(global_model)

        run_id = _resolve_database_run_id(self.config)

        hf_model_meta = {}
        hf_model_id = (self.dataset_args.get("hf_model_id") or "").strip()
        is_hf_run = ("hf" in (self.dataset or "").lower()) and bool(hf_model_id)
        if is_hf_run:
            try:
                hf_model_meta = fetch_hf_model_meta(hf_model_id) or {}
            except Exception as exc:
                hf_model_meta = {
                    "hf_model_id": hf_model_id,
                    "hf_service_meta_json": json.dumps(
                        {
                            "hf_model_id": hf_model_id,
                            "provider": "huggingface_hub",
                            "error": str(exc),
                        },
                        default=str,
                    ),
                }

            user_hf_task = self.dataset_args.get("hf_task") or self.config.get("hf_task")
            if not user_hf_task:
                pipeline_task = (
                    self.dataset_args.get("hf_pipeline_tag")
                    or self.config.get("hf_pipeline_tag")
                    or self.meta.get("hf_pipeline_tag")
                    or self.dataset_args.get("hf_task")
                    or self.config.get("hf_task")
                    or self.meta.get("hf_task")
                )
                normalized_pipeline_task = normalize_hf_task(pipeline_task)
                if normalized_pipeline_task and normalized_pipeline_task != "unknown":
                    self.hf_task = normalized_pipeline_task
                    self.task_family = canonical_task_family(self.task_type, self.hf_task)
                    if hasattr(self.strategy, "hf_task"):
                        self.strategy.hf_task = self.hf_task


        db_path = self.config.get("db_path", "federated2.db")
        update_signature_cfg = self._update_signature_config(db_path)
        writer = make_writer("sqlite", db_path=db_path)
        skip_reason = None
        completed_all_rounds = False
        final_client_outcomes = []
        final_round_idx = None
        final_model_params_manifest = None
        writer.start()
        try:
            # Seed metric dictionary (recommended)
            if hasattr(writer, "seed_metrics"):
                writer.seed_metrics()

            # --- runs dimension
            writer.write_run(
                {
                    "run_id": run_id,
                    "dataset": self.dataset,
                    "task_type": self.task_type,
                    "model_type": self.model_type,
                    "num_clients": self.knobs["num_clients"],
                    "num_rounds": self.knobs["num_rounds"],
                }
            )

            # --- run_params (normalised config)
            # scope suggestions: runner/dataset/adapter/aggregator/splitter
            if hasattr(writer, "write_run_param"):
                # runner level
                writer.write_run_param(run_id, "runner", "seed", self.config.get("seed", 42))
                writer.write_run_param(run_id, "runner", "client_dropout_rate", self.config.get("client_dropout_rate", 0.0))
                writer.write_run_param(run_id, "runner", "save_weights", self.save_weights)
                writer.write_run_param(run_id, "runner", "save_final_model_params", self.save_final_model_params)
                writer.write_run_param(run_id, "runner", "final_model_params_dir", self.final_model_params_dir)
                writer.write_run_param(run_id, "runner", "update_signature_enabled", update_signature_cfg["enabled"])
                writer.write_run_param(run_id, "runner", "update_signature_dim", update_signature_cfg["dim"])
                writer.write_run_param(run_id, "runner", "update_signature_dir", update_signature_cfg["dir"])
                writer.write_run_param(run_id, "runner", "update_signature_max_source_elements", update_signature_cfg["max_source_elements"])
                writer.write_run_param(run_id, "runner", "enable_perturbation_metrics", self.config.get("enable_perturbation_metrics", True))
                writer.write_run_param(run_id, "runner", "perturbation_final_round_only", self.config.get("perturbation_final_round_only", True))
                writer.write_run_param(run_id, "runner", "perturbation_sample_count", self.config.get("perturbation_sample_count", 1))
                writer.write_run_param(run_id, "runner", "perturbation_trust_trials", self.config.get("perturbation_trust_trials", 2))
                writer.write_run_param(run_id, "runner", "perturbation_target_units", self.config.get("perturbation_target_units", 1))
                writer.write_run_param(run_id, "runner", "perturbation_candidate_units", self.config.get("perturbation_candidate_units", 4))
                writer.write_run_param(run_id, "runner", "perturbation_random_strength", self.config.get("perturbation_random_strength", 0.02))
                writer.write_run_param(run_id, "runner", "perturbation_progress_logging", self.config.get("perturbation_progress_logging", False))
                writer.write_run_param(run_id, "runner", "perturbation_progress_sample_interval", self.config.get("perturbation_progress_sample_interval", 1))
                writer.write_run_param(run_id, "runner", "explainability_meaningful_drop_threshold", self.config.get("explainability_meaningful_drop_threshold", 0.2))
                writer.write_run_param(run_id, "runner", "explainability_selectivity_floor", self.config.get("explainability_selectivity_floor", 0.5))

                manifest_metadata = {}
                for key in (
                    "run_regime",
                    "service_source",
                    "model_role",
                    "input_schema",
                    "fit_decision",
                    "fit_reason",
                    "realism_score",
                    "domain_alignment",
                    "dataset_hint",
                    "modality",
                    "hf_pipeline_tag",
                    "hf_downloads",
                    "hf_likes",
                    "hf_author",
                    "hf_url",
                    "hf_service_meta_json",
                    "case_name",
                    "run_group_id",
                    "external_run_id",
                ):
                    value = self.config.get(key)
                    if value is None and isinstance(self.dataset_args, dict):
                        value = self.dataset_args.get(key)
                    if value is not None:
                        manifest_metadata[key] = value

                if "service_source" not in manifest_metadata:
                    manifest_metadata["service_source"] = "huggingface_hub" if is_hf_run else "generic"
                if "run_regime" not in manifest_metadata:
                    if is_hf_run:
                        manifest_metadata["run_regime"] = "inference_only" if inference_only else "finetune_transfer"
                    else:
                        manifest_metadata["run_regime"] = "generic"
                manifest_metadata["weights_exported"] = bool(self.save_weights or self.save_final_model_params)

                runner_metadata_keys = {"run_regime", "service_source", "weights_exported", "case_name", "run_group_id", "external_run_id"}
                adapter_metadata_keys = {"model_role", "fit_decision", "fit_reason", "realism_score"}
                dataset_metadata_keys = set(manifest_metadata) - runner_metadata_keys - adapter_metadata_keys
                for key in sorted(runner_metadata_keys & set(manifest_metadata)):
                    writer.write_run_param(run_id, "runner", key, manifest_metadata[key])
                for key in sorted(adapter_metadata_keys & set(manifest_metadata)):
                    writer.write_run_param(run_id, "adapter", key, manifest_metadata[key])
                for key in sorted(dataset_metadata_keys):
                    writer.write_run_param(run_id, "dataset", key, manifest_metadata[key])

                # benchmark identity for cross-run comparisons
                benchmark_identity = {
                    "dataset": self.dataset,
                    "model": self.model_type,
                    "clients": self.knobs.get("num_clients"),
                    "rounds": self.knobs.get("num_rounds"),
                    "batch": self.knobs.get("batch_size"),
                    "max_length": self.dataset_args.get("max_length", self.config.get("max_length", self.meta.get("max_length"))),
                    "device": execution_device,
                }
                for key, value in benchmark_identity.items():
                    writer.write_run_param(run_id, "runner", f"benchmark_{key}", value)

                # splitter / distribution
                writer.write_run_param(run_id, "splitter", "distribution_type", self.knobs.get("distribution_type"))
                writer.write_run_param(run_id, "splitter", "distribution_param", self.knobs.get("distribution_param"))
                writer.write_run_param(run_id, "splitter", "skew_axis", self.knobs.get("skew_axis"))
                writer.write_run_param(run_id, "splitter", "resolved_skew_axis", (split_info or {}).get("skew_axis"))
                writer.write_run_param(run_id, "splitter", "distribution_bins", self.knobs.get("distribution_bins"))
                writer.write_run_param(run_id, "splitter", "split_source", split_source_name)
                writer.write_run_param(run_id, "splitter", "loaded_source_samples", loaded_split_source_samples)
                writer.write_run_param(run_id, "splitter", "sample_size", self.knobs.get("sample_size"))
                writer.write_run_param(run_id, "splitter", "requested_samples_per_client", self.knobs.get("sample_size"))
                writer.write_run_param(run_id, "splitter", "requested_partition_samples_total", requested_partition_samples)
                writer.write_run_param(run_id, "splitter", "effective_sample_size_total", (split_info or {}).get("effective_sample_size_total"))
                writer.write_run_param(run_id, "splitter", "effective_partition_samples", split_partition_samples)
                writer.write_run_param(
                    run_id,
                    "splitter",
                    "resampled_with_replacement",
                    (
                        requested_partition_samples > loaded_split_source_samples
                        if requested_partition_samples is not None
                        else None
                    ),
                )
                writer.write_run_param(run_id, "splitter", "sample_frac", self.knobs.get("sample_frac"))

                params_by_scope = self.strategy.loggable_run_params()
                for scope, kv in (params_by_scope or {}).items():
                    for k, v in (kv or {}).items():
                        writer.write_run_param(run_id, scope, k, v)

                # dataset args (store as JSON)
                writer.write_run_param(run_id, "dataset", "dataset_args", self.dataset_args)
                if isinstance(self.meta, dict) and isinstance(self.meta.get("accounting"), dict):
                    writer.write_run_param(run_id, "dataset", "accounting", self.meta["accounting"])

                if is_hf_run:
                    if self.hf_task and self.hf_task != "unknown":
                        writer.write_run_param(run_id, "dataset", "hf_task", self.hf_task)
                    def _is_present(value):
                        if value is None:
                            return False
                        if isinstance(value, str):
                            return bool(value.strip())
                        if isinstance(value, (dict, list, tuple, set)):
                            return len(value) > 0
                        return True

                    def _pick_hf_value(key):
                        for source in (self.dataset_args, self.config, self.meta, hf_model_meta):
                            if isinstance(source, dict) and key in source:
                                value = source.get(key)
                                if _is_present(value):
                                    return value
                        return None

                    hf_metadata_keys = [
                        "hf_model_id",
                        "hf_pipeline_tag",
                        "hf_downloads",
                        "hf_likes",
                        "hf_last_modified",
                        "hf_author",
                        "hf_url",
                        "hf_service_meta_json",
                    ]
                    for key in hf_metadata_keys:
                        value = _pick_hf_value(key)
                        if _is_present(value):
                            writer.write_run_param(run_id, "dataset", key, value)


                # hardware snapshot / params count (optional)
                writer.write_run_param(run_id, "runner", "params_count", params_count)
                writer.write_run_param(run_id, "runner", "hardware_snapshot", hardware_snapshot)

            run_init_measurements = {
                **self._canonical_run_metadata(),
                **run_stage_measurements,
                "run_start_ts": run_start_ts,
            }
            writer.write_measurements(
                run_id=run_id,
                round=None,
                client_id=None,
                values=run_init_measurements,
            )

            # --- clients dimension
            for client_id, data in clients.items():
                writer.write_client(
                    {
                        "run_id": run_id,
                        "client_id": client_id,
                        "data_distribution_json": json.dumps(client_distributions.get(client_id, {})),
                        "samples_count": int(len(data["y"])),
                    }
                )

            participated_counts = {cid: 0 for cid in clients.keys()}
            dynamics_tolerance = float(self.config.get("federated_dynamics_tolerance", DEFAULT_TOLERANCE))
            previous_round_global_weights = None
            previous_global_metrics = None

            for round_num in range(self.knobs["num_rounds"]):
                round_idx = round_num + 1
                print(f"--- Round {round_idx} ---")
                round_total_timer = _StageTimer()
                round_log_time_s = 0.0
                round_pre_overhead_timer = _StageTimer()

                client_payloads = []
                client_outcomes = []
                round_metrics = []
                round_qos_records = []
                skipped_clients = 0

                down_bytes = self.strategy.comm_down_bytes(global_model)
                round_start_weights = snapshot_model_weights(global_model)
                carry_forward_values = carry_forward_metrics(
                    previous_round_global_weights,
                    round_start_weights,
                    tolerance=dynamics_tolerance,
                )
                if verbose_progress:
                    print(
                        f"Round {round_idx}: scheduled_clients={len(clients)}, "
                        f"phase={phase_label}, comm_down_bytes={int(down_bytes)}"
                    )

                # Round dimension row
                write_round_timer = _StageTimer()
                writer.write_round(
                    {
                        "run_id": run_id,
                        "round": round_idx,
                        "scheduled_clients": len(clients),
                        "attempted_clients": None,
                        "participating_clients": None,
                        "dropped_clients": None,
                    }
                )
                round_log_time_s += write_round_timer.elapsed_s()
                round_pre_overhead_s = round_pre_overhead_timer.elapsed_s()

                for client_id, data in clients.items():
                    dist = client_distributions.get(client_id)
                    n_samples = int(len(data["y"]))

                    # dropout
                    if self.rng.random() < self.config.get("client_dropout_rate", 0.0):
                        skipped_clients += 1
                        # record dropout as measurements
                        write_dropout_timer = _StageTimer()
                        writer.write_measurements(
                            run_id=run_id,
                            round=round_idx,
                            client_id=client_id,
                            values={
                                "participated_flag": False,
                                "fail_reason": "client_dropout",
                                "samples_count": n_samples,
                                "comm_bytes_down": int(down_bytes),
                                "comm_bytes_up": 0,
                                "compute_time_s": 0.0,
                                "effective_batch_size": int(self.knobs.get("batch_size") or 0),
                                "tokens_in": 0,
                                "tokens_out": 0,
                                "avg_seq_len": 0.0,
                                "truncation_rate": 0.0,
                                "oom_count": 0,
                                "nan_count": 0,
                                "fail_reason_category": "dropout",
                            },
                        )
                        round_log_time_s += write_dropout_timer.elapsed_s()
                        print(f"{client_id} dropped out")
                        continue

                    next_rounds_so_far = participated_counts[client_id] + 1
                    sample_label = "eval_samples" if inference_only else "train_samples"
                    print(
                        f"{client_id} {phase_label}... "
                        f"({sample_label}={n_samples}, round={round_idx}, participation_count={next_rounds_so_far})"
                    )
                    if verbose_progress:
                        action = "evaluate_client" if bool(getattr(self.strategy, "inference_only", False)) else "train_client"
                        print(f"{client_id}: invoking strategy.{action}")

                    outcome = self.strategy.train_client(
                        client_id=client_id,
                        x=data["x"],
                        y=data["y"],
                        global_model=global_model,
                        round_idx=round_idx,
                        rounds_so_far=next_rounds_so_far,
                        comm_down=down_bytes,
                    )
                    setattr(outcome, "client_id", client_id)

                    if not outcome.participated and is_rocm_miopen_runtime_error(outcome.fail_reason):
                        raise _RunSkipped(
                            "skipped_rocm_miopen_runtime_error: "
                            f"{outcome.fail_reason or 'MIOpen/HIPRTC runtime failure'}"
                        )

                    if verbose_progress:
                        status = "participated" if outcome.participated else f"skipped ({outcome.fail_reason or 'unknown reason'})"
                        print(
                            f"{client_id}: strategy completed, status={status}, "
                            f"duration_s={float(outcome.duration):.3f}, "
                            f"metric={self.metric_key}:{float(outcome.metric_value) if outcome.metric_value == outcome.metric_value else 'nan'}"
                        )
                        extras = outcome.extras if isinstance(getattr(outcome, "extras", None), dict) else {}
                        if extras:
                            perf_keys = (
                                "cold_start_time",
                                "tokenizer_load_s",
                                "model_load_s",
                                "tokenizer_cache_hit",
                                "model_cache_hit",
                                "eval_latency_ms_mean",
                                "eval_latency_ms_p95",
                                "eval_latency_ms_steady_mean",
                                "eval_latency_ms_steady_p95",
                                "eval_throughput_eps",
                                "tokens_per_second",
                                "eval_sequence_count",
                                "client_partition_sample_count",
                                "batch_size",
                                "device",
                            )
                            perf_parts = [f"{k}={extras[k]}" for k in perf_keys if k in extras and extras[k] is not None]
                            if perf_parts:
                                print(f"{client_id}: perf " + ", ".join(perf_parts))

                    client_outcomes.append(outcome)

                    client_values = {
                        "participated_flag": bool(outcome.participated),
                        "fail_reason": outcome.fail_reason if not outcome.participated else None,
                        "samples_count": int(outcome.samples_count),
                        "sequence_count": int(outcome.sequence_count) if getattr(outcome, "sequence_count", None) is not None else None,
                        "supervised_token_count": int(outcome.supervised_token_count) if getattr(outcome, "supervised_token_count", None) is not None else None,
                        "aggregation_weight_unit": getattr(outcome, "aggregation_weight_unit", None),
                        "aggregation_weight_value": float(outcome.aggregation_weight_value) if getattr(outcome, "aggregation_weight_value", None) is not None else None,
                        "compute_time_s": float(outcome.duration),
                        "comm_bytes_down": int(outcome.comm_down),
                        "comm_bytes_up": int(outcome.comm_up),
                        "loss": float(outcome.loss) if outcome.loss == outcome.loss else None,
                        self.metric_key: float(outcome.metric_value) if outcome.metric_value == outcome.metric_value else None,
                        "metric_score": float(outcome.metric_score) if outcome.metric_score == outcome.metric_score else None,
                        "extra_metric": float(outcome.extra_metric) if outcome.extra_metric == outcome.extra_metric else None,
                        "cpu_time_s": float(outcome.cpu_time_s) if outcome.cpu_time_s is not None else None,
                        "cpu_utilization": float(outcome.cpu_utilization) if outcome.cpu_utilization is not None else None,
                        "memory_used_mb": float(outcome.memory_used_mb) if outcome.memory_used_mb is not None else None,
                        "memory_utilization": float(outcome.memory_utilization) if outcome.memory_utilization is not None else None,
                        "gpu_utilization": float(outcome.gpu_utilization) if outcome.gpu_utilization is not None else None,
                        "gpu_memory_used_mb": float(outcome.gpu_memory_used_mb) if outcome.gpu_memory_used_mb is not None else None,
                        "gpu_memory_utilization": float(outcome.gpu_memory_utilization) if outcome.gpu_memory_utilization is not None else None,
                    }
                    client_values.update(
                        client_update_metrics(
                            round_start_weights,
                            outcome.payload,
                            tolerance=dynamics_tolerance,
                        )
                    )
                    if (
                        update_signature_cfg["enabled"]
                        and bool(getattr(outcome, "participated", False))
                        and getattr(outcome, "payload", None) is not None
                    ):
                        try:
                            client_values.update(
                                compute_and_store_update_signature(
                                    round_start_weights,
                                    outcome.payload,
                                    output_dir=update_signature_cfg["dir"],
                                    run_id=run_id,
                                    round_idx=round_idx,
                                    client_id=client_id,
                                    dim=update_signature_cfg["dim"],
                                    seed=int(self.config.get("seed", 42) or 42),
                                    max_source_elements=update_signature_cfg["max_source_elements"],
                                )
                            )
                        except Exception as exc:
                            client_values["update_signature_error"] = type(exc).__name__
                    client_values.update(self._extract_dynamic_metrics(outcome))
                    client_values.update(self._normalize_outcome_extras(outcome))
                    client_values = self._drop_non_final_trust_metrics(client_values, round_idx)

                    client_values = {
                        key: value
                        for key, value in client_values.items()
                        if self._safe_metric_value(value) is not None
                    }

                    if outcome.participated:
                        round_qos_records.append(client_values)

                    # write client-round measurements
                    write_client_timer = _StageTimer()
                    writer.write_measurements(
                        run_id=run_id,
                        round=round_idx,
                        client_id=client_id,
                        values=client_values,
                    )
                    round_log_time_s += write_client_timer.elapsed_s()

                    round_metrics.append(
                        {
                            "participated": bool(outcome.participated),
                            "duration": outcome.duration,
                            "cpu_utilization": outcome.cpu_utilization,
                            "memory_utilization": outcome.memory_utilization,
                            "memory_used_mb": outcome.memory_used_mb,
                            "gpu_utilization": outcome.gpu_utilization,
                            "gpu_memory_utilization": outcome.gpu_memory_utilization,
                            "gpu_memory_used_mb": outcome.gpu_memory_used_mb,
                            "cpu_time_s": outcome.cpu_time_s,
                        }
                    )

                    if outcome.participated:
                        participated_counts[client_id] = next_rounds_so_far

                    if outcome.payload is not None:
                        client_payloads.append(outcome.payload)

                # aggregate & evaluate globally
                if verbose_progress:
                    print(
                        f"Round {round_idx}: aggregating {len(client_payloads)} payload(s) from "
                        f"{sum(1 for o in client_outcomes if getattr(o, 'participated', False))} participating client(s)."
                    )
                aggregation_timer = _StageTimer()
                global_weights_before_aggregation = snapshot_model_weights(global_model)
                loss, global_metric, global_score, global_extra = self.strategy.aggregate_and_eval(
                    global_model=global_model,
                    client_payloads=client_payloads,
                    client_outcomes=client_outcomes,
                    round_idx=round_idx,
                    x_train=self.x_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                )
                global_weights_after_aggregation = snapshot_model_weights(global_model)
                global_update_values = global_update_metrics(
                    global_weights_before_aggregation,
                    global_weights_after_aggregation,
                    tolerance=dynamics_tolerance,
                )
                expected_weight_update = self._round_expects_weight_update(client_payloads)
                current_global_metrics = {
                    "loss": loss,
                    "metric": global_metric,
                    "score": global_score,
                    "extra": global_extra,
                }
                repeated_values = repeated_round_metrics(
                    previous_global_metrics,
                    current_global_metrics,
                    expected_update=expected_weight_update,
                    global_weights_changed=global_update_values.get("round_global_weight_changed_flag"),
                    tolerance=dynamics_tolerance,
                )
                previous_global_metrics = current_global_metrics
                previous_round_global_weights = global_weights_after_aggregation
                round_aggregation_s = aggregation_timer.elapsed_s()

                round_usage_summary = summarize_round_usage(
                    round_metrics,
                    scheduled_clients=len(clients),
                    skipped_clients=skipped_clients,
                )

                # update round dimension with aggregates
                write_round_summary_timer = _StageTimer()
                writer.write_round(
                    {
                        "run_id": run_id,
                        "round": round_idx,
                        "scheduled_clients": len(clients),
                        "attempted_clients": int(len(clients) - skipped_clients),
                        "participating_clients": int(sum(1 for o in client_outcomes if getattr(o, "participated", False))),
                        "dropped_clients": int(skipped_clients),
                    }
                )
                round_log_time_s += write_round_summary_timer.elapsed_s()

                round_client_compute_s = float(
                    sum(float(getattr(outcome, "duration", 0.0) or 0.0) for outcome in client_outcomes)
                )
                round_total_s = round_total_timer.elapsed_s()
                round_overhead_s = max(
                    0.0,
                    round_total_s - round_client_compute_s - round_aggregation_s - round_log_time_s,
                )

                # write round-level measurements (client_id NULL)
                write_round_metrics_timer = _StageTimer()
                writer.write_measurements(
                    run_id=run_id,
                    round=round_idx,
                    client_id=None,
                    values={
                        "global_loss": loss,
                        f"global_{self.metric_key}": global_metric,
                        "global_metric_score": global_score,
                        "global_aux_metric": global_extra,
                        "round_aggregation_s": round_aggregation_s,
                        "round_logging_s": round_log_time_s,
                        "round_overhead_s": round_overhead_s,
                        "round_overhead_pre_client_loop_s": round_pre_overhead_s,
                        "round_resource_summary": round_usage_summary,
                        "federated_update_expected_flag": expected_weight_update,
                        "aggregation_payload_count": int(len(client_payloads)),
                        **carry_forward_values,
                        **global_update_values,
                        **repeated_values,
                        **self._round_qos_rollups(round_qos_records),
                    },
                )
                round_log_time_s += write_round_metrics_timer.elapsed_s()

                if self.task_type == "regression" and self.target_scaler and self.target_scaler.get("type") == "standard":
                    rmse_std = float(global_metric)
                    rmse_orig = rmse_std * float(self.target_scaler["std"])
                    print(f"Global model {self.metric_label}: {rmse_std:.6f} (standardized) | {rmse_orig:.2f} (original units)")
                else:
                    print(f"Global model {self.metric_label}: {global_metric}")

                if self.task_type == "regression":
                    print(f"Global metric score: {global_score}")
                if global_extra is not None:
                    print(f"Global auxiliary metric: {global_extra}")
                if verbose_progress:
                    print(
                        f"Round {round_idx} complete: attempted={int(len(clients) - skipped_clients)}, "
                        f"participating={int(sum(1 for o in client_outcomes if getattr(o, 'participated', False)))}, "
                        f"dropped={int(skipped_clients)}"
                    )
                    print("----------------------------------------")

                final_client_outcomes = list(client_outcomes)
                final_round_idx = round_idx

            completed_all_rounds = True

        except _RunSkipped as exc:
            skip_reason = str(exc)
            print(f"Run skipped before database commit: {skip_reason}")
            if hasattr(writer, "abort"):
                writer.abort()
            else:
                writer.finish()
            return {
                "run_id": None,
                "db_path": db_path,
                "rounds": self.knobs["num_rounds"],
                "clients": self.knobs["num_clients"],
                "status": "skipped",
                "skip_reason": skip_reason,
            }
        finally:
            if skip_reason is None:
                if self.save_final_model_params and completed_all_rounds:
                    try:
                        final_model_params_manifest = self._save_final_model_parameters(
                            run_id=run_id,
                            round_idx=final_round_idx,
                            global_model=global_model,
                            client_outcomes=final_client_outcomes,
                        )
                        if final_model_params_manifest:
                            print(f"Final model parameters saved: {final_model_params_manifest}")
                    except Exception as exc:
                        print(f"Warning: failed to save final model parameters: {exc}")
                run_end_epoch = time.time()
                run_end_ts = datetime.now(timezone.utc).isoformat()
                run_end_values = {
                    "run_end_ts": run_end_ts,
                    "run_total_runtime_s": float(run_end_epoch - run_start_epoch),
                }
                if final_model_params_manifest:
                    run_end_values["final_model_params_manifest"] = final_model_params_manifest
                writer.write_measurements(
                    run_id=run_id,
                    round=None,
                    client_id=None,
                    values=run_end_values,
                )
                writer.finish()

        print("Federated Learning Process Complete!\n")
        return {
            "run_id": run_id,
            "db_path": db_path,
            "rounds": self.knobs["num_rounds"],
            "clients": self.knobs["num_clients"],
        }
