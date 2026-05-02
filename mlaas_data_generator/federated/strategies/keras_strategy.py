# classification_keras.py
from ...models.train_eval import train_local_model, evaluate_model, aggregate_weights
from ..system_metrics import ResourceTracker
from ..model_params import extract_model_parameters
from .base import TaskStrategy, ClientOutcome, metric_score_value, _nanmean, _is_keras_like, weights_size
import time, json
import numpy as np


def _bounded_score(value, default=0.0):
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(np.clip(parsed, 0.0, 1.0))


def _importance_proxy_score(model):
    arrays = []
    for attr in ("feature_importances_", "coef_", "cluster_centers_"):
        if hasattr(model, attr):
            try:
                arrays.append(np.asarray(getattr(model, attr), dtype="float64").reshape(-1))
            except Exception:
                pass
    get_weights = getattr(model, "get_weights", None)
    if callable(get_weights):
        try:
            arrays.extend(np.asarray(w, dtype="float64").reshape(-1) for w in get_weights())
        except Exception:
            pass

    if not arrays:
        return False, 0.0

    values = np.abs(np.concatenate([arr for arr in arrays if arr.size]))
    values = values[np.isfinite(values)]
    if values.size == 0 or float(values.sum()) <= 0.0:
        return True, 0.0

    probs = values / float(values.sum())
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(values.size)) if values.size > 1 else 1.0
    concentration = 1.0 - (entropy / max(max_entropy, 1e-12))
    return True, float(np.clip(concentration, 0.0, 1.0))


def _generic_runtime_metrics(model, *, eval_latency_s, metric_score, loss, include_trust_metrics=True):
    metrics = {
        "inference_latency_s": float(eval_latency_s),
        "inference_latency_s_p95": float(eval_latency_s),
    }
    if include_trust_metrics:
        explainability_supported, explainability_score = _importance_proxy_score(model)
        trust_quality = _bounded_score(metric_score)
        loss_penalty = 1.0 / (1.0 + max(0.0, float(loss))) if loss == loss else trust_quality
        trust_score = float(np.clip((trust_quality + loss_penalty) / 2.0, 0.0, 1.0))
        metrics.update(
            {
                "explainability_supported_flag": bool(explainability_supported),
                "explainability_method": "parameter_importance_proxy" if explainability_supported else "unsupported",
                "explainability_score": float(explainability_score),
                "trust_score": trust_score,
                "trust_method": "quality_loss_proxy",
            }
        )
    return metrics


def _merge_runtime_metrics(extras, runtime_metrics):
    merged = dict(extras or {})
    for key, value in runtime_metrics.items():
        merged.setdefault(key, value)
    return merged


class ClassificationStrategy(TaskStrategy):
    def task_type(self) -> str: return "classification"

    def loggable_run_params(self):
        params = super().loggable_run_params()
        mt = (self.config.get("model_type") or "").lower()
        params["aggregator"] = {
            "strategy": "client_metric_average_no_weight_updates" if mt == "randomforest" else "fedavg_uniform",
            "aggregation_weight_unit": "client_uniform",
        }
        return params

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        samples_count = len(y)

        if _is_keras_like(local_model) and _is_keras_like(global_model):
            try:
                local_model.set_weights(global_model.get_weights())
            except Exception:
                pass
        
        start = time.time()
        tracker = ResourceTracker()
        tracker.start()

        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            usage = tracker.stop(duration)
            eval_start = time.time()
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="classification")
            eval_latency_s = time.time() - eval_start
            mscore = metric_score_value("classification", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            extras = _merge_runtime_metrics(
                self.perturbation_metrics(local_model, client_id=client_id, round_idx=round_idx),
                _generic_runtime_metrics(
                    local_model,
                    eval_latency_s=eval_latency_s,
                    metric_score=mscore,
                    loss=loss,
                    include_trust_metrics=self.should_run_perturbation_metrics(round_idx),
                ),
            )
            model_params = None
            if weights is None:
                model_params = extract_model_parameters(
                    model=local_model,
                    config=self.config,
                    metadata={"client_id": client_id, "round": round_idx},
                )

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=weights, extras=extras,  # accuracy/f1 added in records builder
                model_params=model_params,
            )
        except Exception as e:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason=repr(e), samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test,):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if client_payloads:
            new_global_weights = aggregate_weights(client_payloads)
            # Keep parity with your existing set_weights(list_ordered)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            if self.save_weights:
                with open(f"weights/global_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in new_global_weights.items()}, f, indent=4)
        else:
            print("No participating clients provided weights; using client metrics fallback.")
            if participated:
                loss = _nanmean([o.loss for o in participated])
                metric_value = _nanmean([o.metric_value for o in participated])
                extra_metric = _nanmean([o.extra_metric for o in participated])
                mscore = metric_score_value("classification", metric_value)
                return loss, metric_value, mscore, extra_metric

        loss, metric_value, extra_metric = evaluate_model(global_model, x_test, y_test, task_type="classification")
        mscore = metric_score_value("classification", metric_value)
        return loss, metric_value, mscore, extra_metric


class RegressionStrategy(TaskStrategy):
    def task_type(self) -> str: return "regression"

    def loggable_run_params(self):
        params = super().loggable_run_params()
        mt = (self.config.get("model_type") or "").lower()
        params["aggregator"] = {
            "strategy": "client_metric_average_no_weight_updates" if mt == "randomforest" else "fedavg_uniform",
            "aggregation_weight_unit": "client_uniform",
        }
        return params

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        local_model = self.build_model()
        samples_count = len(y)
        if _is_keras_like(local_model) and _is_keras_like(global_model):
            try:
                local_model.set_weights(global_model.get_weights())
            except Exception:
                pass
        start = time.time()
        tracker = ResourceTracker()
        tracker.start()
        try:
            weights = train_local_model(
                local_model, x, y,
                epochs=self.knobs["local_epochs"],
                batch_size=self.knobs["batch_size"],
            )
            duration = time.time() - start
            usage = tracker.stop(duration)
            eval_start = time.time()
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="regression")
            eval_latency_s = time.time() - eval_start
            mscore = metric_score_value("regression", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in weights.items()}, f, indent=4)

            extras = _merge_runtime_metrics(
                self.perturbation_metrics(local_model, client_id=client_id, round_idx=round_idx),
                _generic_runtime_metrics(
                    local_model,
                    eval_latency_s=eval_latency_s,
                    metric_score=mscore,
                    loss=loss,
                    include_trust_metrics=self.should_run_perturbation_metrics(round_idx),
                ),
            )
            model_params = None
            if weights is None:
                model_params = extract_model_parameters(
                    model=local_model,
                    config=self.config,
                    metadata={"client_id": client_id, "round": round_idx},
                )

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=weights, extras=extras,
                model_params=model_params,
            )
        except Exception:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason="error", samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test,):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if client_payloads:
            new_global_weights = aggregate_weights(client_payloads)
            # Keep parity with your existing set_weights(list_ordered)
            global_model.set_weights([new_global_weights[f"layer_{i}"] for i in range(len(new_global_weights))])
            if self.save_weights:
                with open(f"weights/global_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in new_global_weights.items()}, f, indent=4)
        else:
            print("No participating clients provided weights; using client metrics fallback.")
            if participated:
                loss = _nanmean([o.loss for o in participated])
                metric_value = _nanmean([o.metric_value for o in participated])
                extra_metric = _nanmean([o.extra_metric for o in participated])
                mscore = metric_score_value("regression", metric_value)
                return loss, metric_value, mscore, extra_metric
        
        print(f"[DEBUG] Model compiled: {hasattr(global_model, 'optimizer') and global_model.optimizer is not None}")

        loss, metric_value, extra_metric = evaluate_model(global_model, x_test, y_test, task_type="regression")
        mscore = metric_score_value("regression", metric_value)
        return loss, metric_value, mscore, extra_metric
