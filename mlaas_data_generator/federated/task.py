# task.py
from __future__ import annotations
from dataclasses import dataclass
import json, time, numpy as np

from ..models.train_eval import train_local_model, evaluate_model, aggregate_weights
from ..models.builders import create_model
from .system_metrics import ResourceTracker

def metric_score_value(task_type: str, metric_value: float | None) -> float:
    """Map raw metric to a [0,1]-ish score. For classification/clustering, identity; for regression, 1/(1+rmse)."""
    mv = float(metric_value) if metric_value is not None else np.nan
    if mv != mv:  # NaN
        return np.nan
    if task_type == "regression":
        return 1.0 / (1.0 + mv)
    return mv

def weights_size(weights_dict_or_list) -> int:
    if not weights_dict_or_list:
        return 0
    arrays = weights_dict_or_list.values() if isinstance(weights_dict_or_list, dict) else weights_dict_or_list
    return int(sum(np.asarray(w).nbytes for w in arrays))

def _is_keras_like(m) -> bool:
    return hasattr(m, "get_weights") and callable(getattr(m, "get_weights", None)) \
        and hasattr(m, "set_weights") and callable(getattr(m, "set_weights", None))

def _nanmean(values):
    cleaned = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))

@dataclass
class ClientOutcome:
    participated: bool
    fail_reason: str
    samples_count: int
    duration: float
    loss: float
    metric_value: float 
    metric_score: float 
    extra_metric: float 
    rounds_so_far: int
    comm_down: int
    comm_up: int
    cpu_time_s: float | None
    cpu_utilization: float | None
    memory_used_mb: float | None
    memory_utilization: float | None
    gpu_utilization: float | None
    gpu_memory_utilization: float | None
    gpu_memory_used_mb: float | None
    payload: dict | list | None   # weights for NN, None for clustering
    extras: dict                  # extra columns per task

class TaskStrategy:
    """Base class: thin wrapper around your existing per-task logic."""
    def __init__(self, meta, knobs, config, x_test, y_test, metric_key, save_weights: bool):
        self.meta = meta
        self.knobs = knobs
        self.config = config
        self.x_test = x_test
        self.y_test = y_test
        self.metric_key = metric_key
        self.save_weights = save_weights

    # ---- shared helpers
    def build_model(self):
        extra = {}
        if self.task_type() == "clustering":
            for key in ("clustering_k","clustering_init","clustering_n_init","clustering_max_iter","clustering_tol","seed","random_state"):
                if key in self.config:
                    extra[key] = self.config[key]
        else:
            for key in ("rf_trees", "rf_max_depth", "mobilenet_trainable", "n_estimators", "max_depth",
                        "hf_model_id", "max_length", "device"):
                if key in self.config:
                    extra[key] = self.config[key]

        if "batch_size" in self.knobs:
            extra["batch_size"] = self.knobs["batch_size"]

        ds_args = self.config.get("dataset_args", {}) or {}

        for key in ("hf_model_id", "max_length", "device"):
            if key in ds_args and key not in extra:
                extra[key] = ds_args[key]

        # HF adapter uses batch_size at construction time
        if "batch_size" in self.knobs and "batch_size" not in extra:
            extra["batch_size"] = self.knobs["batch_size"]

        return create_model(
            input_shape=tuple(self.meta["input_shape"]),
            num_classes=self.meta.get("num_classes"),
            hidden_layers=self.knobs["hidden_layers"],
            learning_rate=self.knobs["learning_rate"],
            activation=self.knobs["activation"],
            dropout=self.knobs["dropout"],
            weight_decay=self.knobs["weight_decay"],
            optimizer=self.knobs["optimizer"],
            task_type=self.task_type(),
            model_type=self.config.get("model_type"),
            **extra, 
        )

    def comm_down_bytes(self, global_model):
        # number of bytes when broadcasting global weights (0 for clustering adapter if no weights)
        try:
            return weights_size(global_model.get_weights())
        except Exception:
            return 0
        
    def task_type(self) -> str: ...
    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome: ...
    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        """Return (loss, metric_value, metric_score, extra_metric)."""
        ...

# -------------------- Classification --------------------

class ClassificationStrategy(TaskStrategy):
    def task_type(self) -> str: return "classification"

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
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="classification")
            mscore = metric_score_value("classification", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: v.tolist() for k, v in weights.items()}, f, indent=4)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=weights, extras={},  # accuracy/f1 added in records builder
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] {client_id} failed in round {round_idx}: {repr(e)}")
            print(tb)

            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
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


class HFInferenceClassificationStrategy(TaskStrategy):
    def task_type(self) -> str:
        return "classification"

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
        # x is list[str], y is np.ndarray[int]
        samples_count = len(y)

        start = time.time()
        tracker = ResourceTracker()
        tracker.start()

        try:
            # For HF inference-only we treat "global_model" as the service itself (adapter instance)
            adapter = global_model
            if adapter is None:
                adapter = self.build_model()

            batch_debug = adapter.core.debug_first_processed_batch(x, y, inference_only=False)
            print(
                "[evaluate_client] first processed batch | "
                f"input_ids_shape={batch_debug.get('input_ids_shape')} | "
                f"attention_mask_shape={batch_debug.get('attention_mask_shape')} | "
                f"labels_shape={batch_debug.get('labels_shape')} | "
                f"finite_ok={batch_debug.get('finite_ok')} | "
                f"nested_object_keys={batch_debug.get('nested_object_keys')}"
            )
            print(f"[evaluate_client] token example: {batch_debug.get('token_example')}")
            print(f"[evaluate_client] ner_tags example: {batch_debug.get('ner_tags_example')}")

            loss, acc, f1, qos = adapter.evaluate(x, y)

            duration = time.time() - start
            usage = tracker.stop(duration)

            mscore = metric_score_value("classification", acc)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=acc, metric_score=mscore, extra_metric=f1,
                rounds_so_far=rounds_so_far, comm_down=0, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=None,
                extras=qos,
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] {client_id} failed in round {round_idx}: {repr(e)}")
            print(tb)

            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason=repr(e), samples_count=samples_count, duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=0, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        # No weights, so aggregate metrics from participating clients
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if not participated:
            return np.nan, np.nan, np.nan, np.nan

        loss = _nanmean([o.loss for o in participated])
        acc = _nanmean([o.metric_value for o in participated])
        f1 = _nanmean([o.extra_metric for o in participated])
        mscore = metric_score_value("classification", acc)
        return loss, acc, mscore, f1

# -------------------- Regression --------------------

class RegressionStrategy(TaskStrategy):
    def task_type(self) -> str: return "regression"

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
            loss, metric_value, extra_metric = evaluate_model(local_model, self.x_test, self.y_test, task_type="regression")
            mscore = metric_score_value("regression", metric_value)

            if self.save_weights and weights is not None:
                with open(f"weights/{client_id}_round_{round_idx}.json", "w") as f:
                    json.dump({k: np.asarray(v).tolist() for k, v in weights.items()}, f, indent=4)

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=samples_count, duration=duration,
                loss=loss, metric_value=metric_value, metric_score=mscore, extra_metric=extra_metric,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=weights_size(weights),
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=weights, extras={},
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

# -------------------- Clustering --------------------

class ClusteringStrategy(TaskStrategy):
    def task_type(self) -> str: return "clustering"

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down) -> ClientOutcome:
        # local-only KMeans adapter path
        X = x
        t0 = time.time()
        tracker = ResourceTracker()
        tracker.start()
        try:
            local_model = self.build_model()  # returns KMeansAdapter
            local_model.fit(X)
            duration = time.time() - t0
            usage = tracker.stop(duration)

            try:
                loss, sil, inertia = local_model.evaluate(self.x_test)
            except Exception:
                loss, sil, inertia = (np.nan, np.nan, np.nan)
            ari = nmi = np.nan

            try:
                if self.y_test is not None:
                    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                    preds = local_model.predict(self.x_test)
                    if preds.shape[0] == self.y_test.shape[0]:
                        ari = float(adjusted_rand_score(self.y_test, preds))
                        nmi = float(normalized_mutual_info_score(self.y_test, preds))
            except Exception:
                pass

            try:
                comm_up = weights_size(local_model.get_weights())
            except Exception:
                comm_up = 0

            mscore = metric_score_value("clustering", sil)
            return ClientOutcome(
                participated=True, fail_reason="", samples_count=len(X), duration=duration,
                loss=np.nan, metric_value=sil, metric_score=mscore, extra_metric=inertia,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=comm_up,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=None,
                extras={
                    "silhouette": sil,
                    "inertia": inertia,
                    "ari": ari,
                    "nmi": nmi,
                    "clustering_k": getattr(local_model, "k", np.nan),
                    "clustering_agg": "local_only",
                },
            )
        except Exception:
            duration = time.time() - t0
            usage = tracker.stop(duration or 1e-9)
            return ClientOutcome(
                participated=False, fail_reason="error", samples_count=len(X), duration=duration,
                loss=np.nan, metric_value=np.nan, metric_score=np.nan, extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1, comm_down=comm_down, comm_up=0,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                payload=None, extras={},
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]

        if participated:
            sil = _nanmean([o.metric_value for o in participated])
            inertia = _nanmean([o.extra_metric for o in participated])
        else:
            try:
                tmp = self.build_model()   # fresh adapter with same knobs
                tmp.fit(x_train)           # fit on full training set
                _, sil, inertia = tmp.evaluate(x_test)
            except Exception:
                sil, inertia = (np.nan, np.nan)

        # Map silhouette to a score (just identity for clustering)
        mscore = metric_score_value("clustering", sil)

        # The ‘loss’ concept doesn’t exist for clustering, so use NaN
        loss = np.nan
        # Bundle inertia as auxiliary info
        global_extra = {"inertia": inertia}

        return loss, sil, mscore, global_extra

# -------------------- factory --------------------

def make_task_strategy(task_type: str, meta: dict, knobs: dict, config: dict, x_test, y_test, metric_key: str, save_weights: bool) -> TaskStrategy:
    if task_type == "classification":
        mt = (config.get("model_type") or "").lower()
        if mt in ("hf", "hf_text", "transformers"):
            return HFInferenceClassificationStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
        return ClassificationStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    if task_type == "regression":
        return RegressionStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    if task_type == "clustering":
        return ClusteringStrategy(meta, knobs, config, x_test, y_test, metric_key, save_weights)
    raise ValueError(f"Unknown task type: {task_type}")
