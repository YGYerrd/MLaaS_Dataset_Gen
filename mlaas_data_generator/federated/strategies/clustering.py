from .base import TaskStrategy, ClientOutcome, metric_score_value, _nanmean, weights_size
import time
import numpy as np
from ..system_metrics import ResourceTracker
from ..model_params import extract_model_parameters
from .keras_strategy import _generic_runtime_metrics, _merge_runtime_metrics

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

            eval_start = time.time()
            try:
                loss, sil, inertia = local_model.evaluate(self.x_test)
            except Exception:
                loss, sil, inertia = (np.nan, np.nan, np.nan)
            eval_latency_s = time.time() - eval_start
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
            model_params = extract_model_parameters(
                model=local_model,
                config=self.config,
                metadata={"client_id": client_id, "round": round_idx},
            )

            mscore = metric_score_value("clustering", sil)
            extras = _merge_runtime_metrics(
                {
                    "silhouette": sil,
                    "inertia": inertia,
                    "ari": ari,
                    "nmi": nmi,
                    "clustering_k": getattr(local_model, "k", np.nan),
                    "clustering_agg": "local_only",
                },
                _generic_runtime_metrics(
                    local_model,
                    eval_latency_s=eval_latency_s,
                    metric_score=mscore,
                    loss=loss,
                    include_trust_metrics=self.should_run_perturbation_metrics(round_idx),
                ),
            )

            return ClientOutcome(
                participated=True, fail_reason="", samples_count=len(X), duration=duration,
                loss=np.nan, metric_value=sil, metric_score=mscore, extra_metric=inertia,
                rounds_so_far=rounds_so_far, comm_down=comm_down, comm_up=comm_up,
                cpu_time_s=usage.cpu_time_s, cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb, memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization, gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb, peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None,
                extras=extras,
                model_params=model_params,
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
                gpu_memory_used_mb=usage.gpu_memory_used_mb,peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None, extras={},
            )
    
    def loggable_run_params(self):
        adapter = {}
        for key in ("clustering_k","clustering_init","clustering_n_init","clustering_max_iter","clustering_tol","random_state","seed"):
            if key in self.config and self.config[key] is not None:
                adapter[key] = self.config[key]

        return {
            "adapter": adapter,
            "aggregator": {
                "strategy": "local_only_metric_average_no_weight_updates",
                "aggregation_weight_unit": "client_uniform",
            },
        }
    
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
