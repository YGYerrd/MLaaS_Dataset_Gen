# hf_strategy.py
import time
import math
import numpy as np

from .base import TaskStrategy, ClientOutcome, _nanmean, weights_size, metric_score_value, canonical_task_family
from ..system_metrics import ResourceTracker
from ...models.train_eval import aggregate_state_dict


class HFStrategy(TaskStrategy):
    """
    Single HF strategy that covers:
      - inference-only sequence classification (hf / transformers)
      - fine-tune sequence classification (hf_finetune / transformers_finetune)
      - fine-tune token classification (hf_task=token_classification)
    Behaviour is driven by config + dataset_args.
    """

    def task_type(self):
        base_task = str((self.meta or {}).get("task_type") or "").strip().lower()
        resolved = canonical_task_family(base_task, self.hf_task)
        return resolved if resolved != "unknown" else (base_task or "classification")
        
    def __init__(self, meta, knobs, config, x_test, y_test, metric_key, save_weights):
        super().__init__(meta, knobs, config, x_test, y_test, metric_key, save_weights)

        mt = (self.config.get("model_type") or "").lower()
        self.inference_only = mt in ("hf", "hf_text", "transformers")

        ds_args = self.config.get("dataset_args", {}) or {}
        self.hf_task = (ds_args.get("hf_task") or self.config.get("hf_task") or "sequence_classification").lower()
        self.runtime_adjustments = []
        self._apply_runtime_safety_overrides(ds_args)

    def _apply_runtime_safety_overrides(self, ds_args):
        task = str(self.hf_task or "").strip().lower().replace("-", "_")
        batch_cap_defaults = {
            "image_segmentation": ("max_segmentation_batch_size", 1),
            "semantic_segmentation": ("max_segmentation_batch_size", 1),
            "image_detection": ("max_detection_batch_size", 2),
            "object_detection": ("max_detection_batch_size", 2),
            "image_captioning": ("max_multimodal_batch_size", 2),
            "text_image_retrieval": ("max_multimodal_batch_size", 2),
            "visual_question_answering": ("max_multimodal_batch_size", 2),
        }
        if task in batch_cap_defaults:
            try:
                requested_batch_size = int(self.knobs.get("batch_size", 1) or 1)
            except Exception:
                requested_batch_size = 1
            cap_key, default_cap = batch_cap_defaults[task]
            max_safe_batch_size = int(
                self.config.get(
                    cap_key,
                    (ds_args or {}).get(cap_key, default_cap),
                )
                or default_cap
            )
            max_safe_batch_size = max(1, max_safe_batch_size)
            if requested_batch_size > max_safe_batch_size:
                self.knobs["requested_batch_size"] = requested_batch_size
                self.knobs["batch_size"] = max_safe_batch_size
                self.runtime_adjustments.append(
                    f"capped {task} batch_size {requested_batch_size}->{max_safe_batch_size}"
                )

        model_id = str((ds_args or {}).get("hf_model_id") or self.config.get("hf_model_id") or "").strip().lower()
        explicit_device = (ds_args or {}).get("device") or self.config.get("device")
        if task in {"image_detection", "object_detection"} and model_id.startswith("facebook/detr-") and not explicit_device:
            self.config = dict(self.config)
            copied_ds_args = dict(ds_args or {})
            copied_ds_args["device"] = "cpu"
            self.config["dataset_args"] = copied_ds_args
            self.config["device"] = "cpu"
            self.runtime_adjustments.append("forced facebook/detr object detection device=cpu")

    # -------------------------
    # Logging + scoring policies
    # -------------------------
    def _weighting_policy(self):
        task = str(self.hf_task or "").lower()
        token_weighted_tasks = {
            "token_classification",
            "token-cls",
            "ner",
            "fill_mask",
            "masked_lm",
            "mlm",
            "causal_lm_generation",
            "causal-lm",
            "language-modeling",
            "language_modeling",
            "seq2seq_generation",
            "image_captioning",
        }
        label_format = str((self.meta or {}).get("label_format") or "").strip().lower()
        if task == "visual_question_answering" and label_format == "vqa_token_index":
            return "supervised_token_count"
        sequence_weighted_tasks = {
            "sequence_classification",
            "text_classification",
            "sentence_similarity",
            "image_classification",
            "image_detection",
            "object_detection",
            "image_segmentation",
            "text_image_retrieval",
            "visual_question_answering",
        }
        if task in token_weighted_tasks:
            return "supervised_token_count"
        if task in sequence_weighted_tasks:
            return "sequence_count"

        accounting = (self.meta or {}).get("accounting") if isinstance(self.meta, dict) else {}
        if not isinstance(accounting, dict):
            accounting = {}
        if accounting.get("supervised_token_count"):
            return "supervised_token_count"
        return "sequence_count"

    def _resolve_client_weighting(self, samples_count, extras=None):
        extras = extras if isinstance(extras, dict) else {}

        def _intish(*keys):
            for key in keys:
                value = extras.get(key)
                if value is None:
                    continue
                try:
                    return int(value)
                except Exception:
                    continue
            return None

        sequence_count = _intish(
            "train_sequence_count",
            "eval_sequence_count",
            "sequence_count",
            "eval_samples",
            "train_samples",
        )
        if sequence_count is None:
            sequence_count = int(samples_count)

        supervised_token_count = _intish(
            "train_supervised_token_count",
            "eval_supervised_token_count",
            "supervised_token_count",
            "tokens_total",
            "train_loss_denominator_count",
        )

        weight_unit = self._weighting_policy()
        if weight_unit == "supervised_token_count":
            weight_value = supervised_token_count
        else:
            weight_unit = "sequence_count"
            weight_value = sequence_count

        if weight_value is None or weight_value <= 0:
            weight_unit = "sequence_count"
            weight_value = sequence_count if sequence_count and sequence_count > 0 else int(samples_count)

        return {
            "sequence_count": int(sequence_count) if sequence_count is not None else None,
            "supervised_token_count": int(supervised_token_count) if supervised_token_count is not None else None,
            "aggregation_weight_unit": weight_unit,
            "aggregation_weight_value": float(weight_value),
        }

    def _metric_score(self, primary_metric_value):
        if primary_metric_value != primary_metric_value:
            return np.nan

        if self.hf_task in ("causal_lm_generation", "causal-lm", "language-modeling", "language_modeling"):
            return metric_score_value("regression", float(primary_metric_value))

        # token classification primary is typically F1 already in [0,1]
        if self.hf_task in ("token_classification", "token-cls", "ner"):
            return float(primary_metric_value)

        if self.hf_task == "sentence_similarity" and self.task_type() == "regression":
            return metric_score_value("regression", float(primary_metric_value))

        # sequence classification primary is typically accuracy
        return metric_score_value("classification", float(primary_metric_value))

    def _effective_finetune_lr(self):
        requested_lr = self.knobs.get("learning_rate", 5e-5)
        try:
            requested = float(requested_lr)
        except Exception:
            requested = np.nan

        task = str(self.hf_task or "").lower()
        # Full-model transformer vision fine-tunes are unstable at the generic
        # tabular/CNN learning rates used elsewhere in the manifest generator.
        # Clamp object detection to a safer range so pretrained detectors do not
        # immediately collapse during one-round transfer runs.
        if task in {"image_detection", "object_detection"}:
            safe_default = 5e-5
            safe_max = 1e-4
            if not np.isfinite(requested) or requested <= 0:
                return safe_default, safe_default, True
            if requested > safe_max:
                return safe_max, requested, True
            return requested, requested, False

        return requested, requested, False

    def loggable_run_params(self):
        ds_args = self.config.get("dataset_args", {}) or {}
        task = str(self.hf_task or "").strip().lower().replace("-", "_")

        inferred_modality = ((self.meta or {}).get("modality") if isinstance(self.meta, dict) else None)
        inferred_modality = str(inferred_modality or "").strip().lower()
        if inferred_modality not in {"text", "image", "multimodal"}:
            image_tasks = {"image_classification", "image_detection", "image_segmentation"}
            multimodal_tasks = {"image_captioning", "text_image_retrieval", "visual_question_answering", "multimodal"}
            if task in image_tasks:
                inferred_modality = "image"
            elif task in multimodal_tasks:
                inferred_modality = "multimodal"
            else:
                inferred_modality = "text"

        uses_tokenization = inferred_modality in {"text", "multimodal"}

        hf_model_id = ds_args.get("hf_model_id") or self.config.get("hf_model_id")
        max_length  = ds_args.get("max_length") or self.config.get("max_length")
        device      = ds_args.get("device") or self.config.get("device")
        effective_lr, requested_lr, lr_adjusted = self._effective_finetune_lr()
        max_train_time_s = self.knobs.get("max_train_time_s", self.config.get("max_train_time_s", 60))

        adapter = {
            "inference_only": self.inference_only,
            "fine_tune": (not self.inference_only),
            "hf_task": self.hf_task,
            "hf_model_id": hf_model_id,
            "max_length": max_length,
            "device": device,
            "batch_size": self.knobs.get("batch_size"),
            "requested_batch_size": self.knobs.get("requested_batch_size"),
            "local_epochs": self.knobs.get("local_epochs"),
            "lr": effective_lr,
            "requested_lr": requested_lr if lr_adjusted else None,
            "learning_rate_adjusted": bool(lr_adjusted) if lr_adjusted else None,
            "max_new_tokens": ds_args.get("max_new_tokens") or self.config.get("max_new_tokens"),
            "num_beams": ds_args.get("num_beams") or self.config.get("num_beams"),
            "do_sample": ds_args.get("do_sample") if ds_args.get("do_sample") is not None else self.config.get("do_sample"),
            "temperature": ds_args.get("temperature") or self.config.get("temperature"),
            "top_k": ds_args.get("top_k") or self.config.get("top_k"),
            "top_p": ds_args.get("top_p") or self.config.get("top_p"),
            "length_penalty": ds_args.get("length_penalty") or self.config.get("length_penalty"),
            "max_train_time_s": max_train_time_s,
            "aggregation_weight_unit": self._weighting_policy(),
            "runtime_adjustments": self.runtime_adjustments or None,
        }
        if uses_tokenization:
            adapter["padding_mode"] = ("dynamic" if ds_args.get("dynamic_padding") else "max_length")

        meta_train_split = (self.meta or {}).get("train_split") if isinstance(self.meta, dict) else None
        meta_test_split = (self.meta or {}).get("test_split") if isinstance(self.meta, dict) else None
        requested_train_split = ds_args.get("train_split")
        requested_test_split = ds_args.get("test_split")
        resolved_train_split = meta_train_split or requested_train_split
        resolved_test_split = meta_test_split or requested_test_split

        dataset = {
            "dataset_name": ds_args.get("dataset_name"),
            "dataset_config": ds_args.get("dataset_config"),
            "train_split": resolved_train_split,
            "test_split": resolved_test_split,
            "requested_train_split": (
                requested_train_split
                if meta_train_split is not None and meta_train_split != requested_train_split
                else None
            ),
            "requested_test_split": (
                requested_test_split
                if meta_test_split is not None and meta_test_split != requested_test_split
                else None
            ),
            "max_samples": ds_args.get("max_samples"),
        }
        if inferred_modality in {"text", "multimodal"}:
            dataset["text_column"] = ds_args.get("text_column")
            dataset["tokens_column"] = ds_args.get("tokens_column")
        if inferred_modality in {"image", "multimodal"}:
            dataset["image_column"] = ds_args.get("image_column")
            dataset["boxes_column"] = ds_args.get("boxes_column")
            dataset["classes_column"] = ds_args.get("classes_column")
            dataset["mask_column"] = ds_args.get("mask_column")
        if inferred_modality in {"image", "multimodal"} or task in {"sequence_classification", "token_classification", "sentence_similarity"}:
            dataset["label_column"] = ds_args.get("label_column")
        if inferred_modality == "multimodal":
            dataset["missing_pair_handling"] = ds_args.get("missing_pair_handling")
            dataset["question_column"] = ds_args.get("question_column")
            dataset["answer_column"] = ds_args.get("answer_column")
            dataset["ranking_label_column"] = ds_args.get("ranking_label_column")
            dataset["vqa_label_mode"] = ds_args.get("vqa_label_mode")
            dataset["vqa_answer_vocab_size"] = ds_args.get("vqa_answer_vocab_size")
            dataset["vqa_unseen_answer_policy"] = ds_args.get("vqa_unseen_answer_policy")
            dataset["retrieval_positive_policy"] = ds_args.get("retrieval_positive_policy")
        if uses_tokenization:
            dataset["dynamic_padding"] = ds_args.get("dynamic_padding")
            dataset["padding_mode"] = ("dynamic" if ds_args.get("dynamic_padding") else "max_length")

        aggregator = {
            "strategy": "weighted_metric_average_no_weight_updates" if self.inference_only else "fedavg_weighted",
            "aggregation_weight_unit": self._weighting_policy(),
        }

        adapter = {k: v for k, v in adapter.items() if v is not None}
        dataset = {k: v for k, v in dataset.items() if v is not None}
        aggregator = {k: v for k, v in aggregator.items() if v is not None}
        return {"adapter": adapter, "dataset": dataset, "aggregator": aggregator}

    # -------------------------
    # Federation-safe metric stats
    # -------------------------

    def _collect_metric_stats(self, outcomes):
        stats = {}
        for o in outcomes:
            extras = getattr(o, "extras", {}) or {}
            if not isinstance(extras, dict):
                continue
            for k, v in extras.items():
                if not str(k).startswith("metric_stat_"):
                    continue
                key = str(k)[12:]
                try:
                    stats[key] = float(stats.get(key, 0.0)) + float(v)
                except Exception:
                    continue
        return stats

    def _metrics_from_stats(self, stats):
        if not stats:
            return None
        task = str(self.hf_task or "").lower()

        if task in {"image_classification"}:
            total = float(stats.get("total", 0.0))
            if total <= 0:
                return None
            top1 = float(stats.get("top1_correct", 0.0)) / total
            labels = set()
            for key in stats:
                key = str(key)
                for suffix in ("_tp", "_pred_total", "_target_total"):
                    if key.startswith("class_") and key.endswith(suffix):
                        labels.add(key[len("class_"):-len(suffix)])
            f1_scores = []
            for label in sorted(labels):
                tp = float(stats.get(f"class_{label}_tp", 0.0))
                pred_total = float(stats.get(f"class_{label}_pred_total", 0.0))
                target_total = float(stats.get(f"class_{label}_target_total", 0.0))
                if pred_total <= 0 and target_total <= 0:
                    continue
                precision = tp / pred_total if pred_total > 0 else 0.0
                recall = tp / target_total if target_total > 0 else 0.0
                f1_scores.append(0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall))
            macro_f1 = float(np.mean(f1_scores)) if f1_scores else np.nan
            return top1, macro_f1

        if task in {"image_detection", "object_detection"}:
            gt = float(stats.get("gt", 0.0))
            if gt <= 0:
                return None
            vals = []
            for thr in (0.5, 0.75, 0.95):
                tp = float(stats.get(f"tp_{thr}", 0.0))
                fp = float(stats.get(f"fp_{thr}", 0.0))
                vals.append(tp / max(gt + fp, 1e-9))
            return float(np.mean(vals)), float(vals[0])

        if task in {"image_segmentation", "semantic_segmentation"}:
            class_prefix = "class_"
            intersection_suffix = "_intersection"
            pred_total_suffix = "_pred_total"
            target_total_suffix = "_target_total"
            per_class_iou = []
            per_class_dice = []

            for key, value in stats.items():
                key = str(key)
                if not (key.startswith(class_prefix) and key.endswith(intersection_suffix)):
                    continue
                label_token = key[len(class_prefix):-len(intersection_suffix)]
                try:
                    intersection = float(value)
                    pred_total = float(stats.get(f"{class_prefix}{label_token}{pred_total_suffix}", 0.0))
                    target_total = float(stats.get(f"{class_prefix}{label_token}{target_total_suffix}", 0.0))
                except Exception:
                    continue

                union = pred_total + target_total - intersection
                denom = pred_total + target_total
                if union > 0:
                    per_class_iou.append(intersection / union)
                if denom > 0:
                    per_class_dice.append((2.0 * intersection) / max(denom, 1e-9))

            if per_class_iou and per_class_dice:
                return float(np.mean(per_class_iou)), float(np.mean(per_class_dice))

            inter = float(stats.get("intersection", 0.0))
            union = float(stats.get("union", 0.0))
            pred_total = float(stats.get("pred_total", 0.0))
            target_total = float(stats.get("target_total", 0.0))
            if union <= 0 or (pred_total + target_total) <= 0:
                return None
            iou = inter / union
            dice = (2.0 * inter) / max(pred_total + target_total, 1e-9)
            return iou, dice

        if task in {"text_image_retrieval", "image_text_retrieval"}:
            total = float(stats.get("total", 0.0))
            if total <= 0:
                return None
            r1 = float(stats.get("r1_correct", 0.0)) / total
            r5 = float(stats.get("r5_correct", 0.0)) / total
            return r1, r5

        return None

    @staticmethod
    def _finite_float(value):
        try:
            parsed = float(value)
        except Exception:
            return np.nan
        return parsed if np.isfinite(parsed) else np.nan

    def _coerce_image_classification_metrics(self, primary, secondary, extras):
        task = str(self.hf_task or "").lower()
        if task not in {"image_classification"}:
            return primary, secondary

        extras = extras if isinstance(extras, dict) else {}
        primary_val = self._finite_float(primary)
        secondary_val = self._finite_float(secondary)

        if primary_val != primary_val:
            for key in ("accuracy", "top1_accuracy"):
                candidate = self._finite_float(extras.get(key))
                if candidate == candidate:
                    primary_val = candidate
                    break

        if secondary_val != secondary_val:
            for key in ("f1", "macro_f1", "weighted_f1", "top5_accuracy"):
                candidate = self._finite_float(extras.get(key))
                if candidate == candidate:
                    secondary_val = candidate
                    break

        if secondary_val != secondary_val and primary_val == primary_val:
            secondary_val = primary_val

        return primary_val, secondary_val

    def _weighted_metric_from_outcome_extras(self, outcomes, weights, keys):
        pairs = []
        for outcome, weight in zip(outcomes, weights):
            if weight <= 0:
                continue
            extras = getattr(outcome, "extras", None)
            if not isinstance(extras, dict):
                continue
            for key in keys:
                candidate = self._finite_float(extras.get(key))
                if candidate == candidate:
                    pairs.append((candidate, float(weight)))
                    break
        if not pairs:
            return np.nan
        vals, wts = zip(*pairs)
        return float(np.average(vals, weights=wts))

    # -------------------------
    # Model/adapter management
    # -------------------------
    def comm_down_bytes(self, global_model):
        # In inference mode you currently treat comms as 0 (no payload exchange)
        if self.inference_only:
            return 0

        try:
            w = global_model.get_weights()
            return weights_size(w)
        except Exception:
            return 0

    def _get_client_adapter(self, client_id):
        local_adapter = getattr(self, "_client_adapters", {}).get(client_id)
        if local_adapter is None:
            if not hasattr(self, "_client_adapters"):
                self._client_adapters = {}
            local_adapter = self.build_model()
            self._client_adapters[client_id] = local_adapter
        return local_adapter

    # -------------------------
    # Train/eval logic
    # -------------------------
    def _train_eval(self, adapter, x_train, y_train):
        """
        Expected adapter API:
          - fit(x, y, epochs, lr) -> dict qos
          - evaluate(x, y) -> (loss, primary, secondary, qos)

        For token classification: primary is assumed F1, secondary assumed accuracy (or similar).
        """
        effective_lr, requested_lr, lr_adjusted = self._effective_finetune_lr()
        if lr_adjusted:
            print(
                "[HFStrategy] adjusted hf fine-tune learning rate "
                f"for task={self.hf_task}: requested={requested_lr} effective={effective_lr}"
            )
        train_qos = adapter.fit(
            x_train,
            y_train,
            epochs=self.knobs.get("local_epochs", 1),
            lr=effective_lr,
            max_train_time_s=self.knobs.get("max_train_time_s", self.config.get("max_train_time_s", 60)),
        )
        if isinstance(train_qos, dict):
            train_qos["requested_learning_rate"] = float(requested_lr) if requested_lr == requested_lr else np.nan
            train_qos["effective_learning_rate"] = float(effective_lr) if effective_lr == effective_lr else np.nan
            train_qos["learning_rate_adjusted"] = bool(lr_adjusted)
        loss, primary, secondary, eval_qos = adapter.evaluate(self.x_test, self.y_test)
        return loss, primary, secondary, train_qos, eval_qos

    def train_client(self, client_id, x, y, global_model, round_idx, rounds_so_far, comm_down):
        samples_count = len(y)
        if samples_count == 0:
            weighting = self._resolve_client_weighting(samples_count, {})
            return ClientOutcome(
                participated=False,
                fail_reason="Client received zero samples after preprocessing/partitioning",
                samples_count=samples_count,
                duration=0.0,
                loss=np.nan,
                metric_value=np.nan,
                metric_score=np.nan,
                extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1,
                comm_down=(0 if self.inference_only else comm_down),
                comm_up=0,
                cpu_time_s=0.0,
                cpu_utilization=0.0,
                memory_used_mb=0.0,
                memory_utilization=0.0,
                gpu_utilization=0.0,
                gpu_memory_utilization=0.0,
                gpu_memory_used_mb=0.0,
                peak_vram_mb=0.0,
                avg_vram_mb=0.0,
                peak_host_ram_mb=0.0,
                avg_host_ram_mb=0.0,
                payload=None,
                extras={},
                **weighting,
            )

        start = time.time()
        tracker = ResourceTracker()
        tracker.start()

        try:
            if self.inference_only:
                adapter = global_model if global_model is not None else self.build_model()
                loss, primary, secondary, qos = adapter.evaluate(
                    x,
                    y,
                    inference_only=True,
                    max_eval_time_s=self.knobs.get("max_eval_time_s", self.config.get("max_eval_time_s")),
                    progress_log_interval=self.knobs.get(
                        "eval_progress_log_interval",
                        self.config.get("eval_progress_log_interval", 10),
                    ),
                )
                if isinstance(qos, dict) and qos.get("label_space_warning"):
                    print(f"[HFStrategy] {qos.get('label_space_warning')}")
                primary, secondary = self._coerce_image_classification_metrics(primary, secondary, qos)

                duration = time.time() - start
                usage = tracker.stop(duration)

                mscore = self._metric_score(primary)
                weighting = self._resolve_client_weighting(samples_count, qos)
                extras = qos if isinstance(qos, dict) else {}
                extras = dict(extras)
                extras["client_partition_sample_count"] = int(samples_count)
                extras.update(
                    self.perturbation_metrics(
                        adapter,
                        client_id=client_id,
                        round_idx=round_idx,
                        x_eval=x,
                        y_eval=y,
                    )
                )

                return ClientOutcome(
                    participated=True,
                    fail_reason="",
                    samples_count=samples_count,
                    duration=duration,
                    loss=loss,
                    metric_value=float(primary) if primary == primary else np.nan,
                    metric_score=float(mscore) if mscore == mscore else np.nan,
                    extra_metric=float(secondary) if secondary == secondary else np.nan,
                    rounds_so_far=rounds_so_far,
                    comm_down=0,
                    comm_up=0,
                    cpu_time_s=usage.cpu_time_s,
                    cpu_utilization=usage.cpu_utilization,
                    memory_used_mb=usage.memory_used_mb,
                    memory_utilization=usage.memory_utilization,
                    gpu_utilization=usage.gpu_utilization,
                    gpu_memory_utilization=usage.gpu_memory_utilization,
                    gpu_memory_used_mb=usage.gpu_memory_used_mb,
                    peak_vram_mb=usage.peak_vram_mb,
                    avg_vram_mb=usage.avg_vram_mb,
                    peak_host_ram_mb=usage.peak_host_ram_mb,
                    avg_host_ram_mb=usage.avg_host_ram_mb,
                    payload=None,
                    extras=extras,
                    **weighting,
                )

            # fine-tune mode
            local_adapter = self._get_client_adapter(client_id)

            if global_model is not None:
                local_adapter.set_weights(global_model.get_weights())

            loss, primary, secondary, train_qos, eval_qos = self._train_eval(local_adapter, x, y)

            duration = time.time() - start
            usage = tracker.stop(duration)

            payload = local_adapter.get_weights()
            mscore = self._metric_score(primary)

            extras = {}
            if isinstance(train_qos, dict):
                extras.update(train_qos)
            if isinstance(eval_qos, dict):
                extras.update(eval_qos)
            primary, secondary = self._coerce_image_classification_metrics(primary, secondary, extras)
            extras.update(self.perturbation_metrics(local_adapter, client_id=client_id, round_idx=round_idx))
            weighting = self._resolve_client_weighting(samples_count, extras)

            return ClientOutcome(
                participated=True,
                fail_reason="",
                samples_count=samples_count,
                duration=duration,
                loss=loss,
                metric_value=float(primary) if primary == primary else np.nan,
                metric_score=float(mscore) if mscore == mscore else np.nan,
                extra_metric=float(secondary) if secondary == secondary else np.nan,
                rounds_so_far=rounds_so_far,
                comm_down=comm_down,
                comm_up=weights_size(payload),
                cpu_time_s=usage.cpu_time_s,
                cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb,
                memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization,
                gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=payload,
                extras=extras,
                **weighting,
            )

        except Exception as e:
            duration = time.time() - start
            usage = tracker.stop(duration or 1e-9)

            weighting = self._resolve_client_weighting(samples_count, {})

            return ClientOutcome(
                participated=False,
                fail_reason=repr(e),
                samples_count=samples_count,
                duration=duration,
                loss=np.nan,
                metric_value=np.nan,
                metric_score=np.nan,
                extra_metric=np.nan,
                rounds_so_far=rounds_so_far - 1,
                comm_down=(0 if self.inference_only else comm_down),
                comm_up=0,
                cpu_time_s=usage.cpu_time_s,
                cpu_utilization=usage.cpu_utilization,
                memory_used_mb=usage.memory_used_mb,
                memory_utilization=usage.memory_utilization,
                gpu_utilization=usage.gpu_utilization,
                gpu_memory_utilization=usage.gpu_memory_utilization,
                gpu_memory_used_mb=usage.gpu_memory_used_mb,
                peak_vram_mb=usage.peak_vram_mb,
                avg_vram_mb=usage.avg_vram_mb,
                peak_host_ram_mb=usage.peak_host_ram_mb,
                avg_host_ram_mb=usage.avg_host_ram_mb,
                payload=None,
                extras={},
                **weighting,
            )

    def aggregate_and_eval(self, global_model, client_payloads, client_outcomes, round_idx, x_train, x_test, y_test):
        participated = [o for o in (client_outcomes or []) if getattr(o, "participated", False)]
        if not participated:
            return np.nan, np.nan, np.nan, np.nan

        if self.inference_only:
            weights = []
            for o in participated:
                value = getattr(o, "aggregation_weight_value", None)
                if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                    value = getattr(o, "sequence_count", None) or getattr(o, "samples_count", 1)
                weights.append(float(value))

            def _weighted(values, ws):
                pairs = [(float(v), float(w)) for v, w in zip(values, ws) if v == v and w > 0]
                if not pairs:
                    return np.nan
                vals, wts = zip(*pairs)
                return float(np.average(vals, weights=wts))

            loss = _weighted([o.loss for o in participated], weights)
            stats = self._collect_metric_stats(participated)
            derived = self._metrics_from_stats(stats)
            if derived is not None:
                primary, secondary = derived
            else:
                primary = _weighted([o.metric_value for o in participated], weights)
                secondary = _weighted([o.extra_metric for o in participated], weights)
                if str(self.hf_task or "").lower() in {"image_classification"}:
                    if primary != primary:
                        primary = self._weighted_metric_from_outcome_extras(
                            participated,
                            weights,
                            keys=("accuracy", "top1_accuracy"),
                        )
                    if secondary != secondary:
                        secondary = self._weighted_metric_from_outcome_extras(
                            participated,
                            weights,
                            keys=("f1", "macro_f1", "weighted_f1", "top5_accuracy"),
                        )
                    if secondary != secondary and primary == primary:
                        secondary = primary
            mscore = self._metric_score(primary)
            return loss, primary, mscore, secondary

        adapter = global_model if global_model is not None else self.build_model()

        payloads = [o.payload for o in participated if o.payload is not None]
        weights = []
        for o in participated:
            if o.payload is None:
                continue
            value = getattr(o, "aggregation_weight_value", None)
            if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                value = getattr(o, "sequence_count", None) or getattr(o, "samples_count", 1)
            weights.append(float(value))

        if payloads:
            agg = aggregate_state_dict(payloads, weights=weights)
            adapter.set_weights(agg)

        loss, primary, secondary, _qos = adapter.evaluate(x_test, y_test)
        mscore = self._metric_score(primary)
        return loss, primary, mscore, secondary
