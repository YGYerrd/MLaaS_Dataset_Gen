"""Bounded-memory task evaluation on one reconstructed model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

from .contracts import TaskContract, TaskContractError, target_fingerprints


class ConfusionAccumulator:
    """Corpus statistics: batch boundaries do not change F1/IoU/Dice."""
    def __init__(self):
        self.counts = {}

    def update(self, truth, predictions, *, ignore_index=-100):
        truth, predictions = np.asarray(truth), np.asarray(predictions)
        if truth.shape != predictions.shape:
            raise TaskContractError("prediction_alignment_mismatch", "prediction and target shapes differ")
        mask = truth != ignore_index
        yt, yp = truth[mask], predictions[mask]
        if not np.isfinite(yt).all() or not np.isfinite(yp).all() or np.any(yt < 0) or np.any(yp < 0):
            raise TaskContractError("invalid_predictions", "class IDs must be finite and nonnegative")
        if np.any(yt != np.floor(yt)) or np.any(yp != np.floor(yp)):
            raise TaskContractError("invalid_predictions", "class IDs must be integers")
        if yt.size:
            pairs, counts = np.unique(np.stack((yt, yp), axis=1), axis=0, return_counts=True)
            for pair, count in zip(pairs, counts):
                key = tuple(map(int, pair))
                self.counts[key] = self.counts.get(key, 0) + int(count)

    def finalise(self):
        total = sum(self.counts.values())
        if not total:
            raise TaskContractError("insufficient_audit_support", "no valid evaluation positions")
        labels = sorted({v for pair in self.counts for v in pair})
        per_class = {}
        for label in labels:
            tp = self.counts.get((label, label), 0)
            support = sum(n for (t, _), n in self.counts.items() if t == label)
            predicted = sum(n for (_, p), n in self.counts.items() if p == label)
            per_class[str(label)] = {"support": support, "predicted": predicted,
                "f1": 2 * tp / (support + predicted), "iou": tp / (support + predicted - tp),
                "recall": tp / support if support else None}
        return {"accuracy": sum(n for (t, p), n in self.counts.items() if t == p) / total,
                "f1": sum(v["f1"] * v["support"] for v in per_class.values()) / total,
                "iou": float(np.mean([v["iou"] for v in per_class.values()])),
                "dice": float(np.mean([v["f1"] for v in per_class.values()])),
                "supervision_count": total, "per_class": per_class}


@dataclass(frozen=True)
class TaskEvaluation:
    source_ids: tuple[str, ...]
    target_fingerprints: tuple[dict, ...]
    contract: dict
    metrics: dict
    inference_latency_s: float
    predictions: tuple = ()
    targets: tuple = ()

    def validated_metrics(self, *, source_ids, target_labels, task_family):
        contract = TaskContract.from_dict(self.contract)
        if tuple(source_ids) != self.source_ids or list(target_labels) != list(self.target_fingerprints):
            raise TaskContractError("prediction_alignment_mismatch", "evaluation IDs or target hashes differ from the audit")
        if contract.family != task_family:
            raise TaskContractError("target_semantics_mismatch", "evaluation task differs from selected task")
        for metric in (contract.evaluation["primary"], contract.evaluation["secondary"]):
            if not np.isfinite(self.metrics.get(metric, np.nan)):
                raise TaskContractError("invalid_metric", f"required metric {metric} is not finite")
        if self.metrics.get("supervision_count", 0) < 1:
            raise TaskContractError("insufficient_audit_support", "no usable supervision")
        return dict(self.metrics)


def evaluate_task_model(model, x, y, contract: TaskContract, source_ids, *, collect=False,
                        max_eval_time_s=None, progress_seconds=10, prediction_sink=None):
    core = getattr(model, "core", None)
    if core is None or contract.family not in {"token_classification", "segmentation", "generation", "fill_mask"}:
        raise TaskContractError("validation_not_implemented", "structured task evaluation requires its HF backend")
    torch = core.torch
    from ..models.adapters.microbatch import evaluation_batches
    state = {"eval_batch_size_requested": int(core.batch_size),
             "eval_batch_size_effective": int(core.batch_size), "eval_oom_retry_count": 0,
             "eval_memory_fallback_applied": False}
    core.evaluation_memory_metadata = state
    core.model.eval()
    ignore = contract.semantics["ignore_index"]
    contract.supervision_count(y)
    fingerprint = target_fingerprints(y)
    if len(source_ids) != len(fingerprint) or len(set(source_ids)) != len(source_ids):
        raise TaskContractError("prediction_alignment_mismatch", "unique source IDs must align to targets")
    accumulator = ConfusionAccumulator()
    count = supervision = generated_tokens = input_tokens = empty = 0
    loss_total = latency = rouge1 = rouge2 = repetition = 0.0
    predictions, targets = [], []
    start = last = time.perf_counter()
    prediction_digest = hashlib.sha256()
    from ..shared_parent import prepared_data_hash

    def record_prediction(position, prediction, target):
        # Ignore padding/void predictions; batch padding must not affect identity.
        meaningful = prediction if contract.family == "generation" else prediction[target != ignore]
        prediction_digest.update(prepared_data_hash((str(source_ids[position]), meaningful)).encode("ascii"))
        if prediction_sink:
            prediction_sink(position, prediction, target)

    def sync():
        if str(core.device).startswith("cuda"):
            torch.cuda.synchronize(core.device)

    def evaluate_batch(xb, yb):
        accumulator = ConfusionAccumulator()
        count = supervision = generated_tokens = input_tokens = empty = 0
        loss_total = latency = rouge1 = rouge2 = repetition = 0.0
        records = []
        def record_prediction(position, prediction, target):
            records.append((prediction, target))
        if max_eval_time_s is not None and time.perf_counter() - start > max_eval_time_s:
            raise TimeoutError("task evaluation exceeded the complete-partition time budget")
        enc, labels, extra = core.task_spec.encode_batch(core.tokenizer, xb, yb, core.max_length,
            torch, core.device, ignore_index=ignore, inference_only=False)
        if labels is None:
            raise TaskContractError("insufficient_supervision", "batch has no labels")
        batch_count = len(labels)
        if not bool((labels != ignore).any()):
            raise TaskContractError("insufficient_supervision", "batch contains only ignored targets")
        sync()
        before = time.perf_counter()
        if contract.family == "generation":
            # Only source inputs reach generate; reference labels are used in a separate forward.
            with core._generation_inference_mode():
                generated = core._run_with_precision_context(lambda: core.task_spec.generate_predictions(
                    core.model, enc, core.tokenizer, torch, contract.evaluation["decoding"]))
            if generated is None or getattr(generated, "ndim", 0) != 2 or generated.shape[0] != batch_count or generated.shape[1] == 0:
                raise TaskContractError("invalid_predictions", "generation must return a nonempty token sequence, including EOS, for every source")
            sync()
            latency += time.perf_counter() - before
            from ..models.adapters.hf_task import _decode_token_id_batch, _rouge_from_texts, _text_metric_tokens
            texts = _decode_token_id_batch(core.tokenizer, generated.detach().cpu().numpy(), ignore_index=ignore)
            refs = _decode_token_id_batch(core.tokenizer, labels.detach().cpu().numpy(), ignore_index=ignore)
            if texts is None or refs is None or len(texts) != batch_count:
                raise TaskContractError("invalid_predictions", "decoder did not return one text per source")
            r1, r2, _ = _rouge_from_texts(texts, refs)
            rouge1 += r1 * batch_count
            rouge2 += r2 * batch_count
            for text in texts:
                words = _text_metric_tokens(text)
                empty += int(not words)
                repetition += 1 - len(set(words)) / len(words) if words else 0
            generated_tokens += int((generated != core.tokenizer.pad_token_id).sum().item())
            input_tokens += int(enc["attention_mask"].sum().item())
            for offset, (prediction, target) in enumerate(zip(texts, refs)):
                record_prediction(count + offset, prediction, target)
            teacher_inputs = core.task_spec.build_forward_inputs(enc, labels_t=labels, inference_only=False)
            outputs = core._run_with_precision_context(lambda: core.model(**teacher_inputs))
            logits = core._extract_logits(outputs)
            if logits is None or logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(labels.shape):
                raise TaskContractError("prediction_alignment_mismatch", "teacher-forced logits do not align to all target positions")
        else:
            inputs = core.task_spec.build_forward_inputs(enc, labels_t=None, inference_only=True)
            outputs = core._run_with_precision_context(lambda: core.model(**inputs))
            logits = core._extract_logits(outputs)
            sync()
            latency += time.perf_counter() - before
            if contract.family == "segmentation":
                labels = core.task_spec._resize_labels_to_logits(torch, labels, logits.shape[-2:])
                predicted = logits.argmax(dim=1)
            else:
                predicted = logits.argmax(dim=-1)
            truth = labels.detach().cpu().numpy()
            pred = predicted.detach().cpu().numpy()
            accumulator.update(truth, pred, ignore_index=ignore)
            for offset, (prediction, target) in enumerate(zip(pred, truth)):
                record_prediction(count + offset, prediction, target)
        if not bool(torch.isfinite(logits).all()):
            raise TaskContractError("invalid_predictions", "model logits are not finite")
        valid = int((labels != ignore).sum().item())
        if not valid:
            raise TaskContractError("insufficient_supervision", "no targets remain after spatial/token transforms")
        loss = core.task_spec.extract_loss(torch, outputs, logits, labels, extra)
        if loss is None or not bool(torch.isfinite(loss)):
            raise TaskContractError("invalid_metric", "evaluation loss is missing or non-finite")
        loss_total += float(loss.item()) * valid
        supervision += valid
        count += batch_count
        return {"counts": accumulator.counts, "count": count, "supervision": supervision,
                "generated_tokens": generated_tokens, "input_tokens": input_tokens, "empty": empty,
                "loss_total": loss_total, "latency": latency, "rouge1": rouge1, "rouge2": rouge2,
                "repetition": repetition, "records": records}

    with torch.no_grad():
        for batch in evaluation_batches(core, core._batch_iter(x, y), state, evaluate_batch):
            for pair, amount in batch["counts"].items():
                accumulator.counts[pair] = accumulator.counts.get(pair, 0) + amount
            for offset, (prediction, target) in enumerate(batch["records"]):
                record_prediction(count + offset, prediction, target)
                if collect:
                    predictions.append(prediction)
                    targets.append(target)
            count += batch["count"]
            supervision += batch["supervision"]
            generated_tokens += batch["generated_tokens"]
            input_tokens += batch["input_tokens"]
            empty += batch["empty"]
            loss_total += batch["loss_total"]
            latency += batch["latency"]
            rouge1 += batch["rouge1"]
            rouge2 += batch["rouge2"]
            repetition += batch["repetition"]
            if time.perf_counter() - last >= progress_seconds:
                print(f"[task evaluation] {contract.task}: {count}/{len(source_ids)} examples", flush=True)
                last = time.perf_counter()
    if count != len(source_ids) or not supervision:
        raise TaskContractError("prediction_alignment_mismatch", "model did not evaluate the complete partition")
    if contract.family == "generation":
        metrics = {"rouge1": rouge1 / count, "rouge2": rouge2 / count,
                   "empty_output_fraction": empty / count, "repetition_fraction": repetition / count,
                   "input_tokens": input_tokens, "generated_tokens": generated_tokens}
    else:
        metrics = accumulator.finalise()
        if contract.family == "fill_mask":
            metrics.update(masked_accuracy=metrics["accuracy"],
                           perplexity_proxy=float(np.exp(np.clip(loss_total / supervision, -50, 50))))
    metrics.update(loss=loss_total / supervision, example_count=count, supervision_count=supervision,
                   prediction_hash=prediction_digest.hexdigest(),
                   inference_latency_s=latency, evaluation_wall_s=time.perf_counter() - start,
                   quality_score=metrics[contract.evaluation["primary"]],
                   primary_metric_name=contract.evaluation["primary"], secondary_metric_name=contract.evaluation["secondary"],
                   task_semantics_hash=contract.semantics_hash, evaluation_protocol_hash=contract.protocol_hash)
    metrics.update(state)
    return TaskEvaluation(tuple(source_ids), tuple(fingerprint), contract.as_dict(), metrics, latency,
                          tuple(predictions), tuple(targets))
