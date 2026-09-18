"""Task output contracts and recorded operating perturbations for evaluation v2.

These adapters preserve complete outputs. Unsupported contracts are explicit;
neither missing distributions nor a singleton retrieval pool imply stability.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class UnsupportedTaskOutput(ValueError):
    """The model or dataset lacks the contract needed for this measurement."""


@dataclass
class TaskOutput:
    task_family: str
    prediction: Any
    scores: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PerturbedInput:
    sample: Any
    operator: str
    strength: float
    changed: bool
    status: str = "complete"
    reason: str | None = None
    regime: str = "operating"
    metadata: dict = field(default_factory=dict)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if hasattr(value, "float") and str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _settings(config=None, meta=None):
    return {**(meta or {}), **(config or {})}


def _family(value):
    value = str(value or "classification")
    return {"tabular_regression": "regression", "image_classification": "classification",
            "text_classification": "classification", "image_segmentation": "segmentation",
            "image_detection": "detection", "seq2seq_generation": "generation",
            "causal_lm_generation": "generation", "text_image_retrieval": "retrieval"}.get(value, value)


def _probabilities(values, *, logits=False):
    values = _array(values).astype(np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite_or_empty_model_scores")
    if not logits and np.all((values >= 0) & (values <= 1)) and np.allclose(values.sum(axis=-1), 1, atol=1e-5):
        return values
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def output_from_raw(raw, *, task_family, x_sample=None, config=None, meta=None, batched=True):
    """Adapt a single output (a leading singleton batch is removed if requested)."""
    family = _family(task_family)
    cfg = _settings(config, meta)
    metadata = {"output_adapter": f"{family}_v2"}
    ignored = set(int(value) for value in cfg.get("evaluation_ignore_label_ids", [-100]))
    ignored.update(int(cfg[key]) for key in ("ignore_index", "label_pad_value") if cfg.get(key) is not None)
    metadata["ignore_label_ids"] = sorted(ignored)
    if family == "detection":
        if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], dict):
            raw = raw[0]
        if not isinstance(raw, dict) or "boxes" not in raw or not any(k in raw for k in ("labels", "classes")):
            raise UnsupportedTaskOutput("detection_requires_boxes_labels_scores")
        boxes = _array(raw["boxes"]).astype(float).reshape(-1, 4)
        labels = _array(raw.get("labels", raw.get("classes"))).reshape(-1)
        scores = _array(raw.get("scores", np.ones(len(labels)))).astype(float).reshape(-1)
        if len(boxes) != len(labels) or len(scores) != len(labels) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("invalid_detection_outputs")
        box_format = raw.get("box_format", cfg.get("detection_box_format", "xyxy"))
        if box_format == "cxcywh":
            center, size = boxes[:, :2].copy(), boxes[:, 2:].copy()
            boxes = np.concatenate((center - size / 2, center + size / 2), axis=-1)
        elif box_format != "xyxy":
            raise UnsupportedTaskOutput("detection_box_format_must_be_xyxy_or_cxcywh")
        image_size = raw.get("image_size", cfg.get("detection_image_size"))
        normalized = raw.get("normalized", cfg.get("detection_boxes_normalized", True))
        if not normalized:
            if image_size is None:
                raise UnsupportedTaskOutput("pixel_detection_boxes_require_image_size")
            h, w = image_size
            boxes /= np.asarray([w, h, w, h], dtype=float)
        if np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError("detection_box_negative_extent")
        threshold = float(cfg.get("evaluation_detection_score_threshold", 0.05))
        keep = scores >= threshold
        metadata.update(box_format="xyxy", boxes_normalized=True, score_threshold=threshold,
                        match_iou_threshold=float(cfg.get("evaluation_detection_match_iou", 0.5)))
        return TaskOutput(family, {"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]}, metadata=metadata)
    if family == "generation":
        if isinstance(raw, str):
            prediction = tuple(raw.split())
            metadata["sequence_kind"] = "words"
        else:
            arr = _array(raw)
            if batched and arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 0 and arr.dtype.kind in "USO":
                prediction = tuple(str(arr.item()).split())
                metadata["sequence_kind"] = "words"
            elif arr.ndim == 1 and arr.dtype.kind in "iu":
                ignored = set(cfg.get("generation_special_token_ids", []))
                prediction = tuple(int(v) for v in arr if int(v) >= 0 and int(v) not in ignored)
                metadata.update(sequence_kind="token_ids", ignored_token_ids=sorted(ignored))
            elif arr.ndim == 1 and arr.dtype.kind in "US":
                prediction = tuple(str(value) for value in arr)
                metadata["sequence_kind"] = "words"
            else:
                raise UnsupportedTaskOutput("generation_requires_generated_text_or_token_ids")
        return TaskOutput(family, prediction, metadata=metadata)
    arr = _array(raw)
    if batched and arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    if not arr.size or (arr.dtype.kind in "fciu" and not np.isfinite(arr).all()):
        raise ValueError("nonfinite_or_empty_model_output")
    if family == "regression":
        prediction = arr.astype(float)
        metadata["target_scale"] = cfg.get("regression_target_scale", cfg.get("evaluation_regression_target_scale"))
        return TaskOutput(family, prediction, metadata=metadata)
    if family == "classification":
        if cfg.get("label_format") in {"multilabel", "multihot"}:
            raise UnsupportedTaskOutput("multilabel_requires_independent_label_probability_adapter")
        if arr.size == 1:
            value = float(arr.reshape(-1)[0])
            if arr.dtype.kind in "iu" and not cfg.get("evaluation_scores_are_probabilities", False):
                raise UnsupportedTaskOutput("classification_requires_probability_distribution")
            probability = value if 0 <= value <= 1 and not cfg.get("evaluation_scores_are_logits") else 1 / (1 + math.exp(-max(-700, min(700, value))))
            probs = np.asarray([1 - probability, probability])
        elif arr.ndim == 1:
            if cfg.get("evaluation_scores_are_probabilities") and (np.any(arr < 0) or np.any(arr > 1) or not np.isclose(arr.sum(), 1, atol=1e-5)):
                raise ValueError("invalid_classification_probability_distribution")
            probs = _probabilities(arr, logits=bool(cfg.get("evaluation_scores_are_logits", False)))
        else:
            raise UnsupportedTaskOutput("classification_output_must_be_one_distribution")
        classes = cfg.get("evaluation_class_labels", list(range(len(probs))))
        if len(classes) != len(probs):
            raise ValueError("classification_class_labels_mismatch")
        metadata["class_labels"] = list(classes)
        return TaskOutput(family, classes[int(np.argmax(probs))], probs, metadata=metadata)
    if family in {"token_classification", "fill_mask"}:
        if arr.ndim == 2:
            prediction, probs = arr.argmax(axis=-1), None
        elif arr.ndim == 1 and arr.dtype.kind in "iu":
            prediction, probs = arr.astype(int), None
        else:
            raise UnsupportedTaskOutput("token_output_requires_sequence_or_sequence_class_scores")
        valid = np.ones(prediction.shape, dtype=bool)
        if isinstance(x_sample, dict):
            for key, invert in (("attention_mask", False), ("special_tokens_mask", True)):
                if key in x_sample:
                    mask = _array(x_sample[key]).reshape(-1).astype(bool)
                    if mask.shape != valid.shape:
                        raise UnsupportedTaskOutput("token_input_output_alignment_missing")
                    valid &= ~mask if invert else mask
            if family == "fill_mask":
                token_id = cfg.get("evaluation_mask_token_id")
                if token_id is None:
                    raise UnsupportedTaskOutput("fill_mask_requires_mask_token_id")
                valid &= _array(x_sample["input_ids"]).reshape(-1) == int(token_id)
        elif family == "fill_mask":
            raise UnsupportedTaskOutput("fill_mask_requires_encoded_input_alignment")
        if not valid.any():
            raise UnsupportedTaskOutput("no_valid_output_tokens")
        if arr.ndim == 2:
            score_values = arr[valid] if family == "fill_mask" else arr
            probs = _probabilities(score_values, logits=bool(cfg.get("evaluation_scores_are_logits", False)))
            if family == "fill_mask":
                # Vocabulary distributions are needed only for actual masked
                # queries. Keep full predictions, but avoid L x vocab buffers.
                metadata["score_positions"] = np.flatnonzero(valid).tolist()
        return TaskOutput(family, prediction, probs, valid, metadata)
    if family == "segmentation":
        if arr.ndim == 3:
            class_axis = int(cfg.get("evaluation_segmentation_class_axis", -1))
            probs = _probabilities(np.moveaxis(arr, class_axis, -1), logits=bool(cfg.get("evaluation_scores_are_logits", False)))
            prediction = probs.argmax(axis=-1)
        elif arr.ndim == 2 and arr.dtype.kind in "iub":
            prediction, probs = arr.astype(int), None
        else:
            raise UnsupportedTaskOutput("segmentation_requires_complete_mask_or_dense_scores")
        return TaskOutput(family, prediction, probs, np.ones(prediction.shape, dtype=bool), metadata)
    if family == "retrieval":
        candidates = cfg.get("evaluation_retrieval_candidate_ids")
        if arr.ndim != 1 or len(arr) < 2 or candidates is None or len(candidates) != len(arr):
            raise UnsupportedTaskOutput("retrieval_requires_frozen_candidate_pool_with_at_least_two_items")
        if len(set(candidates)) != len(candidates):
            raise UnsupportedTaskOutput("retrieval_candidate_ids_must_be_unique")
        metadata["candidate_ids"] = list(candidates)
        return TaskOutput(family, tuple(candidates[int(i)] for i in np.argsort(-arr, kind="stable")), arr.astype(float), metadata=metadata)
    raise UnsupportedTaskOutput(f"no_output_adapter_for_{family}")


def _batch(samples):
    first = samples[0]
    if isinstance(first, dict):
        return {key: _batch([sample[key] for sample in samples]) for key in first}
    if isinstance(first, str):
        return list(samples)
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
        return list(samples)
    return np.stack([_array(sample) for sample in samples])


def predict_task_output(model, x_sample, y_sample=None, *, task_family, hf_task=None, config=None, meta=None):
    x_sample = prepare_task_input(x_sample, y_sample, task_family=task_family, hf_task=hf_task, model=model)
    return predict_task_outputs(model, [x_sample], task_family=task_family, hf_task=hf_task, config=config, meta=meta)[0]


def prepare_task_input(x_sample, y_sample=None, *, task_family=None, hf_task=None, model=None):
    """Remove causal target tokens using only the structural supervision mask.

    Some cached causal rows concatenate prompt and target. All retained token
    positions are reconstructed before output/explanation evaluation, and no
    target values are supplied to the model. Length is retained for batching.
    """
    from .perturbation import _copy_sample

    spec_name = str(getattr(getattr(getattr(model, "core", None), "task_spec", None), "name", ""))
    causal = str(hf_task or "") == "causal_lm_generation" or spec_name == "causal_lm_generation"
    if not causal or not isinstance(x_sample, dict) or "input_ids" not in x_sample or y_sample is None:
        return x_sample
    ids = _array(x_sample["input_ids"])
    labels = _array(y_sample)
    if labels.shape != ids.shape:
        raise UnsupportedTaskOutput("causal_prompt_supervision_alignment_missing")
    attention = _array(x_sample.get("attention_mask", np.ones(ids.shape))).astype(bool)
    ignore = int(getattr(getattr(model, "core", None), "label_pad_value", -100))
    positions = (attention & (labels == ignore)).reshape(-1)
    if not positions.any():
        raise UnsupportedTaskOutput("causal_prompt_boundary_missing")
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = getattr(tokenizer, "eos_token_id", None) if pad is None else pad
    if pad is None:
        raise UnsupportedTaskOutput("causal_prompt_padding_token_missing")
    out = _copy_sample(x_sample)
    for key in ("input_ids", "attention_mask", "token_type_ids", "special_tokens_mask"):
        if key not in out:
            continue
        values = _array(out[key])
        if values.shape != ids.shape:
            raise UnsupportedTaskOutput("causal_prompt_input_alignment_missing")
        packed = np.full(values.size, int(pad) if key == "input_ids" else 0, dtype=values.dtype)
        packed[:int(positions.sum())] = values.reshape(-1)[positions]
        out[key] = packed.reshape(values.shape)
    if "attention_mask" not in out:
        out["attention_mask"] = np.r_[np.ones(int(positions.sum()), dtype=int), np.zeros(ids.size - int(positions.sum()), dtype=int)].reshape(ids.shape)
    return out


def predict_task_outputs(model, samples, *, task_family, hf_task=None, config=None, meta=None):
    """Batch compatible inputs; exceptions are left visible for evidence tracking."""
    samples = list(samples)
    if not samples:
        return []
    cfg = _settings(config, meta)
    family = _family(task_family)
    core = getattr(model, "core", None)
    if core is not None:
        return _predict_hf_outputs(core, samples, family, hf_task, cfg)
    method = getattr(model, "predict_proba", None) if family == "classification" else None
    if callable(method):
        raw = method(_batch(samples))
        cfg["evaluation_scores_are_probabilities"] = True
        if hasattr(model, "classes_"):
            cfg["evaluation_class_labels"] = _array(model.classes_).tolist()
    else:
        method = getattr(model, "generate", None) if family == "generation" else None
        method = method if callable(method) else getattr(model, "predict", None)
        if not callable(method):
            raise UnsupportedTaskOutput("model_has_no_task_prediction_method")
        if family == "generation" and method == getattr(model, "generate", None):
            raw = method(_batch(samples))
        else:
            try:
                raw = method(_batch(samples), verbose=0)
            except TypeError as error:
                if "verbose" not in str(error):
                    raise
                raw = method(_batch(samples))
    if isinstance(raw, dict) and len(samples) == 1:
        values = [raw]
    else:
        values = list(raw) if not np.isscalar(raw) else [raw]
    if len(values) != len(samples):
        raise ValueError("prediction_batch_length_mismatch")
    results = [output_from_raw(value, task_family=family, x_sample=sample, config=cfg, batched=False)
               for sample, value in zip(samples, values)]
    for result in results:
        result.metadata["inference_backend"] = "generic_batched"
    return results


def _predict_hf_outputs(core, samples, family, hf_task, cfg):
    torch, spec = core.torch, core.task_spec
    model = core.model
    was_training = bool(getattr(model, "training", False))
    cfg = dict(cfg)
    cfg.setdefault("label_pad_value", getattr(core, "label_pad_value", -100))
    tokenizer = getattr(core, "tokenizer", None)
    if family in {"classification", "token_classification"} and bool(getattr(spec, "multilabel", False)):
        raise UnsupportedTaskOutput("multilabel_requires_independent_label_probability_adapter")
    if family == "retrieval":
        # This core's retrieval spec scores the current in-batch pairs. That
        # pool changes with perturbation batching and cannot establish ranking
        # consistency against frozen distractors.
        raise UnsupportedTaskOutput("hf_retrieval_requires_frozen_external_candidate_pool_adapter")
    if family == "fill_mask":
        cfg["evaluation_mask_token_id"] = getattr(tokenizer, "mask_token_id", None)
    try:
        model.eval()
        with torch.no_grad():
            if bool(getattr(spec, "supports_generation", False)):
                ensure_padding = getattr(core, "_ensure_left_padding_for_decoder_only_generation", None)
                if callable(ensure_padding):
                    ensure_padding()
            input_batch = _batch(samples)
            if isinstance(input_batch, dict):
                # This mask is evidence metadata, not a transformer forward arg.
                input_batch = {key: value for key, value in input_batch.items() if key != "special_tokens_mask"}
            enc, _, extra = spec.encode_batch(tokenizer, input_batch, None, core.max_length, torch,
                                              core.device, ignore_index=getattr(core, "label_pad_value", -100), inference_only=True)
            if bool(getattr(spec, "supports_generation", False)):
                generation_context = getattr(core, "_generation_inference_mode", None)
                with generation_context() if callable(generation_context) else contextlib.nullcontext():
                    generation_config = dict(getattr(core, "generation_config", {}) or {})
                    generation_config.update(do_sample=False)
                    raw = spec.generate_predictions(model, enc, tokenizer, torch, generation_config)
                cfg["generation_special_token_ids"] = list(getattr(tokenizer, "all_special_ids", []) or [])
                results = [output_from_raw(value, task_family="generation", config=cfg, batched=False) for value in _array(raw)]
                decoder = getattr(tokenizer, "decode", None)
                if callable(decoder):
                    for result in results:
                        result.metadata["decoded_text"] = decoder(list(result.prediction), skip_special_tokens=True)
            else:
                forward = lambda: model(**spec.build_forward_inputs(enc, labels_t=None, inference_only=True))
                precision = getattr(core, "_run_with_precision_context", None)
                outputs = precision(forward) if callable(precision) else forward()
                extract = getattr(core, "_extract_logits", None)
                logits = extract(outputs) if callable(extract) else spec.extract_logits(outputs)
                cfg["evaluation_scores_are_logits"] = True
                if family == "detection":
                    if not hasattr(outputs, "pred_boxes"):
                        raise UnsupportedTaskOutput("detection_model_lacks_box_predictions")
                    processor = getattr(spec, "_image_processor", None)
                    postprocess = getattr(processor, "post_process_object_detection", None)
                    if callable(postprocess):
                        processed = postprocess(outputs, threshold=0.0, target_sizes=torch.ones((len(samples), 2), device=core.device))
                        results = [output_from_raw({**row, "box_format": "xyxy", "normalized": True},
                                                  task_family=family, x_sample=sample, config=cfg, batched=False)
                                   for row, sample in zip(processed, samples)]
                        for result in results:
                            result.metadata["detection_postprocessor"] = type(processor).__qualname__
                    else:
                        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
                        if model_type not in {"detr", "yolos", "deformable_detr", "conditional_detr", "dab-detr", "deta"} and cfg.get("evaluation_detection_output_contract") != "detr_softmax_background":
                            raise UnsupportedTaskOutput("detection_requires_model_postprocessor_or_explicit_logits_contract")
                        probs = _probabilities(_array(logits), logits=True)
                        boxes = _array(outputs.pred_boxes)
                        results = []
                        for row, boxrow, sample in zip(probs, boxes, samples):
                            labels = row[:, :-1].argmax(axis=-1)
                            remap = getattr(spec, "_remap_predicted_class_indices_if_needed", None)
                            if callable(remap):
                                labels = remap(labels, num_pred_classes=row.shape[-1] - 1)
                            values = {"boxes": boxrow, "labels": labels, "scores": row[:, :-1].max(axis=-1),
                                      "box_format": "cxcywh", "normalized": True}
                            results.append(output_from_raw(values, task_family=family, x_sample=sample, config=cfg, batched=False))
                else:
                    if family == "segmentation":
                        cfg["evaluation_segmentation_class_axis"] = 0
                    results = []
                    for index, value in enumerate(_array(logits)):
                        encoded_sample = {key: _array(val)[index] for key, val in enc.items()}
                        if family == "token_classification" and isinstance(samples[index], dict) and "special_tokens_mask" in samples[index]:
                            encoded_sample["special_tokens_mask"] = samples[index]["special_tokens_mask"]
                        elif family == "token_classification" and "input_ids" in encoded_sample:
                            boundary_ids = [getattr(tokenizer, name, None) for name in
                                            ("cls_token_id", "sep_token_id", "bos_token_id", "eos_token_id", "pad_token_id")]
                            boundary_ids = [int(value) for value in boundary_ids if value is not None]
                            if boundary_ids:
                                encoded_sample["special_tokens_mask"] = np.isin(encoded_sample["input_ids"], boundary_ids)
                        results.append(output_from_raw(value, task_family=family, x_sample=encoded_sample, config=cfg, batched=False))
            if len(results) != len(samples):
                raise ValueError("prediction_batch_length_mismatch")
            for result in results:
                result.metadata.update(inference_backend="hf_batched", hf_task=hf_task)
            return results
    finally:
        if was_training:
            model.train()


def _target_scale(clean, target_scale):
    value = target_scale if target_scale is not None else clean.metadata.get("target_scale")
    if value is None or not np.isfinite(value) or float(value) <= 0:
        raise UnsupportedTaskOutput("regression_requires_fixed_positive_target_scale")
    return float(value)


def _overlap(a, b):
    classes = np.union1d(a, b)
    per_class = {}
    for label in classes:
        aa, bb = a == label, b == label
        intersection = int((aa & bb).sum())
        union = int((aa | bb).sum())
        per_class[str(label)] = {"iou": intersection / union, "dice": 2 * intersection / int(aa.sum() + bb.sum()),
                                 "intersection": intersection, "union": union}
    return per_class


def _sequence_similarity(a, b):
    # Exact edit distance over the entire generated sequence; no prefix cap.
    a, b = tuple(a), tuple(b)
    if a == b:
        return 1.0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return 1 - previous[-1] / max(len(a), len(b), 1)


def _scores_at(output, valid_mask):
    positions = output.metadata.get("score_positions")
    if positions is None:
        return output.scores[valid_mask]
    row_for_position = {int(pos): i for i, pos in enumerate(positions)}
    try:
        rows = [row_for_position[int(pos)] for pos in np.flatnonzero(valid_mask)]
    except KeyError as exc:
        raise UnsupportedTaskOutput("task_query_distribution_alignment_missing") from exc
    return output.scores[rows]


def _box_iou(left, right):
    overlap = np.maximum(0, np.minimum(left[:, None, 2:], right[None, :, 2:]) - np.maximum(left[:, None, :2], right[None, :, :2]))
    intersection = overlap.prod(axis=-1)
    left_area = np.maximum(0, left[:, 2:] - left[:, :2]).prod(axis=-1)
    right_area = np.maximum(0, right[:, 2:] - right[:, :2]).prod(axis=-1)
    return intersection / np.maximum(left_area[:, None] + right_area[None, :] - intersection, 1e-12)


def _detection_comparison(clean, perturbed):
    from scipy.optimize import linear_sum_assignment

    a, b = clean.prediction, perturbed.prediction
    na, nb = len(a["boxes"]), len(b["boxes"])
    ious = _box_iou(a["boxes"], b["boxes"])
    threshold = float(clean.metadata.get("match_iou_threshold", 0.5))
    # Matching maximizes count above threshold first, then overlap; labels are
    # deliberately omitted from assignment so class changes remain observable.
    rows, cols = linear_sum_assignment(-((ious >= threshold) * (min(na, nb) + 1) + ious))
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if ious[i, j] >= threshold]
    agreement = [float(a["labels"][i] == b["labels"][j]) for i, j in matches]
    matched_fraction = 2 * len(matches) / (na + nb) if na + nb else 1.0
    score = 2 * sum(ious[i, j] * correct for (i, j), correct in zip(matches, agreement)) / (na + nb) if na + nb else 1.0
    return score, {"matched_object_fraction": matched_fraction, "matched_object_count": len(matches),
                   "clean_object_count": na, "perturbed_object_count": nb,
                   "class_agreement": float(np.mean(agreement)) if matches else (1.0 if na + nb == 0 else 0.0),
                   "matched_mean_iou": float(np.mean([ious[i, j] for i, j in matches])) if matches else None,
                   "matched_box_movement": float(np.mean([np.abs(a["boxes"][i] - b["boxes"][j]).mean() for i, j in matches])) if matches else None,
                   "empty_output_pair": na + nb == 0}


def compare_task_outputs(clean, perturbed, *, target_scale=None, **kwargs):
    if clean.task_family != perturbed.task_family:
        raise UnsupportedTaskOutput("task_output_family_mismatch")
    family = clean.task_family
    if family == "classification":
        if clean.scores is None or perturbed.scores is None:
            raise UnsupportedTaskOutput("classification_requires_probability_distribution")
        if clean.scores.shape != perturbed.scores.shape or clean.metadata.get("class_labels") != perturbed.metadata.get("class_labels"):
            raise UnsupportedTaskOutput("classification_label_space_changed")
        tv = float(np.abs(clean.scores - perturbed.scores).sum() / 2)
        agreement = float(clean.prediction == perturbed.prediction)
        components = {"class_agreement": agreement, "probability_total_variation": tv}
        score = (agreement + 1 - tv) / 2
    elif family == "regression":
        a, b = _array(clean.prediction), _array(perturbed.prediction)
        if a.shape != b.shape:
            raise UnsupportedTaskOutput("regression_output_shape_changed")
        scale = _target_scale(clean, target_scale)
        delta = float(np.abs(a - b).mean())
        components = {"prediction_absolute_delta": delta, "prediction_scaled_delta": delta / scale, "target_scale": scale}
        score = max(0.0, 1 - delta / scale)
    elif family in {"token_classification", "fill_mask", "segmentation"}:
        a, b = _array(clean.prediction), _array(perturbed.prediction)
        if a.shape != b.shape:
            raise UnsupportedTaskOutput("unaligned_sequence_or_mask_output")
        va = clean.valid_mask if clean.valid_mask is not None else np.ones(a.shape, dtype=bool)
        vb = perturbed.valid_mask if perturbed.valid_mask is not None else np.ones(b.shape, dtype=bool)
        if family == "fill_mask" and np.all(vb[va]):
            # The original masked positions define the task query. Additional
            # masks introduced by an explanation do not create new targets.
            vb = va
        if not np.array_equal(va, vb):
            raise UnsupportedTaskOutput("valid_output_alignment_changed")
        a, b = a[va], b[va]
        if not a.size:
            raise UnsupportedTaskOutput("no_valid_output_positions")
        agreement = float((a == b).mean())
        components = {"valid_position_count": int(a.size), "pixel_agreement" if family == "segmentation" else "token_agreement": agreement,
                      "complete_output_agreement": float(np.array_equal(a, b))}
        score = agreement
        if family == "segmentation":
            overlap = _overlap(a, b)
            components.update(per_class_overlap=overlap, mean_iou=float(np.mean([v["iou"] for v in overlap.values()])))
            score = components["mean_iou"]
        if clean.scores is not None and perturbed.scores is not None:
            clean_scores, perturbed_scores = _scores_at(clean, va), _scores_at(perturbed, vb)
            if clean_scores.shape != perturbed_scores.shape:
                raise UnsupportedTaskOutput("dense_output_label_space_changed")
            tv = float(np.abs(clean_scores - perturbed_scores).sum(axis=-1).mean() / 2)
            components["probability_total_variation"] = tv
            if family == "fill_mask":
                score = (score + 1 - tv) / 2
    elif family == "detection":
        score, components = _detection_comparison(clean, perturbed)
    elif family == "generation":
        score = _sequence_similarity(clean.prediction, perturbed.prediction)
        components = {"sequence_edit_similarity": score, "sequence_exact_agreement": float(clean.prediction == perturbed.prediction),
                      "clean_sequence_length": len(clean.prediction), "perturbed_sequence_length": len(perturbed.prediction)}
    elif family == "retrieval":
        if clean.metadata.get("candidate_ids") != perturbed.metadata.get("candidate_ids"):
            raise UnsupportedTaskOutput("retrieval_candidate_pool_changed")
        left, right = clean.prediction, perturbed.prediction
        if len(left) < 2 or len(left) != len(right):
            raise UnsupportedTaskOutput("retrieval_ranking_requires_comparable_candidates")
        rank = {item: i for i, item in enumerate(right)}
        distance = sum(abs(i - rank[item]) for i, item in enumerate(left))
        score = 1 - distance / (len(left) ** 2 // 2)
        components = {"rank_footrule_similarity": score, "top1_agreement": float(left[0] == right[0]),
                      "candidate_count": len(left)}
    else:
        raise UnsupportedTaskOutput(f"no_stability_adapter_for_{family}")
    return {"stability_score": float(np.clip(score, 0, 1)), "score": float(np.clip(score, 0, 1)), "supported": True, "components": components}


def task_output_quality(output, y_sample, *, target_scale=None, **kwargs):
    if y_sample is None:
        return {"score": None, "metric": None, "reason": "missing_reference_labels"}
    family = output.task_family
    if family == "classification":
        y = _array(y_sample)
        if y.size == len(output.metadata.get("class_labels", [])) and y.size > 1:
            y = output.metadata["class_labels"][int(y.argmax())]
        elif y.size == 1:
            y = y.reshape(-1)[0]
        else:
            return {"score": None, "metric": None, "reason": "unaligned_reference_labels"}
        return {"score": float(output.prediction == y), "metric": "accuracy"}
    if family == "regression":
        y = _array(y_sample).astype(float).reshape(-1)
        p = _array(output.prediction).reshape(-1)
        if p.shape != y.shape or not np.isfinite(y).all():
            return {"score": None, "metric": None, "reason": "invalid_regression_reference"}
        scale = _target_scale(output, target_scale)
        error = float(np.abs(p - y).mean())
        return {"score": max(0.0, 1 - error / scale), "metric": "fixed_scale_absolute_quality", "mae": error, "target_scale": scale}
    if family in {"token_classification", "fill_mask", "segmentation"}:
        labels = _array(y_sample)
        prediction = _array(output.prediction)
        valid = np.array(output.valid_mask, copy=True) if output.valid_mask is not None else np.ones(prediction.shape, dtype=bool)
        if labels.shape != prediction.shape and family == "segmentation" and labels.ndim == prediction.ndim == 2:
            # Evaluate every reference pixel, rather than downsampling away small regions.
            ys = np.minimum((np.arange(labels.shape[0]) * prediction.shape[0] / labels.shape[0]).astype(int), prediction.shape[0] - 1)
            xs = np.minimum((np.arange(labels.shape[1]) * prediction.shape[1] / labels.shape[1]).astype(int), prediction.shape[1] - 1)
            prediction, valid = prediction[np.ix_(ys, xs)], valid[np.ix_(ys, xs)]
        if labels.shape != prediction.shape:
            return {"score": None, "metric": None, "reason": "unaligned_reference_labels"}
        valid &= np.isfinite(labels) & ~np.isin(labels, output.metadata.get("ignore_label_ids", [-100]))
        if not valid.any():
            return {"score": None, "metric": None, "reason": "no_valid_reference_labels"}
        if family == "segmentation":
            overlap = _overlap(prediction[valid], labels[valid])
            return {"score": float(np.mean([v["iou"] for v in overlap.values()])), "metric": "mean_iou",
                    "per_class_overlap": overlap, "pixel_accuracy": float((prediction[valid] == labels[valid]).mean())}
        return {"score": float((prediction[valid] == labels[valid]).mean()), "metric": "valid_token_accuracy"}
    if family == "generation":
        if isinstance(y_sample, str) and output.metadata.get("sequence_kind") == "token_ids":
            text = output.metadata.get("decoded_text")
            if text is None:
                return {"score": None, "metric": None, "reason": "generation_reference_tokenization_missing"}
            return {"score": _sequence_similarity(text.split(), y_sample.split()), "metric": "reference_word_edit_similarity"}
        reference = output_from_raw(y_sample, task_family="generation", batched=False,
                                    config={"generation_special_token_ids": output.metadata.get("ignored_token_ids", [])})
        return {"score": _sequence_similarity(output.prediction, reference.prediction), "metric": "reference_sequence_edit_similarity"}
    if family == "retrieval":
        labels = np.asarray(y_sample).reshape(-1).tolist()
        return {"score": float(output.prediction[0] in labels), "metric": "recall_at_1"}
    if family == "detection" and isinstance(y_sample, dict):
        try:
            reference = output_from_raw(y_sample, task_family="detection", batched=False,
                                        config={"evaluation_detection_score_threshold": 0, "detection_boxes_normalized": y_sample.get("normalized", False)})
            score, components = _detection_comparison(reference, output)
            return {"score": score, "metric": "matched_class_iou_f1", "components": components}
        except (UnsupportedTaskOutput, ValueError):
            return {"score": None, "metric": None, "reason": "detection_reference_coordinate_contract_missing"}
    return {"score": None, "metric": None, "reason": "unsupported_reference_quality"}


def candidate_units(x_sample, *, task_family=None, meta=None, limit=64):
    """Candidate groups depend solely on inputs, never relevance annotations."""
    from .perturbation import _limited, _meaningful_units, _image_array_view, _modality

    modality = _modality(x_sample, task_family=task_family, meta=meta)
    if modality == "tokens":
        cfg = meta or {}
        ids = _array(x_sample["input_ids"]).reshape(-1)
        valid = _array(x_sample.get("attention_mask", np.ones(ids.shape))).reshape(-1).astype(bool)
        if valid.shape != ids.shape:
            raise UnsupportedTaskOutput("candidate_token_attention_alignment_missing")
        if "special_tokens_mask" in x_sample:
            special = _array(x_sample["special_tokens_mask"]).reshape(-1).astype(bool)
            if special.shape != ids.shape:
                raise UnsupportedTaskOutput("candidate_special_token_alignment_missing")
            valid &= ~special
        valid &= ~np.isin(ids, cfg.get("evaluation_boundary_token_ids", []))
        mask_id = cfg.get("evaluation_mask_token_id")
        if _family(task_family) == "fill_mask" and mask_id is not None:
            valid &= ids != int(mask_id)
        return _limited([("token", int(i)) for i in np.flatnonzero(valid)], limit)
    if modality == "numeric":
        cfg = meta or {}
        size = _array(x_sample).size
        configured = cfg.get("feature_groups", [])
        configured = list(configured.values()) if isinstance(configured, dict) else list(configured)
        configured += list(cfg.get("one_hot_feature_groups", []))
        immutable = set(int(i) for i in cfg.get("immutable_feature_indices", []))
        covered, groups = set(), []
        for group in configured:
            indices = tuple(dict.fromkeys(int(i) for i in group))
            if not indices or any(i < 0 or i >= size for i in indices):
                raise UnsupportedTaskOutput("invalid_explanation_feature_groups")
            if any(i in covered for i in indices):
                raise UnsupportedTaskOutput("overlapping_explanation_feature_groups")
            covered.update(indices)
            if not immutable.intersection(indices):
                groups.append(("feature_group", indices))
        groups.extend(("feature", i) for i in range(size) if i not in covered and i not in immutable)
        return _limited(groups, limit)
    if modality != "image":
        return _meaningful_units(x_sample, None, task_family=task_family, meta=meta, limit=limit)
    arr = _image_array_view(x_sample)
    if arr is None or arr.ndim != 3:
        return []
    channel_first = arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4)
    height, width = arr.shape[1:3] if channel_first else arr.shape[:2]
    grid = max(1, math.ceil(math.sqrt(limit)))
    units = []
    for iy in range(min(grid, height)):
        for ix in range(min(grid, width)):
            y0, y1 = round(iy * height / min(grid, height)), round((iy + 1) * height / min(grid, height))
            x0, x1 = round(ix * width / min(grid, width)), round((ix + 1) * width / min(grid, width))
            units.append(("patch", y0, y1, x0, x1, channel_first))
    return _limited(units, limit)


def derive_input_calibration(x_eval, *, task_family=None, meta=None):
    """Freeze feature scales/ranges from the shared reference inputs once."""
    from .perturbation import _get_sample, _modality, _sample_count

    count = _sample_count(x_eval)
    if count < 2:
        return {}
    first = _get_sample(x_eval, 0)
    if _modality(first, task_family=task_family, meta=meta) != "numeric":
        return {}
    values = np.stack([_array(_get_sample(x_eval, i)).astype(float).reshape(-1) for i in range(count)])
    if not np.isfinite(values).all():
        return {}
    discrete = [i for i in range(values.shape[1]) if np.all(values[:, i] == np.round(values[:, i]))
                and len(np.unique(values[:, i])) <= min(20, max(2, count // 10))]
    return {"feature_scales": values.std(axis=0).tolist(), "feature_min": values.min(axis=0).tolist(),
            "feature_max": values.max(axis=0).tolist(), "categorical_feature_indices": discrete,
            "input_calibration_provenance": "frozen_heldout_feature_distribution"}


def resolve_image_calibration(model, *, config=None, meta=None):
    """Resolve physical image normalization without altering frozen loader data.

    Explicit parameters and frozen preprocessing metadata take precedence. A
    missing processor contract is read only from local model/cache assets; this
    helper never downloads a processor or substitutes guessed normalization.
    Returned effective parameters belong in the evaluation definition hash.
    """
    config, meta = config or {}, meta or {}
    settings = _settings(config, meta)
    frozen = {}
    for source in (meta, config):
        processor_config = source.get("processor_config")
        if isinstance(processor_config, dict):
            frozen.update(processor_config.get("image_processor", processor_config))
        normalization = source.get("normalization")
        if isinstance(normalization, dict):
            frozen.update({{"mean": "image_mean", "std": "image_std", "enabled": "do_normalize"}.get(key, key): value
                           for key, value in normalization.items()})
    aliases = {"evaluation_image_mean": "image_mean", "evaluation_image_std": "image_std",
               "evaluation_image_value_range": "image_value_range", "image_do_normalize": "do_normalize",
               "image_do_rescale": "do_rescale", "image_rescale_factor": "rescale_factor"}
    for source in (meta, config):
        for key in ("image_mean", "image_std", "image_value_range", "do_normalize", "do_rescale", "rescale_factor", *aliases):
            if source.get(key) is not None:
                frozen[aliases.get(key, key)] = source[key]
    core = getattr(model, "core", None)
    contract = {}
    origin = None
    complete_explicit = frozen.get("image_mean") is not None and frozen.get("image_std") is not None
    if complete_explicit or frozen.get("do_normalize") is False:
        origin = "explicit_or_frozen_image_preprocessing"
    else:
        spec = getattr(core, "task_spec", None)
        owners = (core, spec, model)
        processor = next((getattr(owner, attr) for owner in owners if owner is not None
                          for attr in ("image_processor", "_image_processor", "processor", "tokenizer")
                          if getattr(owner, attr, None) is not None and hasattr(getattr(owner, attr), "image_mean")), None)
        if processor is not None:
            contract = processor.to_dict() if callable(getattr(processor, "to_dict", None)) else {
                key: getattr(processor, key, None) for key in ("image_mean", "image_std", "do_normalize", "do_rescale", "rescale_factor")}
            origin = "runtime_image_processor"
        else:
            sources = list(dict.fromkeys(str(value) for value in (
                getattr(core, "model_source", None), settings.get("hf_processor"), settings.get("hf_model_id"),
                getattr(core, "model_id", None)) if value))
            if not sources:
                # Numeric/raw-image adapters can infer their already-physical
                # bounds from the input; no HF normalization contract exists.
                if not frozen:
                    return {}
            errors = []
            if sources:
                from transformers import AutoImageProcessor
                for source in sources:
                    try:
                        kwargs = {"local_files_only": True, "use_fast": False}
                        revision = settings.get("hf_model_revision") or getattr(core, "model_revision", None)
                        if revision:
                            kwargs["revision"] = revision
                        processor = AutoImageProcessor.from_pretrained(source, **kwargs)
                        contract = processor.to_dict()
                        origin = "cached_image_processor"
                        contract["evaluation_processor_source"] = source
                        contract["evaluation_processor_revision"] = revision
                        break
                    except (OSError, ValueError, TypeError) as exc:
                        errors.append(f"{source}: {type(exc).__name__}")
                if origin is None:
                    raise UnsupportedTaskOutput("image_processor_calibration_unavailable_locally: " + "; ".join(errors))
    parameters = {**contract, **frozen}
    if origin is None:
        origin = "explicit_image_value_range"
    # hf_image's processor invocation requests do_normalize=True. An explicit
    # frozen preprocessing override remains authoritative if one is supplied.
    normalizes = bool(frozen.get("do_normalize", True if settings.get("hf_processor") else parameters.get("do_normalize", complete_explicit)))
    mean, std = parameters.get("image_mean"), parameters.get("image_std")
    if (mean is None) != (std is None):
        raise UnsupportedTaskOutput("image_normalization_parameters_incomplete")
    if normalizes and (mean is None or std is None):
        raise UnsupportedTaskOutput("image_normalization_parameters_missing")
    result = {}
    if normalizes:
        mean, std = np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
        if not mean.size or not std.size or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise UnsupportedTaskOutput("invalid_image_normalization_parameters")
        result.update(image_mean=mean.tolist(), image_std=std.tolist())
    elif complete_explicit:
        # A frozen processor can publish mean/std defaults even when its actual
        # preprocessing disables normalization. Override them with identity
        # parameters so downstream merged metadata cannot reactivate them.
        result.update(image_mean=[0.0], image_std=[1.0])
    bounds = frozen.get("image_value_range")
    if bounds is None:
        factor = parameters.get("rescale_factor")
        rescales = bool(parameters.get("do_rescale", normalizes))
        if rescales:
            factor = 1 / 255 if factor is None else float(factor)
            bounds = [0.0, 255.0 * factor]
        else:
            bounds = [0.0, 255.0]
    bounds = np.asarray(bounds, dtype=float).reshape(-1)
    if bounds.size != 2 or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise UnsupportedTaskOutput("invalid_image_calibration_bounds")
    contract_hash = hashlib.sha256(json.dumps({"contract": contract, "overrides": frozen}, sort_keys=True, default=str).encode()).hexdigest()
    result.update(image_value_range=bounds.tolist(), input_calibration_provenance=origin,
                  image_calibration_contract={"source": origin, "sha256": contract_hash,
                                              "do_normalize": normalizes,
                                              "processor_source": contract.get("evaluation_processor_source"),
                                              "processor_revision": contract.get("evaluation_processor_revision"),
                                              "local_files_only": True})
    return result


def _changed(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() != right.keys() or any(_changed(left[key], right[key]) for key in left)
    try:
        return not np.array_equal(_array(left), _array(right), equal_nan=True)
    except TypeError:
        return not np.array_equal(_array(left), _array(right))


def perturb_input(x_sample, rng, *, task_family=None, trial_index=0, config=None, meta=None, model=None):
    """Produce calibrated operating changes, recording excluded/unsupported trials."""
    from .perturbation import _copy_sample, _modality, _image_array_view, _set_image_array

    cfg = _settings(config, meta)
    modality = _modality(x_sample, task_family=task_family, meta=meta)
    strengths = cfg.get("evaluation_perturbation_strengths", [0.01, 0.025, 0.05])
    strengths = [float(value) for value in strengths]
    if not strengths or any(not np.isfinite(v) or v < 0 for v in strengths):
        raise ValueError("invalid_evaluation_perturbation_strengths")
    out = _copy_sample(x_sample)
    operator = "unavailable"
    strength = strengths[(int(trial_index) // 4) % len(strengths)]
    metadata = {"definition_version": "operating_perturbations_v2", "calibration_provenance": cfg.get("input_calibration_provenance", "dataset_contract")}

    def unavailable(reason):
        return PerturbedInput(out, operator, strength, False, "unsupported", reason, metadata=metadata)

    if modality == "image":
        arr = _image_array_view(x_sample)
        if arr is None or arr.ndim != 3 or not np.isfinite(arr).all():
            return unavailable("image_requires_finite_three_dimensional_array")
        operators = cfg.get("evaluation_image_operators", ["gaussian_noise", "brightness", "gaussian_blur", "jpeg_compression"])
        if not operators:
            return unavailable("no_image_operators_configured")
        operator = str(operators[int(trial_index) % len(operators)])
        strength = strengths[(int(trial_index) // len(operators)) % len(strengths)]
        channel_first = arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4)
        values = np.moveaxis(arr, 0, -1).astype(float) if channel_first else arr.astype(float)
        image_mean = cfg.get("image_mean", cfg.get("evaluation_image_mean"))
        image_std = cfg.get("image_std", cfg.get("evaluation_image_std"))
        if (image_mean is None) != (image_std is None):
            return unavailable("image_normalization_requires_both_mean_and_std")
        if image_mean is not None:
            image_mean, image_std = np.asarray(image_mean, dtype=float), np.asarray(image_std, dtype=float)
            if image_mean.size not in (1, values.shape[-1]) or image_std.size not in (1, values.shape[-1]) or not np.isfinite(image_mean).all() or not np.isfinite(image_std).all() or np.any(image_std <= 0):
                return unavailable("invalid_image_normalization_contract")
            values = values * image_std + image_mean
            metadata.update(image_mean=image_mean.tolist(), image_std=image_std.tolist(), corruption_space="physical_pixels")
        bounds = cfg.get("image_value_range", cfg.get("evaluation_image_value_range"))
        if bounds is None:
            if image_mean is not None:
                bounds = [0, 1]
            elif arr.dtype.kind in "ui":
                bounds = [0, int(np.iinfo(arr.dtype).max)]
            elif arr.min() >= 0 and arr.max() <= 1:
                bounds = [0, 1]
            elif arr.min() >= 0 and 1 < arr.max() <= 255:
                bounds = [0, 255]
            else:
                return unavailable("normalized_image_requires_calibrated_value_range")
        low, high = map(float, bounds)
        if not np.isfinite([low, high]).all() or high <= low:
            return unavailable("invalid_image_value_range")
        tolerance = 1e-6 * (high - low)
        if values.min() < low - tolerance or values.max() > high + tolerance:
            return PerturbedInput(out, operator, strength, False, "excluded", "input_outside_calibrated_image_range", metadata=metadata)
        metadata["value_range"] = [low, high]
        span = high - low
        if operator == "gaussian_noise":
            values += rng.normal(0, strength * span, size=values.shape)
            metadata["noise_std"] = strength * span
        elif operator == "brightness":
            sign = int(rng.choice([-1, 1]))
            values += sign * strength * span
            metadata["brightness_offset"] = sign * strength * span
        elif operator == "gaussian_blur":
            from scipy.ndimage import gaussian_filter
            sigma = strength * 20
            values = gaussian_filter(values, sigma=(sigma, sigma, 0), mode="reflect")
            metadata["blur_sigma_pixels"] = sigma
        elif operator == "jpeg_compression":
            from PIL import Image
            quality = max(1, min(100, int(round(100 - strength * 600))))
            if values.shape[-1] not in (1, 3):
                return unavailable("jpeg_requires_one_or_three_image_channels")
            pixels = np.round(np.clip((values - low) / span, 0, 1) * 255).astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="JPEG", quality=quality)
            stream.seek(0)
            with Image.open(stream) as reconstructed:
                pixels = np.asarray(reconstructed).copy()
            if pixels.ndim == 2:
                pixels = pixels[..., None]
            values = pixels.astype(float) / 255 * span + low
            metadata["jpeg_quality"] = quality
        else:
            return unavailable("unknown_image_operator")
        values = np.clip(values, low, high)
        if image_mean is not None:
            values = (values - image_mean) / image_std
        if arr.dtype.kind in "ui":
            values = np.round(values)
        values = values.astype(arr.dtype)
        if channel_first:
            values = np.moveaxis(values, -1, 0)
        out = _set_image_array(out, values)
    elif modality == "numeric":
        operator = "feature_scaled_gaussian_noise"
        strength = strengths[int(trial_index) % len(strengths)]
        arr = _array(x_sample)
        values = arr.astype(float).reshape(-1)
        if any(key not in cfg for key in ("feature_scales", "feature_min", "feature_max")):
            return unavailable("tabular_requires_frozen_feature_scales_and_ranges")
        scales, lower, upper = [np.asarray(cfg[key], dtype=float).reshape(-1) for key in ("feature_scales", "feature_min", "feature_max")]
        if any(v.shape != values.shape for v in (scales, lower, upper)) or not all(np.isfinite(v).all() for v in (scales, lower, upper, values)) or np.any(scales < 0) or np.any(lower > upper):
            return unavailable("invalid_tabular_feature_calibration")
        if np.any(values < lower) or np.any(values > upper):
            return PerturbedInput(out, operator, strength, False, "excluded", "input_outside_calibrated_feature_range", metadata=metadata)
        fixed = set(int(i) for i in cfg.get("categorical_feature_indices", [])) | set(int(i) for i in cfg.get("immutable_feature_indices", []))
        for group in cfg.get("one_hot_feature_groups", []):
            fixed.update(int(i) for i in group)
        if any(i < 0 or i >= len(values) for i in fixed):
            return unavailable("invalid_categorical_feature_indices")
        scales = scales.copy()
        scales[list(fixed)] = 0
        candidate = np.clip(values + rng.normal(0, strength * scales), lower, upper)
        candidate[list(fixed)] = values[list(fixed)]
        for index in cfg.get("integer_feature_indices", []):
            candidate[int(index)] = np.clip(round(candidate[int(index)]), lower[int(index)], upper[int(index)])
        constraints = cfg.get("feature_constraints", [])
        for constraint in constraints:
            coefficients = np.asarray(constraint["coefficients"], dtype=float)
            value = float(coefficients @ candidate)
            if value < float(constraint.get("min", -np.inf)) or value > float(constraint.get("max", np.inf)):
                return PerturbedInput(out, operator, strength, False, "excluded", "feature_relationship_constraint", metadata=metadata)
        validator = cfg.get("evaluation_feature_validator")
        if callable(validator) and not bool(validator(candidate.reshape(arr.shape))):
            return PerturbedInput(out, operator, strength, False, "excluded", "feature_relationship_constraint", metadata=metadata)
        # Continuous perturbations must not disappear through integer casting.
        out = candidate.reshape(arr.shape).astype(arr.dtype if arr.dtype.kind == "f" else float)
        metadata.update(fixed_feature_indices=sorted(fixed), feature_constraint_count=len(constraints))
    elif modality in {"text", "tokens"}:
        operator = "validated_text_replacement"
        transformations = cfg.get("evaluation_text_transformations", [])
        transformations = [item for item in transformations if isinstance(item, dict) and item.get("validation_id") and item.get("meaning_preserving") is True]
        if not transformations:
            return unavailable("no_validated_meaning_preserving_text_transformations")
        item = transformations[int(trial_index) % len(transformations)]
        metadata.update(validation_id=str(item["validation_id"]), provenance=item.get("provenance"),
                        alignment_preserved=bool(item.get("alignment_preserved", False)))
        if _family(task_family) in {"token_classification", "fill_mask"} and not item.get("alignment_preserved"):
            return unavailable("token_task_requires_validated_alignment_preserving_transform")
        if modality == "text":
            source, replacement = item.get("source"), item.get("replacement")
            if not isinstance(source, str) or not source or not isinstance(replacement, str):
                return unavailable("invalid_validated_text_transform")
            # Only exact word sequences are replaced; no accidental substrings.
            import re
            out = re.sub(r"(?<!\w)" + re.escape(source) + r"(?!\w)", lambda _: replacement, str(out), count=1)
        else:
            source, replacement = item.get("source_token_ids"), item.get("replacement_token_ids")
            if not source or replacement is None or len(source) != len(replacement):
                return unavailable("tokenized_text_transform_requires_equal_length_validated_token_ids")
            from .perturbation import _valid_replacement_token_id
            if any(_valid_replacement_token_id(model, token) is None for token in replacement):
                return unavailable("validated_replacement_token_outside_model_vocabulary")
            ids = _array(out["input_ids"]).copy()
            flat = ids.reshape(-1)
            attention = _array(out.get("attention_mask", np.ones(ids.shape))).reshape(-1).astype(bool)
            for start in range(len(flat) - len(source) + 1):
                if attention[start:start + len(source)].all() and np.array_equal(flat[start:start + len(source)], source):
                    flat[start:start + len(source)] = replacement
                    break
            out["input_ids"] = flat.reshape(ids.shape)
    else:
        return unavailable("no_operating_perturbation_for_modality")
    changed = _changed(x_sample, out)
    return PerturbedInput(out, operator, strength, changed, "complete" if changed else "excluded",
                          None if changed else "unchanged_input", metadata=metadata)


task_stability = compare_task_outputs
task_quality = task_output_quality
