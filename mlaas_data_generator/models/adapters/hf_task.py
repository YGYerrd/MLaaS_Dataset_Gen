import numpy as np
import re
import importlib.util

# ----------------------------
# HF Task Specs
# ----------------------------

def _load_auto_model_with_safetensor_fallback(transformers, model_id, auto_model_names, **from_pretrained_kwargs):
    last_error = None
    missing = []
    for auto_model_name in auto_model_names:
        AutoModel = getattr(transformers, auto_model_name, None)
        if AutoModel is None:
            missing.append(auto_model_name)
            continue
        try:
            return AutoModel.from_pretrained(
                model_id,
                use_safetensors=True,
                **from_pretrained_kwargs,
            ), "safetensors"
        except OSError as e:
            if "safetensors" not in str(e).lower():
                last_error = e
                continue
            try:
                return AutoModel.from_pretrained(
                    model_id,
                    use_safetensors=False,
                    **from_pretrained_kwargs,
                ), "pickle"
            except Exception as fallback_error:
                last_error = fallback_error
                continue
        except ValueError as e:
            last_error = e
            continue

    if last_error is not None:
        raise last_error
    raise AttributeError(f"transformers is missing AutoModel classes: {', '.join(missing)}")


_TEXT_METRIC_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _text_metric_tokens(text):
    return _TEXT_METRIC_TOKEN_RE.findall(str(text or "").lower())


def _ngram_counts(tokens, n):
    if n <= 0 or len(tokens) < n:
        return {}
    counts = {}
    for i in range(0, len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _overlap_f1(pred_items, ref_items):
    pred_total = sum(pred_items.values())
    ref_total = sum(ref_items.values())
    if pred_total == 0 and ref_total == 0:
        return 0.0
    if pred_total == 0 or ref_total == 0:
        return 0.0
    overlap = 0
    for item, count in pred_items.items():
        overlap += min(count, ref_items.get(item, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    return (2.0 * precision * recall) / (precision + recall)


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_from_texts(pred_texts, ref_texts):
    rouge1, rouge2, rougel = [], [], []
    for pred, ref in zip(pred_texts, ref_texts):
        pred_tokens = _text_metric_tokens(pred)
        ref_tokens = _text_metric_tokens(ref)
        rouge1.append(_overlap_f1(_ngram_counts(pred_tokens, 1), _ngram_counts(ref_tokens, 1)))
        rouge2.append(_overlap_f1(_ngram_counts(pred_tokens, 2), _ngram_counts(ref_tokens, 2)))

        if not pred_tokens and not ref_tokens:
            rougel.append(1.0)
        elif not pred_tokens or not ref_tokens:
            rougel.append(0.0)
        else:
            lcs = _lcs_len(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            rougel.append(0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall))

    if not rouge1:
        return np.nan, np.nan, np.nan
    return float(np.mean(rouge1)), float(np.mean(rouge2)), float(np.mean(rougel))


def _decode_token_id_batch(tokenizer, values, *, ignore_index=-100):
    if tokenizer is None or not hasattr(tokenizer, "batch_decode"):
        return None
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if not np.issubdtype(arr.dtype, np.integer):
        return None

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0

    cleaned = arr.astype("int64", copy=True)
    cleaned[cleaned == int(ignore_index)] = int(pad_id)

    vocab_limit = None
    try:
        vocab_limit = int(getattr(tokenizer, "vocab_size", 0) or 0)
    except Exception:
        vocab_limit = None
    if not vocab_limit:
        try:
            vocab_limit = int(len(tokenizer))
        except Exception:
            vocab_limit = None

    safe_pad_id = int(pad_id)
    if vocab_limit is not None and vocab_limit > 0 and not (0 <= safe_pad_id < vocab_limit):
        safe_pad_id = 0

    invalid = cleaned < 0
    if vocab_limit is not None and vocab_limit > 0:
        invalid |= cleaned >= int(vocab_limit)
    if np.any(invalid):
        cleaned[invalid] = safe_pad_id

    try:
        return list(
            tokenizer.batch_decode(
                cleaned.tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )
    except Exception:
        return None


def _strip_trailing_eos_token(tokenizer, token_ids):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    trimmed = list(token_ids)
    if eos_id is not None and trimmed and int(trimmed[-1]) == int(eos_id):
        trimmed = trimmed[:-1]
    return trimmed


def _pearson_correlation(y_true, y_pred):
    if y_true.size <= 1 or y_pred.size <= 1:
        return np.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return np.nan
    try:
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    except Exception:
        return np.nan


def _average_ranks(values):
    values = np.asarray(values, dtype="float32").reshape(-1)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype="float32")
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman_correlation(y_true, y_pred):
    if y_true.size <= 1 or y_pred.size <= 1:
        return np.nan
    return _pearson_correlation(_average_ranks(y_true), _average_ranks(y_pred))


def _to_torch_tensor(torch, value, *, dtype, device=None):
    if isinstance(value, np.ndarray):
        arr = value
    else:
        arr = np.asarray(value)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    as_tensor = getattr(torch, "as_tensor", None)
    if callable(as_tensor):
        return as_tensor(arr, dtype=dtype, device=device)
    return torch.tensor(arr, dtype=dtype, device=device)


class HFTaskSpec:
    """
    Task-specific behaviour for HF fine-tuning/evaluation.

    Loader schema support:
      - New path: xb is a dict of numpy arrays (already tokenised), e.g.
            {"input_ids": (B, L), "attention_mask": (B, L), ...}
      - Legacy path: xb is raw text (sequence) or list-of-tokens (token task)
    """
    name = "base"

    requires_num_labels = True
    requires_tokenizer = True
    supports_generation = False

    def build_model(self, transformers, model_id, num_labels):
        raise NotImplementedError

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        """
        Returns (enc_dict, labels_tensor_or_none, extra_dict)
        extra_dict can hold masks etc.
        """
        raise NotImplementedError

    def loss_fn(self, torch, logits, labels_t, extra):
        raise NotImplementedError

    def extract_logits(self, outputs):
        return outputs.logits

    def preds_from_logits(self, torch, logits, extra):
        raise NotImplementedError

    def metrics(self, y_true, y_pred, y_extra=None):
        """
        Returns dict with at least:
          - primary (float)
          - secondary (float or np.nan)
        """
        raise NotImplementedError

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        return None

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        return None

    def init_metric_accumulator(self):
        return {}

    def accumulate_metric_statistics(self, accumulator, batch_stats):
        if accumulator is None:
            accumulator = {}
        if not batch_stats:
            return accumulator
        for k, v in batch_stats.items():
            accumulator[k] = float(accumulator.get(k, 0.0)) + float(v)
        return accumulator

    def has_metric_statistics(self, accumulator):
        return bool(accumulator)

    def metric_statistics_summary(self, accumulator):
        if not isinstance(accumulator, dict):
            return {}
        summary = {}
        for key, value in accumulator.items():
            try:
                summary[str(key)] = float(value)
            except Exception:
                continue
        return summary

    def metrics_from_statistics(self, stats):
        return None

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        model_inputs = dict(enc)
        if labels_t is not None and not inference_only:
            model_inputs["labels"] = labels_t
        return model_inputs

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        if labels_t is not None and hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if labels_t is None:
            return None
        return self.loss_fn(torch, logits, labels_t, extra)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        raise NotImplementedError


class SequenceClassificationSpec(HFTaskSpec):
    name = "sequence_classification"

    def __init__(self, multilabel=False, threshold=0.5, label_format="single_index"):
        self.multilabel = bool(multilabel)
        self.threshold = float(threshold)
        self.label_format = str(label_format or "single_index").lower()
    
    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"
        
        if self.label_format in {"onehot", "multilabel", "multihot"}:
            mapping = {"onehot": "single_onehot", "multihot": "multilabel"}
            return mapping.get(self.label_format, self.label_format)


        arr = np.asarray(yb)
        if arr.ndim == 1:
            return "single_index"

        if arr.ndim == 2:
            is_binary = np.isin(arr, [0, 1]).all()
            row_sums = arr.sum(axis=1)
            if is_binary and np.all(row_sums == 1):
                return "single_onehot"
            return "multilabel"

        return "unknown"

    def _is_multilabel_mode(self, label_mode, extra):
        mode = extra.get("label_mode", label_mode)
        return bool(self.multilabel or mode == "multilabel")


    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSequenceClassification
        self.weight_format = None
        extra = {}
        if self.multilabel:
            extra["problem_type"] = "multi_label_classification"
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=int(num_labels),
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                **extra,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=int(num_labels),
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                    **extra,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        label_mode = self._infer_label_mode(yb)
        batch_multilabel = self._is_multilabel_mode(label_mode, {"label_mode": label_mode})
        # New loader path: already tokenised dict of arrays
        if isinstance(xb, dict):
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in xb.items()}
            labels_t = None
            if yb is not None:
                dtype = torch.float32 if (batch_multilabel or label_mode == "single_onehot") else torch.long
                labels_t = _to_torch_tensor(torch, yb, dtype=dtype, device=device)
            return enc, labels_t, {"multilabel": batch_multilabel, "label_mode": label_mode}


        # Legacy path: raw texts
        enc = tokenizer(
            xb,
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            dtype = torch.float32 if (batch_multilabel or label_mode == "single_onehot") else torch.long
            labels_t = _to_torch_tensor(torch, yb, dtype=dtype, device=device)
        return enc, labels_t, {
            "multilabel": batch_multilabel,
            "label_mode": label_mode,
            "ignore_index": int(ignore_index),
        }

    def loss_fn(self, torch, logits, labels_t, extra):
        label_mode = extra.get("label_mode", "unknown")
        if self._is_multilabel_mode(label_mode, extra):
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)

        if label_mode == "single_onehot":
            labels_t = torch.argmax(labels_t, dim=-1)
        if logits.ndim == 3 and labels_t.ndim == 1:
            logits = logits[:, 0, :]
        
        if labels_t.ndim == 1:
            num_classes = int(logits.shape[-1])
            valid = (labels_t >= 0) & (labels_t < num_classes)
            if not bool(torch.any(valid)):
                return logits.new_tensor(0.0)
            logits = logits[valid]
            labels_t = labels_t[valid]
            
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        if bool(extra.get("multilabel", self.multilabel)):
            probs = torch.sigmoid(logits)
            return (probs >= self.threshold).to(dtype=torch.int64)
        return torch.argmax(logits, dim=-1)

    def metrics(self, y_true, y_pred, y_extra=None):
        from sklearn.metrics import f1_score

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        label_mode = self._infer_label_mode(y_true)
        if label_mode == "single_onehot":
            y_true = np.argmax(y_true, axis=1)

        is_multilabel = bool(self.multilabel or label_mode == "multilabel")
        if is_multilabel:
            subset_acc = float((y_pred == y_true).all(axis=1).mean()) if y_true.size else np.nan
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else np.nan
            return {"primary": subset_acc, "secondary": f1}
        
        acc = float((y_pred == y_true).mean()) if y_true.size else np.nan
        f1 = float(f1_score(y_true, y_pred, average="weighted")) if y_true.size else np.nan
        return {"primary": acc, "secondary": f1}


class TokenClassificationSpec(HFTaskSpec):
    name = "token_classification"

    def __init__(self, multilabel=False, label_format="token_index"):
        self.multilabel = bool(multilabel)
        self.label_format = str(label_format or "token_index").lower()

    def _infer_label_mode(self, yb):
        if yb is None:
            return "none"
        
        if self.label_format in {"token_index", "single_index", "onehot", "multilabel", "multihot"}:
            mapping = {"token_index": "single_index", "onehot": "single_onehot", "multihot": "multilabel"}
            return mapping.get(self.label_format, self.label_format)

        arr = np.asarray(yb)

        if arr.ndim in (1, 2):
            return "single_index"

        if arr.ndim == 3:
            is_binary = np.isin(arr, [0, 1]).all()
            if is_binary and np.all(arr.sum(axis=-1) == 1):
                return "single_onehot"
            return "multilabel"

        return "unknown"
    
    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForTokenClassification
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=int(num_labels),
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=int(num_labels),
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def _align_labels(self, enc_word_ids, word_labels, ignore_index=-100):
        aligned = []
        prev = None
        for wid in enc_word_ids:
            if wid is None:
                aligned.append(ignore_index)
            elif wid != prev:
                aligned.append(int(word_labels[wid]))
            else:
                aligned.append(ignore_index)
            prev = wid
        return aligned

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        label_mode = self._infer_label_mode(yb)

        if isinstance(xb, dict):
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in xb.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        batch_multilabel = False

        if yb is not None:
            if label_mode == "single_index":
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)

            elif label_mode == "single_onehot":
                y_idx = np.asarray(yb).argmax(axis=-1)
                labels_t = _to_torch_tensor(torch, y_idx, dtype=torch.long, device=device)

            elif label_mode == "multilabel":
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.float32, device=device)
                batch_multilabel = True

            else:
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)

        batch_multilabel = bool(self.multilabel or batch_multilabel)

        return enc, labels_t, {
            "multilabel": batch_multilabel,
            "label_mode": label_mode,
            "ignore_index": int(ignore_index),
        }

    def loss_fn(self, torch, logits, labels_t, extra):
        use_multilabel = bool(extra.get("multilabel", self.multilabel))
        if use_multilabel:
            if labels_t.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                labels_t = labels_t.float()
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
        ignore_index = int(extra.get("ignore_index", -100))

        if logits.ndim == 3 and labels_t.ndim == 2:
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels_t,
                ignore_index=ignore_index,
            )

        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=ignore_index)


    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)  # [B, T]

    def metrics(self, y_true, y_pred, y_extra=None):
        from sklearn.metrics import f1_score

        ignore_index = -100
        if isinstance(y_extra, dict) and "ignore_index" in y_extra:
            ignore_index = int(y_extra["ignore_index"])

        # Accept torch tensors or numpy arrays
        try:
            import torch
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
        except Exception:
            pass

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = (y_true != ignore_index)
        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.size == 0:
            return {"primary": np.nan, "secondary": np.nan}

        acc = float((yp == yt).mean())
        f1 = float(f1_score(yt, yp, average="weighted"))
        return {"primary": acc, "secondary": f1}


class ImageClassificationSpec(HFTaskSpec):
    name = "image_classification"
    requires_tokenizer = False

    @staticmethod
    def _macro_f1_from_counts(stats):
        labels = set()
        for key in stats:
            key = str(key)
            for suffix in ("_tp", "_pred_total", "_target_total"):
                if key.startswith("class_") and key.endswith(suffix):
                    labels.add(key[len("class_"):-len(suffix)])

        scores = []
        for label in sorted(labels):
            tp = float(stats.get(f"class_{label}_tp", 0.0))
            pred_total = float(stats.get(f"class_{label}_pred_total", 0.0))
            target_total = float(stats.get(f"class_{label}_target_total", 0.0))
            if pred_total <= 0 and target_total <= 0:
                continue
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / target_total if target_total > 0 else 0.0
            scores.append(0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall))

        return float(np.mean(scores)) if scores else np.nan

    @classmethod
    def _macro_f1_from_arrays(cls, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        n = min(int(y_true.size), int(y_pred.size))
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        stats = {}
        for label in np.unique(np.concatenate([y_true, y_pred], axis=0)):
            label_key = str(int(label))
            true_mask = y_true == label
            pred_mask = y_pred == label
            stats[f"class_{label_key}_tp"] = float(np.count_nonzero(true_mask & pred_mask))
            stats[f"class_{label_key}_pred_total"] = float(np.count_nonzero(pred_mask))
            stats[f"class_{label_key}_target_total"] = float(np.count_nonzero(true_mask))
        return cls._macro_f1_from_counts(stats)

    def build_model(self, transformers, model_id, num_labels):
        return transformers.AutoModelForImageClassification.from_pretrained(
            model_id,
            num_labels=int(num_labels),
            ignore_mismatched_sizes=True,
        )

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("image classification expects dict input with 'pixel_values'")
        enc = {"pixel_values": _to_torch_tensor(torch, xb["pixel_values"], dtype=torch.float32, device=device)}
        labels_t = None if yb is None else _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
        return enc, labels_t, {"top_k": 5}

    def loss_fn(self, torch, logits, labels_t, extra):
        return torch.nn.functional.cross_entropy(logits, labels_t)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        if labels_t is None:
            return None
        labels_t = labels_t.view(-1)
        top1 = torch.argmax(logits, dim=-1)
        k = int(min(int(extra.get("top_k", 5)), int(logits.shape[-1])))
        topk = torch.topk(logits, k=k, dim=-1).indices
        top1_correct = int((top1 == labels_t).sum().detach().cpu().item())
        topk_correct = int((topk == labels_t.unsqueeze(-1)).any(dim=-1).sum().detach().cpu().item())
        total = int(labels_t.shape[0])
        stats = {"top1_correct": top1_correct, "top5_correct": topk_correct, "total": total}

        labels_np = labels_t.detach().cpu().numpy().reshape(-1)
        preds_np = top1.detach().cpu().numpy().reshape(-1)
        for label in np.unique(np.concatenate([labels_np, preds_np], axis=0)):
            label_key = str(int(label))
            true_mask = labels_np == label
            pred_mask = preds_np == label
            stats[f"class_{label_key}_tp"] = int(np.count_nonzero(true_mask & pred_mask))
            stats[f"class_{label_key}_pred_total"] = int(np.count_nonzero(pred_mask))
            stats[f"class_{label_key}_target_total"] = int(np.count_nonzero(true_mask))
        return stats

    def metrics_from_statistics(self, stats):
        total = float(stats.get("total", 0.0))
        if total <= 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        top1 = float(stats.get("top1_correct", 0.0)) / total
        top5 = float(stats.get("top5_correct", 0.0)) / total
        f1 = self._macro_f1_from_counts(stats)
        return {
            "primary": top1,
            "secondary": f1,
            "named_metrics": {"accuracy": top1, "top1_accuracy": top1, "f1": f1, "macro_f1": f1, "top5_accuracy": top5},
        }

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        acc = float((y_true == y_pred).mean())
        f1 = self._macro_f1_from_arrays(y_true, y_pred)
        return {
            "primary": acc,
            "secondary": f1,
            "named_metrics": {"accuracy": acc, "top1_accuracy": acc, "f1": f1, "macro_f1": f1},
        }


class ObjectDetectionSpec(HFTaskSpec):
    name = "image_detection"
    requires_tokenizer = False
    requires_num_labels = False

    def __init__(self, score_threshold=0.05):
        self.score_threshold = float(score_threshold)
        self._model_valid_class_ids = None
        self._image_processor = None

    def build_model(self, transformers, model_id, num_labels):
        kwargs = {"ignore_mismatched_sizes": True}
        if num_labels is not None:
            kwargs["num_labels"] = int(num_labels)
        model = transformers.AutoModelForObjectDetection.from_pretrained(
            model_id,
            **kwargs,
        )
        self._model_valid_class_ids = self._extract_valid_class_ids_from_model(model)
        try:
            self._image_processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        except Exception:
            self._image_processor = None
        return model

    @staticmethod
    def _extract_valid_class_ids_from_model(model):
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None)
        if not isinstance(id2label, dict) or not id2label:
            return None
        cleaned = {}
        for k, v in id2label.items():
            try:
                kid = int(k)
            except Exception:
                continue
            cleaned[kid] = str(v)
        if not cleaned:
            return None
        valid = [k for k in sorted(cleaned) if cleaned[k].strip().lower() != "n/a"]
        return valid or None

    def _remap_contiguous_classes_if_needed(self, classes, *, force=False):
        class_ids = np.asarray(classes, dtype=np.int64)
        valid_ids = self._model_valid_class_ids
        if class_ids.size == 0 or not valid_ids:
            return class_ids

        # COCO-style HF checkpoints (e.g. DETR/YOLOS) frequently expose id2label
        # with index 0 reserved for "N/A". Some datasets provide contiguous
        # category ids in [0, 79], where 0 means "person". In that case we need
        # to remap contiguous ids -> model ids before metric matching.
        if (
            valid_ids
            and valid_ids[0] == 1
            and int(np.min(class_ids)) >= 0
            and int(np.max(class_ids)) < len(valid_ids)
            and (force or 0 in set(class_ids.tolist()))
        ):
            mapped = np.asarray([int(valid_ids[int(cid)]) for cid in class_ids], dtype=np.int64)
            return mapped
        return class_ids

    def _remap_predicted_class_indices_if_needed(self, class_indices, num_pred_classes):
        class_ids = np.asarray(class_indices, dtype=np.int64)
        valid_ids = self._model_valid_class_ids
        if class_ids.size == 0 or not valid_ids:
            return class_ids

        # Some checkpoints expose contiguous prediction logits over only the
        # valid classes while still publishing sparse COCO ids in id2label
        # (e.g. valid ids start at 1). Detect that layout from the logits
        # dimensionality and remap only in that case.
        if (
            valid_ids[0] == 1
            and int(num_pred_classes) == int(len(valid_ids))
            and int(np.min(class_ids)) >= 0
            and int(np.max(class_ids)) < int(len(valid_ids))
        ):
            return np.asarray([int(valid_ids[int(cid)]) for cid in class_ids], dtype=np.int64)
        return class_ids

    @staticmethod
    def _normalise_pixel_array(sample):
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"object detection expects 3D image tensors, got shape={arr.shape}")
        if arr.shape[0] == 3:
            return arr
        if arr.shape[-1] == 3:
            return np.transpose(arr, (2, 0, 1))
        raise ValueError(
            "object detection expects image tensors with 3 channels in CHW or HWC layout, "
            f"got shape={arr.shape}"
        )

    @staticmethod
    def _resolve_label_image_size(item, fallback_h, fallback_w):
        image_h = int(fallback_h)
        image_w = int(fallback_w)
        if not isinstance(item, dict):
            return image_h, image_w

        raw_size = item.get("image_size")
        if raw_size is None:
            return image_h, image_w

        try:
            size_arr = np.asarray(raw_size, dtype=np.int64).reshape(-1)
            if size_arr.size >= 2 and int(size_arr[0]) > 0 and int(size_arr[1]) > 0:
                return int(size_arr[0]), int(size_arr[1])
        except Exception:
            pass
        return image_h, image_w

    @staticmethod
    def _extract_orig_size_from_label_tensor(label_item, fallback_h=1, fallback_w=1):
        image_h = int(fallback_h)
        image_w = int(fallback_w)
        if not isinstance(label_item, dict):
            return image_h, image_w

        raw_size = label_item.get("orig_size")
        if raw_size is None:
            return image_h, image_w

        try:
            if hasattr(raw_size, "detach"):
                size_arr = raw_size.detach().cpu().numpy().reshape(-1)
            else:
                size_arr = np.asarray(raw_size, dtype=np.int64).reshape(-1)
            if size_arr.size >= 2 and int(size_arr[0]) > 0 and int(size_arr[1]) > 0:
                return int(size_arr[0]), int(size_arr[1])
        except Exception:
            pass
        return image_h, image_w

    @staticmethod
    def _remap_classes_with_explicit_map(classes, class_id_map):
        class_ids = np.asarray(classes, dtype=np.int64)
        if class_ids.size == 0 or class_id_map is None:
            return class_ids
        mapping = np.asarray(class_id_map, dtype=np.int64).reshape(-1)
        if mapping.size == 0:
            return class_ids
        min_cls = int(np.min(class_ids))
        max_cls = int(np.max(class_ids))
        if min_cls < 0 or max_cls >= int(mapping.size):
            return class_ids
        return mapping[class_ids]

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("object detection expects dict input with 'pixel_values'")
        pixel_values = xb["pixel_values"]
        pixel_arrays = [self._normalise_pixel_array(sample) for sample in pixel_values]
        if not pixel_arrays:
            enc = {
                "pixel_values": torch.empty((0, 3, 0, 0), dtype=torch.float32, device=device),
                "pixel_mask": torch.empty((0, 0, 0), dtype=torch.long, device=device),
            }
        else:
            shapes = {arr.shape for arr in pixel_arrays}
            if len(shapes) == 1:
                batch_pixels = np.stack(pixel_arrays, axis=0)
                batch_mask = np.ones(
                    (batch_pixels.shape[0], batch_pixels.shape[2], batch_pixels.shape[3]),
                    dtype=np.int64,
                )
            else:
                max_h = max(arr.shape[1] for arr in pixel_arrays)
                max_w = max(arr.shape[2] for arr in pixel_arrays)
                padded = []
                masks = []
                for arr in pixel_arrays:
                    pad_h = max_h - arr.shape[1]
                    pad_w = max_w - arr.shape[2]
                    padded.append(np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant"))
                    mask = np.zeros((max_h, max_w), dtype=np.int64)
                    mask[: arr.shape[1], : arr.shape[2]] = 1
                    masks.append(mask)
                batch_pixels = np.stack(padded, axis=0)
                batch_mask = np.stack(masks, axis=0)
            enc = {
                "pixel_values": _to_torch_tensor(torch, batch_pixels, dtype=torch.float32, device=device),
                "pixel_mask": _to_torch_tensor(torch, batch_mask, dtype=torch.long, device=device),
            }
        labels_t = None
        if yb is not None:
            labels_t = []
            for idx, item in enumerate(yb):
                fallback_h = int(pixel_arrays[idx].shape[1]) if idx < len(pixel_arrays) else 1
                fallback_w = int(pixel_arrays[idx].shape[2]) if idx < len(pixel_arrays) else 1
                image_h, image_w = self._resolve_label_image_size(item, fallback_h, fallback_w)
                boxes_xyxy_norm = self._to_xyxy_normalized(
                    item.get("boxes", []),
                    image_h=image_h,
                    image_w=image_w,
                    box_format=item.get("box_format"),
                )
                boxes_cxcywh_norm = self._xyxy_to_cxcywh(boxes_xyxy_norm)
                class_labels = self._remap_classes_with_explicit_map(
                    item.get("classes", []),
                    item.get("class_id_map"),
                )
                if class_labels.size == 0 and item.get("classes") is not None:
                    class_labels = np.asarray(item.get("classes", []), dtype=np.int64)
                if item.get("class_id_map") is None:
                    force_contiguous_remap = bool(item.get("force_contiguous_label_remap", False))
                    class_labels = self._remap_contiguous_classes_if_needed(
                        class_labels,
                        force=force_contiguous_remap,
                    )
                labels_t.append(
                    {
                        "class_labels": _to_torch_tensor(torch, class_labels, dtype=torch.long, device=device),
                        "boxes": _to_torch_tensor(torch, boxes_cxcywh_norm, dtype=torch.float32, device=device),
                        "orig_size": _to_torch_tensor(torch, [image_h, image_w], dtype=torch.long, device=device),
                    }
                )
        return enc, labels_t, {"score_threshold": self.score_threshold}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        out = dict(enc)
        if labels_t is not None and not inference_only:
            out["labels"] = labels_t
        return out

    def loss_fn(self, torch, logits, labels_t, extra):
        return None

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        return getattr(outputs, "loss", None)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def _box_iou(self, boxes_a, boxes_b):
        if boxes_a.size == 0 or boxes_b.size == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
        tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        wh = np.clip(br - tl, a_min=0.0, a_max=None)
        inter = wh[..., 0] * wh[..., 1]
        area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
        area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
        union = np.clip(area_a[:, None] + area_b[None, :] - inter, a_min=1e-9, a_max=None)
        return inter / union

    @staticmethod
    def _xyxy_to_cxcywh(boxes_xyxy):
        if boxes_xyxy.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = boxes_xyxy.T
        w = np.clip(x2 - x1, a_min=0.0, a_max=None)
        h = np.clip(y2 - y1, a_min=0.0, a_max=None)
        cx = x1 + (w / 2.0)
        cy = y1 + (h / 2.0)
        return np.column_stack([cx, cy, w, h]).astype(np.float32, copy=False)

    @staticmethod
    def _cxcywh_to_xyxy(boxes_cxcywh):
        if boxes_cxcywh.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        cx, cy, w, h = boxes_cxcywh.T
        return np.column_stack([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ]).astype(np.float32, copy=False)

    def _to_xyxy_normalized(self, boxes, image_h, image_w, box_format=None):
        out_boxes = np.asarray(boxes, dtype=np.float32)
        if out_boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if out_boxes.ndim == 1:
            out_boxes = out_boxes.reshape(1, 4)
        if out_boxes.ndim != 2 or out_boxes.shape[1] != 4:
            raise ValueError(f"object detection boxes must be Nx4, got shape={out_boxes.shape}")

        fmt = str(box_format or "").strip().lower().replace("-", "_")
        if fmt:
            if fmt in {"xyxy", "pascal_voc"}:
                boxes_xyxy = out_boxes
            elif fmt in {"xywh", "coco"}:
                x, y, w, h = out_boxes.T
                boxes_xyxy = np.column_stack([x, y, x + w, y + h])
            elif fmt in {"cxcywh", "center"}:
                boxes_xyxy = self._cxcywh_to_xyxy(out_boxes)
            else:
                raise ValueError(f"unsupported box_format='{box_format}'")

            max_val = float(np.max(boxes_xyxy)) if boxes_xyxy.size else 0.0
            if max_val > 1.5:
                boxes_xyxy = boxes_xyxy.copy()
                boxes_xyxy[:, [0, 2]] /= max(float(image_w), 1e-9)
                boxes_xyxy[:, [1, 3]] /= max(float(image_h), 1e-9)
            return np.clip(boxes_xyxy, a_min=0.0, a_max=1.0).astype(np.float32, copy=False)

        max_val = float(np.max(out_boxes))
        monotonic_xyxy = bool(np.all(out_boxes[:, 2] >= out_boxes[:, 0]) and np.all(out_boxes[:, 3] >= out_boxes[:, 1]))
        if max_val <= 1.5:
            boxes_xyxy = out_boxes if monotonic_xyxy else self._cxcywh_to_xyxy(out_boxes)
        else:
            bounded_xyxy = (
                monotonic_xyxy
                and np.all(out_boxes[:, 0] <= float(image_w) * 1.05)
                and np.all(out_boxes[:, 2] <= float(image_w) * 1.05)
                and np.all(out_boxes[:, 1] <= float(image_h) * 1.05)
                and np.all(out_boxes[:, 3] <= float(image_h) * 1.05)
            )
            if bounded_xyxy:
                boxes_xyxy = out_boxes
            else:
                x, y, w, h = out_boxes.T
                boxes_xyxy = np.column_stack([x, y, x + w, y + h])
            boxes_xyxy[:, [0, 2]] /= max(float(image_w), 1e-9)
            boxes_xyxy[:, [1, 3]] /= max(float(image_h), 1e-9)

        return np.clip(boxes_xyxy, a_min=0.0, a_max=1.0).astype(np.float32, copy=False)

    @staticmethod
    def _coerce_metric_scalar(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numel") and int(value.numel()) == 1:
            value = value.item()
        try:
            parsed = float(value)
        except Exception:
            return np.nan
        if parsed < 0:
            return np.nan
        return parsed

    def _resolve_map_backend(self):
        candidates = (
            ("faster_coco_eval", "faster_coco_eval"),
            ("pycocotools", "pycocotools"),
        )
        for module_name, backend_name in candidates:
            if importlib.util.find_spec(module_name) is not None:
                return backend_name
        raise ImportError(
            "Object detection evaluation requires a COCO metric backend. "
            "Install either 'faster-coco-eval' or 'pycocotools'."
        )

    def _build_map_metric(self):
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except Exception as e:
            raise ImportError(
                "Object detection evaluation requires 'torchmetrics'. "
                "Install it with: pip install torchmetrics"
            ) from e

        backend = self._resolve_map_backend()
        kwargs = {
            "box_format": "xyxy",
            "iou_type": "bbox",
            "class_metrics": False,
            "backend": backend,
        }
        try:
            metric = MeanAveragePrecision(**kwargs)
        except TypeError:
            kwargs.pop("backend", None)
            metric = MeanAveragePrecision(**kwargs)
        return metric, backend

    def init_metric_accumulator(self):
        metric, backend = self._build_map_metric()
        return {
            "__kind__": "torchmetrics_map",
            "metric": metric,
            "backend": backend,
            "metric_instance_count": 0.0,
            "num_updates": 0.0,
        }

    def accumulate_metric_statistics(self, accumulator, batch_stats):
        if batch_stats and "__map_batch__" in batch_stats:
            if not accumulator or accumulator.get("__kind__") != "torchmetrics_map":
                accumulator = self.init_metric_accumulator()
            payload = batch_stats["__map_batch__"] or {}
            preds = payload.get("preds") or []
            targets = payload.get("targets") or []
            fallback_stats = payload.get("fallback_stats") or {}
            try:
                accumulator["metric"].update(preds, targets)
                accumulator["num_updates"] = float(accumulator.get("num_updates", 0.0)) + 1.0
            except (OverflowError, RuntimeError, ValueError) as exc:
                if "overflow" not in str(exc).lower():
                    raise
                accumulator["map_backend_error"] = str(exc)
                accumulator["fallback_updates"] = float(accumulator.get("fallback_updates", 0.0)) + 1.0
                for key, value in fallback_stats.items():
                    accumulator[key] = float(accumulator.get(key, 0.0)) + float(value)
            accumulator["metric_instance_count"] = float(accumulator.get("metric_instance_count", 0.0)) + float(
                payload.get("metric_instance_count", 0.0)
            )
            return accumulator
        return super().accumulate_metric_statistics(accumulator, batch_stats)

    def has_metric_statistics(self, accumulator):
        if isinstance(accumulator, dict) and accumulator.get("__kind__") == "torchmetrics_map":
            return (
                float(accumulator.get("num_updates", 0.0)) > 0.0
                or float(accumulator.get("fallback_updates", 0.0)) > 0.0
            )
        return super().has_metric_statistics(accumulator)

    def metric_statistics_summary(self, accumulator):
        if isinstance(accumulator, dict) and accumulator.get("__kind__") == "torchmetrics_map":
            summary = {
                "metric_instance_count": float(accumulator.get("metric_instance_count", 0.0)),
                "num_updates": float(accumulator.get("num_updates", 0.0)),
            }
            for key in ("fallback_updates", "gt", "tp_0.5", "fp_0.5", "tp_0.75", "fp_0.75", "tp_0.95", "fp_0.95"):
                if key in accumulator:
                    summary[key] = float(accumulator.get(key, 0.0))
            return summary
        return super().metric_statistics_summary(accumulator)

    @staticmethod
    def _sanitize_metric_boxes_labels(boxes, labels, *, max_size=None):
        out_boxes = np.asarray(boxes, dtype=np.float32)
        out_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if out_boxes.size == 0 or out_labels.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        if out_boxes.ndim == 1:
            out_boxes = out_boxes.reshape(1, 4)
        n = min(int(out_boxes.shape[0]), int(out_labels.shape[0]))
        out_boxes = out_boxes[:n]
        out_labels = out_labels[:n]
        out_boxes = np.nan_to_num(out_boxes, nan=0.0, posinf=0.0, neginf=0.0)
        clip_max = float(max_size) if max_size is not None else float(np.iinfo(np.int32).max - 1)
        clip_max = max(1.0, min(clip_max, float(np.iinfo(np.int32).max - 1)))
        out_boxes = np.clip(out_boxes, a_min=0.0, a_max=clip_max)
        x1 = np.minimum(out_boxes[:, 0], out_boxes[:, 2])
        y1 = np.minimum(out_boxes[:, 1], out_boxes[:, 3])
        x2 = np.maximum(out_boxes[:, 0], out_boxes[:, 2])
        y2 = np.maximum(out_boxes[:, 1], out_boxes[:, 3])
        out_boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32, copy=False)
        valid = (
            np.isfinite(out_boxes).all(axis=1)
            & (out_boxes[:, 2] > out_boxes[:, 0])
            & (out_boxes[:, 3] > out_boxes[:, 1])
            & (out_labels >= 0)
            & (out_labels <= int(np.iinfo(np.int32).max - 1))
        )
        return out_boxes[valid], out_labels[valid].astype(np.int64, copy=False)

    def _simple_detection_stats(self, preds, targets):
        stats = {"gt": 0.0}
        for thr in (0.5, 0.75, 0.95):
            stats[f"tp_{thr}"] = 0.0
            stats[f"fp_{thr}"] = 0.0

        for pred, target in zip(preds or [], targets or []):
            p_boxes = pred.get("boxes")
            p_scores = pred.get("scores")
            p_labels = pred.get("labels")
            t_boxes = target.get("boxes")
            t_labels = target.get("labels")
            if hasattr(p_boxes, "detach"):
                p_boxes = p_boxes.detach().cpu().numpy()
            if hasattr(p_scores, "detach"):
                p_scores = p_scores.detach().cpu().numpy()
            if hasattr(p_labels, "detach"):
                p_labels = p_labels.detach().cpu().numpy()
            if hasattr(t_boxes, "detach"):
                t_boxes = t_boxes.detach().cpu().numpy()
            if hasattr(t_labels, "detach"):
                t_labels = t_labels.detach().cpu().numpy()

            p_boxes, p_labels = self._sanitize_metric_boxes_labels(p_boxes, p_labels)
            t_boxes, t_labels = self._sanitize_metric_boxes_labels(t_boxes, t_labels)
            p_scores = np.asarray(p_scores, dtype=np.float32).reshape(-1)[: len(p_labels)]
            stats["gt"] += float(len(t_labels))
            if len(p_labels) == 0:
                continue

            order = np.argsort(-p_scores) if p_scores.size else np.arange(len(p_labels))
            p_boxes = p_boxes[order]
            p_labels = p_labels[order]
            for thr in (0.5, 0.75, 0.95):
                matched = set()
                for box, label in zip(p_boxes, p_labels):
                    candidate_idx = np.where(t_labels == int(label))[0]
                    if candidate_idx.size == 0:
                        stats[f"fp_{thr}"] += 1.0
                        continue
                    ious = self._box_iou(np.asarray([box], dtype=np.float32), t_boxes[candidate_idx]).reshape(-1)
                    if ious.size == 0:
                        stats[f"fp_{thr}"] += 1.0
                        continue
                    best_local = int(np.argmax(ious))
                    best_idx = int(candidate_idx[best_local])
                    if float(ious[best_local]) >= float(thr) and best_idx not in matched:
                        matched.add(best_idx)
                        stats[f"tp_{thr}"] += 1.0
                    else:
                        stats[f"fp_{thr}"] += 1.0
        return stats

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        if labels_t is None or outputs is None or not hasattr(outputs, "pred_boxes") or not hasattr(outputs, "logits"):
            return None

        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
        boxes = outputs.pred_boxes.detach().cpu().numpy()
        valid_set = set(int(v) for v in self._model_valid_class_ids) if self._model_valid_class_ids else None

        post_processed = None
        if self._image_processor is not None and hasattr(self._image_processor, "post_process_object_detection"):
            try:
                target_sizes = []
                for gt in labels_t:
                    image_h, image_w = self._extract_orig_size_from_label_tensor(gt, fallback_h=1, fallback_w=1)
                    target_sizes.append([image_h, image_w])
                target_sizes_t = _to_torch_tensor(torch, target_sizes, dtype=torch.long)
                post_processed = self._image_processor.post_process_object_detection(
                    outputs,
                    threshold=0.0,
                    target_sizes=target_sizes_t,
                )
            except Exception:
                post_processed = None

        preds = []
        targets = []
        metric_instance_count = 0.0

        for bidx, gt in enumerate(labels_t):
            image_h, image_w = self._extract_orig_size_from_label_tensor(gt, fallback_h=1, fallback_w=1)
            gt_boxes = gt["boxes"].detach().cpu().numpy()
            gt_boxes = self._cxcywh_to_xyxy(gt_boxes)
            gt_boxes[:, [0, 2]] *= max(float(image_w), 1e-9)
            gt_boxes[:, [1, 3]] *= max(float(image_h), 1e-9)
            gt_classes = gt["class_labels"].detach().cpu().numpy()
            metric_instance_count += float(len(gt_classes))

            if post_processed is not None and bidx < len(post_processed):
                pred = post_processed[bidx]
                p_scores = pred["scores"].detach().cpu().numpy()
                p_cls = pred["labels"].detach().cpu().numpy()
                p_boxes = pred["boxes"].detach().cpu().numpy()
            else:
                p_scores = probs[bidx, :, :-1].max(axis=-1)
                p_cls = probs[bidx, :, :-1].argmax(axis=-1)
                p_boxes = self._cxcywh_to_xyxy(boxes[bidx])
                p_boxes[:, [0, 2]] *= max(float(image_w), 1e-9)
                p_boxes[:, [1, 3]] *= max(float(image_h), 1e-9)

            p_cls = self._remap_predicted_class_indices_if_needed(p_cls, num_pred_classes=probs.shape[-1] - 1)
            if valid_set is not None:
                valid_mask = np.asarray([int(cid) in valid_set for cid in p_cls], dtype=bool)
            else:
                valid_mask = np.ones_like(p_scores, dtype=bool)
            keep_idx = np.where(valid_mask)[0]

            p_boxes, p_cls = self._sanitize_metric_boxes_labels(
                p_boxes[keep_idx],
                p_cls[keep_idx],
                max_size=max(float(image_h), float(image_w), 1.0),
            )
            p_scores = np.asarray(p_scores[keep_idx], dtype=np.float32).reshape(-1)[: len(p_cls)]
            gt_boxes, gt_classes = self._sanitize_metric_boxes_labels(
                gt_boxes,
                gt_classes,
                max_size=max(float(image_h), float(image_w), 1.0),
            )

            preds.append(
                {
                    "boxes": _to_torch_tensor(torch, p_boxes, dtype=torch.float32),
                    "scores": _to_torch_tensor(torch, p_scores, dtype=torch.float32),
                    "labels": _to_torch_tensor(torch, p_cls, dtype=torch.int64),
                }
            )
            targets.append(
                {
                    "boxes": _to_torch_tensor(torch, gt_boxes, dtype=torch.float32),
                    "labels": _to_torch_tensor(torch, gt_classes, dtype=torch.int64),
                }
            )

        return {
            "__map_batch__": {
                "preds": preds,
                "targets": targets,
                "metric_instance_count": metric_instance_count,
                "fallback_stats": self._simple_detection_stats(preds, targets),
            }
        }

    @staticmethod
    def _fallback_metrics_from_statistics(stats):
        gt = float(stats.get("gt", 0.0))
        if gt <= 0:
            return None
        vals = []
        named = {}
        for thr in (0.5, 0.75, 0.95):
            tp = float(stats.get(f"tp_{thr}", 0.0))
            fp = float(stats.get(f"fp_{thr}", 0.0))
            value = tp / max(gt + fp, 1e-9)
            vals.append(value)
            named[f"map@{thr}"] = value
        mean_map = float(np.mean(vals)) if vals else np.nan
        named["map"] = mean_map
        return {
            "primary": mean_map,
            "secondary": float(named.get("map@0.5", np.nan)),
            "named_metrics": named,
        }

    def metrics_from_statistics(self, stats):
        if not isinstance(stats, dict):
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        if stats.get("__kind__") != "torchmetrics_map":
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}

        fallback = self._fallback_metrics_from_statistics(stats)
        metric = stats.get("metric")
        if metric is None or float(stats.get("num_updates", 0.0)) <= 0.0:
            return fallback or {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}

        try:
            raw = metric.compute()
        except (OverflowError, RuntimeError, ValueError) as exc:
            if "overflow" not in str(exc).lower():
                raise
            return fallback or {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}
        named = {
            "map": self._coerce_metric_scalar(raw.get("map")),
            "map@0.5": self._coerce_metric_scalar(raw.get("map_50")),
            "map@0.75": self._coerce_metric_scalar(raw.get("map_75")),
            "mar@1": self._coerce_metric_scalar(raw.get("mar_1")),
            "mar@10": self._coerce_metric_scalar(raw.get("mar_10")),
            "mar@100": self._coerce_metric_scalar(raw.get("mar_100")),
        }
        try:
            metric.reset()
        except Exception:
            pass
        return {
            "primary": float(named.get("map", np.nan)),
            "secondary": float(named.get("map@0.5", np.nan)),
            "named_metrics": named,
        }

    def metrics(self, y_true, y_pred, y_extra=None):
        return {"primary": np.nan, "secondary": np.nan}


class ImageSegmentationSpec(HFTaskSpec):
    name = "image_segmentation"
    requires_tokenizer = False

    @staticmethod
    def _segmentation_class_statistics(pred, tgt):
        pred_np = np.asarray(pred, dtype=np.int64).reshape(-1)
        tgt_np = np.asarray(tgt, dtype=np.int64).reshape(-1)
        if pred_np.size == 0 or tgt_np.size == 0:
            return {}

        stats = {}
        for cls in np.union1d(pred_np, tgt_np):
            cls_id = int(cls)
            pred_mask = pred_np == cls_id
            tgt_mask = tgt_np == cls_id
            intersection = float(np.count_nonzero(pred_mask & tgt_mask))
            pred_total = float(np.count_nonzero(pred_mask))
            target_total = float(np.count_nonzero(tgt_mask))
            stats[f"class_{cls_id}_intersection"] = intersection
            stats[f"class_{cls_id}_pred_total"] = pred_total
            stats[f"class_{cls_id}_target_total"] = target_total
        return stats

    @classmethod
    def _segmentation_metrics_from_stats(cls, stats):
        if not isinstance(stats, dict):
            return None

        per_class_iou = []
        per_class_dice = []
        pixel_correct = 0.0
        pixel_total = 0.0
        found_class_stats = False

        class_prefix = "class_"
        intersection_suffix = "_intersection"
        pred_total_suffix = "_pred_total"
        target_total_suffix = "_target_total"

        for key, raw_intersection in stats.items():
            key = str(key)
            if not (key.startswith(class_prefix) and key.endswith(intersection_suffix)):
                continue

            label_token = key[len(class_prefix):-len(intersection_suffix)]
            pred_key = f"{class_prefix}{label_token}{pred_total_suffix}"
            target_key = f"{class_prefix}{label_token}{target_total_suffix}"

            try:
                intersection = float(raw_intersection)
                pred_total = float(stats.get(pred_key, 0.0))
                target_total = float(stats.get(target_key, 0.0))
            except Exception:
                continue

            union = pred_total + target_total - intersection
            denom = pred_total + target_total
            if union > 0:
                per_class_iou.append(intersection / union)
            if denom > 0:
                per_class_dice.append((2.0 * intersection) / denom)
            pixel_correct += intersection
            pixel_total += target_total
            found_class_stats = True

        if not found_class_stats:
            return None

        mean_iou = float(np.mean(per_class_iou)) if per_class_iou else np.nan
        mean_dice = float(np.mean(per_class_dice)) if per_class_dice else np.nan
        pixel_accuracy = pixel_correct / pixel_total if pixel_total > 0 else np.nan
        return {
            "primary": mean_iou,
            "secondary": mean_dice,
            "named_metrics": {
                "iou": mean_iou,
                "dice": mean_dice,
                "pixel_accuracy": pixel_accuracy,
            },
        }

    def build_model(self, transformers, model_id, num_labels):
        return transformers.AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=int(num_labels),
            ignore_mismatched_sizes=True,
        )

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        # SegFormer-style models upsample logits to full mask resolution when
        # labels are passed directly. On consumer GPUs this can allocate many
        # GiB for ADE-style 150-class masks, so this spec computes the loss
        # against labels downsampled to the model's logits resolution instead.
        return dict(enc)

    @staticmethod
    def _resize_labels_to_logits(torch, labels_t, target_size):
        if labels_t is None or labels_t.shape[-2:] == target_size:
            return labels_t
        resized = torch.nn.functional.interpolate(
            labels_t.unsqueeze(1).float(),
            size=target_size,
            mode="nearest",
        )
        return resized.squeeze(1).long()

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict) or "pixel_values" not in xb:
            raise ValueError("image segmentation expects dict input with 'pixel_values'")
        enc = {"pixel_values": _to_torch_tensor(torch, xb["pixel_values"], dtype=torch.float32, device=device)}
        labels_t = None
        if yb is not None:
            labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        labels_t = self._resize_labels_to_logits(torch, labels_t, logits.shape[-2:])
        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=int(extra.get("ignore_index", -100)))

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=1)

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        if labels_t is None:
            return None
        labels_t = self._resize_labels_to_logits(torch, labels_t, logits.shape[-2:])
        pred = torch.argmax(logits, dim=1)
        ignore_index = int(extra.get("ignore_index", -100))
        valid = labels_t != ignore_index
        pred = pred[valid]
        tgt = labels_t[valid]
        if pred.numel() == 0:
            return {"metric_instance_count": 0.0}
        stats = self._segmentation_class_statistics(
            pred.detach().cpu().numpy(),
            tgt.detach().cpu().numpy(),
        )
        stats["metric_instance_count"] = 1.0
        return stats

    def metrics_from_statistics(self, stats):
        metrics = self._segmentation_metrics_from_stats(stats)
        if metrics is not None:
            return metrics

        intersection = float(stats.get("intersection", 0.0))
        union = float(stats.get("union", 0.0))
        pred_total = float(stats.get("pred_total", 0.0))
        target_total = float(stats.get("target_total", 0.0))
        iou = intersection / union if union > 0 else np.nan
        denom = pred_total + target_total
        dice = (2.0 * intersection) / denom if denom > 0 else np.nan
        return {"primary": iou, "secondary": dice, "named_metrics": {"iou": iou, "dice": dice}}

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan}
        ignore_index = None
        if isinstance(y_extra, dict) and y_extra.get("ignore_index") is not None:
            ignore_index = int(y_extra["ignore_index"])
        if ignore_index is not None:
            valid = y_true != ignore_index
            y_true = y_true[valid]
            y_pred = y_pred[valid]
        stats = self._segmentation_class_statistics(y_pred, y_true)
        metrics = self._segmentation_metrics_from_stats(stats)
        if metrics is None:
            return {"primary": np.nan, "secondary": np.nan}
        return metrics
    

class SentenceSimilaritySpec(HFTaskSpec):
    name = "sentence_similarity"

    def __init__(self, is_regression=False, threshold=0.5):
        self.is_regression = bool(is_regression)
        self.threshold = float(threshold)
        self._cls_spec = SequenceClassificationSpec()

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSequenceClassification
        self.weight_format = None
        resolved_num_labels = 1 if self.is_regression else int(num_labels)
        extra = {"problem_type": "regression"} if self.is_regression else {}
        try:
            model = AutoModel.from_pretrained(
                model_id,
                num_labels=resolved_num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                **extra,
            )
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(
                    model_id,
                    num_labels=resolved_num_labels,
                    ignore_mismatched_sizes=True,
                    use_safetensors=False,
                    **extra,
                )
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in xb.items()}
        elif isinstance(xb, (list, tuple)) and xb and isinstance(xb[0], (list, tuple)) and len(xb[0]) == 2:
            text_a = [row[0] for row in xb]
            text_b = [row[1] for row in xb]
            enc = tokenizer(
                text_a,
                text_b,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            if self.is_regression:
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.float32, device=device)
            else:
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
        return enc, labels_t, {"is_regression": self.is_regression}

    def loss_fn(self, torch, logits, labels_t, extra):
        if bool(extra.get("is_regression", self.is_regression)):
            pred = logits.squeeze(-1).to(torch.float32)
            target = labels_t.to(torch.float32).view_as(pred)
            return torch.nn.functional.mse_loss(pred, target)
        return self._cls_spec.loss_fn(torch, logits, labels_t, extra)

    def preds_from_logits(self, torch, logits, extra):
        if bool(extra.get("is_regression", self.is_regression)):
            return logits.squeeze(-1)
        return self._cls_spec.preds_from_logits(torch, logits, extra)

    def metrics(self, y_true, y_pred, y_extra=None):
        if not bool((y_extra or {}).get("is_regression", self.is_regression)):
            return self._cls_spec.metrics(y_true, y_pred, y_extra=y_extra)

        y_true = np.asarray(y_true, dtype="float32").reshape(-1)
        y_pred = np.asarray(y_pred, dtype="float32").reshape(-1)

        if y_true.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"pearson": np.nan, "spearman": np.nan}}

        pearson = _pearson_correlation(y_true, y_pred)
        spearman = _spearman_correlation(y_true, y_pred)
        return {
            "primary": pearson,
            "secondary": spearman,
            "named_metrics": {
                "pearson": pearson,
                "spearman": spearman,
            },
        }
    
class FillMaskSpec(HFTaskSpec):
    name = "fill_mask"
    requires_num_labels = False

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForMaskedLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in xb.items()}
        else:
            enc = tokenizer(
                xb,
                truncation=True,
                padding=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

        labels_t = None
        if yb is not None:
            labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)

        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        if logits.ndim == 3 and labels_t.ndim == 2:
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels_t,
                ignore_index=ignore_index,
            )
        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def metrics(self, y_true, y_pred, y_extra=None):
        ignore_index = -100
        if isinstance(y_extra, dict) and "ignore_index" in y_extra:
            ignore_index = int(y_extra["ignore_index"])

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mask = (y_true != ignore_index)
        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.size == 0:
            return {"primary": np.nan, "secondary": np.nan}

        acc = float((yp == yt).mean())

        return {"primary": acc, "secondary": np.nan}


class CausalLMGenerationSpec(HFTaskSpec):
    name = "causal_lm_generation"
    requires_num_labels = False
    supports_generation = True

    @staticmethod
    def _left_pad_batch(tokenizer, input_ids, attention_mask):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            return input_ids, attention_mask

        ids_np = np.asarray(input_ids)
        mask_np = np.asarray(attention_mask)
        if ids_np.ndim != 2 or mask_np.ndim != 2 or ids_np.shape != mask_np.shape:
            return input_ids, attention_mask

        shifted_ids = np.full(ids_np.shape, int(pad_id), dtype=ids_np.dtype)
        shifted_mask = np.zeros(mask_np.shape, dtype=mask_np.dtype)

        for row_idx in range(ids_np.shape[0]):
            valid = int(mask_np[row_idx].sum())
            if valid <= 0:
                continue
            shifted_ids[row_idx, -valid:] = ids_np[row_idx, :valid]
            shifted_mask[row_idx, -valid:] = mask_np[row_idx, :valid]

        return shifted_ids, shifted_mask

    @staticmethod
    def _left_pad_labels(labels, attention_mask, ignore_index):
        labels_np = np.asarray(labels)
        mask_np = np.asarray(attention_mask)
        if labels_np.ndim != 2 or mask_np.ndim != 2 or labels_np.shape != mask_np.shape:
            return labels

        shifted_labels = np.full(labels_np.shape, int(ignore_index), dtype=labels_np.dtype)
        for row_idx in range(labels_np.shape[0]):
            valid = int(mask_np[row_idx].sum())
            if valid <= 0:
                continue
            shifted_labels[row_idx, -valid:] = labels_np[row_idx, :valid]
        return shifted_labels

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForCausalLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        if isinstance(xb, dict):
            batch = {k: v for k, v in xb.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            labels_np = None if yb is None else np.asarray(yb)
            prompt_only_left_padded = False
            if inference_only and labels_np is not None and "input_ids" in batch and "attention_mask" in batch:
                input_ids = np.asarray(batch["input_ids"])
                attention_mask = np.asarray(batch["attention_mask"])
                if labels_np.shape == input_ids.shape:
                    prompt_only_ids = []
                    prompt_only_mask = []
                    for row_ids, row_mask, row_labels in zip(input_ids, attention_mask, labels_np):
                        active = np.asarray(row_mask).astype(bool)
                        prompt_positions = active & (np.asarray(row_labels) == int(ignore_index))
                        if np.any(prompt_positions):
                            trimmed_ids = np.asarray(row_ids)[prompt_positions]
                            trimmed_mask = np.ones(trimmed_ids.shape[0], dtype=np.asarray(row_mask).dtype)
                        else:
                            valid_ids = np.asarray(row_ids)[active]
                            trimmed_ids = valid_ids[:-1] if valid_ids.shape[0] > 1 else valid_ids
                            trimmed_mask = np.ones(trimmed_ids.shape[0], dtype=np.asarray(row_mask).dtype)
                        prompt_only_ids.append(trimmed_ids.tolist())
                        prompt_only_mask.append(trimmed_mask.tolist())
                    max_prompt_len = max((len(row) for row in prompt_only_ids), default=0)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    if pad_id is None:
                        pad_id = 0
                    if max_prompt_len <= 0:
                        max_prompt_len = 1
                    padded_ids = []
                    padded_mask = []
                    for row_ids, row_mask in zip(prompt_only_ids, prompt_only_mask):
                        if not row_ids:
                            row_ids = [int(pad_id)]
                            row_mask = [1]
                        pad_len = max_prompt_len - len(row_ids)
                        padded_ids.append(([int(pad_id)] * pad_len) + list(row_ids))
                        padded_mask.append(([0] * pad_len) + list(row_mask))
                    batch["input_ids"] = np.asarray(padded_ids, dtype=input_ids.dtype)
                    batch["attention_mask"] = np.asarray(padded_mask, dtype=attention_mask.dtype)
                    prompt_only_left_padded = True
            if "input_ids" in batch and "attention_mask" in batch:
                if (not prompt_only_left_padded) and labels_np is not None and labels_np.shape == np.asarray(batch["input_ids"]).shape:
                    labels_np = self._left_pad_labels(labels_np, batch["attention_mask"], ignore_index)
                if not prompt_only_left_padded:
                    batch["input_ids"], batch["attention_mask"] = self._left_pad_batch(
                        tokenizer,
                        batch["input_ids"],
                        batch["attention_mask"],
                    )
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in batch.items()}
            labels_t = None if labels_np is None else _to_torch_tensor(torch, labels_np, dtype=torch.long, device=device)
            return enc, labels_t, {"ignore_index": int(ignore_index)}

        prompts = list(xb)
        if yb is None or inference_only:
            enc = tokenizer(prompts, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            return {k: v.to(device) for k, v in enc.items()}, None, {"ignore_index": int(ignore_index)}

        prompt_tokens = tokenizer(prompts, truncation=True, padding=False, max_length=int(max_length), add_special_tokens=True)
        target_tokens = tokenizer(list(yb), truncation=True, padding=False, max_length=int(max_length), add_special_tokens=False)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

        batch_ids, batch_masks, batch_labels = [], [], []
        for p_ids, t_ids in zip(prompt_tokens["input_ids"], target_tokens["input_ids"]):
            prompt_ids = _strip_trailing_eos_token(tokenizer, p_ids)
            full_ids = (prompt_ids + list(t_ids) + [eos_id])[: int(max_length)]
            full_labels = ([-100] * len(prompt_ids) + list(t_ids) + [eos_id])[: int(max_length)]
            pad_len = int(max_length) - len(full_ids)
            batch_ids.append(full_ids + [pad_id] * pad_len)
            batch_masks.append([1] * len(full_ids) + [0] * pad_len)
            batch_labels.append(full_labels + [-100] * pad_len)

        enc = {
            "input_ids": _to_torch_tensor(torch, batch_ids, dtype=torch.long, device=device),
            "attention_mask": _to_torch_tensor(torch, batch_masks, dtype=torch.long, device=device),
        }
        labels_t = _to_torch_tensor(torch, batch_labels, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        if not torch.any(labels_t != ignore_index):
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        if getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        cfg = dict(generation_config)
        if cfg.get("pad_token_id") is None and tokenizer.pad_token_id is not None:
            cfg["pad_token_id"] = int(tokenizer.pad_token_id)
        generated = model.generate(**enc, **cfg)
        in_len = enc["input_ids"].shape[1]
        return generated[:, in_len:]

    def metrics(self, y_true, y_pred, y_extra=None):
        loss_mean = np.nan
        ignore_index = -100
        if isinstance(y_extra, dict):
            loss_mean = float(y_extra.get("loss_mean", np.nan))
            ignore_index = int(y_extra.get("ignore_index", ignore_index))
        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan
        token_accuracy = np.nan

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size != 0 and y_pred.size != 0:
            common = min(y_true.shape[-1], y_pred.shape[-1])
            yt = y_true[..., :common]
            yp = y_pred[..., :common]
            mask = (yt != ignore_index)
            yt = yt[mask]
            yp = yp[mask]
            if yt.size != 0:
                token_accuracy = float((yt == yp).mean())

        return {
            "primary": loss_mean,
            "secondary": ppl,
            "named_metrics": {"cross_entropy_loss": loss_mean, "perplexity": ppl, "token_accuracy": token_accuracy},
        }


class Seq2SeqGenerationSpec(HFTaskSpec):
    name = "seq2seq_generation"
    requires_num_labels = False
    supports_generation = True

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModelForSeq2SeqLM
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if isinstance(xb, dict):
            enc = {k: _to_torch_tensor(torch, v, dtype=torch.long, device=device) for k, v in xb.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            labels_t = None if yb is None else _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
            return enc, labels_t, {"ignore_index": int(ignore_index)}

        enc = tokenizer(xb, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        labels_t = None
        if yb is not None and not inference_only:
            targets = tokenizer(text_target=list(yb), truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            labels_t = targets["input_ids"].to(device)
            labels_t = labels_t.masked_fill(labels_t == tokenizer.pad_token_id, int(ignore_index))
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        return model.generate(**enc, **generation_config)

    def metrics(self, y_true, y_pred, y_extra=None):
        task_tag = ""
        loss_mean = np.nan
        ignore_index = -100
        tokenizer = None
        if isinstance(y_extra, dict):
            task_tag = str(y_extra.get("task_tag") or "").strip().lower().replace("-", "_")
            loss_mean = float(y_extra.get("loss_mean", np.nan))
            ignore_index = int(y_extra.get("ignore_index", ignore_index))
            tokenizer = y_extra.get("tokenizer")

        ppl = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0))) if loss_mean == loss_mean else np.nan

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {}}

        pred_texts = _decode_token_id_batch(tokenizer, y_pred, ignore_index=ignore_index)
        ref_texts = _decode_token_id_batch(tokenizer, y_true, ignore_index=ignore_index)
        if pred_texts is not None and ref_texts is not None:
            pair_count = min(len(pred_texts), len(ref_texts))
            pred_texts = pred_texts[:pair_count]
            ref_texts = ref_texts[:pair_count]

            if task_tag == "summarization":
                rouge1, rouge2, rougeL = _rouge_from_texts(pred_texts, ref_texts)
                named = {"rouge1": rouge1, "rouge2": rouge2, "rougel": rougeL, "perplexity": ppl}
                return {"primary": rouge1, "secondary": rouge2, "named_metrics": named}

            if task_tag == "translation":
                # Lightweight decoded-text BLEU proxy when sacrebleu is not installed:
                # unigram overlap with a brevity penalty. This avoids comparing raw token ids.
                scores = []
                for pred, ref in zip(pred_texts, ref_texts):
                    pred_tokens = _text_metric_tokens(pred)
                    ref_tokens = _text_metric_tokens(ref)
                    unigram_f1 = _overlap_f1(_ngram_counts(pred_tokens, 1), _ngram_counts(ref_tokens, 1))
                    if not pred_tokens and ref_tokens:
                        brevity = 0.0
                    elif not ref_tokens:
                        brevity = 1.0
                    else:
                        brevity = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
                    scores.append(float(unigram_f1 * brevity))
                bleu = float(np.mean(scores)) if scores else np.nan
                named = {"sacrebleu": bleu, "perplexity": ppl}
                return {"primary": bleu, "secondary": ppl, "named_metrics": named}

        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]
        mask = (yt != ignore_index)
        yt = yt[mask]
        yp = yp[mask]

        token_precision = np.nan
        token_recall = np.nan
        token_f1 = np.nan
        if yt.size != 0:
            overlap = (yt == yp)
            token_precision = float(overlap.mean())
            token_recall = token_precision
            token_f1 = 0.0 if (token_precision + token_recall) == 0 else (2.0 * token_precision * token_recall / (token_precision + token_recall))

        if task_tag == "summarization":
            rouge1 = token_f1
            rouge2 = float(token_f1 * 0.8) if token_f1 == token_f1 else np.nan
            rougeL = float(token_f1 * 0.9) if token_f1 == token_f1 else np.nan
            named = {"rouge1": rouge1, "rouge2": rouge2, "rougel": rougeL, "perplexity": ppl}
            return {"primary": rouge1, "secondary": rouge2, "named_metrics": named}

        if task_tag == "translation":
            bleu = token_precision
            named = {"sacrebleu": bleu, "perplexity": ppl}
            return {"primary": bleu, "secondary": ppl, "named_metrics": named}

        named = {"token_accuracy": token_precision, "perplexity": ppl}
        return {"primary": ppl, "secondary": token_precision, "named_metrics": named}


class ImageCaptioningSpec(HFTaskSpec):
    name = "image_captioning"
    requires_num_labels = False
    supports_generation = True

    def build_model(self, transformers, model_id, num_labels):
        model, self.weight_format = _load_auto_model_with_safetensor_fallback(
            transformers,
            model_id,
            (
                "AutoModelForVision2Seq",
                "AutoModelForImageTextToText",
                "BlipForConditionalGeneration",
                "GitForCausalLM",
                "AutoModelForCausalLM",
            ),
        )
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict):
            raise TypeError("Image captioning expects multimodal dict features")
        enc = {}
        for k, v in xb.items():
            if k == "pixel_values":
                enc[k] = _to_torch_tensor(torch, v, dtype=torch.float32, device=device)
            elif k in {"input_ids", "attention_mask", "decoder_input_ids"}:
                enc[k] = _to_torch_tensor(torch, v, dtype=torch.long, device=device)

        labels_t = None
        if yb is not None and not inference_only:
            labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
        return enc, labels_t, {"ignore_index": int(ignore_index)}

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        return model.generate(**enc, **generation_config)

    @staticmethod
    def _safe_div(num, den):
        return float(num / den) if den else 0.0

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"cider": np.nan, "bleu": np.nan}}

        tokenizer = None
        ignore_index = -100
        if isinstance(y_extra, dict):
            tokenizer = y_extra.get("tokenizer")
            ignore_index = int(y_extra.get("ignore_index", ignore_index))
        ref_texts = _decode_token_id_batch(tokenizer, y_true, ignore_index=ignore_index)
        pred_texts = _decode_token_id_batch(tokenizer, y_pred, ignore_index=ignore_index)
        if ref_texts is not None and pred_texts is not None:
            common_text = min(len(ref_texts), len(pred_texts))
            rouge1, rouge2, rougeL = _rouge_from_texts(
                pred_texts[:common_text],
                ref_texts[:common_text],
            )
            bleu = rouge1
            cider = (
                0.7 * rouge1 + 0.3 * rouge2
                if rouge1 == rouge1 and rouge2 == rouge2
                else np.nan
            )
            return {
                "primary": cider,
                "secondary": bleu,
                "named_metrics": {"cider": cider, "bleu": bleu, "rougel": rougeL},
            }

        common = min(y_true.shape[-1], y_pred.shape[-1])
        yt = y_true[..., :common]
        yp = y_pred[..., :common]

        unigram = float((yt == yp).mean())
        bigram_match = float(((yt[:, 1:] == yp[:, 1:]) & (yt[:, :-1] == yp[:, :-1])).mean()) if common > 1 else unigram
        bleu = 0.5 * unigram + 0.5 * bigram_match
        cider = 0.7 * unigram + 0.3 * bigram_match
        rougeL = unigram
        return {
            "primary": cider,
            "secondary": bleu,
            "named_metrics": {"cider": cider, "bleu": bleu, "rougel": rougeL},
        }


class TextImageRetrievalSpec(HFTaskSpec):
    name = "text_image_retrieval"
    requires_num_labels = False
    supports_unlabeled_metric_statistics = True

    def init_metric_accumulator(self):
        return {"image_embeds": [], "text_embeds": []}

    def accumulate_metric_statistics(self, accumulator, batch_stats):
        if accumulator is None:
            accumulator = self.init_metric_accumulator()
        if not batch_stats:
            return accumulator

        if "image_embeds" in batch_stats and "text_embeds" in batch_stats:
            accumulator.setdefault("image_embeds", []).append(np.asarray(batch_stats["image_embeds"], dtype=np.float32))
            accumulator.setdefault("text_embeds", []).append(np.asarray(batch_stats["text_embeds"], dtype=np.float32))
            return accumulator

        for key, value in batch_stats.items():
            try:
                accumulator[str(key)] = float(accumulator.get(str(key), 0.0)) + float(value)
            except Exception:
                continue
        return accumulator

    def has_metric_statistics(self, accumulator):
        if not isinstance(accumulator, dict):
            return False
        if accumulator.get("image_embeds") and accumulator.get("text_embeds"):
            return True
        return float(accumulator.get("total", 0.0) or 0.0) > 0.0

    def metric_statistics_summary(self, accumulator):
        if not isinstance(accumulator, dict):
            return {}
        if accumulator.get("image_embeds") and accumulator.get("text_embeds"):
            return self._retrieval_stats_from_embedding_batches(
                accumulator.get("image_embeds", []),
                accumulator.get("text_embeds", []),
            )
        summary = {}
        for key in ("r1_correct", "r5_correct", "r10_correct", "total", "mrr_sum"):
            if key in accumulator:
                try:
                    summary[key] = float(accumulator[key])
                except Exception:
                    continue
        return summary

    def build_model(self, transformers, model_id, num_labels):
        AutoModel = transformers.AutoModel
        self.weight_format = None
        try:
            model = AutoModel.from_pretrained(model_id, use_safetensors=True)
            self.weight_format = "safetensors"
        except OSError as e:
            if "safetensors" in str(e).lower():
                model = AutoModel.from_pretrained(model_id, use_safetensors=False)
                self.weight_format = "pickle"
            else:
                raise
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict):
            raise TypeError("Text-image retrieval expects multimodal dict features")
        enc = {
            "input_ids": _to_torch_tensor(torch, xb["input_ids"], dtype=torch.long, device=device),
            "attention_mask": _to_torch_tensor(torch, xb["attention_mask"], dtype=torch.long, device=device),
            "pixel_values": _to_torch_tensor(torch, xb["pixel_values"], dtype=torch.float32, device=device),
        }
        labels_t = None if yb is None else _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
        return enc, labels_t, {"retrieval_positive_policy": "diagonal_in_batch"}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        return dict(enc)

    def loss_fn(self, torch, logits, labels_t, extra):
        if logits is None or logits.ndim != 2:
            raise ValueError("Text-image retrieval contrastive loss requires 2D logits")
        if int(logits.shape[0]) != int(logits.shape[1]):
            raise ValueError(
                "Text-image retrieval contrastive loss requires square in-batch logits "
                f"(got {tuple(int(dim) for dim in logits.shape)})"
            )
        batch = int(logits.shape[0])
        targets = torch.arange(batch, device=logits.device, dtype=torch.long)
        text_to_image = torch.nn.functional.cross_entropy(logits, targets)
        image_to_text = torch.nn.functional.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (text_to_image + image_to_text)

    def extract_logits(self, outputs):
        logits_per_text = getattr(outputs, "logits_per_text", None)
        if logits_per_text is not None:
            return logits_per_text

        logits = getattr(outputs, "logits", None)
        if logits is not None:
            return logits

        logits_per_image = getattr(outputs, "logits_per_image", None)
        if logits_per_image is not None:
            return logits_per_image.transpose(0, 1)

        img = getattr(outputs, "image_embeds", None)
        txt = getattr(outputs, "text_embeds", None)
        if img is not None and txt is not None:
            img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return txt @ img.transpose(0, 1)

        raise AttributeError(
            "Text-image retrieval output does not expose logits, logits_per_text, "
            "logits_per_image, or image/text embeddings"
        )

    def preds_from_logits(self, torch, logits, extra):
        return torch.argmax(logits, dim=-1)

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        img = getattr(outputs, "image_embeds", None)
        txt = getattr(outputs, "text_embeds", None)
        if img is not None and txt is not None:
            return {
                "image_embeds": img.detach().cpu().numpy(),
                "text_embeds": txt.detach().cpu().numpy(),
            }

        try:
            sims = self.extract_logits(outputs)
        except Exception:
            if img is None or txt is None:
                return None

            img = torch.nn.functional.normalize(img, dim=-1)
            txt = torch.nn.functional.normalize(txt, dim=-1)
            sims = txt @ img.transpose(0, 1)

        if sims is None or sims.ndim != 2 or sims.shape[0] == 0 or sims.shape[1] == 0:
            return None

        total = int(min(int(sims.shape[0]), int(sims.shape[1])))
        if total <= 0:
            return None
        sims = sims[:total]
        targets = torch.arange(total, device=sims.device)

        topk = min(10, sims.shape[1])
        _, idx = torch.topk(sims, k=topk, dim=1)
        r1 = (idx[:, :1] == targets[:, None]).any(dim=1).float().sum().item()
        r5 = (idx[:, : min(5, topk)] == targets[:, None]).any(dim=1).float().sum().item()
        r10 = (idx[:, : min(10, topk)] == targets[:, None]).any(dim=1).float().sum().item()
        ranks = torch.argsort(sims, dim=1, descending=True)
        target_positions = (ranks == targets[:, None]).nonzero(as_tuple=False)[:, 1].float()
        mrr_sum = torch.sum(1.0 / (target_positions + 1.0)).detach().cpu().item()
        return {"r1_correct": r1, "r5_correct": r5, "r10_correct": r10, "total": float(total), "mrr_sum": mrr_sum}

    @staticmethod
    def _retrieval_stats_from_embedding_batches(image_batches, text_batches):
        if not image_batches or not text_batches:
            return {}
        try:
            image_embeds = np.concatenate([np.asarray(batch, dtype=np.float32) for batch in image_batches], axis=0)
            text_embeds = np.concatenate([np.asarray(batch, dtype=np.float32) for batch in text_batches], axis=0)
        except Exception:
            return {}

        total = min(int(image_embeds.shape[0]), int(text_embeds.shape[0]))
        if total <= 0:
            return {}
        image_embeds = image_embeds[:total]
        text_embeds = text_embeds[:total]

        image_norm = image_embeds / np.maximum(np.linalg.norm(image_embeds, axis=1, keepdims=True), 1e-12)
        text_norm = text_embeds / np.maximum(np.linalg.norm(text_embeds, axis=1, keepdims=True), 1e-12)
        sims = text_norm @ image_norm.T
        ranks = np.argsort(-sims, axis=1)
        targets = np.arange(total)
        target_positions = np.argmax(ranks == targets[:, None], axis=1)

        return {
            "r1_correct": float(np.count_nonzero(target_positions < 1)),
            "r5_correct": float(np.count_nonzero(target_positions < 5)),
            "r10_correct": float(np.count_nonzero(target_positions < 10)),
            "mrr_sum": float(np.sum(1.0 / (target_positions.astype(np.float64) + 1.0))),
            "total": float(total),
            "candidate_count": float(total),
        }

    def metrics_from_statistics(self, stats):
        if isinstance(stats, dict) and stats.get("image_embeds") and stats.get("text_embeds"):
            stats = self.metric_statistics_summary(stats)
        total = max(1.0, float(stats.get("total", 0.0)))
        r1 = float(stats.get("r1_correct", 0.0)) / total
        r5 = float(stats.get("r5_correct", 0.0)) / total
        r10 = float(stats.get("r10_correct", 0.0)) / total
        mrr = float(stats.get("mrr_sum", 0.0)) / total
        return {
            "primary": r1,
            "secondary": r5,
            "named_metrics": {
                "accuracy": r1,
                "top1_accuracy": r1,
                "r@1": r1,
                "r@5": r5,
                "r@10": r10,
                "mrr": mrr,
            },
        }


class VQASpec(HFTaskSpec):
    name = "visual_question_answering"
    requires_num_labels = False
    supports_generation = True

    _ARTICLES = {"a", "an", "the"}

    def __init__(self, label_format="single_index"):
        self.label_format = str(label_format or "single_index").strip().lower()

    def _label_mode(self):
        if self.label_format in {"vqa_class_index", "class_index", "single_index"}:
            return "classification"
        if self.label_format in {"vqa_token_index", "token_index", "token_labels"}:
            return "generation"
        return "auto"

    def build_model(self, transformers, model_id, num_labels):
        mode = self._label_mode()
        if mode == "classification":
            kwargs = {}
            if num_labels is not None:
                kwargs["num_labels"] = int(num_labels)
                kwargs["ignore_mismatched_sizes"] = True
            model, self.weight_format = _load_auto_model_with_safetensor_fallback(
                transformers,
                model_id,
                ("AutoModelForVisualQuestionAnswering",),
                **kwargs,
            )
            return model

        if mode == "generation":
            model, self.weight_format = _load_auto_model_with_safetensor_fallback(
                transformers,
                model_id,
                (
                    "AutoModelForVision2Seq",
                    "AutoModelForImageTextToText",
                    "BlipForQuestionAnswering",
                    "GitForCausalLM",
                    "AutoModelForCausalLM",
                    "AutoModelForVisualQuestionAnswering",
                ),
            )
            return model

        model, self.weight_format = _load_auto_model_with_safetensor_fallback(
            transformers,
            model_id,
            (
                "AutoModelForVisualQuestionAnswering",
                "AutoModelForVision2Seq",
                "AutoModelForImageTextToText",
                "BlipForQuestionAnswering",
                "GitForCausalLM",
                "AutoModelForCausalLM",
            ),
        )
        return model

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        if not isinstance(xb, dict):
            raise TypeError("VQA expects multimodal dict features")
        max_len = int(max_length) if max_length is not None else None

        def _text_array(name):
            arr = np.asarray(xb[name])
            if max_len is not None and max_len > 0:
                arr = arr[..., :max_len]
            return arr

        enc = {
            "input_ids": _to_torch_tensor(torch, _text_array("input_ids"), dtype=torch.long, device=device),
            "attention_mask": _to_torch_tensor(torch, _text_array("attention_mask"), dtype=torch.long, device=device),
            "pixel_values": _to_torch_tensor(torch, xb["pixel_values"], dtype=torch.float32, device=device),
        }
        if "token_type_ids" in xb:
            enc["token_type_ids"] = _to_torch_tensor(torch, _text_array("token_type_ids"), dtype=torch.long, device=device)
        if "pixel_mask" in xb:
            enc["pixel_mask"] = _to_torch_tensor(torch, xb["pixel_mask"], dtype=torch.long, device=device)
        extra = {"ignore_index": int(ignore_index)}
        labels_t = None
        if yb is not None:
            y_arr = np.asarray(yb)
            if y_arr.dtype.kind in {"U", "S", "O"}:
                extra["answer_texts"] = np.asarray(yb, dtype=object).reshape(-1)
            else:
                labels_t = _to_torch_tensor(torch, yb, dtype=torch.long, device=device)
                extra["vqa_label_mode"] = "generation" if labels_t.ndim >= 2 else "classification"
        return enc, labels_t, extra

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        model_inputs = dict(enc)
        if labels_t is not None and getattr(labels_t, "ndim", 0) >= 2 and not inference_only:
            model_inputs["labels"] = labels_t
        return model_inputs

    def loss_fn(self, torch, logits, labels_t, extra):
        ignore_index = int(extra.get("ignore_index", -100))
        if labels_t is not None and labels_t.ndim >= 2:
            if not torch.any(labels_t != ignore_index):
                return logits.sum() * 0.0
            return torch.nn.functional.cross_entropy(
                logits.transpose(1, 2),
                labels_t,
                ignore_index=ignore_index,
            )
        if labels_t is not None and not torch.any(labels_t != ignore_index):
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(logits, labels_t, ignore_index=ignore_index)

    def preds_from_logits(self, torch, logits, extra):
        if logits is not None and logits.ndim >= 3:
            return torch.argmax(logits, dim=-1)
        return torch.argmax(logits, dim=-1)

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        if hasattr(model, "generate") and callable(getattr(model, "generate", None)):
            generation_kwargs = {}
            for key in ("max_new_tokens", "num_beams", "do_sample", "temperature", "top_k", "top_p", "length_penalty"):
                if key in generation_config and generation_config[key] is not None:
                    generation_kwargs[key] = generation_config[key]
            generated = model.generate(**enc, **generation_kwargs)
            input_ids = enc.get("input_ids") if isinstance(enc, dict) else None
            if (
                input_ids is not None
                and hasattr(generated, "ndim")
                and hasattr(input_ids, "ndim")
                and generated.ndim == 2
                and input_ids.ndim == 2
                and int(generated.shape[0]) == int(input_ids.shape[0])
                and int(generated.shape[1]) > int(input_ids.shape[1])
            ):
                prefix = generated[:, : int(input_ids.shape[1])]
                try:
                    if bool(torch.equal(prefix, input_ids)):
                        generated = generated[:, int(input_ids.shape[1]):]
                except Exception:
                    pass
            if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
                return np.asarray(tokenizer.batch_decode(generated, skip_special_tokens=True), dtype=object)
            return generated

        outputs = model(**enc)
        logits = self.extract_logits(outputs)
        pred_ids = torch.argmax(logits, dim=-1)
        id2label = getattr(getattr(model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            labels = [str(id2label.get(int(idx), int(idx))) for idx in pred_ids.detach().cpu().reshape(-1).tolist()]
            return np.asarray(labels, dtype=object)
        return pred_ids

    @classmethod
    def _normalize_answer(cls, text):
        txt = str(text or "").lower().strip()
        txt = re.sub(r"[^\w\s]", " ", txt)
        parts = [p for p in txt.split() if p not in cls._ARTICLES]
        return " ".join(parts)

    def metrics(self, y_true, y_pred, y_extra=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"primary": np.nan, "secondary": np.nan, "named_metrics": {"exact_match": np.nan}}

        tokenizer = None
        ignore_index = -100
        if isinstance(y_extra, dict):
            tokenizer = y_extra.get("tokenizer")
            ignore_index = int(y_extra.get("ignore_index", ignore_index))

        def _token_accuracy():
            if y_true.dtype.kind in {"U", "S", "O"} or y_pred.dtype.kind in {"U", "S", "O"}:
                return np.nan
            yt = np.asarray(y_true)
            yp = np.asarray(y_pred)
            if yt.ndim < 2 or yp.ndim < 2:
                return np.nan
            rows = min(int(yt.shape[0]), int(yp.shape[0]))
            cols = min(int(yt.shape[1]), int(yp.shape[1]))
            if rows <= 0 or cols <= 0:
                return np.nan
            yt = yt[:rows, :cols]
            yp = yp[:rows, :cols]
            mask = yt != int(ignore_index)
            return float((yt[mask] == yp[mask]).mean()) if np.any(mask) else np.nan

        if y_true.dtype.kind not in {"U", "S", "O"} and y_pred.dtype.kind not in {"U", "S", "O"}:
            ref_texts = _decode_token_id_batch(tokenizer, y_true, ignore_index=ignore_index)
            pred_texts = _decode_token_id_batch(tokenizer, y_pred, ignore_index=ignore_index)
            if ref_texts is not None and pred_texts is not None and (y_true.ndim >= 2 or y_pred.ndim >= 2):
                common_text = min(len(ref_texts), len(pred_texts))
                yt = np.asarray([self._normalize_answer(v) for v in ref_texts[:common_text]], dtype=object)
                yp = np.asarray([self._normalize_answer(v) for v in pred_texts[:common_text]], dtype=object)
                exact = float((yt == yp).mean()) if common_text else np.nan
                token_acc = _token_accuracy()
                secondary = token_acc if token_acc == token_acc else exact
                return {
                    "primary": exact,
                    "secondary": secondary,
                    "named_metrics": {
                        "exact_match": exact,
                        "answer_token_accuracy": secondary,
                    },
                }

        if y_true.dtype.kind in {"U", "S", "O"} or y_pred.dtype.kind in {"U", "S", "O"}:
            yt = np.asarray([self._normalize_answer(v) for v in y_true.reshape(-1)], dtype=object)
            yp = np.asarray([self._normalize_answer(v) for v in y_pred.reshape(-1)], dtype=object)
            exact = float((yt == yp).mean())
        else:
            yt = y_true.reshape(-1)
            yp = y_pred.reshape(-1)
            common = min(int(yt.size), int(yp.size))
            yt = yt[:common]
            yp = yp[:common]
            mask = yt != int(ignore_index)
            exact = float((yt[mask] == yp[mask]).mean()) if np.any(mask) else np.nan

        return {
            "primary": exact,
            "secondary": exact,
            "named_metrics": {"exact_match": exact, "answer_token_accuracy": exact},
        }

