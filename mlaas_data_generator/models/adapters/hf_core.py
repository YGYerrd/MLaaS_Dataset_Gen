
from mlaas_data_generator.terminal import debug_print
import contextlib
from itertools import islice
import os
import time
import numpy as np

from .hf_task import SequenceClassificationSpec
from .hf_cache import get_cached_tokenizer
from ...data.prefetch import prefetch_batches


class HFCore:
    """
    Framework-agnostic HF training loop wrapper.

    Loader schema support:
      - xs can be dict-of-arrays (preferred, from loader preprocessors)
      - xs can be list of raw texts or list-of-token-lists (legacy)
    """

    def __init__(
        self,
        model_id,
        num_labels=None,
        max_length=128,
        batch_size=16,
        device=None,
        mixed_precision=None,
        precision_type="fp16",
        task_spec=None,
        label_pad_value=-100,
        generation_config=None,
        task_tag=None,
        model_revision=None,
        mobilebert_head_normalization=None,
        max_grad_norm=None,
        dataset_id=None,
    ):
        if os.name != "nt":
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            import torch
            import transformers
        except Exception as e:
            raise ImportError(
                "HF adapters require 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from e

        self.torch = torch
        self.transformers = transformers

        self.model_id = model_id
        self.dataset_id = dataset_id
        self.model_revision = model_revision
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        if self.max_grad_norm is not None and (not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")
        self.model_source = model_id
        if model_revision and not os.path.exists(str(model_id)):
            try:
                from huggingface_hub import snapshot_download
                self.model_source = snapshot_download(repo_id=str(model_id), revision=str(model_revision))
            except Exception as exc:
                raise RuntimeError(f"Unable to resolve pinned Hugging Face model revision {model_id}@{model_revision}") from exc
        self.requested_max_length = int(max_length)
        self.max_length = int(max_length)
        self.model_text_max_length = None
        self.max_length_adjusted = False
        self.batch_size = int(batch_size)
        self.label_pad_value = int(label_pad_value)
        self.mixed_precision = bool(mixed_precision)
        self.precision_type = self._normalize_precision_type(precision_type)
        self.requested_mixed_precision = bool(mixed_precision)
        self.requested_precision_type = self.precision_type

        self.device = self._resolve_device(device)

        self.task_spec = task_spec or SequenceClassificationSpec()
        self.generation_config = self._resolve_generation_config(generation_config)
        self.task_tag = (task_tag or "").strip().lower().replace("-", "_") or None
        tokenizer_required = bool(getattr(self.task_spec, "requires_tokenizer", True))
        if tokenizer_required:
            self.tokenizer, self.tokenizer_load_s, self.tokenizer_cache_hit = get_cached_tokenizer(
                hf_model_id=self.model_source,
                task=getattr(self.task_spec, "name", None),
                device=self.device,
                transformers_module=transformers,
            )
        else:
            self.tokenizer = None
            self.tokenizer_load_s = 0.0
            self.tokenizer_cache_hit = True

        self.model = None
        self.weight_format = None
        self.model_load_s = 0.0
        self.model_cache_hit = False
        self.autocast_enabled = False
        self.autocast_dtype = None
        self.grad_scaler = None
        self.effective_mixed_precision = False
        self.effective_precision_type = "fp32"
        self.precision_fallback_reason = None
        self.gradient_checkpointing_enabled = False
        needs_num_labels = bool(getattr(self.task_spec, "requires_num_labels", True))
        if num_labels is not None or not needs_num_labels:
            model_load_start = time.perf_counter()
            self.model = self.task_spec.build_model(transformers, self.model_source, num_labels)
            saved_normalization = getattr(
                getattr(self.model, "config", None), "mobilebert_head_normalization", None)
            if (saved_normalization not in {None, "none"} and mobilebert_head_normalization is not None
                    and mobilebert_head_normalization != saved_normalization):
                raise ValueError("MobileBERT normalization differs from the saved checkpoint contract")
            if mobilebert_head_normalization is None:
                mobilebert_head_normalization = saved_normalization
            if mobilebert_head_normalization is not None:
                from .mobilebert_stability import apply_mobilebert_head_normalization
                apply_mobilebert_head_normalization(self.model, mobilebert_head_normalization)
            self.model_load_s = float(time.perf_counter() - model_load_start)
            self.weight_format = getattr(self.task_spec, "weight_format", None)
            self.model.to(self.device)
            self._configure_memory_optimizations()
            self.sync_effective_max_length()

    def _device_type(self):
        device = self.device
        if hasattr(device, "type"):
            return str(device.type).lower()
        return str(device).lower()

    def _task_name(self):
        return str(getattr(self.task_spec, "name", "") or "").strip().lower()

    def _should_enable_mixed_precision(self):
        if self._device_type() != "cuda":
            return False
        if self.mixed_precision is False:
            return False
        return True

    @staticmethod
    def _tensor_to_numpy(value, *, dtype=None):
        if hasattr(value, "detach"):
            value = value.detach()
        tensor_dtype = str(getattr(value, "dtype", "")).lower()
        if "bfloat16" in tensor_dtype and hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.asarray(value)
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

    @staticmethod
    def _normalize_precision_type(value):
        text = str(value or "fp16").strip().lower()
        return text if text in {"fp16", "bf16", "fp32"} else "fp16"

    def _cuda_bf16_supported(self):
        cuda = getattr(self.torch, "cuda", None)
        probe = getattr(cuda, "is_bf16_supported", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:
                return False
        return False

    def _disable_mixed_precision(self, reason):
        self.autocast_enabled = False
        self.autocast_dtype = None
        self.grad_scaler = None
        self.effective_mixed_precision = False
        self.effective_precision_type = "fp32"
        self.precision_fallback_reason = str(reason) if reason else None
        self.precision_fallback_error = None

    def _configure_precision_mode(self):
        self.requested_mixed_precision = bool(self.mixed_precision)
        self.requested_precision_type = self._normalize_precision_type(getattr(self, "precision_type", "fp16"))
        self.precision_type = self.requested_precision_type
        self._disable_mixed_precision(None)
        from ...precision_policy import fp32_policy_reason
        model = getattr(self, "model", None)
        # Autocast controls compute precision, not parameter storage. Some HF
        # checkpoints load in FP16; GradScaler cannot unscale their FP16 grads.
        # Normalize before family/base snapshots as well as before training.
        if model is not None:
            model.float()
        family = getattr(getattr(model, "config", None), "model_type", None)
        policy_reason = fp32_policy_reason(getattr(self, "model_id", None), family,
                                          dataset_id=getattr(self, "dataset_id", None))
        if policy_reason:
            if model is not None:
                model.float()
            self._disable_mixed_precision(policy_reason)
            return
        if self.requested_precision_type == "fp32":
            self._disable_mixed_precision("explicit_fp32")
            return
        if not self._should_enable_mixed_precision():
            if self.requested_mixed_precision and self._device_type() != "cuda":
                self.precision_fallback_reason = "mixed_precision_requires_cuda"
            return

        dtype_name = self.requested_precision_type
        if dtype_name == "bf16":
            dtype = getattr(self.torch, "bfloat16", None)
            if dtype is None or not self._cuda_bf16_supported():
                self._disable_mixed_precision("bf16_unsupported_fallback_fp32")
                return
        else:
            dtype = getattr(self.torch, "float16", None)
            if dtype is None:
                self._disable_mixed_precision("fp16_unsupported_fallback_fp32")
                return

        self.autocast_dtype = dtype
        self.autocast_enabled = True
        self.effective_mixed_precision = True
        self.effective_precision_type = dtype_name
        if dtype_name == "fp16":
            try:
                self.grad_scaler = self.torch.amp.GradScaler("cuda")
            except Exception:
                try:
                    self.grad_scaler = self.torch.cuda.amp.GradScaler()
                except Exception:
                    self.grad_scaler = None

    def _precision_retryable_exception(self, exc):
        text = f"{type(exc).__name__}: {exc}".lower()
        markers = ("bfloat16", "float16", "half", "autocast", "amp", "unsupported", "not implemented", "dtype")
        return any(marker in text for marker in markers)

    def _should_enable_gradient_checkpointing(self):
        return self._task_name() in {
            "image_detection",
            "image_segmentation",
            "image_captioning",
            "text_image_retrieval",
            "visual_question_answering",
            "causal_lm_generation",
            "seq2seq_generation",
        }

    def _make_autocast_context(self):
        if not getattr(self, "autocast_enabled", False):
            return contextlib.nullcontext()
        torch = self.torch
        try:
            return torch.autocast(device_type="cuda", dtype=getattr(self, "autocast_dtype", None), enabled=True)
        except Exception:
            return contextlib.nullcontext()

    def _run_with_precision_context(self, fn):
        if not getattr(self, "autocast_enabled", False):
            return fn()
        try:
            with self._make_autocast_context():
                return fn()
        except Exception as exc:
            if not self._precision_retryable_exception(exc):
                raise
            self._fallback_to_fp32(f"runtime_unsupported_{self.requested_precision_type}")
            self.precision_fallback_error = f"{type(exc).__name__}: {exc}"
            return fn()

    def _fallback_to_fp32(self, reason):
        # Keep scaling until training finishes: previously accumulated gradients
        # are still scaled even after autocast is disabled.
        scaler = getattr(self, "grad_scaler", None)
        self._disable_mixed_precision(reason)
        self.grad_scaler = scaler

    def _validate_eval_tensors(self, logits, loss=None, *, require_loss=False):
        if logits is not None and not bool(self.torch.isfinite(logits).all()):
            raise ValueError("nonfinite_evaluation_logits: evaluation quality is invalid")
        if require_loss and loss is None:
            raise ValueError("missing_evaluation_loss: supervised evaluation requires a loss")
        if loss is not None and not bool(self.torch.isfinite(loss)):
            raise ValueError("nonfinite_evaluation_loss: evaluation quality is invalid")

    def _configure_memory_optimizations(self):
        if self.model is None:
            return

        from .rocm_batchnorm import configure_native_batchnorm
        native_bn_count = configure_native_batchnorm(self.model, self.torch, self.device)
        if native_bn_count:
            print(f"[HFCore] Windows ROCm: using native GPU BatchNorm for {native_bn_count} layers")

        from .rocm_dpt_layout import configure_rocm_convolution_layout, convolution_layout_family
        layout_count = configure_rocm_convolution_layout(self.model, self.torch, self.device)
        if layout_count:
            family = convolution_layout_family(self.model)
            print(f"[HFCore] Windows ROCm: using contiguous {family} convolution inputs for {layout_count} layers")

        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            for attr, value in (("output_hidden_states", False), ("output_attentions", False)):
                if hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, value)
                    except Exception:
                        pass
            if hasattr(cfg, "use_cache"):
                try:
                    cfg.use_cache = False
                except Exception:
                    pass

        if self._should_enable_gradient_checkpointing() and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                self.gradient_checkpointing_enabled = True
            except Exception:
                self.gradient_checkpointing_enabled = False

        self._configure_precision_mode()

    @contextlib.contextmanager
    def _generation_inference_mode(self):
        model = getattr(self, "model", None)
        if model is None:
            yield
            return

        cfg = getattr(model, "config", None)
        prev_use_cache = None
        had_use_cache = False
        if cfg is not None and hasattr(cfg, "use_cache"):
            had_use_cache = True
            prev_use_cache = getattr(cfg, "use_cache", None)
            try:
                cfg.use_cache = True
            except Exception:
                had_use_cache = False

        grad_ckpt_disable_called = False
        grad_ckpt_prev = bool(getattr(self, "gradient_checkpointing_enabled", False))
        if grad_ckpt_prev and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
                self.gradient_checkpointing_enabled = False
                grad_ckpt_disable_called = True
            except Exception:
                grad_ckpt_disable_called = False

        try:
            yield
        finally:
            if had_use_cache:
                try:
                    cfg.use_cache = prev_use_cache
                except Exception:
                    pass
            if grad_ckpt_disable_called and hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable()
                    self.gradient_checkpointing_enabled = grad_ckpt_prev
                except Exception:
                    pass

    def _qos_startup(self):
        return {
            "tokenizer_load_s": float(self.tokenizer_load_s),
            "model_load_s": float(self.model_load_s),
            "cold_start_time": float(self.tokenizer_load_s + self.model_load_s),
            "tokenizer_cache_hit": bool(self.tokenizer_cache_hit),
            "model_cache_hit": bool(self.model_cache_hit),
            "mixed_precision_requested": bool(getattr(self, "requested_mixed_precision", False)),
            "precision_type_requested": str(getattr(self, "requested_precision_type", "fp16")),
            "mixed_precision_effective": bool(getattr(self, "effective_mixed_precision", False)),
            "precision_type_effective": str(getattr(self, "effective_precision_type", "fp32")),
            "precision_fallback_reason": getattr(self, "precision_fallback_reason", None),
            "precision_fallback_error": getattr(self, "precision_fallback_error", None),
        }

    @staticmethod
    def _concat_or_pad_batches(batches, *, empty_dtype="int64", pad_value=0):
        if not batches:
            return np.asarray([], dtype=empty_dtype)

        arrays = []
        for batch in batches:
            arr = np.asarray(batch)
            if arr.dtype == object:
                try:
                    return np.concatenate([np.asarray(item, dtype=object) for item in batches], axis=0)
                except Exception:
                    return np.asarray(batches, dtype=object)
            arrays.append(arr)

        try:
            return np.concatenate(arrays, axis=0)
        except ValueError:
            pass
        except Exception:
            return np.asarray(batches, dtype=object)

        ndim = arrays[0].ndim
        if ndim <= 1 or any(arr.ndim != ndim for arr in arrays):
            return np.asarray(batches, dtype=object)

        target_tail = []
        for axis in range(1, ndim):
            target_tail.append(max(int(arr.shape[axis]) for arr in arrays))

        padded = []
        for arr in arrays:
            target_shape = (int(arr.shape[0]), *target_tail)
            out = np.full(target_shape, pad_value, dtype=arr.dtype)
            slices = tuple(slice(0, size) for size in arr.shape)
            out[slices] = arr
            padded.append(out)

        return np.concatenate(padded, axis=0)

    def _resolve_generation_config(self, generation_config):
        defaults = {
            "max_new_tokens": 64,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "length_penalty": 1.0,
        }
        cfg = dict(defaults)
        if isinstance(generation_config, dict):
            cfg.update({k: generation_config[k] for k in defaults.keys() if k in generation_config and generation_config[k] is not None})
        cfg["max_new_tokens"] = int(cfg["max_new_tokens"])
        cfg["num_beams"] = int(cfg["num_beams"])
        cfg["do_sample"] = bool(cfg["do_sample"])
        cfg["temperature"] = float(cfg["temperature"])
        cfg["top_k"] = int(cfg["top_k"])
        cfg["top_p"] = float(cfg["top_p"])
        cfg["length_penalty"] = float(cfg["length_penalty"])

        if not cfg["do_sample"]:
            cfg["temperature"] = 1.0
        if cfg["num_beams"] > 1:
            cfg["do_sample"] = False

        return cfg

    def _resolve_device(self, device):
        torch = self.torch

        if device is not None:
            return device

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @staticmethod
    def _bounded_positive_int(value):
        try:
            parsed = int(value)
        except Exception:
            return None
        if parsed <= 0 or parsed >= 1_000_000:
            return None
        return parsed

    @classmethod
    def _config_text_length_limits(cls, config):
        if config is None:
            return []
        configs = [config]
        for attr in ("text_config", "encoder", "decoder"):
            nested = getattr(config, attr, None)
            if nested is not None:
                configs.append(nested)

        limits = []
        for cfg in configs:
            for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
                candidate = cls._bounded_positive_int(getattr(cfg, attr, None))
                if candidate is not None:
                    limits.append(candidate)
        return limits

    def sync_effective_max_length(self):
        limits = []
        tokenizer_limit = self._bounded_positive_int(getattr(self.tokenizer, "model_max_length", None))
        if tokenizer_limit is not None:
            limits.append(tokenizer_limit)
        limits.extend(self._config_text_length_limits(getattr(self.model, "config", None)))

        if not limits:
            return self.max_length

        self.model_text_max_length = int(min(limits))
        if self.max_length > self.model_text_max_length:
            self.max_length = int(self.model_text_max_length)
            self.max_length_adjusted = True
        return self.max_length

    def _ensure_left_padding_for_decoder_only_generation(self):
        if not bool(getattr(self.task_spec, "supports_generation", False)):
            return
        if self.tokenizer is None:
            return
        model_cfg = getattr(self.model, "config", None)
        is_encoder_decoder = bool(getattr(model_cfg, "is_encoder_decoder", False))
        if not is_encoder_decoder and getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

    @staticmethod
    def _batch_prefetch_depth(xs):
        from ...data.image_store import DiskArrayStore

        values = xs.values() if isinstance(xs, dict) else (xs,)
        if not any(isinstance(value, DiskArrayStore) for value in values):
            return 0
        depth = int(os.environ.get("MLAAS_PREFETCH_BATCHES", "2"))
        if not 0 <= depth <= 8:
            raise ValueError("MLAAS_PREFETCH_BATCHES must be between 0 and 8")
        return depth

    def _batch_iter(self, xs, ys, *, indices=None, batch_size=None, contrastive=False):
        bs = self.batch_size if batch_size is None else batch_size

        def take_batch(value, start, stop):
            if value is None:
                return None
            if indices is None:
                return value[start:stop]
            batch_indices = indices[start:stop].tolist()
            if isinstance(value, tuple):
                return tuple(value[index] for index in batch_indices)
            if isinstance(value, list):
                return [value[index] for index in batch_indices]
            return value[batch_indices]

        def ranges(n):
            if contrastive:
                from ...contrastive_batch import contrastive_ranges
                return contrastive_ranges(n, bs)
            return ((i, min(n, i + bs)) for i in range(0, n, bs))

        if isinstance(xs, dict):
            n = len(next(iter(xs.values()))) if xs else 0
            for i, stop in ranges(n):
                xb = {k: take_batch(v, i, stop) for k, v in xs.items()}
                yb = take_batch(ys, i, stop)
                xb, span = self._trim_batch_sequence_padding(xb)
                # Seq2seq targets have their own sequence axis.  Trimming them
                # to the source attention span can silently discard supervised
                # target tokens when examples in a batch have different source
                # lengths.  Causal-LM and token-classification labels remain
                # positionally aligned with their inputs and still use the span.
                label_span = None if self._targets_have_independent_sequence_axis() else span
                yb = self._trim_label_padding(yb, span=label_span)
                yield xb, yb
            return

        n = len(xs)
        for i, stop in ranges(n):
            xb = take_batch(xs, i, stop)
            yb = take_batch(ys, i, stop)
            yield xb, yb

    @staticmethod
    def _batch_item_count(xb):
        if isinstance(xb, dict):
            if not xb:
                return 0
            return len(next(iter(xb.values())))
        return len(xb)

    @staticmethod
    def _active_column_span(mask, *, inactive_value=0):
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape[1] == 0:
            return None
        active_cols = np.any(mask != inactive_value, axis=0)
        active_idx = np.flatnonzero(active_cols)
        if active_idx.size == 0:
            return None
        return int(active_idx[0]), int(active_idx[-1]) + 1

    def _trim_batch_sequence_padding(self, xb):
        if not isinstance(xb, dict):
            return xb, None
        attention_mask = xb.get("attention_mask")
        span = self._active_column_span(attention_mask, inactive_value=0)
        if span is None:
            return xb, None
        start, end = span
        width = int(np.asarray(attention_mask).shape[1])
        if start == 0 and end == width:
            return xb, span

        trimmed = {}
        for key, value in xb.items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 2
                and value.shape[0] == attention_mask.shape[0]
                and value.shape[1] == attention_mask.shape[1]
            ):
                trimmed[key] = value[:, start:end]
            else:
                trimmed[key] = value
        return trimmed, span

    def _trim_label_padding(self, yb, *, span=None):
        if not isinstance(yb, np.ndarray) or yb.ndim < 2 or yb.shape[1] == 0:
            return yb
        if span is not None:
            start, end = span
            if 0 <= start < end <= int(yb.shape[1]):
                return yb[:, start:end]
        span = self._active_column_span(yb, inactive_value=self.label_pad_value)
        if span is None:
            return yb
        start, end = span
        if start == 0 and end == int(yb.shape[1]):
            return yb
        return yb[:, start:end]

    def _targets_have_independent_sequence_axis(self):
        """Whether source-input trimming must not change the target sequence."""
        return str(getattr(getattr(self, "task_spec", None), "name", "") or "") == "seq2seq_generation"

    def _labels_to_numpy(self, labels_t):
        if labels_t is None:
            return None
        if hasattr(labels_t, "detach"):
            return self._tensor_to_numpy(labels_t)
        if isinstance(labels_t, (list, tuple)):
            converted = []
            for item in labels_t:
                if isinstance(item, dict):
                    converted.append(
                        {
                            k: (self._tensor_to_numpy(v) if hasattr(v, "detach") else np.asarray(v))
                            for k, v in item.items()
                        }
                    )
                else:
                    converted.append(self._tensor_to_numpy(item) if hasattr(item, "detach") else np.asarray(item))
            return np.asarray(converted, dtype=object)
        return np.asarray(labels_t)

    def _count_supervised_tokens(self, labels_t, ignore_index):
        arr = self._labels_to_numpy(labels_t)
        if arr is None:
            return 0
        arr = np.asarray(arr)
        if arr.ndim < 2 or arr.dtype == object:
            return 0
        try:
            return int(np.count_nonzero(arr != int(ignore_index)))
        except Exception:
            try:
                coerced = arr.astype("int64", copy=False)
            except Exception:
                return 0
            return int(np.count_nonzero(coerced != int(ignore_index)))

    def _debug_shape(self, value):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return tuple(int(dim) for dim in shape)
            except Exception:
                return shape
        try:
            arr = np.asarray(value, dtype=object)
            return tuple(int(dim) for dim in arr.shape)
        except Exception:
            return None

    def _debug_preview(self, value, max_items=12):
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()

        sample = value
        while isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]

        if isinstance(sample, (list, tuple)):
            return list(sample[:max_items])
        return sample

    def debug_first_processed_batch(self, xs, ys, inference_only=False):
        torch = self.torch
        batch_iter = self._batch_iter(xs, ys)
        xb, yb = next(batch_iter)
        enc, labels_t, _ = self.task_spec.encode_batch(
            self.tokenizer,
            xb,
            yb,
            self.max_length,
            torch,
            self.device,
            ignore_index=self.label_pad_value,
            inference_only=bool(inference_only),
        )

        input_ids = enc.get("input_ids") if isinstance(enc, dict) else None
        attention_mask = enc.get("attention_mask") if isinstance(enc, dict) else None

        finite_ok = True
        finite_details = {}
        if isinstance(enc, dict):
            for key, tensor in enc.items():
                if hasattr(tensor, "dtype") and hasattr(torch, "is_floating_point") and torch.is_floating_point(tensor):
                    finite_value = bool(torch.isfinite(tensor).all().detach().cpu().item())
                    finite_details[key] = finite_value
                    finite_ok = finite_ok and finite_value
        if labels_t is not None and hasattr(labels_t, "dtype") and torch.is_floating_point(labels_t):
            labels_finite = bool(torch.isfinite(labels_t).all().detach().cpu().item())
            finite_details["labels"] = labels_finite
            finite_ok = finite_ok and labels_finite

        nested_object_keys = []
        if isinstance(xb, dict):
            for key, value in xb.items():
                arr = np.asarray(value, dtype=object)
                if arr.dtype == object:
                    nested_object_keys.append(str(key))

        token_source = xb.get("tokens") if isinstance(xb, dict) and "tokens" in xb else input_ids
        tag_source = xb.get("ner_tags") if isinstance(xb, dict) and "ner_tags" in xb else (yb if yb is not None else labels_t)

        return {
            "input_ids_shape": self._debug_shape(input_ids),
            "attention_mask_shape": self._debug_shape(attention_mask),
            "labels_shape": self._debug_shape(labels_t),
            "token_example": self._debug_preview(token_source),
            "ner_tags_example": self._debug_preview(tag_source),
            "finite_ok": bool(finite_ok),
            "finite_details": finite_details,
            "nested_object_keys": nested_object_keys,
        }

    def _debug_shape(self, value):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                return tuple(int(dim) for dim in shape)
            except Exception:
                return shape
        try:
            arr = np.asarray(value, dtype=object)
            return tuple(int(dim) for dim in arr.shape)
        except Exception:
            return None

    def _debug_preview(self, value, max_items=12):
        if value is None:
            return None

        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()

        sample = value
        while isinstance(sample, (list, tuple)) and sample:
            sample = sample[0]

        if isinstance(sample, (list, tuple)):
            return list(sample[:max_items])
        return sample

    def debug_first_processed_batch(self, xs, ys, inference_only=False):
        torch = self.torch
        batch_iter = self._batch_iter(xs, ys)
        xb, yb = next(batch_iter)
        enc, labels_t, _ = self.task_spec.encode_batch(
            self.tokenizer,
            xb,
            yb,
            self.max_length,
            torch,
            self.device,
            ignore_index=self.label_pad_value,
            inference_only=bool(inference_only),
        )

        input_ids = enc.get("input_ids") if isinstance(enc, dict) else None
        attention_mask = enc.get("attention_mask") if isinstance(enc, dict) else None

        finite_ok = True
        finite_details = {}
        if isinstance(enc, dict):
            for key, tensor in enc.items():
                if hasattr(tensor, "dtype") and hasattr(torch, "is_floating_point") and torch.is_floating_point(tensor):
                    finite_value = bool(torch.isfinite(tensor).all().detach().cpu().item())
                    finite_details[key] = finite_value
                    finite_ok = finite_ok and finite_value
        if labels_t is not None and hasattr(labels_t, "dtype") and torch.is_floating_point(labels_t):
            labels_finite = bool(torch.isfinite(labels_t).all().detach().cpu().item())
            finite_details["labels"] = labels_finite
            finite_ok = finite_ok and labels_finite

        nested_object_keys = []
        if isinstance(xb, dict):
            for key, value in xb.items():
                arr = np.asarray(value, dtype=object)
                if arr.dtype == object:
                    nested_object_keys.append(str(key))

        token_source = xb.get("tokens") if isinstance(xb, dict) and "tokens" in xb else input_ids
        tag_source = xb.get("ner_tags") if isinstance(xb, dict) and "ner_tags" in xb else (yb if yb is not None else labels_t)

        return {
            "input_ids_shape": self._debug_shape(input_ids),
            "attention_mask_shape": self._debug_shape(attention_mask),
            "labels_shape": self._debug_shape(labels_t),
            "token_example": self._debug_preview(token_source),
            "ner_tags_example": self._debug_preview(tag_source),
            "finite_ok": bool(finite_ok),
            "finite_details": finite_details,
            "nested_object_keys": nested_object_keys,
        }

    def count_params(self):
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters()))

    def get_weights(self):
        sd = self.model.state_dict()
        out = {}
        for k, v in sd.items():
            out[k] = self._tensor_to_numpy(v)
        return out

    def set_weights(self, weights_dict):
        torch = self.torch
        sd = self.model.state_dict()
        new_sd = {}
        for k, v in sd.items():
            if k in weights_dict:
                new_sd[k] = torch.tensor(weights_dict[k], device="cpu")
            else:
                new_sd[k] = v.detach().cpu()
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device)

    def _init_metric_accumulator(self):
        init_fn = getattr(self.task_spec, "init_metric_accumulator", None)
        if callable(init_fn):
            return init_fn()
        return {}

    def _accumulate_metric_statistics(self, accumulator, batch_stats):
        if not batch_stats:
            return accumulator
        update_fn = getattr(self.task_spec, "accumulate_metric_statistics", None)
        if callable(update_fn):
            return update_fn(accumulator, batch_stats)
        if accumulator is None:
            accumulator = {}
        for k, v in batch_stats.items():
            accumulator[k] = float(accumulator.get(k, 0.0)) + float(v)
        return accumulator

    def _has_metric_statistics(self, accumulator):
        has_fn = getattr(self.task_spec, "has_metric_statistics", None)
        if callable(has_fn):
            return bool(has_fn(accumulator))
        return bool(accumulator)

    def _metric_statistics_summary(self, accumulator):
        summary_fn = getattr(self.task_spec, "metric_statistics_summary", None)
        if callable(summary_fn):
            return summary_fn(accumulator)
        if not isinstance(accumulator, dict):
            return {}
        summary = {}
        for key, value in accumulator.items():
            try:
                summary[str(key)] = float(value)
            except Exception:
                continue
        return summary

    def _extract_logits(self, outputs):
        extract_fn = getattr(self.task_spec, "extract_logits", None)
        if callable(extract_fn):
            return extract_fn(outputs)
        return outputs.logits

    def _training_timed_out(self, train_start_ts, max_train_time_s):
        if max_train_time_s is None:
            return False
        return (time.perf_counter() - float(train_start_ts)) > float(max_train_time_s)

    def _build_optimizer(self, *, optimizer, lr, weight_decay):
        torch = self.torch
        name = str(optimizer or "adamw").strip().lower()
        params = self.model.parameters()
        if name in {"none", "null", ""}:
            name = "adamw"
        if name == "sgd":
            return torch.optim.SGD(params, lr=float(lr), weight_decay=float(weight_decay))
        if name == "adam":
            return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=float(lr), weight_decay=float(weight_decay))
        return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    def _backward_microbatch(self, xb, yb, *, loss_weight):
        """One disposable forward/backward frame; returns CPU scalar statistics."""
        if self.task_spec.name == "text_image_retrieval" and self._batch_item_count(xb) < 2:
            raise ValueError("insufficient_contrastive_batch: refusing a forward with fewer than two pairs")
        torch = self.torch
        enc, labels_t, extra = self.task_spec.encode_batch(
            self.tokenizer, xb, yb, self.max_length, torch, self.device,
            ignore_index=self.label_pad_value, inference_only=False,
        )
        if self.task_spec.name in {"token_classification", "image_segmentation", "seq2seq_generation"}:
            if labels_t is None or not bool((labels_t != self.label_pad_value).any()):
                raise ValueError("insufficient_supervision: training batch has no supervised positions")
        model_inputs = self.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=False)

        # A rejected AMP forward can corrupt running statistics even though no
        # optimizer update occurred. Restore them before every precision retry.
        batchnorm_state = [
            (buffer, buffer.detach().clone())
            for module in self.model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for buffer in (module.running_mean, module.running_var, module.num_batches_tracked)
            if buffer is not None
        ]
        forward_calls = 0

        def restore_batchnorm():
            with torch.no_grad():
                for buffer, original in batchnorm_state:
                    buffer.copy_(original)

        def forward():
            nonlocal forward_calls
            if forward_calls:
                restore_batchnorm()
            forward_calls += 1
            outputs = self.model(**model_inputs)
            logits = self._extract_logits(outputs)
            return self.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)

        loss = self._run_with_precision_context(forward)
        if loss is None:
            raise ValueError("Supervised fine-tune mode requires labels/loss-capable batch")
        fp32_retries = 0
        if not bool(torch.isfinite(loss)) and getattr(self, "autocast_enabled", False):
            loss = None
            self._fallback_to_fp32("nonfinite_training_loss_fallback_fp32")
            with torch.autocast(device_type=self._device_type(), enabled=False):
                loss = forward()
            fp32_retries = 1
        if not bool(torch.isfinite(loss)):
            restore_batchnorm()
            raise ValueError("nonfinite_training_loss: no optimizer update was applied for this batch")
        scaled_loss = loss * loss_weight
        scaler = getattr(self, "grad_scaler", None)
        (scaler.scale(scaled_loss) if scaler is not None else scaled_loss).backward()
        tokens = self._count_supervised_tokens(labels_t, self.label_pad_value)
        sequences = self._batch_item_count(xb)
        denominator = tokens if hasattr(labels_t, "ndim") and labels_t.ndim >= 2 else sequences
        return (float(loss.detach().cpu().item()) * max(1, denominator),
                max(1, denominator), tokens, sequences, fp32_retries)

    def finetune(
        self,
        xs,
        ys,
        epochs=1,
        lr=5e-5,
        optimizer="adamw",
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
        max_train_time_s=60,
        progress_log_interval=None,
    ):
        torch = self.torch

        y_local = ys

        self.model.float()
        self.model.train()
        from .singleton_batchnorm import configure_singleton_batchnorm
        configure_singleton_batchnorm(self.model, torch)
        optimizer_name = str(optimizer or "adamw").strip().lower()
        weight_decay = float(weight_decay or 0.0)
        warmup_ratio = max(0.0, float(warmup_ratio or 0.0))
        gradient_accumulation_steps = max(1, int(gradient_accumulation_steps or 1))
        optimizer_obj = self._build_optimizer(optimizer=optimizer_name, lr=float(lr), weight_decay=weight_decay)

        total_loss = 0.0
        train_loss_denominator_count = 0
        train_supervised_token_count = 0
        train_sequence_count = 0
        step_lat_ms = []
        t_start = time.perf_counter()
        timeout_hit = False
        epochs = int(epochs)
        progress_log_interval = int(progress_log_interval) if progress_log_interval is not None else 0
        if isinstance(xs, dict):
            total_sequence_count = len(next(iter(xs.values()))) if xs else 0
        else:
            total_sequence_count = len(xs)
        contrastive = getattr(self.task_spec, "name", None) == "text_image_retrieval"
        train_batch_size = int(self.batch_size)
        if contrastive:
            from ...contrastive_batch import contrastive_ranges, retrieval_batch_target
            params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
            pixels = xs.get("pixel_values") if isinstance(xs, dict) else None
            image_size = max(pixels.shape[-2:]) if pixels is not None else 224
            preferred = retrieval_batch_target(params_m, image_size)
            train_batch_size = min(total_sequence_count, max(train_batch_size, preferred))
            # Even full batches can always fall back to pairs without sample loss.
            if train_batch_size > 3 and train_batch_size % 2:
                train_batch_size -= 1
            contrastive_ranges(total_sequence_count, train_batch_size)
            print(f"[HFCore.finetune] contrastive policy | vram_basis_gib=16 "
                  f"| requested_batch={self.batch_size} | preferred_batch={preferred} "
                  f"| physical_batch={train_batch_size} | minimum_pairs=2", flush=True)
        batches_per_epoch = (
            int(np.ceil(float(total_sequence_count) / float(max(1, train_batch_size))))
            if total_sequence_count
            else 0
        )
        total_batches = int(max(0, epochs) * max(0, batches_per_epoch))
        optimizer_steps_total = epochs * int(np.ceil(batches_per_epoch / gradient_accumulation_steps))
        warmup_steps = int(round(float(optimizer_steps_total) * warmup_ratio)) if optimizer_steps_total else 0
        scheduler = None
        if warmup_steps > 0 or warmup_ratio > 0.0:
            def _lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_obj, lr_lambda=_lr_lambda)
        global_batch_idx = 0
        optimizer_step_count = 0
        successful_optimizer_step_count = 0

        def record_successful_step(*_args):
            nonlocal successful_optimizer_step_count
            successful_optimizer_step_count += 1

        # GradScaler can skip an attempted update. Count actual optimizer calls
        # so scheduler progress follows successful updates.
        step_counter = optimizer_obj.register_step_post_hook(record_successful_step)

        prefetch_depth = self._batch_prefetch_depth(xs)
        print(
            f"[HFCore.finetune] starts | epochs={epochs} | batch_size={train_batch_size} "
            f"| sequence_count={total_sequence_count} | batches_per_epoch={batches_per_epoch} "
            f"| optimizer={optimizer_name} | weight_decay={weight_decay} "
            f"| warmup_ratio={warmup_ratio} | gradient_accumulation_steps={gradient_accumulation_steps} "
            f"| max_train_time_s={max_train_time_s} | prefetch_batches={prefetch_depth}", flush=True,
        )
        completed_epochs = 0
        fp32_loss_retry_count = 0
        last_progress_at = t_start
        optimizer_obj.zero_grad(set_to_none=True)
        from .microbatch import backward_window, TrainingBudgetExpired
        memory_state = {
            "batch_size_requested": int(self.batch_size),
            "micro_batch_size_effective": train_batch_size,
            "logical_batch_size": train_batch_size,
            "gradient_accumulation_steps_requested": gradient_accumulation_steps,
            "gradient_accumulation_steps_effective": gradient_accumulation_steps,
            "effective_optimizer_batch_size": train_batch_size * gradient_accumulation_steps,
            "oom_retry_count": 0,
            "memory_fallback_applied": False,
            "optimizer_step_count": 0,
            "successful_optimizer_step_count": 0,
            "skipped_optimizer_step_count": 0,
        }
        if contrastive:
            memory_state.update(contrastive_min_pairs=2, contrastive_preferred_batch_size=preferred,
                                contrastive_vram_basis_gib=16,
                                contrastive_batch_adjusted=train_batch_size != self.batch_size)
        self.training_memory_metadata = memory_state

        try:
            for epoch_idx in range(epochs):
                # Local-update preparation seeds NumPy with the client's training_seed.
                # Draw a fresh order each epoch without changing partition membership
                # or the source arrays used for provenance and evaluation.
                epoch_indices = np.random.permutation(total_sequence_count)
                with prefetch_batches(
                    self._batch_iter(xs, y_local, indices=epoch_indices, batch_size=train_batch_size,
                                     contrastive=contrastive), depth=prefetch_depth
                ) as batches:
                    batch_idx = 0
                    while window := list(islice(batches, gradient_accumulation_steps)):
                        t0 = time.perf_counter()
                        try:
                            totals = backward_window(self, window, optimizer_obj, memory_state,
                                                     t_start, max_train_time_s)
                        except TrainingBudgetExpired:
                            timeout_hit = True
                            break
                        # A step can allocate optimizer state and partially update parameters.
                        # Never replay it after OOM without a complete transactional snapshot.
                        grad_scaler = getattr(self, "grad_scaler", None)
                        max_grad_norm = getattr(self, "max_grad_norm", None)
                        if max_grad_norm is not None:
                            if grad_scaler is not None:
                                grad_scaler.unscale_(optimizer_obj)
                            # A finite loss can still have invalid gradients. Fail
                            # before mutating parameters rather than corrupting the model.
                            norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm, error_if_nonfinite=True)
                            memory_state["max_grad_norm"] = max_grad_norm
                            memory_state["gradient_norm_max"] = max(
                                memory_state.get("gradient_norm_max", 0.0), float(norm))
                            memory_state["gradient_clipped_step_count"] = (
                                memory_state.get("gradient_clipped_step_count", 0) + int(norm > max_grad_norm))
                        successful_steps_before = successful_optimizer_step_count
                        if grad_scaler is not None:
                            grad_scaler.step(optimizer_obj)
                            grad_scaler.update()
                        else:
                            optimizer_obj.step()
                        optimizer_obj.zero_grad(set_to_none=True)
                        optimizer_step_count += 1
                        if scheduler is not None and successful_optimizer_step_count > successful_steps_before:
                            scheduler.step()
                        total_loss += totals[0]
                        train_loss_denominator_count += totals[1]
                        train_supervised_token_count += totals[2]
                        train_sequence_count += totals[3]
                        fp32_loss_retry_count += totals[4]
                        global_batch_idx += len(window)
                        batch_idx += len(window)
                        memory_state.update(optimizer_step_count=optimizer_step_count,
                                            successful_optimizer_step_count=successful_optimizer_step_count,
                                            skipped_optimizer_step_count=optimizer_step_count - successful_optimizer_step_count)
                        step_lat_ms.append((time.perf_counter() - t0) * 1000.0 / len(window))
                        if progress_log_interval > 0 and (
                            global_batch_idx == len(window) or global_batch_idx == total_batches
                            or time.perf_counter() - last_progress_at >= 10
                        ):
                            print(
                                f"[HFCore.finetune] progress | epoch={epoch_idx + 1}/{max(1, epochs)} "
                                f"| batch={batch_idx}/{max(1, batches_per_epoch)} "
                                f"| global_batch={global_batch_idx}/{max(1, total_batches)} "
                                f"| elapsed_s={time.perf_counter() - t_start:.2f}", flush=True,
                            )
                            last_progress_at = time.perf_counter()

                if timeout_hit:
                    debug_print(
                        f"[HFCore.finetune] timeout | epoch={epoch_idx + 1}/{max(1, epochs)} "
                        f"| completed_batches={global_batch_idx}/{max(1, total_batches)} "
                        f"| elapsed_s={time.perf_counter() - t_start:.2f}"
                    )
                    break
                completed_epochs += 1

        finally:
            optimizer_obj.zero_grad(set_to_none=True)
            step_counter.remove()

        duration_s = time.perf_counter() - t_start
        self.model.eval()
        if contrastive and successful_optimizer_step_count == 0:
            raise ValueError(
                "no_contrastive_optimizer_updates: training completed without a parameter update; "
                "check the training budget and AMP skipped-step counts."
            )
        debug_print(
            f"[HFCore.finetune] ends | completed_batches={global_batch_idx}/{max(1, total_batches)} "
            f"| duration_s={duration_s:.2f} | timeout_hit={timeout_hit}"
        )

        step_mean = float(np.mean(step_lat_ms)) if step_lat_ms else np.nan
        step_p95 = float(np.percentile(step_lat_ms, 95)) if step_lat_ms else np.nan
        steady_steps = step_lat_ms[1:] if len(step_lat_ms) > 1 else []
        steady_step_mean = float(np.mean(steady_steps)) if steady_steps else np.nan
        steady_step_p95 = float(np.percentile(steady_steps, 95)) if steady_steps else np.nan

        train_loss = float(total_loss / max(1, train_loss_denominator_count))
        train_throughput = float(train_sequence_count / max(duration_s, 1e-9))
        token_throughput = (
            float(train_supervised_token_count / max(duration_s, 1e-9))
            if train_supervised_token_count > 0
            else np.nan
        )

        return {
            "train_loss": train_loss,
            "train_time_s": float(duration_s),
            "train_step_latency_ms_mean": step_mean,
            "train_step_latency_ms_p95": step_p95,
            "train_step_latency_ms_steady_mean": steady_step_mean,
            "train_step_latency_ms_steady_p95": steady_step_p95,
            "train_throughput_eps": train_throughput,
            **self._qos_startup(),
            "train_sequence_count": int(train_sequence_count),
            "train_supervised_token_count": int(train_supervised_token_count),
            "train_loss_denominator_count": int(train_loss_denominator_count),
            "tokens_total": int(train_supervised_token_count),
            "tokens_per_second": token_throughput,
            # Deprecated legacy aliases retained for downstream consumers expecting the
            # old sample-count fields; these now map to the explicit unit-specific keys.
            "train_samples": int(train_loss_denominator_count),
            "batch_size": train_batch_size,
            "device": str(self.device),
            "hf_model_id": self.model_id,
            "max_length": int(self.max_length),
            "requested_max_length": int(getattr(self, "requested_max_length", self.max_length)),
            "model_text_max_length": (
                int(getattr(self, "model_text_max_length", None))
                if getattr(self, "model_text_max_length", None) is not None
                else np.nan
            ),
            "max_length_adjusted": bool(getattr(self, "max_length_adjusted", False)),
            "hf_task": getattr(self.task_spec, "name", None),
            "label_pad_value": int(self.label_pad_value),
            "hf_weights_format": self.weight_format,
            "train_timeout_s": (None if max_train_time_s is None else float(max_train_time_s)),
            "train_stopped_early": bool(timeout_hit),
            "completed_training_epochs": completed_epochs,
            "train_stopping_reason": "resource_limit" if timeout_hit else "epochs_completed",
            "train_prefetch_batches": prefetch_depth,
            "optimizer": optimizer_name,
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "warmup_steps": int(warmup_steps),
            "optimizer_step_count": int(optimizer_step_count),
            "planned_optimizer_step_count": int(optimizer_steps_total),
            "successful_optimizer_step_count": int(successful_optimizer_step_count),
            "skipped_optimizer_step_count": int(optimizer_step_count - successful_optimizer_step_count),
            "train_fp32_loss_retry_count": int(fp32_loss_retry_count),
            **memory_state,
            "gradient_accumulation_steps": memory_state["gradient_accumulation_steps_effective"],
            "gradient_accumulation": memory_state["gradient_accumulation_steps_effective"],
        }

    def eval(self, xs, ys, inference_only=False, max_eval_time_s=None, progress_log_interval=None):
        from .microbatch import is_device_oom, release_failed_attempt
        started = time.perf_counter()
        batch_size = int(self.batch_size)
        state = {"eval_batch_size_requested": batch_size,
                 "eval_batch_size_effective": batch_size, "eval_oom_retry_count": 0,
                 "eval_memory_fallback_applied": False, "eval_fp32_retry_count": 0}
        self.evaluation_memory_metadata = state
        while True:
            remaining = (None if max_eval_time_s is None
                         else float(max_eval_time_s) - (time.perf_counter() - started))
            if remaining is not None and remaining <= 0:
                raise TimeoutError("HF evaluation exhausted its time budget during OOM recovery")
            try:
                loss, primary, secondary, qos = self._eval_once(
                    xs, ys, inference_only, remaining, progress_log_interval,
                    batch_size=state["eval_batch_size_effective"],
                )
                elapsed = time.perf_counter() - started
                qos.update(state)
                qos["eval_total_time_s"] = elapsed
                qos["eval_throughput_eps"] = qos["eval_sequence_count"] / max(elapsed, 1e-9)
                qos["tokens_per_second"] = qos["eval_supervised_token_count"] / max(elapsed, 1e-9)
                return loss, primary, secondary, qos
            except Exception as exc:
                if (
                    isinstance(exc, ValueError)
                    and str(exc).startswith(("nonfinite_evaluation_logits:", "nonfinite_evaluation_loss:"))
                    and getattr(self, "autocast_enabled", False)
                ):
                    # Discard all partial metrics and replay within the original
                    # budget, using FP32 consistently for the entire evaluation.
                    release_failed_attempt(self, exc)
                    self._fallback_to_fp32("nonfinite_evaluation_fallback_fp32")
                    state["eval_fp32_retry_count"] += 1
                    print("[HFCore.eval] nonfinite result: restarting evaluation in FP32", flush=True)
                    continue
                if not is_device_oom(exc) or state["eval_batch_size_effective"] <= 1:
                    raise
                release_failed_attempt(self, exc)
                old = state["eval_batch_size_effective"]
                state["eval_batch_size_effective"] = max(1, old // 2)
                state["eval_oom_retry_count"] += 1
                state["eval_memory_fallback_applied"] = True
                print(f"[HFCore.eval] OOM retry | batch_size={old}->{state['eval_batch_size_effective']}", flush=True)

    def _eval_once(self, xs, ys, inference_only=False, max_eval_time_s=None,
                   progress_log_interval=None, *, batch_size=None):
        torch = self.torch
        batch_size = int(self.batch_size if batch_size is None else batch_size)
        # Quality for encoder-decoder tasks must use source-only generation,
        # even after fine-tuning. A separate teacher forward supplies the loss.
        if self.task_spec.name == "seq2seq_generation":
            inference_only = True

        def _is_dense_label_tensor(value):
            return hasattr(value, "ndim") and not isinstance(value, (list, tuple, dict))

        y_true = ys
        self.model.eval()
        prefetch_depth = self._batch_prefetch_depth(xs)

        latencies_ms = []
        total_loss = 0.0
        eval_loss_denominator_count = 0
        eval_supervised_token_count = 0

        preds_all = []
        labels_all = []
        stats_accum = self._init_metric_accumulator()

        if isinstance(xs, dict):
            eval_sequence_count = len(next(iter(xs.values())))
        else:
            eval_sequence_count = len(xs)
        eval_batch_count = int(np.ceil(float(eval_sequence_count) / float(max(1, batch_size)))) if eval_sequence_count else 0

        t_start = time.perf_counter()

        last_extra = {}
        last_progress_at = t_start
        eval_batch_count = int((eval_sequence_count + max(1, batch_size) - 1) / max(1, batch_size))
        progress_log_interval = int(progress_log_interval) if progress_log_interval is not None else 0
        max_eval_time_s = float(max_eval_time_s) if max_eval_time_s is not None else None
        progress_log_every = max(
            1,
            min(25, eval_batch_count // 4 if eval_batch_count > 4 else eval_batch_count or 1),
        )

        debug_print(
            f"[HFCore.eval] dataloader creation starts | inference_only={bool(inference_only)} "
            f"| batch_size={batch_size} | eval_sequence_count={eval_sequence_count} | eval_batch_count={eval_batch_count}"
        )
        first_batch_logged = False

        with torch.no_grad(), prefetch_batches(
            self._batch_iter(xs, y_true, batch_size=batch_size), depth=prefetch_depth
        ) as batches:
            for batch_idx, (xb, yb) in enumerate(batches, start=1):
                if not first_batch_logged:
                    debug_print("[HFCore.eval] first batch pulled")
                if max_eval_time_s is not None and (time.perf_counter() - t_start) > max_eval_time_s:
                    raise TimeoutError(
                        f"HF evaluation exceeded max_eval_time_s={max_eval_time_s} "
                        f"after batch {batch_idx - 1}/{eval_batch_count}"
                    )
                t0 = time.perf_counter()
                labels_recorded = False

                enc, labels_t, extra = self.task_spec.encode_batch(
                    self.tokenizer,
                    xb,
                    yb,
                    self.max_length,
                    torch,
                    self.device,
                    ignore_index=self.label_pad_value,
                    inference_only=bool(inference_only),
                )

                last_extra = dict(extra or {})
                teacher_forced = None
                if bool(inference_only) and bool(getattr(self.task_spec, "supports_generation", False)) and yb is not None:
                    teacher_enc, teacher_labels_t, teacher_extra = self.task_spec.encode_batch(
                        self.tokenizer,
                        xb,
                        yb,
                        self.max_length,
                        torch,
                        self.device,
                        ignore_index=self.label_pad_value,
                        inference_only=False,
                    )
                    if teacher_labels_t is not None:
                        teacher_forced = (teacher_enc, teacher_labels_t, dict(teacher_extra or {}))
                        labels_all.append(self._labels_to_numpy(teacher_labels_t))
                        labels_recorded = True
                        supervised_token_count = self._count_supervised_tokens(teacher_labels_t, self.label_pad_value)
                        eval_supervised_token_count += supervised_token_count
                    elif isinstance(teacher_extra, dict) and teacher_extra.get("answer_texts") is not None:
                        labels_all.append(np.asarray(teacher_extra.get("answer_texts"), dtype=object).reshape(-1))
                        labels_recorded = True

                if bool(inference_only) and bool(getattr(self.task_spec, "supports_generation", False)):
                    self._ensure_left_padding_for_decoder_only_generation()
                    if not first_batch_logged:
                        debug_print("[HFCore.eval] model forward starts")
                    with self._generation_inference_mode():
                        def _generate():
                            pred_t = self.task_spec.generate_predictions(
                                self.model,
                                enc,
                                self.tokenizer,
                                torch,
                                self.generation_config,
                            )
                            return pred_t

                        pred_t = self._run_with_precision_context(_generate)
                    if self.task_spec.name == "seq2seq_generation" and (
                        pred_t is None or getattr(pred_t, "ndim", 0) != 2
                        or pred_t.shape[0] != len(enc["input_ids"]) or pred_t.shape[1] == 0
                    ):
                        raise ValueError("invalid_evaluation_predictions: generation must return tokens for every source")
                    if not first_batch_logged:
                        debug_print("[HFCore.eval] first batch forward ends")
                    if hasattr(pred_t, "detach"):
                        preds_all.append(self._tensor_to_numpy(pred_t))
                    else:
                        preds_all.append(np.asarray(pred_t, dtype=object))

                    if teacher_forced is not None:
                        teacher_enc, teacher_labels_t, teacher_extra = teacher_forced
                        teacher_inputs = self.task_spec.build_forward_inputs(
                            teacher_enc,
                            labels_t=teacher_labels_t,
                            inference_only=False,
                        )
                        def _teacher_forward():
                            outputs = self.model(**teacher_inputs)
                            logits = self._extract_logits(outputs)
                            return outputs, logits

                        outputs, logits = self._run_with_precision_context(_teacher_forward)
                        if self.task_spec.name == "seq2seq_generation" and (
                            logits is None or logits.ndim != 3
                            or tuple(logits.shape[:2]) != tuple(teacher_labels_t.shape)
                        ):
                            raise ValueError("invalid_evaluation_alignment: logits must align to all reference positions")
                        loss = self.task_spec.extract_loss(torch, outputs, logits, teacher_labels_t, teacher_extra)
                        self._validate_eval_tensors(logits, loss, require_loss=True)
                        stat = self.task_spec.batch_metric_statistics(torch, logits, teacher_labels_t, teacher_extra)
                        if stat:
                            stats_accum = self._accumulate_metric_statistics(stats_accum, stat)

                        stat_out = self.task_spec.batch_metric_statistics_from_outputs(torch, outputs, teacher_labels_t, teacher_extra)
                        if stat_out:
                            stats_accum = self._accumulate_metric_statistics(stats_accum, stat_out)

                        loss = self.task_spec.extract_loss(torch, outputs, logits, teacher_labels_t, teacher_extra)
                        if loss is not None:
                            if _is_dense_label_tensor(teacher_labels_t) and teacher_labels_t.ndim >= 2:
                                loss_denominator_count = supervised_token_count
                            elif isinstance(xb, dict):
                                loss_denominator_count = len(next(iter(xb.values())))
                            else:
                                loss_denominator_count = len(xb)

                            total_loss += float(loss.detach().cpu().item()) * float(max(1, loss_denominator_count))
                            eval_loss_denominator_count += int(max(1, loss_denominator_count))
                else:
                    model_inputs = self.task_spec.build_forward_inputs(enc, labels_t=labels_t, inference_only=bool(inference_only))
                    if not first_batch_logged:
                        debug_print("[HFCore.eval] model forward starts")
                    def _eval_forward():
                        outputs = self.model(**model_inputs)
                        logits = self._extract_logits(outputs)
                        pred_t = self.task_spec.preds_from_logits(torch, logits, extra)
                        return outputs, logits, pred_t

                    outputs, logits, pred_t = self._run_with_precision_context(_eval_forward)
                    self._validate_eval_tensors(logits)
                    if not first_batch_logged:
                        debug_print("[HFCore.eval] first batch forward ends")
                    if hasattr(pred_t, "detach"):
                        preds_all.append(self._tensor_to_numpy(pred_t))
                    else:
                        preds_all.append(np.asarray(pred_t, dtype=object))
                    
                    collect_unlabeled_stats = bool(getattr(self.task_spec, "supports_unlabeled_metric_statistics", False))
                    if labels_t is not None or collect_unlabeled_stats:
                        stat = self.task_spec.batch_metric_statistics(torch, logits, labels_t, extra)
                        if stat:
                            stats_accum = self._accumulate_metric_statistics(stats_accum, stat)

                        stat_out = self.task_spec.batch_metric_statistics_from_outputs(torch, outputs, labels_t, extra)
                        if stat_out:
                            stats_accum = self._accumulate_metric_statistics(stats_accum, stat_out)

                    if not bool(inference_only) and self.task_spec.name != "text_image_retrieval":
                        loss = self.task_spec.extract_loss(torch, outputs, logits, labels_t, extra)
                        self._validate_eval_tensors(logits, loss, require_loss=labels_t is not None)
                        if loss is not None and labels_t is not None:
                            labels_all.append(self._labels_to_numpy(labels_t))
                            labels_recorded = True
                            supervised_token_count = self._count_supervised_tokens(labels_t, self.label_pad_value)
                            eval_supervised_token_count += supervised_token_count

                            if _is_dense_label_tensor(labels_t) and labels_t.ndim >= 2:
                                loss_denominator_count = supervised_token_count
                            elif isinstance(xb, dict):
                                loss_denominator_count = len(next(iter(xb.values())))
                            else:
                                loss_denominator_count = len(xb)

                            total_loss += float(loss.detach().cpu().item()) * float(max(1, loss_denominator_count))
                            eval_loss_denominator_count += int(max(1, loss_denominator_count))

                if labels_t is not None and bool(inference_only) and yb is not None and not labels_recorded:
                    labels_all.append(self._labels_to_numpy(labels_t))

                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                if eval_batch_count and (
                    batch_idx == 1
                    or batch_idx == eval_batch_count
                    or batch_idx % progress_log_every == 0
                    or (progress_log_interval > 0 and time.perf_counter() - last_progress_at >= 10)
                ):
                    progress_units_done = min(batch_idx * batch_size, eval_sequence_count)
                    progress_units_label = (
                        "examples_done" if self._has_metric_statistics(stats_accum) else "sequences_done"
                    )
                    visible = progress_log_interval > 0 and (
                        batch_idx == 1 or batch_idx == eval_batch_count
                        or time.perf_counter() - last_progress_at >= 10
                    )
                    emit = print if visible else debug_print
                    emit(
                        "[HFCore.eval] progress | "
                        f"batch={batch_idx}/{eval_batch_count} | "
                        f"{progress_units_label}={progress_units_done}/{eval_sequence_count} | "
                        f"last_batch_ms={latencies_ms[-1]:.2f}", flush=True,
                    )
                    if visible:
                        last_progress_at = time.perf_counter()
                first_batch_logged = True
                del enc, labels_t, extra
                if 'outputs' in locals():
                    del outputs
                if 'logits' in locals():
                    del logits
                if 'pred_t' in locals():
                    del pred_t
                if 'loss' in locals():
                    del loss
                if 'model_inputs' in locals():
                    del model_inputs
                if 'teacher_forced' in locals():
                    del teacher_forced

        duration_s = time.perf_counter() - t_start

        y_true_np = self._concat_or_pad_batches(
            labels_all,
            empty_dtype="int64",
            pad_value=int(self.label_pad_value),
        )
        pred_pad_value = 0
        if self.tokenizer is not None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            pred_pad_value = int(self.tokenizer.pad_token_id)
        y_pred_np = self._concat_or_pad_batches(
            preds_all,
            empty_dtype="int64",
            pad_value=pred_pad_value,
        )
        label_space_warning = None
        label_space_mismatch = False
        model_num_labels = None
        dataset_label_count = None
        pred_label_overlap_count = None
        if bool(inference_only) and getattr(self.task_spec, "name", None) == "image_classification":
            try:
                cfg = getattr(self.model, "config", None)
                model_num_labels_raw = getattr(cfg, "num_labels", None)
                if model_num_labels_raw is not None:
                    model_num_labels = int(model_num_labels_raw)

                y_true_flat = np.asarray(y_true_np).reshape(-1)
                y_pred_flat = np.asarray(y_pred_np).reshape(-1)
                if y_true_flat.size and y_pred_flat.size:
                    true_unique = np.unique(y_true_flat)
                    pred_unique = np.unique(y_pred_flat)
                    dataset_label_count = int(true_unique.size)
                    pred_label_overlap_count = int(np.intersect1d(pred_unique, true_unique).size)
                    max_true = int(np.max(true_unique))
                    contiguous_small_label_ids = max_true < max(1, dataset_label_count * 2)
                    if (
                        model_num_labels is not None
                        and model_num_labels >= max(32, dataset_label_count * 10)
                        and contiguous_small_label_ids
                        and pred_label_overlap_count == 0
                    ):
                        label_space_mismatch = True
                        label_space_warning = (
                            "HF image-classification inference likely has incompatible label spaces: "
                            f"dataset_labels={dataset_label_count}, model_num_labels={model_num_labels}, "
                            "predicted IDs do not overlap dataset IDs. "
                            "Use a checkpoint fine-tuned for this dataset/task or run with model_type=hf_finetune."
                        )
            except Exception:
                label_space_warning = None
                label_space_mismatch = False

        named_metrics = None
        loss_mean = (
            float(total_loss / max(1, eval_loss_denominator_count))
            if eval_loss_denominator_count > 0
            else np.nan
        )
        retrieval_loss_candidates = None
        if self.task_spec.name == "text_image_retrieval" and not inference_only:
            loss_mean, retrieval_loss_candidates = self.task_spec.pooled_evaluation_loss(
                torch, stats_accum, self.model)
            eval_loss_denominator_count = retrieval_loss_candidates
            # Include pooled loss calculation in evaluation runtime.
            duration_s = time.perf_counter() - t_start

        stats_summary = self._metric_statistics_summary(stats_accum)
        m_stats = self.task_spec.metrics_from_statistics(stats_accum) if self._has_metric_statistics(stats_accum) else None
        metric_start = None
        metric_mode = None
        if isinstance(m_stats, dict) and m_stats:
            metric_start = time.perf_counter()
            metric_mode = "statistics"
            debug_print("[HFCore.eval] metric computation starts | mode=statistics", flush=True)
            primary = float(m_stats.get("primary", np.nan))
            secondary = float(m_stats.get("secondary", np.nan))
            named_metrics = m_stats.get("named_metrics") if isinstance(m_stats, dict) else None
        elif y_true_np.size == 0 or y_pred_np.size == 0:
            debug_print(
                "[HFCore.eval] metric computation skipped | "
                f"reason=empty_arrays | y_true_size={y_true_np.size} | y_pred_size={y_pred_np.size}",
                flush=True,
            )
            primary = np.nan
            secondary = np.nan
        else:
            metric_start = time.perf_counter()
            metric_mode = "arrays"
            debug_print(
                "[HFCore.eval] metric computation starts | "
                f"mode=arrays | y_true_shape={getattr(y_true_np, 'shape', None)} "
                f"| y_pred_shape={getattr(y_pred_np, 'shape', None)}",
                flush=True,
            )
            metrics_extra = dict(last_extra or {})
            metrics_extra["task_tag"] = self.task_tag
            metrics_extra["loss_mean"] = loss_mean
            if getattr(self.task_spec, "supports_generation", False):
                metrics_extra["tokenizer"] = self.tokenizer
            m = self.task_spec.metrics(y_true_np, y_pred_np, y_extra=metrics_extra)
            primary = float(m.get("primary", np.nan))
            secondary = float(m.get("secondary", np.nan))
            named_metrics = m.get("named_metrics") if isinstance(m, dict) else None

            if getattr(self.task_spec, "name", None) == "fill_mask" and loss_mean == loss_mean:
                try:
                    secondary = float(np.exp(np.clip(loss_mean, a_min=-50.0, a_max=50.0)))
                except Exception:
                    secondary = np.nan

        if metric_start is not None:
            debug_print(
                "[HFCore.eval] metric computation ends | "
                f"mode={metric_mode} | primary={primary} | secondary={secondary} "
                f"| metric_s={time.perf_counter() - metric_start:.2f}",
                flush=True,
            )

        lat_mean = float(np.mean(latencies_ms)) if latencies_ms else np.nan
        lat_p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else np.nan
        steady_lat = latencies_ms[1:] if len(latencies_ms) > 1 else []
        lat_steady_mean = float(np.mean(steady_lat)) if steady_lat else np.nan
        lat_steady_p95 = float(np.percentile(steady_lat, 95)) if steady_lat else np.nan
        throughput = float(eval_sequence_count / max(duration_s, 1e-9))
        token_throughput = (
            float(eval_supervised_token_count / max(duration_s, 1e-9))
            if eval_supervised_token_count > 0
            else np.nan
        )

        metric_instance_count = None
        if stats_summary:
            for metric_count_key in ("metric_instance_count", "total"):
                metric_count_value = stats_summary.get(metric_count_key)
                if metric_count_value is not None:
                    metric_instance_count = int(metric_count_value)
                    break

        qos = {
            "eval_latency_ms_mean": lat_mean,
            "eval_latency_ms_p95": lat_p95,
            "eval_latency_ms_steady_mean": lat_steady_mean,
            "eval_latency_ms_steady_p95": lat_steady_p95,
            "eval_throughput_eps": throughput,
            **self._qos_startup(),
            "eval_sequence_count": int(eval_sequence_count),
            "eval_batch_count": int(eval_batch_count),
            "eval_prefetch_batches": prefetch_depth,
            "eval_supervised_token_count": int(eval_supervised_token_count),
            "tokens_total": int(eval_supervised_token_count),
            "tokens_per_second": token_throughput,
            # Deprecated legacy aliases retained for downstream consumers expecting the
            # old sample-count fields; these now map to the explicit unit-specific keys.
            "eval_samples": int(eval_sequence_count),
            "batch_size": int(self.batch_size),
            "device": str(self.device),
            "hf_model_id": self.model_id,
            "max_length": int(self.max_length),
            "requested_max_length": int(getattr(self, "requested_max_length", self.max_length)),
            "model_text_max_length": (
                int(getattr(self, "model_text_max_length", None))
                if getattr(self, "model_text_max_length", None) is not None
                else np.nan
            ),
            "max_length_adjusted": bool(getattr(self, "max_length_adjusted", False)),
            "hf_task": getattr(self.task_spec, "name", None),
            "label_pad_value": int(self.label_pad_value),
            "hf_weights_format": self.weight_format,
            "inference_only": bool(inference_only),
        }
        if metric_instance_count is not None:
            qos["metric_instance_count"] = int(metric_instance_count)
        if retrieval_loss_candidates is not None:
            qos.update(evaluation_loss_protocol="pooled_symmetric_contrastive_v1",
                       eval_loss_candidate_count=retrieval_loss_candidates,
                       eval_loss_denominator_count=eval_loss_denominator_count)
        if model_num_labels is not None:
            qos["model_num_labels"] = int(model_num_labels)
        if dataset_label_count is not None:
            qos["dataset_label_count"] = int(dataset_label_count)
        if pred_label_overlap_count is not None:
            qos["pred_label_overlap_count"] = int(pred_label_overlap_count)
        if label_space_mismatch:
            qos["label_space_mismatch"] = bool(label_space_mismatch)
        if label_space_warning:
            qos["label_space_warning"] = str(label_space_warning)

        if named_metrics and isinstance(named_metrics, dict):
            for mk, mv in named_metrics.items():
                if mv is not None and not (isinstance(mv, float) and np.isnan(mv)):
                    qos[str(mk).lower()] = float(mv)

        if stats_summary:
            for sk, sv in stats_summary.items():
                qos[f"metric_stat_{sk}"] = float(sv)

        if getattr(self.task_spec, "supports_generation", False):
            qos.update({f"generation_{k}": v for k, v in self.generation_config.items()})
        if self.task_spec.name == "seq2seq_generation":
            qos["evaluation_protocol"] = "source_only_generation_v1"
            qos["evaluation_loss_protocol"] = "teacher_forced_v1"

        return loss_mean, primary, secondary, qos
    
