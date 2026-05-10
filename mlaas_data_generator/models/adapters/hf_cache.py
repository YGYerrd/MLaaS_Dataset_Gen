import threading
import time

from ...data.preprocessors.hf_text_generation import _load_auto_tokenizer


_TOKENIZER_CACHE = {}
_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()
_GENERATION_TASKS = {
    "causal_lm_generation",
    "seq2seq_generation",
    "image_captioning",
    "visual_question_answering",
}


def _cache_key(hf_model_id, task, device):
    return (str(hf_model_id), str(task or "").strip().lower(), str(device))


def _should_left_pad_tokenizer(*, hf_model_id, task, transformers_module):
    task_name = str(task or "").strip().lower()
    if task_name == "causal_lm_generation":
        return True

    auto_config = getattr(transformers_module, "AutoConfig", None)
    if auto_config is None or not hasattr(auto_config, "from_pretrained"):
        return False

    try:
        config = auto_config.from_pretrained(hf_model_id)
    except Exception:
        return False

    if bool(getattr(config, "is_encoder_decoder", False)):
        return False
    if task_name in _GENERATION_TASKS:
        return True
    if bool(getattr(config, "is_decoder", False)):
        return True
    return False


def get_cached_tokenizer(*, hf_model_id, task, device, transformers_module):
    """
    Returns (tokenizer, load_s, cache_hit).
    """
    key = _cache_key(hf_model_id, task, device)
    with _CACHE_LOCK:
        tok = _TOKENIZER_CACHE.get(key)
    if tok is not None:
        return tok, 0.0, True

    t0 = time.time()
    tok = _load_auto_tokenizer(hf_model_id)

    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token
    if _should_left_pad_tokenizer(
        hf_model_id=hf_model_id,
        task=task,
        transformers_module=transformers_module,
    ):
        tok.padding_side = "left"

    load_s = float(time.time() - t0)
    with _CACHE_LOCK:
        _TOKENIZER_CACHE.setdefault(key, tok)
        tok = _TOKENIZER_CACHE[key]
    return tok, load_s, False


def get_cached_model(*, hf_model_id, task, device, loader_fn):
    """
    Returns (model, load_s, cache_hit).
    """
    key = _cache_key(hf_model_id, task, device)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
    if model is not None:
        return model, 0.0, True

    t0 = time.time()
    model = loader_fn()
    load_s = float(time.time() - t0)

    with _CACHE_LOCK:
        _MODEL_CACHE.setdefault(key, model)
        model = _MODEL_CACHE[key]
    return model, load_s, False

