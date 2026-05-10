import sys
import types

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf
from mlaas_data_generator.models.adapters import hf_cache


class DummySplit:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.features = {}

    def __getitem__(self, key):
        return [r.get(key) for r in self._rows]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    mask_token_id = 99
    vocab_size = 128

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def _encode(self, text):
        toks = [min(30, len(tok) + 2) for tok in str(text).split()]
        return toks or [3]

    def __call__(self, text_a, text_b=None, truncation=True, padding=False, max_length=16, return_attention_mask=True, return_special_tokens_mask=False, **kwargs):
        if isinstance(text_a, str):
            text_a = [text_a]
        if text_b is not None and isinstance(text_b, str):
            text_b = [text_b]

        ids, mask = [], []
        for i, a in enumerate(text_a):
            row = [1] + self._encode(a)
            if text_b is not None:
                row += [2] + self._encode(text_b[i])
            row = row[: int(max_length)]
            if padding == "max_length":
                row = row + [self.pad_token_id] * max(0, int(max_length) - len(row))
            ids.append(row)
            mask.append([1 if x != self.pad_token_id else 0 for x in row])

        out = {"input_ids": ids, "attention_mask": mask}
        if return_special_tokens_mask:
            out["special_tokens_mask"] = [[1 if tok in (0, 1, 2) else 0 for tok in row] for row in ids]
        return out

    def pad(self, encodings, padding="max_length", max_length=None, return_tensors=None, **kwargs):
        if padding != "max_length":
            raise ValueError("fake tokenizer only supports max_length padding")
        pad_to = int(max_length)
        out = {}
        for key, rows in encodings.items():
            padded = []
            pad_val = 0
            if key == "input_ids":
                pad_val = self.pad_token_id
            for row in rows:
                r = list(row)[:pad_to]
                r += [pad_val] * max(0, pad_to - len(r))
                padded.append(r)
            out[key] = padded

        if return_tensors == "np":
            import numpy as np

            out = {k: np.asarray(v) for k, v in out.items()}
        return out


def _install_fake_transformers():
    hf_cache._TOKENIZER_CACHE.clear()
    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)


def _install_fake_transformers_with_decoder_config(*, is_decoder):
    hf_cache._TOKENIZER_CACHE.clear()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return types.SimpleNamespace(is_encoder_decoder=False, is_decoder=bool(is_decoder))

    sys.modules["transformers"] = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoConfig=FakeAutoConfig,
    )


def test_sequence_dynamic_padding_metadata():
    _install_fake_transformers()
    train, test, meta = preprocess_hf(
        (DummySplit([{"text": "short", "label": 0}, {"text": "a much longer sample", "label": 1}]), None),
        (DummySplit([{"text": "tiny", "label": 0}]), None),
        {"hf_task": "sequence_classification", "modality": "text", "max_length": 12, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        text_column="text",
        label_column="label",
        dynamic_padding=True,
    )
    assert train[0]["input_ids"].shape[1] <= 12
    assert meta["dynamic_padding"] is True
    assert meta["padding_mode"] == "dynamic"


def test_fill_mask_dynamic_padding_metadata():
    _install_fake_transformers()
    train, test, meta = preprocess_hf(
        (DummySplit([{"text": "mask me now"}, {"text": "mask this too"}]), None),
        (DummySplit([{"text": "tiny"}]), None),
        {"hf_task": "fill_mask", "modality": "text", "max_length": 10, "hf_id": "dummy", "seed": 7},
        hf_model_id="dummy/model",
        text_column="text",
        dynamic_padding=True,
    )
    assert train[0]["input_ids"].shape == train[1].shape
    assert meta["padding_mode"] == "dynamic"
    assert meta["dynamic_padding"] is True


def test_similarity_dynamic_padding_metadata():
    _install_fake_transformers()
    train, test, meta = preprocess_hf(
        (DummySplit([
            {"a": "hello world", "b": "hello", "label": 0},
            {"a": "a much longer sentence here", "b": "pair", "label": 1},
        ]), None),
        (DummySplit([{"a": "foo", "b": "bar", "label": 0}]), None),
        {"hf_task": "sentence_similarity", "modality": "text", "max_length": 14, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        text_column=["a", "b"],
        label_column="label",
        dynamic_padding=True,
    )
    assert train[0]["input_ids"].shape[1] <= 14
    assert meta["padding_mode"] == "dynamic"
    assert meta["dynamic_padding"] is True


def test_cached_tokenizer_left_pads_decoder_only_models_even_outside_generation_task():
    _install_fake_transformers_with_decoder_config(is_decoder=True)

    tokenizer, _, _ = hf_cache.get_cached_tokenizer(
        hf_model_id="dummy/model",
        task="sequence_classification",
        device="cpu",
        transformers_module=sys.modules["transformers"],
    )

    assert tokenizer.padding_side == "left"


def test_cached_tokenizer_left_pads_decoder_only_image_caption_generation():
    _install_fake_transformers_with_decoder_config(is_decoder=False)

    tokenizer, _, _ = hf_cache.get_cached_tokenizer(
        hf_model_id="microsoft/git-base-textcaps",
        task="image_captioning",
        device="cpu",
        transformers_module=sys.modules["transformers"],
    )

    assert tokenizer.padding_side == "left"
