import sys
import types

import numpy as np
import pytest

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf
from mlaas_data_generator.data.preprocessors import hf_text_generation


class DummySplit:
    def __init__(self, rows):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return [r.get(key) for r in self._rows]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    padding_side = "right"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def _encode_one(self, text):
        base = [min(50, max(3, len(tok))) for tok in str(text).split()]
        return base or [3]

    def __call__(self, texts=None, text_target=None, truncation=True, padding=False, max_length=8, add_special_tokens=True, return_attention_mask=True, **kwargs):
        seqs = text_target if text_target is not None else texts
        if isinstance(seqs, str):
            seqs = [seqs]
        ids = []
        masks = []
        for t in seqs:
            token_ids = self._encode_one(t)
            if add_special_tokens:
                token_ids = [1] + token_ids
            token_ids = token_ids[: int(max_length)]
            ids.append(token_ids)
            masks.append([1] * len(token_ids))
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = masks
        return out


class EmptyPreservingFakeTokenizer(FakeTokenizer):
    def _encode_one(self, text):
        return [min(50, max(3, len(tok))) for tok in str(text).split()]


class EosAppendingFakeTokenizer(FakeTokenizer):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, texts=None, text_target=None, truncation=True, padding=False, max_length=8, add_special_tokens=True, return_attention_mask=True, **kwargs):
        out = super().__call__(
            texts=texts,
            text_target=text_target,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        if text_target is None and add_special_tokens:
            ids = []
            for token_ids in out["input_ids"]:
                token_ids = list(token_ids)
                if len(token_ids) < int(max_length):
                    token_ids.append(self.eos_token_id)
                else:
                    token_ids[-1] = self.eos_token_id
                ids.append(token_ids)
            out["input_ids"] = ids
            if return_attention_mask and "attention_mask" in out:
                out["attention_mask"] = [[1] * len(token_ids) for token_ids in ids]
        return out


def _install_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
    sys.modules["transformers"] = fake_mod


def _install_empty_preserving_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoTokenizer=EmptyPreservingFakeTokenizer)
    sys.modules["transformers"] = fake_mod


def _install_eos_appending_fake_transformers():
    fake_mod = types.SimpleNamespace(AutoTokenizer=EosAppendingFakeTokenizer)
    sys.modules["transformers"] = fake_mod


def test_seq2seq_generation_preprocessor_falls_back_to_slow_tokenizer():
    calls = []

    class FallbackTokenizer(FakeTokenizer):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs.get("use_fast"))
            if kwargs.get("use_fast"):
                raise TypeError("Input must be a List[Union[str, AddedToken]]")
            return cls()

    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=FallbackTokenizer)
    train_rows = [{"article": "Long article body", "highlights": "Short summary"}]
    test_rows = [{"article": "Held-out article", "highlights": "Held-out summary"}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    assert calls == [True, False]
    assert meta["column_mapping"] == {"source": "article", "target": "highlights"}


def test_causal_lm_generation_preprocessor_prompt_completion_mapping():
    _install_fake_transformers()
    train_rows = [
        {"prompt": "Write a haiku", "completion": "Soft rain at dusk"},
        {"prompt": "Translate hello", "completion": "bonjour"},
    ]
    test_rows = [{"prompt": "Say hi", "completion": "hi"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 12, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=6,
        target_max_length=6,
        dynamic_padding=True,
    )

    x_train, y_train = train
    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape == y_train.shape
    assert meta["column_mapping"]["prompt"] == "prompt"
    assert meta["column_mapping"]["target"] == "completion"
    assert meta["accounting"]["sequence_count"] == 2
    assert meta["accounting"]["supervised_token_count"] > 0


def test_causal_lm_generation_preprocessor_strips_trailing_prompt_eos_before_target():
    _install_eos_appending_fake_transformers()
    train_rows = [{"prompt": "Write a haiku", "completion": "Soft rain"}]
    test_rows = [{"prompt": "Say hi", "completion": "hi"}]

    train, _, _ = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 12, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=6,
        target_max_length=6,
        dynamic_padding=True,
    )

    x_train, y_train = train
    active_ids = x_train["input_ids"][0][x_train["attention_mask"][0] == 1].tolist()
    active_labels = y_train[0][y_train[0] != -100].tolist()

    assert active_ids.count(EosAppendingFakeTokenizer.eos_token_id) == 1
    assert active_labels[-1] == EosAppendingFakeTokenizer.eos_token_id


def test_seq2seq_generation_preprocessor_source_target_mapping():
    _install_fake_transformers()
    train_rows = [
        {"source_text": "summarize this long article", "target_text": "short summary"},
        {"source_text": "another source", "target_text": "target output with extra words"},
    ]
    test_rows = [{"source_text": "src", "target_text": "tgt"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape[1] <= 7
    assert y_train.shape[1] <= 5
    assert x_train["input_ids"].shape[0] == y_train.shape[0]
    assert x_test["input_ids"].shape[0] == y_test.shape[0]
    assert np.any(y_train == -100)
    assert meta["column_mapping"]["source"] == "source_text"
    assert meta["column_mapping"]["target"] == "target_text"


def test_causal_lm_generation_preprocessor_single_text_column():
    _install_fake_transformers()
    train_rows = [
        {"text": "The quick brown fox"},
        {"text": "Jumps over lazy dogs"},
    ]
    test_rows = [{"text": "Single column inference text"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 9, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert set(x_train.keys()) >= {"input_ids", "attention_mask"}
    assert x_train["input_ids"].shape == y_train.shape
    assert x_test["input_ids"].shape == y_test.shape
    assert meta["generation_mode"] == "single_text"
    assert meta["column_mapping"] == {"text": "text"}

    non_pad_train = x_train["attention_mask"] == 1
    non_pad_test = x_test["attention_mask"] == 1
    assert np.array_equal(y_train[non_pad_train], x_train["input_ids"][non_pad_train])
    assert np.array_equal(y_test[non_pad_test], x_test["input_ids"][non_pad_test])
    assert np.all(y_train[~non_pad_train] == -100)
    assert np.all(y_test[~non_pad_test] == -100)


def test_seq2seq_generation_preprocessor_article_highlights_mapping():
    _install_fake_transformers()
    train_rows = [
        {"article": "Long article body", "highlights": "Short summary"},
        {"article": "Another document", "highlights": "Another summary"},
    ]
    test_rows = [{"article": "Held-out article", "highlights": "Held-out summary"}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test

    assert x_train["input_ids"].shape[0] == y_train.shape[0]
    assert x_test["input_ids"].shape[0] == y_test.shape[0]
    assert meta["column_mapping"] == {"source": "article", "target": "highlights"}


def test_seq2seq_generation_preprocessor_column_mapping_overrides_heuristics():
    _install_fake_transformers()
    train_rows = [
        {"article": "Heuristic source", "highlights": "Heuristic target", "src": "Mapped source", "tgt": "Mapped target"},
    ]
    test_rows = [
        {"article": "Heuristic source test", "highlights": "Heuristic target test", "src": "Mapped source test", "tgt": "Mapped target test"},
    ]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        column_mapping={"source": "src", "target": "tgt"},
        source_max_length=7,
        target_max_length=5,
        dynamic_padding=True,
    )

    assert meta["column_mapping"] == {"source": "src", "target": "tgt"}


def test_causal_lm_generation_preprocessor_honors_explicit_same_text_and_label_columns():
    _install_fake_transformers()
    train_rows = [
        {"text": "review text one", "label": 1},
        {"text": "review text two", "label": 0},
    ]
    test_rows = [{"text": "held out review", "label": 1}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        text_column="text",
        label_column="text",
        dynamic_padding=True,
    )

    assert meta["generation_mode"] == "single_text"
    assert meta["column_mapping"] == {"text": "text"}


def test_seq2seq_generation_preprocessor_honors_explicit_text_and_label_columns():
    _install_fake_transformers()
    train_rows = [
        {"article": "Heuristic source", "highlights": "Heuristic target", "src": "Mapped source", "tgt": "Mapped target"},
    ]
    test_rows = [
        {"article": "Held-out source", "highlights": "Held-out target", "src": "Mapped held-out source", "tgt": "Mapped held-out target"},
    ]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        text_column="src",
        label_column="tgt",
        dynamic_padding=True,
    )

    assert meta["column_mapping"] == {"source": "src", "target": "tgt"}


def test_seq2seq_generation_preprocessor_codet5_falls_back_to_roberta_tokenizer():
    calls = []

    class BrokenAutoTokenizer(FakeTokenizer):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("auto", kwargs.get("use_fast")))
            raise TypeError("Input must be a List[Union[str, AddedToken]]")

    sys.modules["transformers"] = types.SimpleNamespace(
        AutoTokenizer=BrokenAutoTokenizer,
        RobertaTokenizer=FakeTokenizer,
    )
    original_loader = hf_text_generation._load_codet5_roberta_tokenizer
    hf_text_generation._load_codet5_roberta_tokenizer = lambda transformers, model_id: calls.append(("roberta", None)) or FakeTokenizer()
    train_rows = [{"article": "Long article body", "highlights": "Short summary"}]
    test_rows = [{"article": "Held-out article", "highlights": "Held-out summary"}]
    try:
        _, _, meta = preprocess_hf(
            (DummySplit(train_rows), None),
            (DummySplit(test_rows), None),
            {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
            hf_model_id="Salesforce/codet5-small",
            source_max_length=7,
            target_max_length=5,
            dynamic_padding=True,
        )
    finally:
        hf_text_generation._load_codet5_roberta_tokenizer = original_loader

    assert calls == [("auto", True), ("auto", False), ("roberta", None)]
    assert meta["column_mapping"] == {"source": "article", "target": "highlights"}


def test_seq2seq_generation_preprocessor_raises_without_plausible_target():
    _install_fake_transformers()
    train_rows = [{"article": "Long article body", "document": "Duplicate candidate source"}]
    test_rows = [{"article": "Held-out article", "document": "Held-out duplicate source"}]

    with pytest.raises(ValueError, match="Could not resolve target column"):
        preprocess_hf(
            (DummySplit(train_rows), None),
            (DummySplit(test_rows), None),
            {"hf_task": "seq2seq_generation", "modality": "text", "max_length": 10, "hf_id": "dummy"},
            hf_model_id="dummy/model",
            dynamic_padding=True,
        )


def test_causal_lm_generation_preprocessor_sets_left_padding_and_meta():
    _install_fake_transformers()
    train_rows = [{"text": "left pad me"}]
    test_rows = [{"text": "and me too"}]

    _, _, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 8, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        dynamic_padding=True,
    )

    assert meta["padding_side"] == "left"
    assert meta["pad_token_id"] == 0


def test_causal_lm_generation_preprocessor_keeps_blank_wikitext_rows_nonempty():
    _install_empty_preserving_fake_transformers()
    train_rows = [{"text": ""}, {"text": "   "}]
    test_rows = [{"text": ""}]

    train, test, meta = preprocess_hf(
        (DummySplit(train_rows), None),
        (DummySplit(test_rows), None),
        {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 8, "hf_id": "dummy"},
        hf_model_id="dummy/model",
        dynamic_padding=True,
    )

    x_train, y_train = train
    x_test, y_test = test
    assert x_train["input_ids"].shape[1] >= 1
    assert x_test["input_ids"].shape[1] >= 1
    assert np.all(x_train["attention_mask"].sum(axis=1) >= 1)
    assert np.all(x_test["attention_mask"].sum(axis=1) >= 1)
    assert meta["accounting"]["supervised_token_count"] == int(np.count_nonzero(y_train != -100))
