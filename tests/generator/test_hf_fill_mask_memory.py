"""Bound corpus tokenization without changing source-row masking semantics."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator.data.preprocessors import hf_text_fill_mask as module


class SliceOnlyDataset:
    column_names = ["text"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        assert isinstance(key, slice), "must not materialize the full text column"
        assert key.stop - key.start <= 256
        return {"text": self.rows[key]}


class Tokenizer:
    mask_token_id = 99
    vocab_size = 100

    def __init__(self, side):
        self.padding_side = side
        self.calls = []

    def __call__(self, texts, *, max_length, **kwargs):
        self.calls.append(len(texts))
        rows = [[1] + [int(t) for t in text.split()] + [2] for text in texts]
        rows = [row[:max_length] for row in rows]
        return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows],
                "special_tokens_mask": [[int(t in (1, 2)) for t in r] for r in rows],
                "token_type_ids": [[0] * len(r) for r in rows]}

    def pad(self, enc, *, max_length, **kwargs):
        def pad(row, key):
            padding = [1 if key == "special_tokens_mask" else 0] * (max_length - len(row))
            return padding + row if self.padding_side == "left" else row + padding
        return {key: np.asarray([pad(row, key) for row in rows], dtype="int64")
                for key, rows in enc.items()}


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_batched_preparation_matches_whole_corpus(monkeypatch, side, dynamic):
    tokenizer = Tokenizer(side)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    monkeypatch.setattr(module, "get_cached_tokenizer", lambda **kwargs: (tokenizer, None, None))
    rows = ["3 4 5 " * (i % 3) for i in range(519)]
    test_rows = ["6 7 8 9 10 11 12 13", ""]
    source_rows = list(range(900, 900 + len(rows)))[::-1]
    width = 10 if dynamic else 16
    expected = []
    for texts, ids, split in ((rows, source_rows, "custom_train"), (test_rows, [70, 80], "test")):
        enc = tokenizer.pad(module._tokenize_texts(tokenizer, texts, 16, True), max_length=width)
        x, y = module._source_mlm_labels(enc, tokenizer, .4, -100, seed=73,
                                       source_ids=[f"{split}:{i}" for i in ids])
        keep = np.any(y != -100, axis=1)
        expected.append(({key: value[keep] for key, value in x.items()}, y[keep]))
    tokenizer.calls.clear()
    train, test, meta = module.preprocess_hf_text_fill_mask(
        (SliceOnlyDataset(rows), None), (SliceOnlyDataset(test_rows), None),
        {"max_length": 16, "seed": 73, "train_split": "custom_train",
         "selected_train_rows": source_rows, "selected_test_rows": [70, 80]},
        hf_model_id="offline", mlm_probability=.4, dynamic_padding=dynamic)
    assert tokenizer.calls == [256, 256, 7, 2]
    assert meta["input_shape"] == (width,)
    for actual, reference in zip((train, test), expected):
        np.testing.assert_array_equal(actual[1], reference[1])
        for key in reference[0]:
            np.testing.assert_array_equal(actual[0][key], reference[0][key])
            assert actual[0][key].dtype == np.int32
    assert meta["selected_train_rows"] == [source_rows[i] for i in meta["accepted_train_positions"]]
