from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.services.perturbation import (
    _mask_token, _mask_token_text, _mask_token_id,
)


def adapter(size=50264, mask=50264, unk=3, pad=1):
    embeddings = torch.nn.Embedding(size, 1)
    tokenizer = SimpleNamespace(mask_token_id=mask, mask_token="<mask>",
                                unk_token_id=unk, unk_token="<unk>",
                                pad_token_id=pad, pad_token="<pad>")
    return SimpleNamespace(core=SimpleNamespace(tokenizer=tokenizer,
        model=SimpleNamespace(get_input_embeddings=lambda: embeddings,
                              config=SimpleNamespace(vocab_size=size))))


def test_distilbart_mask_outside_embeddings_uses_unknown_token():
    model = adapter()
    source = {"input_ids": np.array([0, 42, 2]), "attention_mask": np.ones(3, dtype=int)}
    result = _mask_token(source, 1, model=model)
    assert result["input_ids"].tolist() == [0, 3, 2]
    assert source["input_ids"].tolist() == [0, 42, 2]
    assert _mask_token_id(model) is None
    assert _mask_token_text(model) == "<unk>"
    assert model.core.model.get_input_embeddings()(torch.tensor(result["input_ids"])).shape == (3, 1)


@pytest.mark.parametrize("size,unk,expected", [(50265, 3, 50264), (50264, None, 1)])
def test_valid_mask_is_preserved_and_missing_unknown_falls_back_to_pad(size, unk, expected):
    model = adapter(size=size, unk=unk)
    result = _mask_token({"input_ids": np.array([42])}, 0, model=model)
    assert result["input_ids"].item() == expected


def test_actual_resized_embeddings_take_precedence_over_stale_config():
    model = adapter(size=50265)
    model.core.model.config.vocab_size = 50264
    assert _mask_token_id(model) == 50264
