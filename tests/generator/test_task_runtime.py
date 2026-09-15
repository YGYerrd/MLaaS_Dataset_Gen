from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.tasks import resolve_task_contract, TaskContractError
from mlaas_data_generator.tasks.runtime import evaluate_task_model
from mlaas_data_generator.models.adapters.hf_task import Seq2SeqGenerationSpec


class Tokenizer:
    pad_token_id = 0
    special_tokens_map = {"eos_token": "</s>", "pad_token": "<pad>"}
    padding_side = truncation_side = "right"

    def __len__(self):
        return 10

    def get_vocab(self):
        return {str(i): i for i in range(10)}

    def batch_decode(self, rows, **kwargs):
        return [" ".join(str(v) for v in row if v not in (0, 9)) for row in rows]


class Decoder(torch.nn.Module):
    config = SimpleNamespace(decoder_start_token_id=0, eos_token_id=9, pad_token_id=0)

    def __init__(self, missing=False, short_logits=False):
        super().__init__()
        self.missing, self.short_logits = missing, short_logits
        self.generate_inputs = []

    def generate(self, **kwargs):
        assert "labels" not in kwargs and "decoder_input_ids" not in kwargs
        self.generate_inputs.append(kwargs["input_ids"].clone())
        if self.missing:
            return None
        # The last source emits EOS immediately; this is a valid empty prediction.
        return torch.tensor([[1, 2, 9] if i.item() < 3 else [9, 0, 0] for i in kwargs["input_ids"][:, 0]])

    def forward(self, input_ids, attention_mask, labels):
        length = labels.shape[1] - int(self.short_logits)
        return SimpleNamespace(logits=torch.zeros(len(labels), length, 10), loss=None)


def fixture(*, batch_size=2, **decoder_args):
    core = SimpleNamespace(batch_size=batch_size, torch=torch, model=Decoder(**decoder_args), task_spec=Seq2SeqGenerationSpec(),
        tokenizer=Tokenizer(), device="cpu", max_length=3, label_pad_value=-100,
        _generation_inference_mode=nullcontext, _run_with_precision_context=lambda fn: fn(),
        _extract_logits=lambda result: result.logits)
    from mlaas_data_generator.models.adapters.hf_core import HFCore
    core._batch_item_count = HFCore._batch_item_count
    core._device_type = lambda: "cpu"
    core._batch_iter = lambda x, y: (({k: v[start:start + batch_size] for k, v in x.items()}, y[start:start + batch_size])
        for start in range(0, len(y), batch_size))
    model = SimpleNamespace(core=core)
    config = {"hf_task": "seq2seq_generation", "task_tag": "summarization", "task_type": "generation", "model_type": "hf_finetune"}
    contract = resolve_task_contract(config, model=model)
    x = {"input_ids": np.array([[1, 0], [2, 0], [3, 0]]), "attention_mask": np.array([[1, 0]] * 3)}
    y = np.array([[1, 2, -100], [1, 2, 9], [3, 9, -100]])
    return model, contract, x, y


def test_decoded_metrics_reference_isolation_empty_eos_and_uneven_batches():
    model, contract, x, y = fixture()
    streamed = []
    result = evaluate_task_model(model, x, y, contract, ["a", "b", "c"],
        prediction_sink=lambda i, p, t: streamed.append((i, p, t)))
    assert result.predictions == ()  # streaming does not retain masks/text outputs
    assert [r[0] for r in streamed] == [0, 1, 2]
    assert result.metrics["rouge1"] == pytest.approx(2 / 3)
    assert result.metrics["rouge2"] == pytest.approx(2 / 3)
    assert result.metrics["empty_output_fraction"] == pytest.approx(1 / 3)
    assert result.metrics["supervision_count"] == 7
    assert result.metrics["input_tokens"] == 3
    assert result.metrics["loss"] == pytest.approx(np.log(10))
    model, contract, x, y = fixture(batch_size=3)
    whole = evaluate_task_model(model, x, y, contract, ["a", "b", "c"])
    assert whole.metrics["rouge1"] == result.metrics["rouge1"]
    assert whole.metrics["prediction_hash"] == result.metrics["prediction_hash"]
    assert whole.metrics["loss"] == pytest.approx(result.metrics["loss"])


@pytest.mark.parametrize("failure,code", [("missing", "invalid_predictions"), ("short_logits", "prediction_alignment_mismatch")])
def test_generation_failures_are_not_empty_text_scores(failure, code):
    model, contract, x, y = fixture(**{failure: True})
    with pytest.raises(TaskContractError, match=code):
        evaluate_task_model(model, x, y, contract, ["a", "b", "c"])


def test_all_masked_targets_fail_before_execution():
    model, contract, x, y = fixture()
    y[:] = -100
    with pytest.raises(TaskContractError, match="insufficient_supervision"):
        evaluate_task_model(model, x, y, contract, ["a", "b", "c"])
    assert model.core.model.generate_inputs == []


def test_tokenizer_and_decoding_settings_change_frozen_protocol():
    model, contract, _, _ = fixture()
    config = {"hf_task": "seq2seq_generation", "task_tag": "summarization", "model_type": "hf_finetune"}
    changed = resolve_task_contract({**config, "target_max_length": 12}, model=model)
    assert changed.semantics_hash != contract.semantics_hash
    changed = resolve_task_contract({**config, "generation_config": {"max_new_tokens": 8, "length_penalty": 2}}, model=model)
    assert changed.semantics_hash == contract.semantics_hash
    assert changed.protocol_hash != contract.protocol_hash
    model.core.tokenizer.get_vocab = lambda: {"changed": 1}
    assert resolve_task_contract(config, model=model).semantics_hash != contract.semantics_hash
    with pytest.raises(TaskContractError, match="deterministic"):
        resolve_task_contract({**config, "do_sample": True}, model=model)
