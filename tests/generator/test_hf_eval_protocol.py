"""Exercise evaluation using tiny tensors and a deterministic dummy decoder."""
import numpy as np
import pytest
import torch

from mlaas_data_generator.models.adapters.hf_task import Seq2SeqGenerationSpec
from tests.generator.test_hf_training_precision import tiny_core
from tests.generator.test_task_runtime import Decoder, Tokenizer, fixture


def make_core(**decoder_args):
    core = tiny_core()
    core.model = Decoder(**decoder_args)
    core.model.config.is_encoder_decoder = True
    core.task_spec = Seq2SeqGenerationSpec()
    core.task_tag = "summarization"
    core.tokenizer = Tokenizer()
    core.generation_config = {"max_new_tokens": 3, "do_sample": False}
    core.autocast_enabled = False
    return core


def test_finetuned_seq2seq_quality_uses_generation_and_separate_teacher_loss():
    _, _, x, y = fixture()
    core = make_core()
    loss, rouge1, rouge2, qos = core.eval(x, y, inference_only=False)
    assert len(core.model.generate_inputs) == 2
    assert rouge1 == pytest.approx(2 / 3) and rouge2 == pytest.approx(2 / 3)
    assert loss == pytest.approx(np.log(10))
    assert qos["evaluation_protocol"] == "source_only_generation_v1"
    assert qos["evaluation_loss_protocol"] == "teacher_forced_v1"


@pytest.mark.parametrize("bad", ["loss", "logits"])
def test_nonfinite_teacher_results_cannot_be_saved_as_zero_rouge(bad):
    _, _, x, y = fixture()
    core = make_core()
    original = core.model.forward
    def forward(**kwargs):
        result = original(**kwargs)
        if bad == "loss":
            result.loss = torch.tensor(float("nan"))
        else:
            result.logits[:] = float("inf")
        return result
    core.model.forward = forward
    with pytest.raises(ValueError, match=f"nonfinite_evaluation_{bad}"):
        core.eval(x, y, inference_only=False)


@pytest.mark.parametrize("args,reason", [
    ({"missing": True}, "invalid_evaluation_predictions"),
    ({"short_logits": True}, "invalid_evaluation_alignment"),
])
def test_missing_or_misaligned_predictions_fail(args, reason):
    _, _, x, y = fixture()
    with pytest.raises(ValueError, match=reason):
        make_core(**args).eval(x, y)


@pytest.mark.parametrize("bad", ["loss", "logits"])
@pytest.mark.parametrize("persistent", [False, True])
def test_nonfinite_amp_evaluation_restarts_in_fp32(bad, persistent):
    _, _, x, y = fixture()
    core = make_core()
    core.autocast_enabled = True
    original = core.model.forward
    calls = []

    def forward(**kwargs):
        calls.append(core.autocast_enabled)
        result = original(**kwargs)
        # Fail on the second batch to exercise discarding partial metrics.
        if persistent or (core.autocast_enabled and len(calls) == 2):
            if bad == "loss":
                result.loss = torch.tensor(float("nan"))
            else:
                result.logits[:] = float("inf")
        return result

    core.model.forward = forward
    if persistent:
        with pytest.raises(ValueError, match=f"nonfinite_evaluation_{bad}"):
            core.eval(x, y)
        assert calls == [True, False]
    else:
        _, primary, secondary, qos = core.eval(x, y)
        assert calls == [True, True, False, False]
        assert primary == pytest.approx(2 / 3) and secondary == pytest.approx(2 / 3)
        assert qos["eval_fp32_retry_count"] == 1
        assert qos["eval_sequence_count"] == len(y)
    assert not core.autocast_enabled
    assert core.precision_fallback_reason == "nonfinite_evaluation_fallback_fp32"
