import contextlib
import numpy as np
import torch

from mlaas_data_generator.federated import perturbation as federated_perturbation
from mlaas_data_generator.models.adapters.hf_core import HFCore
from mlaas_data_generator.models.adapters.hf_task import (
    CausalLMGenerationSpec,
    Seq2SeqGenerationSpec,
    TextImageRetrievalSpec,
)
from mlaas_data_generator.services import perturbation as service_perturbation


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.ndim = self.value.ndim
        self.shape = self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)

    def item(self):
        return self.value.item()

    def sum(self):
        return FakeTensor(np.asarray(self.value.sum()))

    def __ne__(self, other):
        return FakeTensor(self.value != other)


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(value, dtype=None, device=None):
        return FakeTensor(value)

    @staticmethod
    @contextlib.contextmanager
    def no_grad():
        yield


class BrokenCountMask:
    def sum(self):
        return FakeTensor(np.asarray(2**62, dtype=np.int64))


class BrokenCountTensor(FakeTensor):
    def __ne__(self, other):
        return BrokenCountMask()


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    padding_side = "right"


class AlwaysOverflowTokenizer(DummyTokenizer):
    vocab_size = 32

    def __len__(self):
        return self.vocab_size

    def batch_decode(self, values, **kwargs):
        raise OverflowError("out of range integral type conversion attempted")


class EosPromptTokenizer(DummyTokenizer):
    def __call__(self, texts=None, text_target=None, truncation=True, padding=False, max_length=8, add_special_tokens=True, **kwargs):
        seqs = text_target if text_target is not None else texts
        if isinstance(seqs, str):
            seqs = [seqs]
        ids = []
        masks = []
        for idx, _ in enumerate(seqs):
            token_ids = [10 + idx, 20 + idx]
            if text_target is None and add_special_tokens:
                token_ids.append(self.eos_token_id)
            ids.append(token_ids[: int(max_length)])
            masks.append([1] * len(ids[-1]))
        out = {"input_ids": ids}
        if kwargs.get("return_attention_mask", True):
            out["attention_mask"] = masks
        return out


class DummyGenerationModel:
    def __init__(self):
        self.forward_calls = 0
        self.generate_use_cache_values = []
        self.gradient_checkpointing_disable_calls = 0
        self.gradient_checkpointing_enable_calls = 0
        self.config = type("Cfg", (), {"is_encoder_decoder": False, "use_cache": False})()

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_use_cache_values.append(bool(getattr(self.config, "use_cache", None)))
        batch = kwargs["input_ids"].numpy()
        generated = []
        for row in batch:
            prompt_tokens = [tok for tok in row.tolist() if tok != 0]
            generated.append(prompt_tokens + [7, 8])
        max_len = max(len(row) for row in generated)
        padded = [row + [0] * (max_len - len(row)) for row in generated]
        return FakeTensor(np.asarray(padded, dtype=np.int64))

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_disable_calls += 1

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enable_calls += 1

    def __call__(self, **kwargs):
        self.forward_calls += 1
        labels = kwargs["labels"].numpy()
        logits = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.float32)
        return type("Out", (), {"logits": FakeTensor(logits), "loss": FakeTensor(np.asarray(0.5, dtype=np.float32))})


class DummyDetectionModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        logits = np.asarray([[0.1, 0.9]], dtype=np.float32)
        return type("Out", (), {"logits": FakeTensor(logits), "pred_boxes": FakeTensor(np.zeros((1, 1, 4), dtype=np.float32))})


class DummyClipModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        import torch

        device = kwargs["input_ids"].device
        logits = torch.tensor([[8.0, 1.0], [0.5, 7.0]], dtype=torch.float32, device=device)
        return type(
            "CLIPOutput",
            (),
            {
                "logits_per_text": logits,
                "logits_per_image": logits.transpose(0, 1),
            },
        )


class DummyClipEmbeddingModel:
    def __init__(self, image_embeds, text_embeds):
        self.image_embeds = image_embeds
        self.text_embeds = text_embeds

    def eval(self):
        return self

    def __call__(self, **kwargs):
        idx = kwargs["input_ids"][:, 0].detach().cpu().numpy().astype(int)
        device = kwargs["input_ids"].device
        image_embeds = torch.tensor(self.image_embeds[idx], dtype=torch.float32, device=device)
        text_embeds = torch.tensor(self.text_embeds[idx], dtype=torch.float32, device=device)
        logits = text_embeds @ image_embeds.transpose(0, 1)
        return type(
            "CLIPOutput",
            (),
            {
                "image_embeds": image_embeds,
                "text_embeds": text_embeds,
                "logits_per_text": logits,
                "logits_per_image": logits.transpose(0, 1),
            },
        )


class DummyGenerationSpec:
    name = "seq2seq_generation"
    supports_generation = True

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        enc = {
            "input_ids": torch.tensor(xb["input_ids"], dtype=torch.long, device=device),
            "attention_mask": torch.tensor(xb["attention_mask"], dtype=torch.long, device=device),
        }
        labels = None
        if yb is not None and not inference_only:
            labels = torch.tensor(yb, dtype=torch.long, device=device)
        return enc, labels, {"ignore_index": int(ignore_index)}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        out = dict(enc)
        if labels_t is not None and not inference_only:
            out["labels"] = labels_t
        return out

    def generate_predictions(self, model, enc, tokenizer, torch, generation_config):
        generated = model.generate(**enc, **generation_config)
        in_len = enc["input_ids"].shape[1]
        return FakeTensor(generated.numpy()[:, in_len:])

    def extract_loss(self, torch, outputs, logits, labels_t, extra):
        return outputs.loss

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        return None

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        return None

    def metrics_from_statistics(self, stats):
        return None

    def metrics(self, y_true, y_pred, y_extra=None):
        common = min(y_true.shape[-1], y_pred.shape[-1])
        score = float((y_true[:, :common] == y_pred[:, :common]).mean())
        return {"primary": score, "secondary": 0.25, "named_metrics": {"token_accuracy": score}}


class DummyDetectionSpec:
    name = "image_detection"
    supports_generation = False

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, ignore_index=-100, inference_only=False):
        enc = {"pixel_values": torch.tensor(xb["pixel_values"], dtype=torch.long, device=device)}
        labels = None if yb is None else [{"classes": torch.tensor([0], dtype=torch.long, device=device)}]
        return enc, labels, {}

    def build_forward_inputs(self, enc, labels_t=None, inference_only=False):
        return dict(enc)

    def preds_from_logits(self, torch, logits, extra):
        return logits

    def batch_metric_statistics(self, torch, logits, labels_t, extra):
        return {"tp": 1.0} if labels_t is not None else None

    def batch_metric_statistics_from_outputs(self, torch, outputs, labels_t, extra):
        return {"gt": 1.0} if labels_t is not None else None

    def metrics_from_statistics(self, stats):
        tp = float(stats.get("tp", 0.0))
        gt = float(stats.get("gt", 0.0))
        score = tp / gt if gt > 0 else np.nan
        return {"primary": score, "secondary": score, "named_metrics": {"map": score}}

    def metrics(self, y_true, y_pred, y_extra=None):
        return {"primary": np.nan, "secondary": np.nan}


def _make_generation_probe_core():
    core = HFCore.__new__(HFCore)
    core.torch = FakeTorch()
    core.task_spec = DummyGenerationSpec()
    core.tokenizer = DummyTokenizer()
    core.model = DummyGenerationModel()
    core.generation_config = {}
    core.batch_size = 1
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True
    core.gradient_checkpointing_enabled = True
    return core


def test_causal_lm_inference_only_strips_supervised_suffix_from_prompt_tokens():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    xb = {
        "input_ids": np.asarray([[11, 12, 21, 22, 99]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 1, 1, 1]], dtype=np.int64),
    }
    yb = np.asarray([[-100, -100, 21, 22, 99]], dtype=np.int64)

    enc, labels_t, extra = spec.encode_batch(
        DummyTokenizer(),
        xb,
        yb,
        max_length=5,
        torch=fake_torch,
        device="cpu",
        inference_only=True,
    )

    assert enc["input_ids"].numpy().tolist() == [[11, 12]]
    assert enc["attention_mask"].numpy().tolist() == [[1, 1]]
    assert labels_t.numpy().tolist() == yb.tolist()
    assert extra["ignore_index"] == -100


def test_hfcore_eval_inference_only_generation_uses_teacher_forced_labels_for_metrics_and_loss():
    core = HFCore.__new__(HFCore)
    core.torch = FakeTorch()
    core.task_spec = DummyGenerationSpec()
    core.tokenizer = DummyTokenizer()
    core.model = DummyGenerationModel()
    core.generation_config = {}
    core.batch_size = 2
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True
    core.gradient_checkpointing_enabled = True

    xs = {
        "input_ids": np.asarray([[5, 6], [7, 8]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1], [1, 1]], dtype=np.int64),
    }
    ys = np.asarray([[7, 8], [7, 8]], dtype=np.int64)

    loss, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isclose(loss, 0.5)
    assert np.isclose(primary, 1.0)
    assert np.isclose(secondary, 0.25)
    assert qos["eval_supervised_token_count"] == 4
    assert qos["tokens_total"] == 4
    assert core.model.forward_calls == 1
    assert core.model.generate_use_cache_values == [True]
    assert core.model.gradient_checkpointing_disable_calls == 1
    assert core.model.gradient_checkpointing_enable_calls == 1
    assert core.model.config.use_cache is False
    assert core.gradient_checkpointing_enabled is True
    assert core.tokenizer.padding_side == "left"


def test_hfcore_batch_iter_trims_common_text_padding_per_batch():
    core = HFCore.__new__(HFCore)
    core.batch_size = 2
    core.label_pad_value = -100

    xs = {
        "input_ids": np.asarray([[11, 12, 0, 0], [21, 22, 23, 0]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
        "token_type_ids": np.asarray([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64),
    }
    ys = np.asarray([[31, 32, -100, -100], [41, 42, 43, -100]], dtype=np.int64)

    xb, yb = next(core._batch_iter(xs, ys))

    assert xb["input_ids"].shape == (2, 3)
    assert xb["attention_mask"].shape == (2, 3)
    assert xb["token_type_ids"].shape == (2, 3)
    assert yb.shape == (2, 3)
    assert xb["input_ids"].tolist() == [[11, 12, 0], [21, 22, 23]]
    assert yb.tolist() == [[31, 32, -100], [41, 42, 43]]


def test_hfcore_batch_iter_trims_common_left_padding_per_batch():
    core = HFCore.__new__(HFCore)
    core.batch_size = 2
    core.label_pad_value = -100

    xs = {
        "input_ids": np.asarray([[0, 0, 11, 12], [0, 21, 22, 23]], dtype=np.int64),
        "attention_mask": np.asarray([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.int64),
    }

    xb, _ = next(core._batch_iter(xs, None))

    assert xb["input_ids"].shape == (2, 3)
    assert xb["attention_mask"].shape == (2, 3)
    assert xb["input_ids"].tolist() == [[0, 11, 12], [21, 22, 23]]
    assert xb["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]


def test_hfcore_batch_iter_keeps_fill_mask_labels_aligned_to_trimmed_inputs():
    core = HFCore.__new__(HFCore)
    core.batch_size = 2
    core.label_pad_value = -100

    xs = {
        "input_ids": np.asarray([[11, 12, 13, 0], [21, 22, 23, 24]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int64),
    }
    ys = np.asarray([[-100, 12, -100, -100], [-100, 22, -100, -100]], dtype=np.int64)

    xb, yb = next(core._batch_iter(xs, ys))

    assert xb["input_ids"].shape == (2, 4)
    assert xb["attention_mask"].shape == (2, 4)
    assert yb.shape == (2, 4)
    assert yb.tolist() == [[-100, 12, -100, -100], [-100, 22, -100, -100]]


def test_service_perturbation_probe_left_pads_decoder_only_generation():
    core = _make_generation_probe_core()
    adapter = type("Adapter", (), {"core": core})()

    probe = service_perturbation._predict_hf_probe(
        adapter,
        {
            "input_ids": np.asarray([5, 6], dtype=np.int64),
            "attention_mask": np.asarray([1, 1], dtype=np.int64),
        },
        task_family="generation",
        hf_task="causal_lm_generation",
    )

    assert probe is not None
    assert core.tokenizer.padding_side == "left"


def test_federated_perturbation_probe_left_pads_decoder_only_generation():
    core = _make_generation_probe_core()
    adapter = type("Adapter", (), {"core": core})()

    probe = federated_perturbation._predict_hf_probe(
        adapter,
        {
            "input_ids": np.asarray([5, 6], dtype=np.int64),
            "attention_mask": np.asarray([1, 1], dtype=np.int64),
        },
        task_family="generation",
        hf_task="causal_lm_generation",
    )

    assert probe is not None
    assert core.tokenizer.padding_side == "left"


def test_service_perturbation_runtime_limits_cap_detection_work():
    candidate_limit, trust_trials, random_trials, budget_fractions, max_duration_s, adjustments = (
        service_perturbation._resolve_runtime_limits(
            {
                "perturbation_max_duration_s": 30,
                "perturbation_detection_max_duration_s": 15,
                "perturbation_detection_candidate_units_cap": 2,
                "perturbation_detection_budget_count_cap": 1,
                "perturbation_detection_random_trials_cap": 2,
                "perturbation_detection_trust_trials_cap": 1,
            },
            task_family="detection",
            candidate_limit=4,
            trust_trials=3,
            random_trials=8,
            budget_fractions=[0.1, 0.2, 0.3],
        )
    )

    assert candidate_limit == 2
    assert trust_trials == 1
    assert random_trials == 2
    assert budget_fractions == [0.1]
    assert max_duration_s == 15
    assert adjustments


def test_federated_perturbation_runtime_limits_cap_detection_work():
    candidate_limit, trust_trials, random_trials, budget_fractions, max_duration_s, adjustments = (
        federated_perturbation._resolve_runtime_limits(
            {
                "perturbation_max_duration_s": 30,
                "perturbation_detection_max_duration_s": 15,
                "perturbation_detection_candidate_units_cap": 2,
                "perturbation_detection_budget_count_cap": 1,
                "perturbation_detection_random_trials_cap": 2,
                "perturbation_detection_trust_trials_cap": 1,
            },
            task_family="detection",
            candidate_limit=4,
            trust_trials=3,
            random_trials=8,
            budget_fractions=[0.1, 0.2, 0.3],
        )
    )

    assert candidate_limit == 2
    assert trust_trials == 1
    assert random_trials == 2
    assert budget_fractions == [0.1]
    assert max_duration_s == 15
    assert adjustments


def test_service_perturbation_skips_heavy_multimodal_tasks_by_default():
    result = service_perturbation.run_perturbation_stage(
        object(),
        x_eval=[{"question": "what color"}],
        y_eval=["blue"],
        task_family="vqa",
        hf_task="visual_question_answering",
        config={"enable_perturbation_metrics": True, "perturbation_sample_count": 1},
    )

    assert result["perturbation_supported_flag"] is False
    assert result["explainability_supported_flag"] is False
    assert result["perturbation_skip_reason"] == "disabled_for_heavy_multimodal_task"
    assert result["perturbation_runtime_policy"] == "skip_heavy_multimodal"


def test_federated_perturbation_skips_heavy_multimodal_tasks_by_default():
    result = federated_perturbation.run_perturbation_stage(
        object(),
        x_eval=[{"question": "what color"}],
        y_eval=["blue"],
        task_family="vqa",
        hf_task="visual_question_answering",
        config={"enable_perturbation_metrics": True, "perturbation_sample_count": 1},
    )

    assert result["perturbation_supported_flag"] is False
    assert result["explainability_supported_flag"] is False
    assert result["perturbation_skip_reason"] == "disabled_for_heavy_multimodal_task"
    assert result["perturbation_runtime_policy"] == "skip_heavy_multimodal"


def test_hfcore_eval_pads_ragged_generated_batches_for_metrics():
    core = HFCore.__new__(HFCore)
    core.torch = FakeTorch()
    core.task_spec = DummyGenerationSpec()
    core.tokenizer = DummyTokenizer()
    core.model = DummyGenerationModel()
    core.generation_config = {}
    core.batch_size = 1
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {
        "input_ids": np.asarray([[5, 0, 0], [7, 8, 9]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 0, 0], [1, 1, 1]], dtype=np.int64),
    }
    ys = np.asarray([[7, 8], [7, 8]], dtype=np.int64)

    loss, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isclose(loss, 0.5)
    assert np.isfinite(primary)
    assert np.isclose(secondary, 0.25)
    assert qos["eval_supervised_token_count"] == 4
    assert core.model.forward_calls == 2


def test_causal_lm_encode_batch_left_pads_dict_inputs_even_without_labels():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    tok = DummyTokenizer()
    xb = {
        "input_ids": np.asarray([[10, 11, 0, 0], [20, 21, 22, 0]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
    }

    enc, labels_t, _ = spec.encode_batch(
        tok,
        xb,
        None,
        max_length=4,
        torch=fake_torch,
        device="cpu",
        inference_only=True,
    )

    assert tok.padding_side == "left"
    assert enc["input_ids"].numpy().tolist() == [[0, 0, 10, 11], [0, 20, 21, 22]]
    assert enc["attention_mask"].numpy().tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
    assert labels_t is None


def test_causal_lm_encode_batch_left_pads_dict_labels_with_inputs():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    tok = DummyTokenizer()
    xb = {
        "input_ids": np.asarray([[10, 11, 0, 0], [20, 21, 22, 0]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64),
    }
    yb = np.asarray([[10, 11, -100, -100], [20, 21, 22, -100]], dtype=np.int64)

    enc, labels_t, _ = spec.encode_batch(
        tok,
        xb,
        yb,
        max_length=4,
        torch=fake_torch,
        device="cpu",
        inference_only=False,
    )

    assert enc["input_ids"].numpy().tolist() == [[0, 0, 10, 11], [0, 20, 21, 22]]
    assert enc["attention_mask"].numpy().tolist() == [[0, 0, 1, 1], [0, 1, 1, 1]]
    assert labels_t.numpy().tolist() == [[-100, -100, 10, 11], [-100, 20, 21, 22]]


def test_causal_lm_inference_only_prompt_extraction_keeps_left_padding_for_mixed_prompt_lengths():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    tok = DummyTokenizer()
    xb = {
        "input_ids": np.asarray([[11, 21, 22, 99], [31, 32, 41, 99]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int64),
    }
    yb = np.asarray([[-100, 21, 22, 99], [-100, -100, 41, 99]], dtype=np.int64)

    enc, labels_t, _ = spec.encode_batch(
        tok,
        xb,
        yb,
        max_length=4,
        torch=fake_torch,
        device="cpu",
        inference_only=True,
    )

    assert enc["input_ids"].numpy().tolist() == [[0, 11], [31, 32]]
    assert enc["attention_mask"].numpy().tolist() == [[0, 1], [1, 1]]
    assert labels_t.numpy().tolist() == yb.tolist()


def test_causal_lm_encode_batch_strips_trailing_prompt_eos_before_appending_target():
    spec = CausalLMGenerationSpec()
    fake_torch = FakeTorch()
    tok = EosPromptTokenizer()

    enc, labels_t, _ = spec.encode_batch(
        tok,
        ["prompt one"],
        ["target one"],
        max_length=8,
        torch=fake_torch,
        device="cpu",
        inference_only=False,
    )

    assert enc["input_ids"].numpy().tolist() == [[10, 20, 10, 20, 99, 0, 0, 0]]
    assert labels_t.numpy().tolist() == [[-100, -100, 10, 20, 99, -100, -100, -100]]


def test_hfcore_eval_inference_only_non_generation_uses_label_stats_for_metrics():
    core = HFCore.__new__(HFCore)
    core.torch = FakeTorch()
    core.task_spec = DummyDetectionSpec()
    core.tokenizer = None
    core.model = DummyDetectionModel()
    core.generation_config = {}
    core.batch_size = 1
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy-det"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {"pixel_values": np.asarray([[[[1.0]]]], dtype=np.float32)}
    ys = [{"classes": np.asarray([0], dtype=np.int64), "boxes": np.asarray([[0, 0, 1, 1]], dtype=np.float32)}]

    _, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isclose(primary, 1.0)
    assert np.isclose(secondary, 1.0)
    assert np.isclose(qos["map"], 1.0)


def test_seq2seq_metrics_falls_back_when_tokenizer_decode_overflows():
    spec = Seq2SeqGenerationSpec()
    out = spec.metrics(
        np.asarray([[11, 12, -100]], dtype=np.int64),
        np.asarray([[11, 12, 10**18]], dtype=np.int64),
        y_extra={
            "task_tag": "summarization",
            "loss_mean": 0.0,
            "ignore_index": -100,
            "tokenizer": AlwaysOverflowTokenizer(),
        },
    )

    assert np.isclose(out["primary"], 1.0)
    assert np.isclose(out["secondary"], 0.8)
    assert np.isclose(out["named_metrics"]["perplexity"], 1.0)


def test_hfcore_count_supervised_tokens_uses_numpy_values_not_backend_sum():
    core = HFCore.__new__(HFCore)
    labels_t = BrokenCountTensor(np.asarray([[255, 1], [2, 255]], dtype=np.int64))

    assert core._count_supervised_tokens(labels_t, 255) == 2


def test_hfcore_eval_clip_retrieval_uses_logits_per_text_for_accuracy():
    import torch

    core = HFCore.__new__(HFCore)
    core.torch = torch
    core.task_spec = TextImageRetrievalSpec()
    core.tokenizer = None
    core.model = DummyClipModel()
    core.generation_config = {}
    core.batch_size = 2
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy-clip"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {
        "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1], [1, 1]], dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 2, 2), dtype=np.float32),
    }
    ys = np.zeros((2,), dtype=np.int64)

    loss, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isnan(loss)
    assert np.isclose(primary, 1.0)
    assert np.isclose(secondary, 1.0)
    assert np.isclose(qos["accuracy"], 1.0)
    assert np.isclose(qos["top1_accuracy"], 1.0)
    assert np.isclose(qos["r@1"], 1.0)


def test_hfcore_eval_clip_retrieval_r5_uses_full_eval_candidate_pool():
    image_embeds = np.eye(6, dtype=np.float32)
    text_embeds = np.eye(6, dtype=np.float32)
    text_embeds[5] = np.asarray([5, 4, 3, 2, 1, 0], dtype=np.float32)

    core = HFCore.__new__(HFCore)
    core.torch = torch
    core.task_spec = TextImageRetrievalSpec()
    core.tokenizer = None
    core.model = DummyClipEmbeddingModel(image_embeds, text_embeds)
    core.generation_config = {}
    core.batch_size = 3
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy-clip"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {
        "input_ids": np.asarray([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], dtype=np.int64),
        "attention_mask": np.ones((6, 2), dtype=np.int64),
        "pixel_values": np.zeros((6, 3, 2, 2), dtype=np.float32),
    }
    ys = np.zeros((6,), dtype=np.int64)

    _, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isclose(primary, 5 / 6)
    assert np.isclose(secondary, 5 / 6)
    assert np.isclose(qos["r@5"], 5 / 6)
    assert np.isclose(qos["metric_stat_candidate_count"], 6.0)


def test_hfcore_configure_precision_mode_disables_mixed_precision_on_cpu():
    core = HFCore.__new__(HFCore)
    core.torch = type("Torch", (), {"float16": object(), "bfloat16": object(), "cuda": type("Cuda", (), {"is_bf16_supported": staticmethod(lambda: True)})()})()
    core.device = "cpu"
    core.mixed_precision = True
    core.precision_type = "bf16"
    core._configure_precision_mode()

    assert core.autocast_enabled is False
    assert core.effective_mixed_precision is False
    assert core.effective_precision_type == "fp32"
    assert core.precision_fallback_reason == "mixed_precision_requires_cuda"


def test_hfcore_configure_precision_mode_enables_bf16_on_supported_cuda():
    class TorchStub:
        float16 = object()
        bfloat16 = object()
        class cuda:
            @staticmethod
            def is_bf16_supported():
                return True
        class amp:
            @staticmethod
            def GradScaler(device_type):
                raise AssertionError("bf16 should not create a GradScaler")

    core = HFCore.__new__(HFCore)
    core.torch = TorchStub()
    core.device = "cuda"
    core.mixed_precision = True
    core.precision_type = "bf16"
    core._configure_precision_mode()

    assert core.autocast_enabled is True
    assert core.effective_mixed_precision is True
    assert core.effective_precision_type == "bf16"
    assert core.grad_scaler is None


def test_hfcore_precision_runtime_falls_back_to_fp32():
    core = HFCore.__new__(HFCore)
    core.autocast_enabled = True
    core.requested_precision_type = "bf16"
    core.effective_mixed_precision = True
    core.effective_precision_type = "bf16"
    core.grad_scaler = object()
    core.precision_fallback_reason = None
    core._make_autocast_context = contextlib.nullcontext

    calls = {"count": 0}

    def flaky_forward():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("bfloat16 unsupported kernel")
        return "ok"

    assert core._run_with_precision_context(flaky_forward) == "ok"
    assert calls["count"] == 2
    assert core.autocast_enabled is False
    assert core.effective_mixed_precision is False
    assert core.effective_precision_type == "fp32"
    assert core.precision_fallback_reason == "runtime_unsupported_bf16"
