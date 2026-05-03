import numpy as np
import pytest

from mlaas_data_generator.models.adapters.hf_task import (
    ImageCaptioningSpec,
    SentenceSimilaritySpec,
    TextImageRetrievalSpec,
    VQASpec,
)
from mlaas_data_generator.models.adapters.hf_core import HFCore


def _dummy_core(torch, task_spec, model, *, max_length=3):
    core = HFCore.__new__(HFCore)
    core.torch = torch
    core.task_spec = task_spec
    core.tokenizer = None
    core.model = model
    core.generation_config = {"max_new_tokens": 4, "num_beams": 1, "do_sample": False}
    core.batch_size = 2
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = max_length
    core.requested_max_length = max_length
    core.model_text_max_length = None
    core.max_length_adjusted = False
    core.model_id = "dummy"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True
    return core


def test_image_captioning_metrics_cider_and_bleu():
    spec = ImageCaptioningSpec()
    y_true = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y_pred = np.array([[1, 2, 9, 4], [5, 0, 7, 8]])
    out = spec.metrics(y_true, y_pred)
    assert "cider" in out["named_metrics"]
    assert "bleu" in out["named_metrics"]
    assert out["primary"] >= out["secondary"]


def test_retrieval_metrics_from_statistics_recall_at_k():
    spec = TextImageRetrievalSpec()
    out = spec.metrics_from_statistics({"r1_correct": 3, "r5_correct": 7, "r10_correct": 9, "total": 10})
    assert np.isclose(out["named_metrics"]["accuracy"], 0.3)
    assert np.isclose(out["named_metrics"]["top1_accuracy"], 0.3)
    assert np.isclose(out["named_metrics"]["r@1"], 0.3)
    assert np.isclose(out["named_metrics"]["r@5"], 0.7)
    assert np.isclose(out["named_metrics"]["r@10"], 0.9)


def test_retrieval_loss_is_symmetric_contrastive_and_backwardable():
    torch = pytest.importorskip("torch")

    spec = TextImageRetrievalSpec()
    logits = torch.tensor(
        [[3.0, 0.5, -1.0], [0.1, 2.5, 0.0], [-0.5, 0.2, 2.0]],
        requires_grad=True,
    )
    labels = torch.zeros((3,), dtype=torch.long)

    loss = spec.loss_fn(torch, logits, labels, {})
    targets = torch.arange(3)
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(logits, targets)
        + torch.nn.functional.cross_entropy(logits.transpose(0, 1), targets)
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sentence_similarity_regression_metrics_use_zero_for_constant_predictions():
    spec = SentenceSimilaritySpec(is_regression=True)
    y_true = np.array([0.0, 0.3, 0.7, 1.0], dtype="float32")
    y_pred = np.array([0.5, 0.5, 0.5, 0.5], dtype="float32")

    out = spec.metrics(y_true, y_pred, y_extra={"is_regression": True})

    assert out["primary"] == pytest.approx(0.0)
    assert out["secondary"] == pytest.approx(0.0)
    assert out["named_metrics"]["pearson"] == pytest.approx(0.0)
    assert out["named_metrics"]["spearman"] == pytest.approx(0.0)


def test_vqa_metrics_exact_match_with_normalization():
    spec = VQASpec()
    y_true = np.array(["The cat", "an apple", "blue"], dtype=object)
    y_pred = np.array(["cat", "apple!", "red"], dtype=object)
    out = spec.metrics(y_true, y_pred)
    assert np.isclose(out["named_metrics"]["exact_match"], 2 / 3)


def test_vqa_metrics_ignore_unseen_classification_labels():
    spec = VQASpec(label_format="vqa_class_index")
    y_true = np.array([0, -100, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 0], dtype=np.int64)

    out = spec.metrics(y_true, y_pred, y_extra={"ignore_index": -100})

    assert np.isclose(out["named_metrics"]["exact_match"], 0.5)


def test_captioning_metrics_decode_token_labels_when_tokenizer_available():
    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def batch_decode(self, rows, **kwargs):
            vocab = {1: "a", 2: "cat", 3: "dog"}
            return [" ".join(vocab.get(int(tok), "") for tok in row).strip() for row in rows]

    spec = ImageCaptioningSpec()
    y_true = np.array([[1, 2, -100]], dtype=np.int64)
    y_pred = np.array([[1, 2, 0]], dtype=np.int64)

    out = spec.metrics(y_true, y_pred, y_extra={"tokenizer": TinyTokenizer(), "ignore_index": -100})

    assert np.isclose(out["named_metrics"]["cider"], 1.0)
    assert np.isclose(out["named_metrics"]["bleu"], 1.0)


def test_vqa_metrics_decode_generative_token_labels():
    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def batch_decode(self, rows, **kwargs):
            vocab = {1: "the", 2: "cat", 3: "dog"}
            return [" ".join(vocab.get(int(tok), "") for tok in row).strip() for row in rows]

    spec = VQASpec(label_format="vqa_token_index")
    y_true = np.array([[1, 2, -100], [3, -100, -100]], dtype=np.int64)
    y_pred = np.array([[2, 0, 0], [2, 0, 0]], dtype=np.int64)

    out = spec.metrics(y_true, y_pred, y_extra={"tokenizer": TinyTokenizer(), "ignore_index": -100})

    assert np.isclose(out["named_metrics"]["exact_match"], 0.5)
    assert np.isclose(out["secondary"], 0.0)
    assert np.isclose(out["named_metrics"]["answer_token_accuracy"], 0.0)


def test_vqa_generate_predictions_trims_decoder_only_prompt_tokens():
    torch = pytest.importorskip("torch")

    class TinyTokenizer:
        def batch_decode(self, rows, **kwargs):
            vocab = {1: "what", 2: "color", 3: "blue"}
            return [" ".join(vocab.get(int(tok), "") for tok in row).strip() for row in rows]

    class PromptEchoModel:
        def generate(self, **kwargs):
            input_ids = kwargs["input_ids"]
            answer = torch.full((input_ids.shape[0], 1), 3, dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat([input_ids, answer], dim=1)

    spec = VQASpec(label_format="vqa_token_index")
    enc = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    preds = spec.generate_predictions(PromptEchoModel(), enc, TinyTokenizer(), torch, {})

    assert preds.tolist() == ["blue"]


def test_hfcore_finetune_dummy_retrieval_model():
    torch = pytest.importorskip("torch")

    class DummyRetrievalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            batch = int(kwargs["input_ids"].shape[0])
            logits = self.scale * torch.eye(batch)
            return type("Out", (), {"logits_per_text": logits})

    core = _dummy_core(torch, TextImageRetrievalSpec(), DummyRetrievalModel())
    xs = {
        "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "attention_mask": np.ones((2, 2), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 2, 2), dtype=np.float32),
    }
    ys = np.zeros((2,), dtype=np.int64)

    out = core.finetune(xs, ys, epochs=1, lr=1e-3, max_train_time_s=10)

    assert out["train_loss"] == out["train_loss"]
    assert out["train_sequence_count"] == 2


def test_hfcore_finetune_dummy_caption_and_vqa_generation_models():
    torch = pytest.importorskip("torch")

    class DummyTokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logit_bias = torch.nn.Parameter(torch.zeros(5))

        def forward(self, **kwargs):
            labels = kwargs["labels"]
            logits = self.logit_bias.reshape(1, 1, -1).repeat(labels.shape[0], labels.shape[1], 1)
            return type("Out", (), {"logits": logits})

    xs = {
        "input_ids": np.asarray([[1, 2, 0], [3, 4, 0]], dtype=np.int64),
        "attention_mask": np.ones((2, 3), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 2, 2), dtype=np.float32),
    }
    ys = np.asarray([[1, 2, -100], [1, 3, -100]], dtype=np.int64)

    caption_core = _dummy_core(torch, ImageCaptioningSpec(), DummyTokenModel())
    vqa_core = _dummy_core(torch, VQASpec(label_format="vqa_token_index"), DummyTokenModel())

    caption_out = caption_core.finetune(xs, ys, epochs=1, lr=1e-3, max_train_time_s=10)
    vqa_out = vqa_core.finetune(xs, ys, epochs=1, lr=1e-3, max_train_time_s=10)

    assert caption_out["train_supervised_token_count"] == 4
    assert vqa_out["train_supervised_token_count"] == 4


def test_hfcore_finetune_dummy_vqa_classification_model():
    torch = pytest.importorskip("torch")

    class DummyVQAClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))

        def forward(self, **kwargs):
            batch = int(kwargs["input_ids"].shape[0])
            return type("Out", (), {"logits": self.logits.reshape(1, -1).repeat(batch, 1)})

    core = _dummy_core(torch, VQASpec(label_format="vqa_class_index"), DummyVQAClassifier())
    xs = {
        "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "attention_mask": np.ones((2, 2), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 2, 2), dtype=np.float32),
    }
    ys = np.asarray([0, 1], dtype=np.int64)

    out = core.finetune(xs, ys, epochs=1, lr=1e-3, max_train_time_s=10)

    assert out["train_loss"] == out["train_loss"]
    assert out["train_sequence_count"] == 2


def test_vqa_encode_batch_truncates_pretokenized_questions_to_effective_length():
    torch = pytest.importorskip("torch")

    spec = VQASpec()
    xs = {
        "input_ids": np.ones((2, 48), dtype=np.int64),
        "attention_mask": np.ones((2, 48), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 4, 4), dtype=np.float32),
    }

    enc, labels_t, extra = spec.encode_batch(
        None,
        xs,
        np.asarray(["cat", "dog"], dtype=object),
        max_length=40,
        torch=torch,
        device="cpu",
    )

    assert enc["input_ids"].shape == (2, 40)
    assert enc["attention_mask"].shape == (2, 40)
    assert labels_t is None
    assert extra["answer_texts"].tolist() == ["cat", "dog"]


def test_hfcore_syncs_max_length_to_model_text_limit():
    core = HFCore.__new__(HFCore)
    core.tokenizer = type("Tokenizer", (), {"model_max_length": 40})()
    core.model = type("Model", (), {"config": type("Config", (), {"max_position_embeddings": 40})()})()
    core.max_length = 48
    core.requested_max_length = 48
    core.model_text_max_length = None
    core.max_length_adjusted = False

    assert core.sync_effective_max_length() == 40
    assert core.max_length == 40
    assert core.requested_max_length == 48
    assert core.model_text_max_length == 40
    assert core.max_length_adjusted is True


def test_hfcore_vqa_inference_keeps_text_answers_for_metrics():
    torch = pytest.importorskip("torch")

    class DummyVQAModel:
        config = type("Config", (), {"id2label": {0: "cat", 1: "home"}})()

        def eval(self):
            return None

        def __call__(self, **kwargs):
            batch = int(kwargs["input_ids"].shape[0])
            logits = torch.zeros((batch, 2), dtype=torch.float32)
            logits[:, 0] = 10.0
            return type("Out", (), {"logits": logits})

    core = HFCore.__new__(HFCore)
    core.torch = torch
    core.task_spec = VQASpec()
    core.tokenizer = None
    core.model = DummyVQAModel()
    core.generation_config = {"max_new_tokens": 4, "num_beams": 1, "do_sample": False}
    core.batch_size = 2
    core.device = "cpu"
    core.label_pad_value = -100
    core.max_length = 4
    core.model_id = "dummy-vqa"
    core.weight_format = None
    core.task_tag = None
    core.tokenizer_load_s = 0.0
    core.model_load_s = 0.0
    core.tokenizer_cache_hit = True
    core.model_cache_hit = True

    xs = {
        "input_ids": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "attention_mask": np.ones((2, 2), dtype=np.int64),
        "pixel_values": np.zeros((2, 3, 4, 4), dtype=np.float32),
    }
    ys = np.asarray(["cat", "cat"], dtype=object)

    loss, primary, secondary, qos = core.eval(xs, ys, inference_only=True)

    assert np.isnan(loss)
    assert np.isclose(primary, 1.0)
    assert np.isclose(secondary, 1.0)
    assert np.isclose(qos["exact_match"], 1.0)
    assert np.isclose(qos["answer_token_accuracy"], 1.0)


def test_vqa_build_model_routes_answer_text_git_models_to_generative_loader():
    class DummyConfig:
        model_type = "git"

    class GitForCausalLM:
        called = 0

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.called += 1
            return {"loader": "git", "model_id": model_id}

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            return DummyConfig()

    TransformersStub = type(
        "TransformersStub",
        (),
        {
            "AutoConfig": AutoConfig,
            "GitForCausalLM": GitForCausalLM,
        },
    )

    spec = VQASpec(label_format="answer_text")
    model = spec.build_model(TransformersStub, "microsoft/git-base-vqav2", num_labels=None)

    assert model["loader"] == "git"
    assert GitForCausalLM.called == 1


def test_vqa_build_model_rejects_classification_labels_for_git_family():
    class DummyConfig:
        model_type = "git"

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            return DummyConfig()

    TransformersStub = type(
        "TransformersStub",
        (),
        {
            "AutoConfig": AutoConfig,
        },
    )

    spec = VQASpec(label_format="vqa_class_index")

    with pytest.raises(ValueError, match="generative VQA architecture family 'git'"):
        spec.build_model(TransformersStub, "microsoft/git-base-vqav2", num_labels=2)


def test_vqa_build_model_routes_answer_text_vilt_models_to_classification_loader():
    class DummyConfig:
        model_type = "vilt"

    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id):
            return DummyConfig()

    class AutoModelForVisualQuestionAnswering:
        called = 0

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.called += 1
            return {"loader": "vilt", "model_id": model_id}

    TransformersStub = type(
        "TransformersStub",
        (),
        {
            "AutoConfig": AutoConfig,
            "AutoModelForVisualQuestionAnswering": AutoModelForVisualQuestionAnswering,
        },
    )

    spec = VQASpec(label_format="answer_text")
    model = spec.build_model(TransformersStub, "dandelin/vilt-b32-finetuned-vqa", num_labels=None)

    assert model["loader"] == "vilt"
    assert AutoModelForVisualQuestionAnswering.called == 1
