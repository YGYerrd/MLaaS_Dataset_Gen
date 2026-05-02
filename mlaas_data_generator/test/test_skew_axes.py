import numpy as np
import pytest

from mlaas_data_generator.data.skew_axes import resolve_skew_axis
from mlaas_data_generator.data.splitters import split_data


def test_resolve_skew_axis_supports_core_task_families():
    cls = resolve_skew_axis(
        np.arange(6).reshape(6, 1),
        np.asarray([0, 1, 0, 1, 2, 2]),
        {},
        split_name="train",
        task_family="classification",
        hf_task="sequence_classification",
    )
    assert cls.effective_axis == "class_label"
    assert cls.bucket_spec["cardinality"] == 3

    token = resolve_skew_axis(
        {"attention_mask": np.ones((4, 3), dtype="int64")},
        np.asarray([[0, 1, -100], [0, 0, -100], [0, 2, 2], [0, 0, -100]], dtype="int64"),
        {"ignore_index": -100},
        split_name="train",
        task_family="classification",
        hf_task="token_classification",
    )
    assert token.effective_axis == "entity_present_sentence"
    assert token.axis_family == "categorical"

    regression = resolve_skew_axis(
        np.arange(10).reshape(10, 1),
        np.linspace(0.0, 1.0, 10),
        {},
        split_name="train",
        task_family="regression",
        hf_task="sentence_similarity",
    )
    assert regression.effective_axis == "score_bin"
    assert regression.axis_family == "ordered_numeric"

    generation = resolve_skew_axis(
        {"attention_mask": np.ones((4, 4), dtype="int64")},
        np.asarray([[1, 2, -100, -100], [1, -100, -100, -100], [1, 2, 3, 4], [1, 2, 3, -100]], dtype="int64"),
        {"ignore_index": -100},
        split_name="train",
        task_family="generation",
        hf_task="causal_lm_generation",
    )
    assert generation.effective_axis == "supervised_token_bucket"

    mlm = resolve_skew_axis(
        {"attention_mask": np.ones((3, 3), dtype="int64")},
        np.asarray([[10, -100, -100], [20, 20, -100], [30, -100, -100]], dtype="int64"),
        {"ignore_index": -100},
        split_name="train",
        task_family="classification",
        hf_task="fill_mask",
    )
    assert mlm.effective_axis == "masked_token_id"


def test_resolve_skew_axis_supports_vqa_question_type_and_retrieval_domain():
    vqa = resolve_skew_axis(
        np.arange(4).reshape(4, 1),
        np.asarray(["yes", "no", "red", "yes"], dtype=object),
        {"split_sidecars": {"train": {"question_text": ["What color", "How many", "Is it red", "Where is it"]}}},
        split_name="train",
        task_family="vqa",
        hf_task="visual_question_answering",
        requested_axis="question_type",
    )
    assert vqa.effective_axis == "question_type"
    assert vqa.bucket_spec["cardinality"] >= 2

    retrieval = resolve_skew_axis(
        {"caption_lengths": np.asarray([2, 7, 3, 9], dtype="int64")},
        np.zeros((4,), dtype="int64"),
        {"split_sidecars": {"train": {"query_domain": ["news", "news", "science", "science"]}}},
        split_name="train",
        task_family="retrieval",
        hf_task="text_image_retrieval",
        requested_axis="query_domain",
    )
    assert retrieval.effective_axis == "query_domain"
    assert retrieval.bucket_spec["cardinality"] == 2


def test_split_data_dirichlet_uses_task_aware_bucket_ids():
    x = {"attention_mask": np.ones((6, 3), dtype="int64")}
    y = np.asarray(
        [
            [0, 1, -100],
            [0, 1, -100],
            [0, 0, -100],
            [0, 0, -100],
            [0, 2, -100],
            [0, 2, -100],
        ],
        dtype="int64",
    )
    clients, resolved = split_data(
        x,
        y,
        2,
        strategy="dirichlet",
        distribution_param=0.2,
        meta={"ignore_index": -100},
        task_family="classification",
        hf_task="token_classification",
    )

    assert resolved["strategy"] == "dirichlet"
    assert resolved["skew_axis"] == "entity_present_sentence"
    assert resolved["bucket_distribution"]
    assert len(clients) == 2


def test_split_data_quantity_skew_preserves_bucket_mix_while_varying_sizes():
    x = np.arange(120).reshape(120, 1)
    y = np.repeat(np.asarray([0, 1]), 60)
    clients, resolved = split_data(
        x,
        y,
        3,
        strategy="quantity_skew",
        distribution_param=0.3,
        rng=np.random.default_rng(7),
        meta={},
        task_family="classification",
        hf_task="sequence_classification",
        skew_axis="class_label",
    )

    sizes = [len(payload["y"]) for payload in clients.values()]
    ratios = []
    for payload in clients.values():
        local = np.asarray(payload["y"])
        ratios.append(float(np.mean(local == 1)) if len(local) else 0.0)
    assert max(sizes) - min(sizes) >= 10
    assert all(abs(ratio - 0.5) <= 0.15 for ratio in ratios)
    assert resolved["bucket_distribution"] == {"0": 60, "1": 60}


def test_split_data_rejects_incompatible_strategy_axis_combo():
    with pytest.raises(ValueError, match="ordered skew axis"):
        split_data(
            np.arange(6).reshape(6, 1),
            np.asarray(["yes", "no", "yes", "blue", "red", "blue"], dtype=object),
            2,
            strategy="shard",
            meta={},
            task_family="vqa",
            hf_task="visual_question_answering",
            skew_axis="answer_vocab",
        )
