import numpy as np

from mlaas_data_generator.data.distributions import (
    get_data_distribution,
    get_retrieval_pair_stats,
    get_token_label_stats,
    get_vqa_answer_stats,
)


def test_get_token_label_stats_ignores_ignore_and_pad_tokens():
    y = np.asarray([
        [-100, 11, 11, 0],
        [-100, 13, 0, 13],
    ])

    stats = get_token_label_stats(y, ignore_index=-100, pad_token_id=0)

    assert stats["total_tokens"] == 8
    assert stats["supervised_tokens"] == 4
    assert stats["unique_supervised_token_ids"] == 2
    assert stats["top_supervised_token_ids"] == {13: 2, 11: 2} or stats["top_supervised_token_ids"] == {11: 2, 13: 2}
    assert stats["supervised_ratio"] == 0.5


def test_get_data_distribution_detection_dict_targets():
    y = [
        {"boxes": [[0, 0, 10, 10], [10, 10, 20, 20]], "labels": [1, 2]},
        {"boxes": [[0, 0, 8, 8]], "labels": [2]},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 3
    assert stats["avg_boxes_per_sample"] == 1.5
    assert stats["class_counts"] == {1: 1, 2: 2}


def test_get_data_distribution_detection_supports_classes_and_class_labels():
    y = [
        {"boxes": [[0, 0, 10, 10]], "classes": [5]},
        {"bbox": [[1, 1, 8, 8]], "class_labels": [7]},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 2
    assert stats["class_counts"] == {5: 1, 7: 1}


def test_get_data_distribution_detection_supports_nested_annotation_schemas():
    y = [
        {"annotation": {"objects": {"bbox": [[0, 0, 10, 10]], "category": [3]}}},
        {"annotation": {"annotations": {"boxes": [[2, 2, 6, 6]], "category_id": [4]}}},
    ]

    stats = get_data_distribution(y, num_classes=None)

    assert stats["samples"] == 2
    assert stats["total_boxes"] == 2
    assert stats["class_counts"] == {3: 1, 4: 1}


def test_get_data_distribution_counts_object_labels_when_num_classes_unknown():
    stats = get_data_distribution(["cat", "cat", "home", None], num_classes=None, bins=2)

    assert stats["cat"] == 2
    assert stats["home"] == 1


def test_get_retrieval_pair_stats_summarizes_pairs_not_numeric_bins():
    x = {
        "input_ids": np.asarray([[1, 2, 0], [1, 2, 0], [3, 4, 5]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=np.int64),
        "pixel_values": np.asarray(
            [
                np.zeros((3, 2, 2), dtype=np.float32),
                np.zeros((3, 2, 2), dtype=np.float32),
                np.ones((3, 2, 2), dtype=np.float32),
            ]
        ),
        "caption_lengths": np.asarray([2, 4, 6], dtype=np.int64),
        "image_sizes": np.asarray([[10, 20], [20, 30], [30, 40]], dtype=np.int64),
    }

    stats = get_retrieval_pair_stats(x)

    assert set(stats) == {
        "image_caption_pairs",
        "unique_images",
        "unique_captions",
        "caption_length",
        "image_size",
    }
    assert stats["image_caption_pairs"] == 3
    assert stats["unique_images"] == 2
    assert stats["unique_captions"] == 2
    assert stats["caption_length"]["mean"] == 4.0
    assert stats["image_size"]["height"]["mean"] == 20.0
    assert stats["image_size"]["width"]["mean"] == 30.0
    assert stats["image_size"]["area"]["mean"] == (200 + 600 + 1200) / 3


def test_get_vqa_answer_stats_uses_sparse_distribution_for_small_answer_ids():
    y = np.asarray([1, 1, 3, -100], dtype=np.int64)

    stats = get_vqa_answer_stats(y, ignore_index=-100)

    assert stats["samples"] == 4
    assert stats["label_unit"] == "answer_id"
    assert stats["distribution"] == [{"bin": 1, "count": 2}, {"bin": 3, "count": 1}]


def test_get_vqa_answer_stats_uses_histogram_for_large_token_ids():
    y = np.asarray(
        [
            [101, 305, -100],
            [876, 30522, -100],
        ],
        dtype=np.int64,
    )

    stats = get_vqa_answer_stats(y, ignore_index=-100)

    assert stats["samples"] == 2
    assert stats["label_unit"] == "answer_token_id"
    assert stats["supervised_answer_tokens"] == 4
    assert "distribution" not in stats
    assert stats["histogram"]["bin_edges"][0] == 0
    assert sum(stats["histogram"]["counts"]) == 4
    assert stats["top_answer_ids"][0]["count"] == 1
