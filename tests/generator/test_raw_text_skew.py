"""Raw character-length sampling, independent of model tokenization and masking."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from mlaas_data_generator.clients.partitioning import client_partition
from mlaas_data_generator.data.raw_text_axis import AXIS, LENGTH_FIELD
from mlaas_data_generator.data.skew_axes import resolve_skew_axis
from mlaas_data_generator.data.tokenization_selection import training_row_selection
from mlaas_data_generator.data.preprocessors.hf import preprocess_hf
from mlaas_data_generator.tasks.contracts import resolve_task_contract
from tests.generator.test_tokenization_selection import config
from tests.generator.test_hf_fill_mask_memory import Tokenizer


def resolve(x, meta=None):
    return resolve_skew_axis(x, None, meta or {}, split_name="train", task_family="fill_mask", hf_task="fill_mask")


def test_raw_length_axis_uses_fixed_unicode_character_bounds():
    lengths = [0, 1, 64, 65, 256, 257, 1024, 1025, 4096, 4097]
    x = Dataset.from_dict({"body": ["😀" * size for size in lengths]})
    axis = resolve(x, {"dataset_args": {"text_column": "body"}})
    assert axis.effective_axis == AXIS
    assert axis.bucket_ids.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    assert axis.source_fields == ["raw.body"]
    assert axis.bucket_spec["unit"] == "unicode_code_points"
    assert axis.bucket_spec["upper_bounds_inclusive"] == [0, 64, 256, 1024, 4096]
    order = [9, 1, 7, 0, 5, 8, 6, 2, 4, 3]
    reordered = resolve(x.select(order), {"text_column": "body"})
    np.testing.assert_array_equal(reordered.bucket_ids, axis.bucket_ids[order])
    assert reordered.bucket_spec == axis.bucket_spec


@pytest.mark.parametrize("x", [
    {"input_ids": np.ones((10, 3)), "attention_mask": np.ones((10, 3))},
    {"text": ["same length"] * 10},
    {"wrong_column": ["text", "long text" * 20]},
])
def test_unavailable_raw_axis_fails_without_using_prepared_features(x):
    cfg = config(task="fill_mask", strategy="dirichlet", skew_axis=AXIS)
    with pytest.raises(ValueError, match="requested skew axis is unavailable.*raw_text_length_bucket"):
        client_partition(x, np.arange(10), cfg, source_ids=[f"train:{i}" for i in range(10)],
                         meta={}, task_family="fill_mask")


def test_non_text_raw_values_fail_before_any_tokenizer_is_needed():
    with pytest.raises(ValueError, match="requires string values"):
        resolve({"text": ["valid", None]})


@pytest.mark.parametrize("strategy", ["dirichlet", "shard"])
def test_fill_mask_partition_is_independent_of_tokenizer_and_masking(strategy):
    raw = Dataset.from_dict({"text": ["word " * (2, 20, 80, 300)[i % 4] for i in range(600)]})
    cfg = config(task="fill_mask", task_type="fill_mask", strategy=strategy,
                 skew_axis=None, sample_size=73, distribution_param=.2)
    cfg["task_contract"] = resolve_task_contract(cfg).as_dict()
    args = dict(source_ids=[f"train:{i}" for i in range(600)], task_family="fill_mask")
    first = client_partition(raw, None, cfg, meta={"max_length": 32, "seed": 42}, **args)
    second = client_partition(raw, None, {**cfg, "hf_model_id": "different", "mlm_probability": .8},
                              meta={"max_length": 512, "seed": 73}, **args)
    np.testing.assert_array_equal(first["indices"], second["indices"])
    assert first["metadata"] == second["metadata"]
    assert first["metadata"]["skew_axis"] == AXIS
    assert first["metadata"]["effective_strategy"] == strategy
    assert len(first["indices"]) == len(set(first["indices"])) == 73
    if strategy == "dirichlet":
        assert first["metadata"]["dirichlet_alpha"] == .2
    other = client_partition(raw, None, {**cfg, "sample_seed": 79}, meta={}, **args)
    assert first["metadata"]["source_ids"] != other["metadata"]["source_ids"]


def test_group_scans_raw_text_once_then_tokenizes_selected_union_and_replacements(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf_text_fill_mask

    count = 4099
    events, seen = [], []

    def raw_text(index):
        return "" if index % 31 == 0 else f"{index + 10} " * (20, 80, 300, 1200)[index % 4]

    class RawSource:
        column_names = ["body"]

        def __len__(self):
            return count

        def __getitem__(self, key):
            assert isinstance(key, slice) and key.stop - key.start <= 256
            events.append(("raw", key.stop - key.start))
            return {"body": [raw_text(i) for i in range(key.start, key.stop)]}

        def select(self, indices):
            return Dataset.from_dict({"body": [raw_text(i) for i in indices]})

    class RecordingTokenizer(Tokenizer):
        vocab_size = count + 100

        def __call__(self, texts, **kwargs):
            events.append(("tokenize", len(texts)))
            seen.extend(texts)
            return super().__call__(texts, **kwargs)

    tok = RecordingTokenizer("right")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(PreTrainedTokenizerBase=Tokenizer))
    monkeypatch.setattr(hf_text_fill_mask, "get_cached_tokenizer", lambda **kwargs: (tok, None, None))
    evaluation = Dataset.from_dict({"body": ["3 " * 20, "4 " * 20]})
    configs = [config(task="fill_mask", task_type="fill_mask", strategy="dirichlet", skew_axis=None,
        seed=seed, sample_size=73, distribution_param=2.0) for seed in (42, 43)]
    for cfg in configs:
        cfg["task_contract"] = resolve_task_contract(cfg).as_dict()
    with training_row_selection(configs) as selection:
        actual = preprocess_hf((RawSource(), None), (evaluation, None),
            {"hf_task": "fill_mask", "modality": "text", "max_length": 24,
             "selected_test_rows": [70, 71]}, hf_model_id="offline", text_column="body", mlm_probability=.9)
    assert sum(size for kind, size in events if kind == "raw") == count
    first_tokenization = next(i for i, (kind, _) in enumerate(events) if kind == "tokenize")
    assert all(kind == "raw" for kind, _ in events[:first_tokenization])
    assert all(kind == "tokenize" for kind, _ in events[first_tokenization:])
    union = set()
    for cfg in configs:
        split = selection.materialize(cfg["service_id"], actual[0])
        rows = split["resolved"]["accepted_source_row_ids"]
        assert len(rows) == len(set(rows)) == 73
        assert all(row % 31 for row in rows)
        item = selection.selections[cfg["service_id"]]
        expected = [int(i) for i in item["order"] if i % 31][:73]
        assert rows == expected
        assert item["partition"]["metadata"]["effective_strategy"] == "dirichlet"
        union.update(rows)
    training_texts = [text for text in seen if text and int(text.split()[0]) >= 10]
    assert len(training_texts) == len(union)
    assert {int(text.split()[0]) - 10 for text in training_texts} == union
    assert actual[2]["tokenization_selection"]["mode"] == "client_rows_first"
    assert actual[2]["selected_test_rows"] == [70, 71]
    assert actual[2]["accounting"]["tokenized_record_count"] < count
    assert actual[2]["tokenization_selection"]["rejected_training_count"] > 0
    assert actual[2]["split_sidecars"]["train"][LENGTH_FIELD] == [len(raw_text(i)) for i in selection.indices]
    replay = {**configs[0], "_recorded_source_ids": [f"train:{i}" for i in selection.selections[configs[0]["service_id"]]["accepted"]]}
    events.clear()
    with training_row_selection([replay]) as restored:
        reconstructed = preprocess_hf((RawSource(), None), (evaluation, None),
            {"hf_task": "fill_mask", "modality": "text", "max_length": 24},
            hf_model_id="offline", text_column="body", mlm_probability=.9)
    assert not any(kind == "raw" for kind, _ in events)
    original = selection.materialize(configs[0]["service_id"], actual[0])
    replayed = restored.materialize(replay["service_id"], reconstructed[0])
    np.testing.assert_array_equal(original["y_train"], replayed["y_train"])
    for key in original["x_train"]:
        np.testing.assert_array_equal(original["x_train"][key], replayed["x_train"][key])
