"""Evaluation budgets count dataset rows and never follow client training knobs."""
import numpy as np
import pytest

from mlaas_data_generator.evaluation_sampling import (
    evaluation_indices,
    evaluation_settings,
    task_benchmark_sample_size,
)
from mlaas_data_generator.shared_parent import reserve_evaluation
from mlaas_data_generator.config import CONFIG


@pytest.mark.parametrize("count", [1, 71, 1023, 1024, 1025, 2047, 2048, 2049, 25000])
def test_default_cap_and_full_small_split(count):
    rows, info = evaluation_indices(count, {})
    assert len(rows) == len(set(rows)) == min(count, CONFIG["benchmark_sample_size"])
    assert info["available_sample_size_total"] == count
    assert info["evaluation_seed"] == 42
    if count <= CONFIG["benchmark_sample_size"]:
        np.testing.assert_array_equal(rows, np.arange(count))


@pytest.mark.parametrize("cap", [17, 2048])
def test_configurable_cap_is_independent_of_client_and_training_seed(cap):
    config = {"benchmark_sample_size": cap, "evaluation_seed": 0}
    rows, info = evaluation_indices(4096, config)
    other, _ = evaluation_indices(4096, {**config, "service_id": "other", "seed": 999,
        "sample_seed": 56, "training_seed": 12, "sample_size": 5, "hf_model_id": "different"})
    np.testing.assert_array_equal(rows, other)
    assert len(rows) == cap
    assert info["evaluation_seed"] == 0
    changed, _ = evaluation_indices(4096, {**config, "evaluation_seed": 1})
    assert not np.array_equal(rows, changed)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "bad", float("nan")])
def test_invalid_caps_are_rejected(value):
    with pytest.raises(ValueError, match="benchmark_sample_size"):
        evaluation_settings({"benchmark_sample_size": value})


def test_none_uses_config_default_and_legacy_cap_alias_still_works(monkeypatch):
    from mlaas_data_generator.config import CONFIG
    monkeypatch.setitem(CONFIG, "benchmark_sample_size", 37)
    assert evaluation_settings({"benchmark_sample_size": None})["benchmark_sample_size"] == 37
    assert evaluation_settings({"max_eval_samples": 23})["benchmark_sample_size"] == 23
    assert evaluation_settings({"dataset_args": {"benchmark_sample_size": 19}})["benchmark_sample_size"] == 19


@pytest.mark.parametrize("task", [
    "seq2seq_generation",
    "text2text-generation",
    "image_captioning",
    "captioning",
    "visual_question_answering",
    "vqa",
])
def test_generation_heavy_tasks_use_quarter_of_global_benchmark_cap(task, monkeypatch):
    monkeypatch.setitem(CONFIG, "benchmark_sample_size", 2048)
    assert task_benchmark_sample_size({"hf_task": task}) == 512
    assert evaluation_settings({"hf_task": task})["benchmark_sample_size"] == 512


@pytest.mark.parametrize("task", ["causal_lm_generation", "sequence_classification", "image_segmentation"])
def test_other_tasks_keep_global_benchmark_cap(task):
    assert task_benchmark_sample_size({"hf_task": task}, 2048) == 2048


def test_quarter_task_cap_uses_nested_task_and_remains_positive():
    assert task_benchmark_sample_size({"dataset_args": {"task": "image_captioning"}}, 3) == 1


def test_explicit_task_cap_is_not_quartered_again(monkeypatch):
    monkeypatch.setitem(CONFIG, "benchmark_sample_size", 2048)
    config = {"hf_task": "seq2seq_generation", "benchmark_sample_size": 512}
    assert evaluation_settings(config)["benchmark_sample_size"] == 512
    rows, info = evaluation_indices(1000, config)
    assert len(rows) == info["requested_sample_size_total"] == 512


def test_old_manifest_cap_is_clamped_before_family_preprocessing(monkeypatch):
    from mlaas_data_generator.cli.manifest.preparation import contract_config
    from mlaas_data_generator.data.master_loader import prepare_dataset, PREPROCESSOR_REGISTRY
    monkeypatch.setitem(CONFIG, "benchmark_sample_size", 2048)
    config = {"hf_task": "seq2seq_generation", "benchmark_sample_size": 2048}
    resolved = contract_config(config)
    assert resolved["benchmark_sample_size"] == 512
    assert resolved["evaluation_preparation_policy"] == "bounded_raw_pool_v1"
    observed = []
    def prepare(train, test, meta, **kwargs):
        observed.append(len(test[0]))
        return train, test, meta
    monkeypatch.setitem(PREPROCESSOR_REGISTRY, "evaluation_spy", prepare)
    rows = np.arange(13368)
    kwargs = {**config, "_evaluation_preparation_policy": resolved["evaluation_preparation_policy"],
              "preprocessors": [{"name": "evaluation_spy"}]}
    _, test, meta = prepare_dataset("hf", (rows[:312], rows[:312]), (rows, rows),
                                   {"selected_test_rows": rows.tolist()}, **kwargs)
    assert observed == [1024]
    np.testing.assert_array_equal(test[0], meta["selected_test_rows"])
    ids = [f"test:{i}" for i in meta["selected_test_rows"]]
    reservation = reserve_evaluation([], ids, test[1], task_family="generation", seed=42,
        contract={}, benchmark_sample_size=resolved["benchmark_sample_size"])
    assert len(reservation["catalogue_metric_ids"]) == 512
    assert len(reservation["audit_ids"]) == 512
    assert not set(reservation["audit_ids"]) & set(reservation["catalogue_metric_ids"])


@pytest.mark.parametrize("task", ["classification", "regression", "token_classification", "segmentation", "generation", "fill_mask", "retrieval"])
@pytest.mark.parametrize("cap", [None, 37, 2048])
def test_family_reserves_audit_then_caps_remaining_examples(task, cap):
    ids = [f"test:{i}" for i in range(2500)]
    kwargs = dict(task_family=task, seed=42, contract={})
    reservation = reserve_evaluation([], ids, np.arange(len(ids)) % 2,
                                     benchmark_sample_size=cap, **kwargs)
    assert len(reservation["audit_ids"]) == 512
    assert len(reservation["catalogue_metric_ids"]) == min(cap or CONFIG["benchmark_sample_size"], 2500 - 512)
    assert not set(reservation["audit_ids"]) & set(reservation["catalogue_metric_ids"])
    assert reservation == reserve_evaluation([], ids, np.arange(len(ids)) % 2,
                                             benchmark_sample_size=cap, **kwargs)
    # Adjusting the service budget cannot change the separate audit holdout.
    other = reserve_evaluation([], ids, np.arange(len(ids)) % 2, benchmark_sample_size=11, **kwargs)
    assert reservation["audit_ids"] == other["audit_ids"]


def test_small_family_uses_entire_pool_after_audit():
    ids = [f"test:{i}" for i in range(71)]
    reservation = reserve_evaluation([], ids, np.arange(71) % 2,
                                     task_family="classification", seed=42, contract={})
    assert len(reservation["audit_ids"]) == 35
    assert len(reservation["catalogue_metric_ids"]) == 36
    assert set(reservation["catalogue_metric_ids"]) | set(reservation["audit_ids"]) == set(ids)


def test_family_identity_freezes_cap_and_seed_but_not_training_quantity():
    from mlaas_data_generator.cli.manifest.preparation import group_key
    config = {"dataset": "iris", "model_type": "mlp", "training_regime": "generic"}
    key = group_key(config)
    assert group_key({**config, "sample_size": 8, "sample_seed": 17, "training_seed": 99}) == key
    assert group_key({**config, "benchmark_sample_size": 128}) != key
    assert group_key({**config, "evaluation_seed": 1}) != key


def test_runner_counts_examples_not_tokens_or_pixels():
    from mlaas_data_generator.services.runner import _cap_benchmark_split
    y = np.arange(1500)
    x = {"tokens": np.repeat(y[:, None], 12, axis=1), "images": np.zeros((1500, 8, 8))}
    selected, labels, info = _cap_benchmark_split(x, y, config={"sample_size": 4, "benchmark_sample_size": 1024})
    assert selected["tokens"].shape == (1024, 12)
    assert selected["images"].shape == (1024, 8, 8)
    np.testing.assert_array_equal(selected["tokens"][:, 0], labels)
    assert info["effective_sample_size_total"] == 1024


def test_generic_and_hf_sources_select_the_same_rows_before_preprocessing(monkeypatch):
    import datasets
    from mlaas_data_generator.data.master_loader import _select_generic_benchmark
    from mlaas_data_generator.data.sources.huggingface import load_huggingface_source
    rows = np.arange(1500)
    raw = datasets.Dataset.from_dict({"text": list(map(str, rows)), "label": (rows % 2).tolist()})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: raw)
    config = {"benchmark_sample_size": 1024, "evaluation_seed": 7}
    train, test, meta = load_huggingface_source(dataset_name="offline", **config)
    _, generic, generic_meta = _select_generic_benchmark((rows, rows), (rows, rows), {}, config)
    assert len(train[0]) == 1500
    assert len(test[0]) == len(generic[0]) == 1024
    assert meta["selected_test_rows"] == generic_meta["selected_test_rows"]
    np.testing.assert_array_equal(generic[0], meta["selected_test_rows"])
    # Family preparation must retain the complete source for its audit holdout.
    _, full, _ = load_huggingface_source(dataset_name="offline", **config, _defer_evaluation_sampling=True)
    assert len(full[0]) == 1500


def test_fedavg_manifest_budget_does_not_follow_training_quantity():
    from mlaas_data_generator.fedavg_manifest import expand_fedavg_run
    from mlaas_data_generator.provenance import sha256_json
    def expand(quantity, **overrides):
        return expand_fedavg_run({"sample_size": quantity, **overrides}, run_index=0,
            clients_per_run=4, rounds=2, participation_rate=1, seed=99,
            service_id_fn=lambda prefix, value: prefix + sha256_json(value),
            service_config_fn=lambda row: {}, client_knobs_fn=lambda **kwargs: {})
    for quantity in (50, 5000):
        rows = expand(quantity)
        assert {row["benchmark_sample_size"] for row in rows} == {CONFIG["benchmark_sample_size"]}
        assert {row["evaluation_seed"] for row in rows} == {42}
        assert {row["benchmark_sample_seed"] for row in rows} == {42}
    changed = expand(50, benchmark_sample_size=256, evaluation_seed=3)
    assert {row["benchmark_sample_size"] for row in changed} == {256}
    assert {row["evaluation_seed"] for row in changed} == {3}
    assert changed[0]["fl_run_id"] != expand(50)[0]["fl_run_id"]


def test_fedavg_builder_forwards_explicit_evaluation_seed(monkeypatch):
    from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder
    monkeypatch.setattr(builder, "_collect_fedavg_run_seed_rows", lambda **kwargs: [{"sample_size": 50}])
    monkeypatch.setattr(builder, "_fedavg_client_knobs_for_row", lambda **kwargs: {})
    frame = builder.build_hf_manifest(generation_mode="fedavg", fedavg_runs=1,
        fedavg_clients_per_run=4, fedavg_rounds=2, seed=99, evaluation_seed=3)
    assert set(frame["evaluation_seed"]) == {3}
    assert set(frame["benchmark_sample_seed"]) == {3}
    assert set(frame["benchmark_sample_size"]) == {CONFIG["benchmark_sample_size"]}
