"""Raw-source selection and replacement contract, using offline tokenizers."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from mlaas_data_generator.data.preprocessors.hf import preprocess_hf
from mlaas_data_generator.data.tokenization_selection import training_row_selection
from mlaas_data_generator.services.runner import ServiceRunner


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, pairs=None, *, max_length, **kwargs):
        self.calls.append(list(texts))
        rows = [[int(text.split()[0][3:]) + 10] + [3] * (len(text.split()) - 1) for text in texts]
        if pairs is not None:
            rows = [row + [4] * len(pair.split()) for row, pair in zip(rows, pairs)]
        rows = [row[:max_length] for row in rows]
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

    def pad(self, values, *, max_length, **kwargs):
        return {key: np.asarray([row + [0] * (max_length - len(row)) for row in rows], dtype="int32")
                for key, rows in values.items()}


@pytest.fixture
def tokenizer(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf_text_sequence
    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(hf_text_sequence, "get_cached_tokenizer", lambda **kwargs: (tokenizer, None, None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        PreTrainedTokenizerBase=RecordingTokenizer,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)))
    return tokenizer


def config(*, strategy="iid", versioned=True, task="sequence_classification", seed=42, **extra):
    cfg = dict(service_id=f"client_{seed}", dataset="hf", dataset_name="local_fixture",
        model_type="hf_finetune", training_regime="finetune_transfer", hf_task=task,
        modality="text", task_type="classification", no_save_weights=True,
        sample_size=6, sample_seed=seed, seed=seed, split_strategy=strategy,
        client_partition_index=seed, heterogeneity_intensity=0.5)
    if versioned:
        cfg["partition_generator_version"] = "independent_source_pool_v3"
    cfg.update(extra)
    if strategy in {"dirichlet", "shard"}:
        cfg.setdefault("skew_axis", "score_bin" if cfg.get("task_type") == "regression" else "class_label")
    return cfg


def dataset(*, pair=False, regression=False):
    data = {"text": [f"row{i} " + "word " * (i % 5) for i in range(40)],
            "label": [float(i * i) if regression else f"class_{i % 4}" for i in range(40)]}
    if pair:
        data["other"] = ["second sentence"] * 40
    return Dataset.from_dict(data)


def prepare(ds, *, pair=False, regression=False, dynamic=False):
    # Keep test membership/order independent of all client seeds and quantities.
    task = "sentence_similarity" if pair else "sequence_classification"
    return preprocess_hf((ds, None), (ds.select([0, 1, 2, 3]), None),
        {"hf_task": task, "modality": "text", "max_length": 12,
         "task_type": "regression" if regression else "classification",
         "selected_train_rows": list(range(100, 100 + len(ds))),
         "selected_test_rows": [70, 71, 72, 73]},
        hf_model_id="local-tokenizer", text_column=["text", "other"] if pair else "text",
        label_column="label", label_mode="regression" if regression else "categorical",
        dynamic_padding=dynamic)


@pytest.mark.parametrize("strategy", ["iid", "quantity_skew", "dirichlet", "shard"])
@pytest.mark.parametrize("pair,regression", [(False, False), (True, False), (True, True)])
def test_selection_before_tokenization_preserves_shared_labels(tokenizer, strategy, pair, regression):
    ds = dataset(pair=pair, regression=regression)
    cfg = config(strategy=strategy, task_type="regression" if regression else "classification")
    baseline = prepare(ds, pair=pair, regression=regression)
    tokenizer.calls.clear()
    with training_row_selection([cfg]) as selection:
        actual = prepare(ds, pair=pair, regression=regression)
    selected = selection.materialize(cfg["service_id"], actual[0])
    assert len(selected["y_train"]) == 6
    assert len(tokenizer.calls[0]) == 6 and len(tokenizer.calls[1]) == 4
    source = selected["resolved"]["accepted_source_row_ids"]
    np.testing.assert_array_equal(selected["y_train"], baseline[0][1][np.asarray(source) - 100])
    for key in baseline[1][0]:
        np.testing.assert_array_equal(actual[1][0][key], baseline[1][0][key])
    for key in ("label_mapping", "num_classes", "label_format", "label_min", "label_max"):
        assert actual[2].get(key) == baseline[2].get(key)


def test_iid_and_saved_uniform_profile_skip_axis_resolution(monkeypatch):
    from mlaas_data_generator.clients import partitioning
    monkeypatch.setattr(partitioning, "resolve_skew_axis", lambda *a, **k: pytest.fail("IID must not inspect an axis"))
    for strategy, quantity in (("iid", 1000), ("quantity_skew", 37)):
        cfg = config(strategy=strategy, task="causal_lm_generation", sample_size=quantity,
                     skew_axis="supervised_token_bucket")
        got = partitioning.client_partition(None, None, cfg,
            source_ids=[f"train:{i}" for i in range(203037)], meta={}, task_family="generation")
        np.testing.assert_array_equal(got["indices"], np.random.default_rng(42).permutation(203037)[:quantity])
        assert len(set(got["indices"])) == quantity
        assert got["metadata"]["skew_axis"] is None
        assert got["metadata"]["effective_strategy"] == strategy


@pytest.mark.parametrize("strategy", ["dirichlet", "shard"])
def test_skew_requires_an_explicit_available_axis(strategy):
    from mlaas_data_generator.clients.partitioning import client_partition
    cfg = config(strategy=strategy, skew_axis=None)
    with pytest.raises(ValueError, match="explicit skew_axis"):
        with training_row_selection([cfg]):
            prepare(dataset())
    with pytest.raises(ValueError, match="unavailable"):
        client_partition(None, None, {**cfg, "skew_axis": "invented_axis"},
                         source_ids=["a", "b"], meta={}, task_family="generation")


def generation_fixture(monkeypatch, *, count=203037, invalid=(), dynamic=False):
    from mlaas_data_generator.data.preprocessors import hf_text_generation as generation
    from tests.generator.test_hf_generation_preprocessors import FakeTokenizer
    class Tokenizer(FakeTokenizer):
        def __init__(self):
            self.seen = []
        def __call__(self, texts=None, **kwargs):
            self.seen.extend(texts or kwargs.get("text_target", []))
            return super().__call__(texts, **kwargs)
    tok = Tokenizer()
    monkeypatch.setattr(generation, "_load_auto_tokenizer", lambda *a: tok)
    invalid = set(invalid)
    ds = Dataset.from_dict({"article": ["" if i in invalid else f"source{i} article content" + " word" * (i % 8) for i in range(count)]})
    evaluation = Dataset.from_dict({"article": ["evaluation one", "evaluation two", "evaluation three", "evaluation four"]})
    def load():
        return preprocess_hf((ds, None), (evaluation, None),
            {"hf_task": "causal_lm_generation", "modality": "text", "max_length": 12,
             "train_split": "train", "test_split": "test", "selected_train_rows": list(range(count)),
             "selected_test_rows": [40, 41, 42, 43]}, hf_model_id="offline", text_column="article", dynamic_padding=dynamic)
    return tok, load


@pytest.mark.parametrize("saved", [False, True])
def test_arxiv_sized_pool_tokenizes_exactly_1000_selected_rows(monkeypatch, saved):
    tok, load = generation_fixture(monkeypatch)
    cfg = config(task="causal_lm_generation", sample_size=1000, no_save_weights=not saved)
    with training_row_selection([cfg]) as selection:
        actual = load()
    split = selection.materialize(cfg["service_id"], actual[0])
    rows = split["resolved"]["accepted_source_row_ids"]
    assert rows == np.random.default_rng(42).permutation(203037)[:1000].tolist()
    assert len(split["y_train"]) == 1000
    assert len([text for text in tok.seen if text.startswith("source")]) == 1000
    assert actual[2]["selected_test_rows"] == [40, 41, 42, 43]
    assert actual[2]["tokenization_selection"]["source_pool_count"] == 203037
    assert actual[2]["accounting"]["tokenized_record_count"] == 1000


@pytest.mark.parametrize("count,quantity,invalid_count", [(100, 25, 8), (20, 25, 8), (100, 5, 40)])
def test_replacements_and_exhaustion_follow_candidate_order(monkeypatch, count, quantity, invalid_count):
    order = np.random.default_rng(42).permutation(count).tolist()
    invalid = order[:invalid_count]
    tok, load = generation_fixture(monkeypatch, count=count, invalid=invalid)
    cfg = config(task="causal_lm_generation", sample_size=quantity)
    with training_row_selection([cfg]) as selection:
        actual = load()
    got = selection.materialize(cfg["service_id"], actual[0])
    expected = order[invalid_count:invalid_count + quantity]
    assert got["resolved"]["accepted_source_row_ids"] == expected
    assert len(got["y_train"]) == len(expected)
    assert got["resolved"]["rejected_candidate_count"] == invalid_count
    assert actual[2]["accounting"]["tokenized_record_count"] == invalid_count + len(expected)
    assert len([t for t in tok.seen if not t.startswith("evaluation")]) == invalid_count + len(expected)
    assert got["resolved"]["requested_quota"] == quantity


def test_group_union_and_replacements_are_prepared_once(monkeypatch):
    invalid = set(np.random.default_rng(42).permutation(100)[:8])
    tok, load = generation_fixture(monkeypatch, count=100, invalid=invalid)
    configs = [config(task="causal_lm_generation", sample_size=25, seed=s) for s in (42, 43, 44)]
    with training_row_selection(configs) as selection:
        actual = load()
    for cfg in configs:
        expected = [i for i in np.random.default_rng(cfg["sample_seed"]).permutation(100) if i not in invalid][:25]
        got = selection.materialize(cfg["service_id"], actual[0])
        assert got["resolved"]["accepted_source_row_ids"] == expected
    prepared = [text for text in tok.seen if text.startswith("source")]
    assert len(prepared) == len(set(prepared))


def fill_mask_fixture(monkeypatch, *, count, invalid=()):
    from mlaas_data_generator.data.preprocessors import hf_text_fill_mask
    from tests.generator.test_hf_fill_mask_memory import Tokenizer

    class RecordingMaskTokenizer(Tokenizer):
        vocab_size = count + 100

        def __init__(self):
            super().__init__("right")
            self.seen = []

        def __call__(self, texts, **kwargs):
            self.seen.extend(texts)
            return super().__call__(texts, **kwargs)

    class RawSource:
        column_names = ["text"]

        def __len__(self):
            return count

        def __getitem__(self, key):
            pytest.fail("fill-mask must select raw rows before reading texts")

        def select(self, indices):
            return Dataset.from_dict({"text": [
                "" if i in invalid else f"{i + 10} " * 20 for i in indices]})

    tokenizer = RecordingMaskTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(PreTrainedTokenizerBase=Tokenizer))
    monkeypatch.setattr(hf_text_fill_mask, "get_cached_tokenizer", lambda **kwargs: (tokenizer, None, None))
    evaluation = Dataset.from_dict({"text": ["3 " * 20, "4 " * 20]})

    def load():
        return preprocess_hf((RawSource(), None), (evaluation, None),
            {"hf_task": "fill_mask", "modality": "text", "max_length": 24,
             "selected_test_rows": [70, 71]},
            hf_model_id="offline", mlm_probability=.9)

    return tokenizer, load


def test_fill_mask_large_pool_prepares_only_group_union_and_replacements(monkeypatch):
    count = 2_119_719
    configs = [config(task="fill_mask", task_type="fill_mask", sample_size=37, seed=s,
                      skew_axis="masked_token_id") for s in (42, 43)]
    orders = [np.random.default_rng(cfg["sample_seed"]).permutation(count) for cfg in configs]
    invalid = {int(order[0]) for order in orders}
    tokenizer, load = fill_mask_fixture(monkeypatch, count=count, invalid=invalid)
    with training_row_selection(configs) as selection:
        actual = load()

    expected_union = set()
    for cfg, order in zip(configs, orders):
        expected = [int(i) for i in order[:40] if i not in invalid][:37]
        split = selection.materialize(cfg["service_id"], actual[0])
        assert split["resolved"]["accepted_source_row_ids"] == expected
        assert len(split["y_train"]) == 37
        expected_union.update(expected)
    texts = [text for text in tokenizer.seen if text and int(text.split()[0]) >= 10]
    assert len(texts) == len(expected_union)
    assert {int(text.split()[0]) - 10 for text in texts} == expected_union
    assert actual[2]["tokenization_selection"]["mode"] == "client_rows_first"
    assert actual[2]["accounting"]["tokenized_record_count"] == len(expected_union) + len(invalid)
    assert actual[2]["selected_test_rows"] == [70, 71]


@pytest.mark.parametrize("strategy", ["dirichlet", "shard"])
@pytest.mark.parametrize("grouped", [False, True])
def test_masked_token_skew_fails_before_loading_or_tokenizing(monkeypatch, strategy, grouped):
    from mlaas_data_generator.data.tokenization_selection import active_selection

    tokenizer, _ = fill_mask_fixture(monkeypatch, count=100)
    cfg = config(task="fill_mask", strategy=strategy, skew_axis="masked_token_id")
    configs = [config(task="fill_mask", seed=43), cfg] if grouped else [cfg]
    with pytest.raises(ValueError, match="masked_token_id.*incompatible with raw-row selection"):
        with training_row_selection(configs):
            pytest.fail("incompatible skew must fail before dataset loading")
    assert not tokenizer.seen
    assert active_selection() is None
    assert cfg["split_strategy"] == strategy  # Never silently substitute IID.


def test_recorded_fill_mask_rows_replay_without_resolving_historical_skew(monkeypatch):
    tokenizer, load = fill_mask_fixture(monkeypatch, count=100)
    cfg = config(task="fill_mask", strategy="dirichlet", skew_axis="masked_token_id",
                 _recorded_source_ids=["train:43", "train:9", "train:61"])
    with training_row_selection([cfg]) as selection:
        actual = load()
    split = selection.materialize(cfg["service_id"], actual[0])
    assert split["resolved"]["accepted_source_row_ids"] == [43, 9, 61]
    assert len(tokenizer.seen) == 5


def test_replacement_dynamic_padding_and_evaluation_stay_consistent(monkeypatch):
    tok, load = generation_fixture(monkeypatch, count=100, invalid=np.random.default_rng(42).permutation(100)[:8], dynamic=True)
    cfg = config(task="causal_lm_generation", sample_size=25)
    with training_row_selection([cfg]) as selection:
        actual = load()
    split = selection.materialize(cfg["service_id"], actual[0])
    assert len(split["y_train"]) == 25
    assert actual[2]["input_shape"] == actual[0][0]["input_ids"].shape[1:]
    assert len([t for t in tok.seen if t.startswith("evaluation")]) == 4
    assert np.all(split["y_train"][split["x_train"]["attention_mask"] == 0] == -100)


def test_image_decoding_replacement_tracks_original_source_ids(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf_image
    calls = []
    class ImageProcessor:
        def __call__(self, image, **kwargs):
            calls.append(int(np.asarray(image)[0, 0, 0]))
            if calls[-1] in {2, 6}:
                raise ValueError("bad image")
            return {"pixel_values": np.asarray(image).transpose(2, 0, 1)[None]}
        def to_dict(self):
            return {}
    import transformers
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: ImageProcessor())
    raw = Dataset.from_dict({"image": [{"array": np.full((3, 3, 3), i, dtype=np.uint8).tolist()} for i in range(12)], "label": [i % 2 for i in range(12)]})
    cfg = config(task="image_classification", sample_size=9)
    with training_row_selection([cfg]) as selection:
        result = preprocess_hf((raw, None), (raw.select([0, 1]), None),
            {"modality": "image", "hf_task": "image_classification", "selected_train_rows": list(range(100, 112))},
            hf_model_id="offline", training_augmentations=False)
    split = selection.materialize(cfg["service_id"], result[0])
    expected = [100 + i for i in np.random.default_rng(42).permutation(12) if i not in {2, 6}][:9]
    assert split["resolved"]["accepted_source_row_ids"] == expected
    assert len(split["y_train"]) == 9


def test_raw_load_and_generic_preparation_are_separate(monkeypatch):
    from mlaas_data_generator.data import master_loader
    x, y = np.arange(100).reshape(50, 2), np.arange(50) % 2
    monkeypatch.setattr(master_loader, "load_sklearn_source", lambda *a, **k: ((x, y), (x[:4], y[:4]), {"task_type": "classification"}))
    seen = []
    monkeypatch.setitem(master_loader.PREPROCESSOR_REGISTRY, "record", lambda tr, te, m: seen.append(len(tr[1])) or (tr, te, m))
    raw = master_loader.load_raw_dataset("iris")
    assert not seen and len(raw[0][1]) == 50
    cfg = config(sample_size=7)
    with training_row_selection([cfg]) as selection:
        prepared = master_loader.prepare_dataset("iris", *raw, preprocessors=[{"name": "record"}])
    assert seen == [7]
    assert len(selection.materialize(cfg["service_id"], prepared[0])["y_train"]) == 7


def test_loader_selects_before_calling_generation_preprocessor(monkeypatch):
    from mlaas_data_generator.data import master_loader
    raw = Dataset.from_dict({"article": ["raw source"] * 203037})
    monkeypatch.setattr(master_loader, "load_huggingface_source", lambda **k:
        ((raw, None), (raw.select([0, 1]), None), {"hf_task": "causal_lm_generation"}))
    seen = []
    def prepare(tr, te, meta, **kwargs):
        seen.append(len(tr[0]))
        return (np.arange(len(tr[0])), np.arange(len(tr[0]))), (np.arange(2), np.arange(2)), meta
    monkeypatch.setitem(master_loader.PREPROCESSOR_REGISTRY, "hf", prepare)
    with training_row_selection([config(task="causal_lm_generation", sample_size=1000)]):
        train, _, _ = master_loader.load_dataset("hf", hf_model_id="offline")
    assert seen == [1000] and len(train[1]) == 1000


def test_explicit_prepared_feature_axis_is_the_only_full_pool_sampling_path(monkeypatch):
    tok, load = generation_fixture(monkeypatch, count=100)
    cfg = config(strategy="dirichlet", task="causal_lm_generation", skew_axis="supervised_token_bucket")
    with training_row_selection([cfg]) as selection:
        actual = load()
    assert selection.materialize(cfg["service_id"], actual[0]) is None
    assert len([t for t in tok.seen if t.startswith("source")]) == 100
    assert "explicit prepared-feature" in actual[2]["tokenization_selection"]["reason"]


def test_scope_is_reset_after_failure(tokenizer):
    with pytest.raises(RuntimeError):
        with training_row_selection([config()]):
            raise RuntimeError("interrupted")
    assert len(prepare(dataset())[0][1]) == 40


def test_saved_family_execution_prepares_selected_union_and_retains_checkpoints(tokenizer, monkeypatch, tmp_path):
    import json
    import sqlite3
    from mlaas_data_generator.cli.manifest import preparation
    from mlaas_data_generator.cli import run_manifest as manifest
    from mlaas_data_generator.services import runner
    from tests.generator.test_service_runner import DummyModel

    model = DummyModel()
    model.set_weights = lambda weights: None
    seen = []
    model.fit = lambda x, y, **kwargs: seen.append(x["input_ids"][:, 0].tolist())
    model.predict = lambda x, **kwargs: np.tile([0.7, 0.1, 0.1, 0.1], (len(x["input_ids"]), 1))
    monkeypatch.setattr(runner, "_resolve_hf_revisions", lambda cfg: cfg.update(hf_model_revision="a" * 40, hf_dataset_revision="b" * 40))
    monkeypatch.setattr(runner, "_fetch_hf_metadata", lambda *a: {})
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})
    configs = [config(seed=s, no_save_weights=False, weight_retention="checkpoint", model_type="mlp", training_regime="generic",
        hf_model_id="offline", db_path=str(tmp_path / "services.db"), training_epochs=1,
        batch_size=2, enable_perturbation_metrics=False, update_signature_enabled=False,
        measure_system_metrics=False, provenance_artifact_dir=str(tmp_path / "artifacts")) for s in (42, 43)]
    entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=cfg, validation=manifest.RowValidation(True))
               for i, cfg in enumerate(configs)]
    results = preparation.execute_entries(entries, load_dataset=lambda *a, **k: prepare(dataset()),
                                          build_model=lambda *a: model)
    assert all(r["status"] == "success" for r in results), results
    union = set()
    for cfg, trained in zip(configs, seen):
        rows = np.random.default_rng(cfg["sample_seed"]).permutation(40)[:6]
        assert trained == (rows + 10).tolist()
        union.update(rows.tolist())
    assert len(tokenizer.calls[0]) == len(union) < 40
    with sqlite3.connect(configs[0]["db_path"]) as conn:
        assert conn.execute("SELECT count(*) FROM service_checkpoint_storage WHERE round_trip_status='passed'").fetchone()[0] == 2
        for (payload,) in conn.execute("SELECT split_config_json FROM service_split_provenance WHERE split_name='train'"):
            split = json.loads(payload)
            assert split["source_pool_sample_count"] == 40
            assert len(split["partition_source_ids"]) == 6
    # A durable family no longer binds a client union. Reopening one client
    # prepares just its rows under the same family contract.
    cfg = {**configs[0], "_family_reference": {
        "document_ref": configs[0]["family_document_ref"], "document_hash": configs[0]["family_document_hash"]}}
    tokenizer.calls.clear()
    with training_row_selection([cfg]) as selection:
        _, _, _, reloaded, _, _ = preparation.prepare(cfg, load_dataset=lambda *a, **k: prepare(dataset()), build_model=lambda *a: model)
    assert len(reloaded[0][1]) == 6
    assert len(selection.materialize(cfg["service_id"], reloaded[0])["y_train"]) == 6


@pytest.mark.parametrize("grouped", [False, True])
def test_execution_trains_selected_rows_without_sampling_twice(tokenizer, monkeypatch, tmp_path, grouped):
    from mlaas_data_generator.services import runner
    from mlaas_data_generator.cli import run_manifest as manifest
    from tests.generator.test_service_runner import DummyModel
    ds = dataset()
    configs = [config(seed=s, model_type="mlp", training_regime="generic", db_path=str(tmp_path / "services.db"),
        training_epochs=1, enable_perturbation_metrics=False, update_signature_enabled=False,
        measure_system_metrics=False, batch_size=2, provenance_artifact_dir=str(tmp_path / "artifacts")) for s in (42, 43)]
    seen = []
    model = DummyModel()
    model.fit = lambda x, y, **kwargs: seen.append(x["input_ids"].copy())
    model.predict = lambda x, **kwargs: np.tile([0.7, 0.1, 0.1, 0.1], (len(x["input_ids"]), 1))
    model.set_weights = lambda weights: None
    monkeypatch.setattr(runner, "_load_dataset", lambda *args, **kwargs: prepare(ds))
    monkeypatch.setattr(runner, "_resolve_hf_revisions", lambda cfg: None)
    monkeypatch.setattr(runner, "_fetch_hf_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(runner.ServiceRunner, "_build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(manifest, "_build_prepared_hf_model", lambda *args, **kwargs: model)
    if grouped:
        entries = [manifest.ManifestEntry(idx=i, ordinal=i + 1, resolved=cfg, validation=manifest.RowValidation(True))
                   for i, cfg in enumerate(configs)]
        results = manifest._execute_hf_model_group(entries)
        assert all(r["status"] == "success" for r in results), results
    else:
        results = [runner.execute_service(cfg) for cfg in configs]
        assert all(r.status == "success" for r in results), [r.error for r in results]
    for cfg, trained in zip(configs, seen):
        np.testing.assert_array_equal(trained[:, 0] - 10, np.random.default_rng(cfg["sample_seed"]).permutation(40)[:6])


