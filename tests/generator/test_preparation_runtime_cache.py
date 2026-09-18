import pytest

from mlaas_data_generator.data.runtime_cache import cached_preparation, preparation_cache_scope


def test_cache_is_scoped_bounded_and_does_not_retain_failures(monkeypatch):
    from mlaas_data_generator.data import runtime_cache
    monkeypatch.setitem(runtime_cache._LIMITS, "dataset", 2)
    loads = []
    def get(key):
        return cached_preparation("dataset", key, lambda: loads.append(key) or {"rows": [key]})
    with preparation_cache_scope():
        a, hit = get("a")
        a["rows"].append("mutation")
        assert get("a") == ({"rows": ["a"]}, True)
        get("b")
        get("c")
        assert get("a")[1] is False  # LRU eviction.
        with pytest.raises(ValueError):
            cached_preparation("dataset", "bad", lambda: (_ for _ in ()).throw(ValueError()))
        assert get("bad")[1] is False
    with preparation_cache_scope():
        assert get("bad")[1] is False
    assert get("bad")[1] is False  # Calls outside manifest runs are uncached.


def test_raw_split_reuse_preserves_recipe_sampling_revisions_and_formats(monkeypatch):
    import datasets
    from mlaas_data_generator.data.sources.huggingface import load_huggingface_source
    calls = []
    raw = datasets.Dataset.from_dict({"text": [str(i) for i in range(40)], "label": [i % 2 for i in range(40)]})
    def load(name, config, *, split, **kwargs):
        calls.append((name, config, split, kwargs.get("revision")))
        return raw
    monkeypatch.setattr(datasets, "load_dataset", load)
    kwargs = dict(dataset_name="local", dataset_revision="r1", max_samples=8,
                  benchmark_sample_size=4, seed=7)
    expected = load_huggingface_source(**kwargs)
    assert len(expected[0][0]) == 40  # Historical max_samples no longer truncates the source.
    assert len(expected[1][0]) == 4
    calls.clear()
    with preparation_cache_scope():
        first = load_huggingface_source(**kwargs)
        first[0][0].set_format("numpy")
        second = load_huggingface_source(**kwargs)
        assert len(calls) == 2  # One train load, one test load across both recipes.
        assert second[0][0].format["type"] is None
        assert second[0][0].to_dict() == expected[0][0].to_dict()
        assert second[1][0].to_dict() == expected[1][0].to_dict()
        assert second[2] == expected[2]  # Includes source IDs and accounting.
        changed_training = load_huggingface_source(**{**kwargs, "seed": 8})
        assert changed_training[2]["selected_test_rows"] == second[2]["selected_test_rows"]
        changed = load_huggingface_source(**{**kwargs, "evaluation_seed": 8})
        assert len(calls) == 2
        assert changed[2]["selected_train_rows"] == second[2]["selected_train_rows"]
        assert changed[2]["selected_test_rows"] != second[2]["selected_test_rows"]
        load_huggingface_source(**{**kwargs, "dataset_revision": "r2"})
        assert len(calls) == 4
        load_huggingface_source(**{**kwargs, "dataset_config": "other"})
        assert len(calls) == 6


def test_generation_tokenizer_reuse_does_not_leak_padding_policy(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf_text_generation as generation
    from tests.generator.test_hf_generation_preprocessors import FakeTokenizer
    calls = []
    monkeypatch.setattr(generation, "_load_auto_tokenizer_uncached",
                        lambda model: calls.append(model) or FakeTokenizer())
    with preparation_cache_scope():
        first = generation._load_auto_tokenizer("model-a")
        first.padding_side = "left"
        assert generation._load_auto_tokenizer("model-a").padding_side == "right"
        generation._load_auto_tokenizer("model-b")
    assert calls == ["model-a", "model-b"]
