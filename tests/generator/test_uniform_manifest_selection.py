"""Population sampling must not encode size, quality, cost or task-order ranks."""
from collections import Counter
import random

from mlaas_data_generator.cli.manifest.hf_manifest_builder import (
    _balanced_family_limit, _cap_models_per_family, _sort_datasets, _sort_models,
)


def model_order(models, seed):
    return _sort_models(models, task_key="text_classification",
                        selected_training_regimes=["finetune_transfer"],
                        rng=random.Random(seed))


def test_selection_is_invariant_to_size_quality_and_cost():
    models = [{"hf_model_id": str(i), "estimated_params_m": 5 + i * 100}
              for i in range(8)]
    changed = [{**m, "estimated_params_m": 2000 - i * 200,
                "finetune_validated": i % 2 == 0,
                "inference_dataset_keys": ["dataset"] if i % 2 else []}
               for i, m in enumerate(models)]
    for seed in range(20):
        assert [m["hf_model_id"] for m in model_order(models, seed)] == [
            m["hf_model_id"] for m in model_order(changed, seed)]
        first = [{"dataset_name": str(i), "train_examples": 10 ** i,
                  "max_samples": 10 ** i} for i in range(8)]
        second = [{**d, "train_examples": 10 ** (8-i), "max_samples": 10 ** (8-i)}
                  for i, d in enumerate(first)]
        assert [d["dataset_name"] for d in _sort_datasets(first,
            task_key="text_classification", rng=random.Random(seed))] == [
            d["dataset_name"] for d in _sort_datasets(second,
            task_key="text_classification", rng=random.Random(seed))]


def test_models_have_uniform_opportunity_and_regime_filter_survives():
    models = [{"hf_model_id": str(i), "family": str(i // 2),
               "estimated_params_m": 10 ** i} for i in range(6)]
    models.append({"hf_model_id": "inference-only",
                   "allowed_training_regimes": ["inference_only"]})
    counts = Counter()
    for seed in range(1200):
        chosen = _cap_models_per_family(model_order(models, seed), 1)[:2]
        assert len({m["family"] for m in chosen}) == 2
        counts.update(m["hf_model_id"] for m in chosen)
    assert set(counts) == {str(i) for i in range(6)}
    assert all(320 < n < 480 for n in counts.values())
    assert model_order(models, 42) == model_order(models, 42)
    assert model_order(models, 42) != model_order(models, 43)


def test_truncation_balances_tasks_and_keeps_families_together():
    rows = [{"task": task, "hf_model_id": str(f), "dataset_name": "data",
             "client": c} for task in ("text", "image", "regression")
            for f in range(4) for c in range(5)]
    for seed in range(20):
        chosen = _balanced_family_limit(rows, 30, seed=seed)
        assert Counter(r["task"] for r in chosen) == dict.fromkeys(
            ("text", "image", "regression"), 10)
        assert set(Counter((r["task"], r["hf_model_id"]) for r in chosen).values()) == {5}
        assert chosen == _balanced_family_limit(rows, 30, seed=seed)
    first = Counter(_balanced_family_limit(rows, 5, seed=seed)[0]["task"]
                    for seed in range(300))
    assert all(70 < n < 130 for n in first.values())
    assert _balanced_family_limit(rows, 0, seed=42) == []
    assert _balanced_family_limit(rows, len(rows), seed=42) == rows
    assert len(_balanced_family_limit(rows, 7, seed=42)) == 7


def test_task_rotation_handles_unequal_available_families():
    rows = [{"task": task, "hf_model_id": str(f), "client": c}
            for task, count in (("text", 6), ("generic", 1))
            for f in range(count) for c in range(4)]
    chosen = _balanced_family_limit(rows, 20, seed=42)
    assert len(chosen) == 20
    assert sum(r["task"] == "generic" for r in chosen) == 4


def test_incompatible_model_does_not_consume_architecture_cap(monkeypatch):
    from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder

    good = next(m for m in builder._model_candidates("text_classification")
                if m["hf_model_id"] == "albert/albert-base-v2")
    bad = {**good, "hf_model_id": "unusable", "dataset_keys": ["not-registered"]}
    monkeypatch.setattr(builder, "_model_candidates", lambda task: [bad, good])
    monkeypatch.setattr(builder, "_sort_models", lambda models, **kw: models)
    manifest = builder.build_hf_manifest(task_keys=["text_classification"],
        training_regimes=["finetune_transfer"], models_per_task=1,
        max_models_per_family=1, datasets_per_model=1, clients_per_family=4, seed=42)
    assert len(manifest) == 4
    assert set(manifest.hf_model_id) == {good["hf_model_id"]}
