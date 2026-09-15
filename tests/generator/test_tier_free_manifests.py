import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import pandas as pd
import pytest

from mlaas_data_generator.clients import calibration, recipes, training
from mlaas_data_generator.cli.manifest.cmd_hf_manifest import register_hf_manifest
from mlaas_data_generator.cli.manifest.hf_manifest_builder import (
    _sort_models, _sort_datasets, build_hf_manifest, save_manifest,
)
from mlaas_data_generator.cli.manifest.review import expand_review_manifest
from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row
from mlaas_data_generator.cohort_manifest import build_cohort_manifest
from mlaas_data_generator.provenance import sha256_json
from mlaas_data_generator.shared_parent import PARENTS_VERSION


def assert_no_tiers(value):
    if isinstance(value, dict):
        assert not any("tier" in str(key) for key in value)
        for item in value.values():
            assert_no_tiers(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_tiers(item)
    elif isinstance(value, str) and value.startswith(("{", "[")):
        assert_no_tiers(json.loads(value))


def generic_manifest(**kwargs):
    return build_hf_manifest(task_keys=["sklearn_image_classification"],
        clients_per_family=8, seed=42, average_samples=1000, **kwargs)


def test_profiles_do_not_select_training_ranges_or_budgets():
    baseline = generic_manifest(manifest_profile="test")
    for profile in ("balanced", "benchmark"):
        pd.testing.assert_frame_equal(baseline, generic_manifest(manifest_profile=profile))
    assert_no_tiers(baseline.to_dict("records"))
    assert set(baseline["split_strategy"]) == set(recipes.PROFILES)


def test_no_model_or_dataset_is_excluded_by_resource_size():
    models = [{"hf_model_id": f"model-{size}", "estimated_params_m": size}
              for size in (5, 200, 600, 2700)]
    selected = _sort_models(models, task_key="text_classification",
        selected_training_regimes=["finetune_transfer"], rng=random.Random(42))
    assert {r["hf_model_id"] for r in selected} == {r["hf_model_id"] for r in models}
    datasets = [{"dataset_name": str(n), "max_samples": n} for n in (100, 2000, 6000, 20000)]
    selected = _sort_datasets(datasets, task_key="text_classification", rng=random.Random(42))
    assert len(selected) == len(datasets)
    assert_no_tiers(selected)


def test_explicit_budgets_survive_review_roundtrip(tmp_path):
    df = generic_manifest(timeout_s=900, max_train_time_s=300, max_eval_time_s=240)
    path = tmp_path / "clients.csv"
    save_manifest(df, path)
    assert_no_tiers(json.loads(path.with_suffix(".families.json").read_text()))
    expanded, defaults = expand_review_manifest(pd.read_csv(path), path)
    for _, row in expanded.iterrows():
        config = _resolve_row(row, defaults)
        assert_no_tiers(config)
        assert (config["timeout_s"], config["max_train_time_s"], config["max_eval_time_s"]) == (900, 300, 240)
        assert config["training_policy"]["limits"]["max_train_time_s"] == 300
        training.validate_training_recipe(config)
        validation = _validate_row(config)
        assert validation.ok, validation.error


@pytest.mark.parametrize("key", ["timeout_s", "max_train_time_s", "max_eval_time_s"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_explicit_budgets_are_rejected(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        generic_manifest(**{key: value})


def test_removed_cli_and_python_argument_are_rejected():
    parser = argparse.ArgumentParser()
    register_hf_manifest(parser.add_subparsers())
    with pytest.raises(SystemExit):
        parser.parse_args(["hf-manifest", "--resource-tier", "medium"])
    with pytest.raises(TypeError, match="resource_tier"):
        build_hf_manifest(resource_tier="medium")
    args = parser.parse_args(["hf-manifest", "--max-train-time-s", "300"])
    assert args.max_train_time_s == 300


def test_inference_service_identity_includes_explicit_budget():
    options = dict(task_keys=["text2text_generation"], models_per_task=1,
                   datasets_per_model=1, training_regimes=["inference_only"], seed=42)
    first = build_hf_manifest(**options, max_eval_time_s=120)
    second = build_hf_manifest(**options, max_eval_time_s=240)
    assert not first.empty and len(first) == len(second)
    assert set(first.service_id).isdisjoint(second.service_id)


@pytest.mark.parametrize("version", ["conditional_balancing_v1", "conditional_balancing_v2", "conditional_balancing_v3"])
def test_saved_tiered_recipe_validates_without_new_evidence(monkeypatch, version):
    fixture = Path(__file__).with_name("fixtures") / "legacy_calibrated_recipe_v3.json"
    row = json.loads(fixture.read_text())
    if version != row["recipe_sampler_version"]:
        row["recipe_sampler_version"] = version
        row["training_policy"]["version"] = version
        row["training_policy"].pop("calibration")
        row.pop("calibration_band")
        row.pop("calibration_template_id")
        row["training_policy_hash"] = sha256_json(row["training_policy"])
    before = deepcopy(row)
    monkeypatch.setattr(calibration, "load_evidence", lambda: pytest.fail("replay read new evidence"))
    training.validate_training_recipe(row)
    assert row == before
    row["max_train_time_s"] += 1
    with pytest.raises(ValueError, match="training time limit"):
        training.validate_training_recipe(row)


def test_new_cohorts_do_not_inherit_parent_resource_metadata(tmp_path):
    parent = {key: key for key in (
        "parent_state_hash", "document_hash", "family_id", "parent_checkpoint_id",
        "lineage_id", "document_ref", "compatibility_contract_hash")}
    parent["config"] = {"model_type": "randomforest", "training_regime": "generic",
        "task_type": "classification", "dataset": "sklearn", "dataset_name": "digits",
        "resource_tier": "heavy", "model_resource_tier": "light", "dataset_resource_tier": "heavy",
        "hf_service_meta_json": json.dumps({"resource_tier": "heavy", "model_family": "rf"}),
        "service_config": json.dumps({"resource_tier": "heavy"})}
    path = tmp_path / "parents.json"
    path.write_text(json.dumps({"version": PARENTS_VERSION, "parents": [parent]}))
    df = build_cohort_manifest(path, clients_per_parent=8, verify=False)
    assert len(df) == 8
    assert_no_tiers(df.to_dict("records"))
    for row in df.to_dict("records"):
        training.validate_training_recipe(row)
