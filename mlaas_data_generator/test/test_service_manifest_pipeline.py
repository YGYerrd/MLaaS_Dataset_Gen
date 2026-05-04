import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import mlaas_data_generator.cli.run_manifest as run_manifest_module
from mlaas_data_generator.services import runner as service_runner_module
from mlaas_data_generator.cli.manifest.hf_manifest_builder import MANIFEST_COLUMNS, build_hf_manifest
from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row, run_manifest
from mlaas_data_generator.services.runner import ServiceExecutionResult


def test_manifest_builder_emits_service_rows_without_federated_columns():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer", "inference_only"],
        knob_variants_per_pair=2,
        total_services=4,
        seed=123,
        manifest_profile="test",
    )

    assert not df.empty
    assert list(df.columns) == MANIFEST_COLUMNS
    assert df["service_id"].is_unique
    assert {"service_id", "service_config", "training_regime", "dataset_variant", "split_variant", "knob_variant"}.issubset(df.columns)
    assert {"skew_axis", "skew_axis_config"}.issubset(df.columns)
    assert not {"num_rounds", "client_participation_rate", "aggregation"}.intersection(df.columns)
    assert "num_clients" not in df.columns
    assert set(df["training_regime"]).issubset({"finetune_transfer", "inference_only"})
    assert set(df["resource_tier"]) == {"light"}
    for payload in df["service_config"]:
        assert isinstance(json.loads(payload), dict)


def test_manifest_resource_tier_caps_model_and_workload_size():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=3,
        datasets_per_model=2,
        training_regimes=["finetune_transfer"],
        resource_tier="light",
        knob_variants_per_pair=2,
        seed=123,
    )

    assert not df.empty
    assert set(df["resource_tier"]) == {"light"}
    assert set(df["model_resource_tier"]) == {"light"}
    assert df["max_samples"].max() <= 128
    assert df["max_length"].dropna().max() <= 96


def test_manifest_smoketest_expands_to_all_pairs_with_minimum_sample_budgets():
    baseline = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="light",
        seed=123,
    )
    smoke = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="smoketest",
        seed=123,
    )

    assert not smoke.empty
    assert set(smoke["resource_tier"]) == {"smoketest"}
    assert set(smoke["training_regime"]) == {"finetune_transfer"}
    assert smoke["max_samples"].max() <= 8
    assert smoke["sample_size"].dropna().max() <= 7
    assert len(smoke) > len(baseline)
    smoke_pairs = set(zip(smoke["hf_model_id"], smoke["dataset_name"], smoke["dataset_config"], strict=False))
    assert len(smoke_pairs) == len(smoke)
    baseline_pairs = set(zip(baseline["hf_model_id"], baseline["dataset_name"], baseline["dataset_config"], strict=False))
    assert baseline_pairs.issubset(smoke_pairs)


def test_run_manifest_validation_accepts_smoketest_resource_tier():
    row = pd.Series(
        {
            "service_id": "svc_smoketest",
            "enabled": True,
            "dataset": "hf",
            "model_type": "hf_finetune",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "train_split": "train",
            "test_split": "validation",
            "benchmark_split": "validation",
            "text_column": "sentence",
            "label_column": "label",
            "training_regime": "finetune_transfer",
            "resource_tier": "smoketest",
            "training_epochs": 1,
            "batch_size": 4,
            "precision_type": "fp16",
        }
    )

    validation = _validate_row(_resolve_row(row, {}))
    assert validation.ok, validation.error


def test_manifest_knob_variants_are_task_aware_and_distinct():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="medium",
        knob_variants_per_pair=6,
        seed=123,
    )

    assert list(df["knob_variant"]) == [0, 1, 2, 3, 4, 5]
    assert {"adamw", "adam", "sgd"}.issubset(set(df["optimizer"]))
    assert len(set(df["batch_size"])) > 1
    assert len(set(df["learning_rate"])) > 1
    assert len(set(df["sample_size"].dropna())) > 1
    assert len(set(df["sample_seed"])) == len(df)
    assert len(set(df["warmup_ratio"])) > 1
    assert len(set(df["gradient_accumulation_steps"])) > 1
    assert set(df["precision_type"]) == {"fp16"}
    assert set(df["split_strategy"]).issubset({"iid", "dirichlet"})
    assert df["skew_axis"].notna().all()
    for payload in df["service_config"]:
        config = json.loads(payload)
        assert config["resource_tier"] == "medium"
        assert config["max_train_time_s"] == 120
        assert "sample_seed" in config
        assert "warmup_ratio" in config
        assert "gradient_accumulation_steps" in config
        assert "precision_type" in config
        assert "skew_axis" in config


def test_manifest_mixed_precision_distribution_targets_runtime_regime_band():
    df = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="medium",
        knob_variants_per_pair=20,
        seed=123,
    )

    ratio = float(df["mixed_precision"].mean())
    assert 0.70 <= ratio <= 0.80
    assert set(df.loc[df["mixed_precision"], "precision_type"]) == {"fp16"}


def test_manifest_fill_mask_knob_variants_include_mlm_probability():
    df = build_hf_manifest(
        task_keys=["fill_mask"],
        models_per_task=1,
        datasets_per_model=1,
        training_regimes=["finetune_transfer"],
        resource_tier="medium",
        knob_variants_per_pair=4,
        seed=123,
    )

    assert list(df["knob_variant"]) == [0, 1, 2, 3]
    assert list(df["mlm_probability"]) == [0.10, 0.15, 0.20, 0.30]
    for payload, expected in zip(df["service_config"], df["mlm_probability"], strict=False):
        config = json.loads(payload)
        assert config["mlm_probability"] == expected


def test_manifest_inference_uses_explicit_dataset_matches():
    df = build_hf_manifest(
        task_keys=["object_detection"],
        models_per_task=3,
        datasets_per_model=5,
        training_regimes=["inference_only"],
        resource_tier="medium",
        seed=123,
    )

    assert not df.empty
    assert set(df["training_regime"]) == {"inference_only"}
    assert set(df["dataset_name"]) == {"detection-datasets/coco"}
    assert set(df["learning_rate"].dropna()) == set()


def test_manifest_text2text_inference_retiers_slow_summarization_models():
    df = build_hf_manifest(
        task_keys=["text2text_generation"],
        models_per_task=10,
        datasets_per_model=10,
        training_regimes=["inference_only"],
        resource_tier="medium",
        seed=123,
    )

    assert not df.empty
    t5_base = df[df["hf_model_id"] == "t5-base"]
    t5_small = df[df["hf_model_id"] == "t5-small"]

    assert not t5_base.empty
    assert not t5_small.empty
    assert set(t5_base["max_eval_time_s"]) == {240}
    assert set(t5_small["max_eval_time_s"]) == {120}
    assert t5_base["max_samples"].max() <= 384
    assert t5_small["max_samples"].max() >= t5_base["max_samples"].max()


def test_manifest_service_ids_are_deterministic():
    first = build_hf_manifest(task_keys=["text_classification"], models_per_task=1, datasets_per_model=1, total_services=2, seed=99)
    second = build_hf_manifest(task_keys=["text_classification"], models_per_task=1, datasets_per_model=1, total_services=2, seed=99)
    pd.testing.assert_series_equal(first["service_id"], second["service_id"])


def test_manifest_builder_varies_when_seed_is_omitted():
    first = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        total_services=2,
        seed=None,
    )
    second = build_hf_manifest(
        task_keys=["text_classification"],
        models_per_task=1,
        datasets_per_model=1,
        total_services=2,
        seed=None,
    )

    assert not first.empty
    assert not second.empty
    assert not first["service_id"].equals(second["service_id"])


def test_run_manifest_dry_run_validates_service_rows(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "service_id": "svc_manual",
                "enabled": True,
                "dataset": "hf",
                "model_type": "hf",
                "task_type": "classification",
                "hf_task": "sequence_classification",
                "hf_model_id": "distilbert-base-uncased",
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "training_regime": "inference_only",
                "batch_size": 4,
            }
        ]
    ).to_csv(manifest, index=False)

    results_path = run_manifest(str(manifest), dry_run=True, db_path=str(tmp_path / "services.db"))
    results = pd.read_csv(results_path)

    assert results.iloc[0]["service_id"] == "svc_manual"
    assert results.iloc[0]["status"] == "success"
    resolved = json.loads(results.iloc[0]["resolved_config_json"])
    assert resolved["benchmark_split"] == "validation"
    assert "num_rounds" not in resolved
    assert "num_clients" not in resolved


def test_run_manifest_writes_failed_rows_to_retry_manifest(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    results_path = tmp_path / "service_manifest_results.csv"
    failed_manifest_path = tmp_path / "service_manifest_failed.csv"
    db_path = tmp_path / "services.db"
    pd.DataFrame(
        [
            {
                "service_id": "defaults",
                "batch_size": 4,
                "training_regime": "generic",
            },
            {
                "service_id": "svc_retry",
                "enabled": True,
                "dataset": "fixture",
                "model_type": "generic",
                "task_type": "classification",
            },
        ]
    ).to_csv(manifest, index=False)

    def fake_execute_service(config):
        return ServiceExecutionResult(
            service_id=config["service_id"],
            status="failed",
            db_path=str(db_path),
            metrics={},
            error="planned failure",
        )

    monkeypatch.setattr(run_manifest_module, "MANIFEST_RESULTS_PATH", results_path)
    monkeypatch.setattr(run_manifest_module, "FAILED_MANIFEST_PATH", failed_manifest_path)
    monkeypatch.setattr(run_manifest_module, "execute_service", fake_execute_service)

    run_manifest(str(manifest), db_path=str(db_path))

    retry = pd.read_csv(failed_manifest_path)
    assert len(retry) == 1
    assert retry.iloc[0]["service_id"] == "svc_retry"
    assert retry.iloc[0]["dataset"] == "fixture"
    assert retry.iloc[0]["batch_size"] == 4
    assert retry.iloc[0]["db_path"] == str(db_path)


def test_run_manifest_removes_stale_retry_manifest_when_no_rows_fail(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    failed_manifest_path = tmp_path / "service_manifest_failed.csv"
    failed_manifest_path.write_text("service_id\nstale\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "service_id": "svc_ok",
                "enabled": True,
                "dataset": "hf",
                "model_type": "hf",
                "task_type": "classification",
                "hf_task": "sequence_classification",
                "hf_model_id": "distilbert-base-uncased",
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "training_regime": "inference_only",
                "batch_size": 4,
            }
        ]
    ).to_csv(manifest, index=False)

    monkeypatch.setattr(run_manifest_module, "MANIFEST_RESULTS_PATH", tmp_path / "service_manifest_results.csv")
    monkeypatch.setattr(run_manifest_module, "FAILED_MANIFEST_PATH", failed_manifest_path)

    run_manifest(str(manifest), dry_run=True, db_path=str(tmp_path / "services.db"))

    assert not failed_manifest_path.exists()


class TransformersGroupedModel:
    fit_start_weights = []
    fit_lrs = []
    fail_next_fit = False

    def __init__(self):
        self.weight = 0.0

    def fit(self, x, y, epochs=1, lr=5e-5, max_train_time_s=60, progress_log_interval=None):
        self.__class__.fit_start_weights.append(float(self.weight))
        self.__class__.fit_lrs.append(float(lr))
        self.weight += 1.0
        if self.__class__.fail_next_fit:
            self.__class__.fail_next_fit = False
            raise RuntimeError("planned grouped fit failure")
        return {"train_loss": 0.1, "train_sequence_count": len(y), "train_supervised_token_count": len(y)}

    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=None):
        return 0.2, 0.75, 0.5, {"eval_sequence_count": len(y), "eval_batch_count": 1}

    def count_params(self):
        return 64

    def get_weights(self):
        return {"weight": np.asarray([self.weight], dtype="float32")}

    def set_weights(self, weights):
        self.weight = float(np.asarray(weights["weight"]).reshape(-1)[0])


def _grouped_manifest_rows(db_path):
    base = {
        "enabled": True,
        "dataset": "hf",
        "dataset_name": "fixture/dataset",
        "dataset_config": "default",
        "train_split": "train",
        "test_split": "validation",
        "model_type": "hf_finetune",
        "task_type": "classification",
        "hf_task": "sequence_classification",
        "hf_model_id": "fixture/model",
        "text_column": "text",
        "label_column": "label",
        "training_regime": "finetune_transfer",
        "training_epochs": 1,
        "batch_size": 2,
        "max_samples": 4,
        "update_signature_enabled": False,
        "enable_perturbation_metrics": False,
        "measure_system_metrics": False,
        "db_path": str(db_path),
    }
    rows = []
    for idx, lr in enumerate((1e-5, 2e-5)):
        row = dict(base)
        row.update({"service_id": f"svc_grouped_{idx}", "knob_variant": idx, "learning_rate": lr})
        rows.append(row)
    return rows


def test_run_manifest_groups_hf_rows_and_resets_model_between_knobs(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    db_path = tmp_path / "services.db"
    pd.DataFrame(_grouped_manifest_rows(db_path)).to_csv(manifest, index=False)

    calls = {"load_dataset": 0, "create_model": 0}
    TransformersGroupedModel.fit_start_weights = []
    TransformersGroupedModel.fit_lrs = []
    TransformersGroupedModel.fail_next_fit = False

    def fake_load_dataset(name, **kwargs):
        calls["load_dataset"] += 1
        x = {"input_ids": np.arange(8).reshape(4, 2), "attention_mask": np.ones((4, 2), dtype="int64")}
        y = np.asarray([0, 1, 0, 1])
        meta = {"input_shape": (2,), "num_classes": 2, "task_type": "classification", "hf_task": "sequence_classification", "dataset_family": "hf"}
        return (x, y), (x, y), meta

    def fake_create_model(**kwargs):
        calls["create_model"] += 1
        return TransformersGroupedModel()

    monkeypatch.setattr(run_manifest_module, "_ensure_manifest_preflight", lambda enabled_df: None)
    monkeypatch.setattr(service_runner_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(service_runner_module, "create_model", fake_create_model)
    monkeypatch.setattr(service_runner_module, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(service_runner_module, "_service_perturbation_metrics", lambda *args, **kwargs: ({}, None))

    results_path = run_manifest(str(manifest), db_path=str(db_path))
    results = pd.read_csv(results_path)

    assert calls == {"load_dataset": 1, "create_model": 1}
    assert list(results["row_index"]) == [0, 1]
    assert set(results["status"]) == {"success"}
    assert TransformersGroupedModel.fit_start_weights == [0.0, 0.0]
    assert TransformersGroupedModel.fit_lrs == [1e-5, 2e-5]


def test_grouped_hf_failure_does_not_poison_next_service(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    db_path = tmp_path / "services.db"
    pd.DataFrame(_grouped_manifest_rows(db_path)).to_csv(manifest, index=False)

    TransformersGroupedModel.fit_start_weights = []
    TransformersGroupedModel.fit_lrs = []
    TransformersGroupedModel.fail_next_fit = True

    def fake_load_dataset(name, **kwargs):
        x = {"input_ids": np.arange(8).reshape(4, 2), "attention_mask": np.ones((4, 2), dtype="int64")}
        y = np.asarray([0, 1, 0, 1])
        meta = {"input_shape": (2,), "num_classes": 2, "task_type": "classification", "hf_task": "sequence_classification", "dataset_family": "hf"}
        return (x, y), (x, y), meta

    monkeypatch.setattr(run_manifest_module, "_ensure_manifest_preflight", lambda enabled_df: None)
    monkeypatch.setattr(service_runner_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(service_runner_module, "create_model", lambda **kwargs: TransformersGroupedModel())
    monkeypatch.setattr(service_runner_module, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(service_runner_module, "_service_perturbation_metrics", lambda *args, **kwargs: ({}, None))

    results_path = run_manifest(str(manifest), db_path=str(db_path))
    results = pd.read_csv(results_path)

    assert list(results["status"]) == ["failed", "success"]
    assert TransformersGroupedModel.fit_start_weights == [0.0, 0.0]


def test_run_manifest_progress_counts_grouped_rows(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    db_path = tmp_path / "services.db"
    pd.DataFrame(_grouped_manifest_rows(db_path)).to_csv(manifest, index=False)

    class FakeProgress:
        last = None

        def __init__(self, total):
            self.total = total
            self.records = []
            self.starts = []
            self.clears = []
            self.finished = False
            FakeProgress.last = self

        def start(self, worker_label, description):
            self.starts.append((worker_label, description))

        def clear(self, worker_label):
            self.clears.append(worker_label)

        def record(self, status):
            self.records.append(status)

        def finish(self):
            self.finished = True

    def fake_load_dataset(name, **kwargs):
        x = {"input_ids": np.arange(8).reshape(4, 2), "attention_mask": np.ones((4, 2), dtype="int64")}
        y = np.asarray([0, 1, 0, 1])
        meta = {"input_shape": (2,), "num_classes": 2, "task_type": "classification", "hf_task": "sequence_classification", "dataset_family": "hf"}
        return (x, y), (x, y), meta

    monkeypatch.setattr(run_manifest_module, "ManifestProgressTracker", FakeProgress)
    monkeypatch.setattr(run_manifest_module, "_ensure_manifest_preflight", lambda enabled_df: None)
    monkeypatch.setattr(service_runner_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(service_runner_module, "create_model", lambda **kwargs: TransformersGroupedModel())
    monkeypatch.setattr(service_runner_module, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(service_runner_module, "_service_perturbation_metrics", lambda *args, **kwargs: ({}, None))

    run_manifest(str(manifest), db_path=str(db_path), workers=1)

    tracker = FakeProgress.last
    assert tracker is not None
    assert tracker.total == 2
    assert tracker.records == ["success", "success"]
    assert tracker.finished is True


def test_resolved_manifest_row_gets_deterministic_service_id():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "training_regime": "inference_only",
            "batch_size": 4,
        }
    )
    resolved = _resolve_row(row, {})
    assert resolved["service_id"].startswith("classification_")
    assert _validate_row(resolved).ok


def test_resolved_manifest_row_forces_mixed_precision_off_on_cpu():
    row = pd.Series(
        {
            "dataset": "hf",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "dataset_config": "sst2",
            "training_regime": "inference_only",
            "batch_size": 4,
            "device": "cpu",
            "mixed_precision": True,
            "precision_type": "bf16",
        }
    )
    resolved = _resolve_row(row, {})
    assert resolved["mixed_precision"] is False
    assert resolved["precision_type"] == "fp16"
    assert _validate_row(resolved).ok


def test_run_manifest_rejects_legacy_federated_columns():
    row = pd.Series(
        {
            "service_id": "svc_bad",
            "dataset": "hf",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "hf_model_id": "distilbert-base-uncased",
            "dataset_name": "glue",
            "training_regime": "inference_only",
            "batch_size": 4,
            "num_rounds": 2,
        }
    )

    validation = _validate_row(_resolve_row(row, {}))

    assert not validation.ok
    assert "Federated columns" in validation.error


def test_manifest_builder_filters_tiny_detection_and_segmentation_datasets():
    det_df = build_hf_manifest(
        task_keys=["object_detection"],
        models_per_task=3,
        datasets_per_model=10,
        training_regimes=["finetune_transfer"],
        resource_tier="light",
        seed=123,
    )
    seg_df = build_hf_manifest(
        task_keys=["image_segmentation"],
        models_per_task=10,
        datasets_per_model=10,
        training_regimes=["finetune_transfer"],
        resource_tier="light",
        seed=123,
    )

    assert not det_df.empty
    assert not seg_df.empty
    assert "mini" not in set(det_df["dataset_config"].dropna())
    assert "nateraw/ade20k-tiny" not in set(seg_df["dataset_name"])


def test_validate_row_rejects_tiny_detection_manifest_row():
    row = pd.Series(
        {
            "service_id": "svc_tiny_det",
            "dataset": "hf",
            "dataset_name": "keremberke/license-plate-object-detection",
            "dataset_config": "mini",
            "model_type": "hf_finetune",
            "task_type": "detection",
            "hf_task": "image_detection",
            "hf_model_id": "hustvl/yolos-tiny",
            "training_regime": "finetune_transfer",
            "batch_size": 2,
        }
    )

    validation = _validate_row(_resolve_row(row, {}))

    assert not validation.ok
    assert "require at least 32 train examples" in validation.error


def test_validate_row_rejects_upernet_batch_size_one():
    row = pd.Series(
        {
            "service_id": "svc_bad_seg",
            "dataset": "hf",
            "dataset_name": "zhoubolei/scene_parse_150",
            "model_type": "hf_finetune",
            "task_type": "segmentation",
            "hf_task": "image_segmentation",
            "hf_model_id": "openmmlab/upernet-convnext-tiny",
            "training_regime": "finetune_transfer",
            "batch_size": 1,
        }
    )

    validation = _validate_row(_resolve_row(row, {}))

    assert not validation.ok
    assert "batch_size >= 2" in validation.error


def test_manifest_builder_excludes_known_bad_paths():
    df = build_hf_manifest(
        task_keys=["text_classification", "text2text_generation", "image_segmentation", "image_classification"],
        models_per_task=20,
        datasets_per_model=20,
        training_regimes=["finetune_transfer"],
        resource_tier="medium",
        seed=123,
    )

    assert not df.empty
    assert not (df["hf_model_id"] == "squeezebert/squeezebert-uncased").any()
    assert not (
        (df["hf_model_id"] == "Salesforce/codet5-small")
        & (df["task"] == "text2text_generation")
    ).any()
    assert not (
        (df["dataset_name"] == "buddhi19/SyntheticGenV5")
        & (df["task"] == "image_segmentation")
    ).any()
    assert not (
        (df["hf_model_id"] == "microsoft/resnet-50")
        & (df["dataset_name"] == "cifar10")
        & (df["task"] == "image_classification")
    ).any()


@pytest.mark.parametrize(
    ("row", "expected_message"),
    [
        (
            {
                "service_id": "svc_blocked_squeezebert",
                "dataset": "hf",
                "dataset_name": "ag_news",
                "model_type": "hf_finetune",
                "task_type": "classification",
                "hf_task": "sequence_classification",
                "hf_model_id": "squeezebert/squeezebert-uncased",
                "training_regime": "finetune_transfer",
                "batch_size": 8,
            },
            "squeezebert/squeezebert-uncased",
        ),
        (
            {
                "service_id": "svc_blocked_codet5",
                "dataset": "hf",
                "dataset_name": "EdinburghNLP/xsum",
                "model_type": "hf_finetune",
                "task_type": "text2text_generation",
                "task": "text2text_generation",
                "task_tag": "summarization",
                "hf_task": "seq2seq_generation",
                "hf_model_id": "Salesforce/codet5-small",
                "text_column": "document",
                "label_column": "summary",
                "training_regime": "finetune_transfer",
                "batch_size": 4,
            },
            "Salesforce/codet5-small",
        ),
        (
            {
                "service_id": "svc_blocked_seg_schema",
                "dataset": "hf",
                "dataset_name": "buddhi19/SyntheticGenV5",
                "model_type": "hf_finetune",
                "task_type": "segmentation",
                "task": "image_segmentation",
                "hf_task": "image_segmentation",
                "hf_model_id": "nvidia/segformer-b4-finetuned-ade-512-512",
                "image_column": "image",
                "mask_column": "mask",
                "training_regime": "finetune_transfer",
                "batch_size": 4,
            },
            "SyntheticGenV5",
        ),
        (
            {
                "service_id": "svc_blocked_miopen_pair",
                "dataset": "hf",
                "dataset_name": "cifar10",
                "model_type": "hf_finetune",
                "task_type": "classification",
                "task": "image_classification",
                "hf_task": "image_classification",
                "hf_model_id": "microsoft/resnet-50",
                "image_column": "img",
                "label_column": "label",
                "training_regime": "finetune_transfer",
                "batch_size": 8,
            },
            "miopenStatusUnknownError",
        ),
        (
            {
                "service_id": "svc_blocked_mobilenet_pet",
                "dataset": "hf",
                "dataset_name": "timm/oxford-iiit-pet",
                "model_type": "hf_finetune",
                "task_type": "classification",
                "task": "image_classification",
                "hf_task": "image_classification",
                "hf_model_id": "google/mobilenet_v2_1.0_224",
                "image_column": "image",
                "label_column": "label",
                "training_regime": "finetune_transfer",
                "batch_size": 8,
            },
            "miopenStatusUnknownError",
        ),
        (
            {
                "service_id": "svc_blocked_mobilenet_fashion_mnist",
                "dataset": "hf",
                "dataset_name": "zalando-datasets/fashion_mnist",
                "dataset_config": "default",
                "model_type": "hf_finetune",
                "task_type": "classification",
                "task": "image_classification",
                "hf_task": "image_classification",
                "hf_model_id": "google/mobilenet_v2_1.0_224",
                "image_column": "image",
                "label_column": "label",
                "training_regime": "finetune_transfer",
                "batch_size": 8,
            },
            "miopenStatusUnknownError",
        ),
    ],
)
def test_validate_row_rejects_known_bad_paths(row, expected_message):
    validation = _validate_row(_resolve_row(pd.Series(row), {}))

    assert not validation.ok
    assert expected_message in validation.error


def test_run_manifest_preflight_records_one_failure_for_missing_datasets(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    db_path = tmp_path / "services.db"
    pd.DataFrame(
        [
            {
                "service_id": "svc_preflight",
                "enabled": True,
                "dataset": "hf",
                "model_type": "hf",
                "task_type": "classification",
                "hf_task": "sequence_classification",
                "hf_model_id": "distilbert-base-uncased",
                "dataset_name": "glue",
                "dataset_config": "sst2",
                "train_split": "train",
                "test_split": "validation",
                "text_column": "sentence",
                "label_column": "label",
                "training_regime": "inference_only",
                "batch_size": 4,
            }
        ]
    ).to_csv(manifest, index=False)

    real_import_module = run_manifest_module.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("datasets missing for test")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(run_manifest_module.importlib, "import_module", fake_import_module)

    results_path = run_manifest(str(manifest), dry_run=False, db_path=str(db_path))
    results = pd.read_csv(results_path)

    assert len(results) == 1
    assert results.iloc[0]["status"] == "failed"
    assert "datasets" in results.iloc[0]["error_message"]

    with sqlite3.connect(db_path) as conn:
        failure_stage, service_id = conn.execute(
            "SELECT failure_stage, service_id FROM service_failures"
        ).fetchone()
        assert failure_stage == "manifest_preflight"
        assert service_id is None
        assert conn.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 0
