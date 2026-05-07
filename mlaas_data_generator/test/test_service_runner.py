import sqlite3
import json
import types

import numpy as np
import pytest

from mlaas_data_generator.services import runner
from mlaas_data_generator.cli import run_manifest as run_manifest_cli


class DummyModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, x, y, epochs=1, batch_size=32, verbose=0):
        self.fit_calls += 1

    def evaluate(self, x, y, verbose=0):
        return [0.2, 0.75]

    def predict(self, x, verbose=0):
        return np.asarray([[0.1, 0.9], [0.8, 0.2]])

    def count_params(self):
        return 128

    def get_weights(self):
        return [np.asarray([1.0, 2.0])]


def test_service_runner_writes_one_service_record(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = np.asarray([[0.0], [1.0]])
        y_train = np.asarray([0, 1])
        x_test = np.asarray([[0.0], [1.0]])
        y_test = np.asarray([1, 0])
        meta = {"input_shape": (1,), "num_classes": 2, "task_type": "classification", "input_schema": "tabular_features"}
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: DummyModel())
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})

    result = runner.execute_service(
        {
            "service_id": "svc_smoke",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic",
            "model_type": "mlp",
            "task_type": "classification",
            "modality": "tabular",
            "training_regime": "generic",
            "training_epochs": 1,
            "batch_size": 2,
        }
    )

    assert result.status == "success"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM services WHERE service_id='svc_smoke'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM service_metrics WHERE service_id='svc_smoke'").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM service_split_provenance WHERE service_id='svc_smoke'").fetchone()[0] == 2
        metadata_json = conn.execute("SELECT metadata_json FROM services WHERE service_id='svc_smoke'").fetchone()[0]
        assert "hf_dataset_metadata_error" not in json.loads(metadata_json).get("loader_meta", {})
        old_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('rounds','clients','service_client_distributions')"
            )
        }
        assert old_tables == set()


class TransformersSentenceSimilarityModel:
    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=10):
        return 0.2, 0.25, 0.5, {"pearson": 0.25, "spearman": 0.5}

    def count_params(self):
        return 64


class TransformersBrokenMetricModel:
    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=10):
        return 0.2, np.nan, 0.5, {}

    def count_params(self):
        return 64


class TransformersSentenceSimilarityFallbackModel:
    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=10):
        return 0.2, np.nan, 0.5, {"pearson": np.nan, "spearman": 0.5}

    def count_params(self):
        return 64


class TransformersOomOnceModel:
    def __init__(self):
        self.batch_size = 4
        self.fit_calls = 0

    def fit(self, x, y, **kwargs):
        self.fit_calls += 1
        if self.fit_calls == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 44.00 MiB.")
        return {"completed_batches": 1}

    def evaluate(self, x, y, inference_only=False, max_eval_time_s=None, progress_log_interval=10):
        return 0.1, 0.75, 0.5, {}

    def count_params(self):
        return 64


@pytest.fixture(autouse=True)
def _stub_optional_runner_metrics(monkeypatch):
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(runner, "_service_perturbation_metrics", lambda *args, **kwargs: ({}, None))
    monkeypatch.setattr(runner, "_count_model_params", lambda model: 64)


def test_service_runner_prefers_loader_regression_semantics_for_sentence_similarity(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = [("a", "b"), ("c", "d")]
        y_train = np.asarray([0.1, 0.9], dtype="float32")
        x_test = [("e", "f"), ("g", "h")]
        y_test = np.asarray([0.2, 0.8], dtype="float32")
        meta = {
            "input_shape": (2,),
            "num_classes": 1,
            "task_type": "regression",
            "hf_task": "sentence_similarity",
            "input_schema": "text_pair",
            "dataset_family": "synthetic",
        }
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: TransformersSentenceSimilarityModel())

    result = runner.execute_service(
        {
            "service_id": "svc_sts",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic_pairs",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sentence_similarity",
            "modality": "text",
            "training_regime": "inference_only",
            "training_epochs": 0,
            "batch_size": 2,
        }
    )

    assert result.status == "success"
    with sqlite3.connect(db_path) as conn:
        task_family, task_type, functional_attributes_json = conn.execute(
            "SELECT task_family, task_type, functional_attributes_json FROM services WHERE service_id='svc_sts'"
        ).fetchone()
        assert task_family == "regression"
        assert task_type == "regression"
        functional = json.loads(functional_attributes_json)
        assert functional["primary_metric"] == "pearson"
        assert functional["secondary_metric"] == "spearman"

        metric_rows = dict(
            conn.execute(
                "SELECT metric_name, COALESCE(value_num, CAST(value_int AS REAL)) FROM service_metrics WHERE service_id='svc_sts'"
            ).fetchall()
        )
        assert metric_rows["pearson"] == pytest.approx(0.25)
        assert metric_rows["spearman"] == pytest.approx(0.5)
        assert metric_rows["metric_score"] == pytest.approx(0.625)


def test_service_runner_uses_spearman_score_when_sentence_similarity_pearson_is_nan(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = [("a", "b"), ("c", "d")]
        y_train = np.asarray([0.1, 0.9], dtype="float32")
        x_test = [("e", "f"), ("g", "h")]
        y_test = np.asarray([0.2, 0.8], dtype="float32")
        meta = {
            "input_shape": (2,),
            "num_classes": 1,
            "task_type": "regression",
            "hf_task": "sentence_similarity",
            "input_schema": "text_pair",
            "dataset_family": "synthetic",
        }
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: TransformersSentenceSimilarityFallbackModel())

    result = runner.execute_service(
        {
            "service_id": "svc_sts_nan_pearson",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic_pairs",
            "model_type": "hf",
            "task_type": "regression",
            "hf_task": "sentence_similarity",
            "modality": "text",
            "training_regime": "inference_only",
            "training_epochs": 0,
            "batch_size": 2,
        }
    )

    assert result.status == "success"
    with sqlite3.connect(db_path) as conn:
        metric_rows = dict(
            conn.execute(
                "SELECT metric_name, value_num FROM service_metrics WHERE service_id='svc_sts_nan_pearson'"
            ).fetchall()
        )
        pearson_json = conn.execute(
            "SELECT value_json FROM service_metrics WHERE service_id='svc_sts_nan_pearson' AND metric_name='pearson'"
        ).fetchone()[0]
        assert pearson_json == "null"
        assert metric_rows["spearman"] == pytest.approx(0.5)
        assert metric_rows["metric_score"] == pytest.approx(0.75)


def test_service_runner_fails_metric_validation_for_non_finite_primary_metric(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = np.asarray([[0.0], [1.0]])
        y_train = np.asarray([0, 1])
        x_test = np.asarray([[0.0], [1.0]])
        y_test = np.asarray([1, 0])
        meta = {"input_shape": (1,), "num_classes": 2, "task_type": "classification", "dataset_family": "synthetic"}
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: TransformersBrokenMetricModel())

    result = runner.execute_service(
        {
            "service_id": "svc_bad_metric",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic_bad_metric",
            "model_type": "hf",
            "task_type": "classification",
            "modality": "tabular",
            "training_regime": "inference_only",
            "training_epochs": 0,
            "batch_size": 2,
        }
    )

    assert result.status == "failed"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT status FROM services WHERE service_id='svc_bad_metric'").fetchone()[0] == "failed"
        failure_stage = conn.execute(
            "SELECT failure_stage FROM service_failures WHERE service_id='svc_bad_metric'"
        ).fetchone()[0]
        assert failure_stage == "metric_validation"


def test_service_runner_retries_transformers_finetune_once_after_cuda_oom(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"
    model = TransformersOomOnceModel()

    def fake_load_dataset(name, **kwargs):
        x_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.asarray([0, 1, 0, 1])
        x_test = np.asarray([[0.0], [1.0]])
        y_test = np.asarray([1, 0])
        meta = {"input_shape": (1,), "num_classes": 2, "task_type": "classification", "dataset_family": "synthetic"}
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: model)

    result = runner.execute_service(
        {
            "service_id": "svc_oom_retry",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "synthetic_oom",
            "model_type": "hf",
            "task_type": "classification",
            "hf_task": "sequence_classification",
            "modality": "text",
            "training_regime": "finetune_transfer",
            "training_epochs": 1,
            "batch_size": 4,
        }
    )

    assert result.status == "success"
    assert model.fit_calls == 2
    assert model.batch_size == 2


def test_service_runner_rejects_undersized_detection_finetune_run_before_model_build(monkeypatch, tmp_path):
    db_path = tmp_path / "services.db"

    def fake_load_dataset(name, **kwargs):
        x_train = np.asarray([[0.0], [1.0], [2.0]])
        y_train = np.asarray([0, 1, 0])
        x_test = np.asarray([[0.0], [1.0], [2.0]])
        y_test = np.asarray([1, 0, 1])
        meta = {"input_shape": (1,), "num_classes": 2, "task_type": "detection", "dataset_family": "synthetic"}
        return (x_train, y_train), (x_test, y_test), meta

    monkeypatch.setattr(runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        runner,
        "create_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model build should not run for invalid detection rows")),
    )

    result = runner.execute_service(
        {
            "service_id": "svc_tiny_det_runtime",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "tiny_detection",
            "model_type": "hf_finetune",
            "task_type": "detection",
            "hf_task": "image_detection",
            "modality": "image",
            "training_regime": "finetune_transfer",
            "training_epochs": 1,
            "batch_size": 2,
        }
    )

    assert result.status == "failed"
    with sqlite3.connect(db_path) as conn:
        failure_stage = conn.execute(
            "SELECT failure_stage FROM service_failures WHERE service_id='svc_tiny_det_runtime'"
        ).fetchone()[0]
        assert failure_stage == "service_validation"


def test_service_split_dirichlet_and_quantity_skew_change_label_distribution():
    x = np.arange(400).reshape(400, 1)
    y = np.repeat(np.arange(4), 100)
    meta = {"task_type": "classification", "hf_task": "sequence_classification", "num_classes": 4}

    iid_runner = runner.ServiceRunner(
        {
            "service_id": "svc_iid_split",
            "split_strategy": "iid",
            "sample_size": 120,
            "sample_seed": 123,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )
    dirichlet_runner = runner.ServiceRunner(
        {
            "service_id": "svc_dirichlet_split",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 120,
            "sample_seed": 123,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )
    quantity_runner = runner.ServiceRunner(
        {
            "service_id": "svc_quantity_split",
            "split_strategy": "quantity_skew",
            "distribution_param": 0.7,
            "sample_size": 120,
            "sample_seed": 123,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )

    iid_counts = iid_runner._resolve_service_split(x, y, meta)["distribution_map"]["train"]
    dirichlet_counts = dirichlet_runner._resolve_service_split(x, y, meta)["distribution_map"]["train"]
    quantity_counts = quantity_runner._resolve_service_split(x, y, meta)["distribution_map"]["train"]

    def _values(counts):
        return np.asarray([counts.get(str(i), counts.get(i, 0)) for i in range(4)])

    iid_values = _values(iid_counts)
    dirichlet_values = _values(dirichlet_counts)
    quantity_values = _values(quantity_counts)

    assert iid_values.max() - iid_values.min() < 25
    assert dirichlet_values.max() - dirichlet_values.min() >= 50
    assert quantity_values.max() - quantity_values.min() >= 35
    quantity_resolved = quantity_runner._resolve_service_split(x, y, meta)["resolved"]
    assert quantity_resolved["effective_strategy"] == "dirichlet"
    assert "compatibility alias" in str(quantity_resolved["fallback_reason"])


def test_service_split_sample_seed_changes_subset_for_same_strategy():
    x = np.arange(400).reshape(400, 1)
    y = np.repeat(np.arange(4), 100)
    meta = {"task_type": "classification", "hf_task": "sequence_classification", "num_classes": 4}

    first = runner.ServiceRunner(
        {
            "service_id": "svc_split_seed_a",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 120,
            "sample_seed": 111,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )._resolve_service_split(x, y, meta)
    second = runner.ServiceRunner(
        {
            "service_id": "svc_split_seed_b",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 120,
            "sample_seed": 222,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )._resolve_service_split(x, y, meta)
    assert first["distribution_map"]["train"] != second["distribution_map"]["train"]
    assert not np.array_equal(first["x_train"], second["x_train"])


def test_run_manifest_detects_cuda_worker_slots(monkeypatch):
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)

    assert run_manifest_cli._detect_cuda_device_ids() == [0, 1]


def test_run_manifest_auto_gpu_affinity_only_for_auto_like_devices():
    assert run_manifest_cli._entry_supports_auto_gpu_affinity({"device": "auto"}) is True
    assert run_manifest_cli._entry_supports_auto_gpu_affinity({"device": "cuda"}) is True
    assert run_manifest_cli._entry_supports_auto_gpu_affinity({"device": "cpu"}) is False
    assert run_manifest_cli._entry_supports_auto_gpu_affinity({"device": "cuda:1"}) is False


def test_run_manifest_builds_per_gpu_db_paths():
    assert run_manifest_cli._db_path_for_gpu("outputs/services.db", 0).endswith("outputs\\services.gpu0.db")
    assert run_manifest_cli._db_path_for_gpu("outputs/services.sqlite.db", 1).endswith("outputs\\services.gpu1.sqlite.db")


def test_run_manifest_rewrites_entry_db_path_for_gpu():
    entry = run_manifest_cli.ManifestEntry(
        idx=1,
        ordinal=1,
        resolved={"service_id": "svc", "db_path": "outputs/services.db"},
        validation=run_manifest_cli.RowValidation(True),
    )

    gpu_entry = run_manifest_cli._entry_with_gpu_db_path(entry, 1)

    assert gpu_entry.resolved["db_path"].endswith("outputs\\services.gpu1.db")
    assert entry.resolved["db_path"] == "outputs/services.db"


def test_grouped_hf_parallelizes_model_groups(monkeypatch):
    group_calls = []

    def fake_parallel(model_groups, gpu_slots):
        group_calls.append((len(model_groups), len(gpu_slots)))
        return [{"service_id": "svc_grouped", "row_index": 0, "manifest_group_id": "g", "case_name": "c", "status": "success", "error_message": "", "resolved_config_json": "{}"}]

    monkeypatch.setattr(run_manifest_cli, "_execute_hf_groups_with_gpu_affinity", fake_parallel)
    monkeypatch.setattr(run_manifest_cli, "_detect_cuda_device_ids", lambda: [0, 1])

    entries = [
        run_manifest_cli.ManifestEntry(
            idx=i,
            ordinal=i,
            resolved={
                "service_id": f"svc_{i}",
                "dataset": "hf",
                "model_type": "hf",
                "hf_model_id": f"model_{i % 2}",
                "hf_task": "sequence_classification",
                "device": "auto",
                "dataset_args": {"dataset_name": f"data_{i}"},
            },
            validation=run_manifest_cli.RowValidation(True),
        )
        for i in range(2)
    ]

    results = run_manifest_cli._execute_entries_grouped_hf(entries, workers=2)

    assert len(results) == 1
    assert group_calls == [(2, 2)]


def test_hf_group_keys_include_precision_type():
    first = {
        "hf_model_id": "model",
        "hf_task": "sequence_classification",
        "model_type": "hf",
        "device": "auto",
        "mixed_precision": True,
        "precision_type": "fp16",
        "dataset_args": {},
    }
    second = dict(first, precision_type="bf16")
    assert run_manifest_cli._hf_model_group_key(first) != run_manifest_cli._hf_model_group_key(second)


def test_service_split_derives_distinct_seed_when_manifest_has_no_sample_seed():
    x = np.arange(400).reshape(400, 1)
    y = np.repeat(np.arange(4), 100)
    meta = {"task_type": "classification", "hf_task": "sequence_classification", "num_classes": 4}

    first = runner.ServiceRunner(
        {
            "service_id": "svc_missing_seed_a",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 120,
            "seed": 42,
            "knob_variant": 0,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )._resolve_service_split(x, y, meta)
    second = runner.ServiceRunner(
        {
            "service_id": "svc_missing_seed_b",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 120,
            "seed": 42,
            "knob_variant": 1,
            "task_type": "classification",
            "hf_task": "sequence_classification",
        }
    )._resolve_service_split(x, y, meta)

    assert first["resolved"]["sample_seed"] != second["resolved"]["sample_seed"]
    assert first["distribution_map"]["train"] != second["distribution_map"]["train"]


def test_service_split_token_classification_uses_task_aware_axis_without_iid_fallback():
    x = {"attention_mask": np.asarray([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype="int64")}
    y = np.asarray(
        [
            [0, 1, -100],
            [0, 0, -100],
            [0, 2, 2],
            [0, 0, -100],
        ],
        dtype="int64",
    )
    meta = {"task_type": "classification", "hf_task": "token_classification", "ignore_index": -100}
    split = runner.ServiceRunner(
        {
            "service_id": "svc_token_axis",
            "split_strategy": "dirichlet",
            "distribution_param": 0.2,
            "sample_size": 3,
            "sample_seed": 7,
            "task_type": "classification",
            "hf_task": "token_classification",
        }
    )._resolve_service_split(x, y, meta)

    assert split["resolved"]["effective_strategy"] == "dirichlet"
    assert split["resolved"]["effective_axis"] == "entity_present_sentence"
    assert split["resolved"]["sample_strategy_effective"] == "dirichlet"
    assert split["resolved"]["bucket_distribution"]


def test_service_split_generation_vqa_and_retrieval_use_task_aware_axes():
    cases = [
        (
            "causal_lm_generation",
            {"attention_mask": np.asarray([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype="int64")},
            np.asarray([[10, 11, -100], [12, -100, -100], [13, 14, 15], [16, -100, -100]], dtype="int64"),
            {"task_type": "generation", "hf_task": "causal_lm_generation", "ignore_index": -100},
            "supervised_token_bucket",
        ),
        (
            "visual_question_answering",
            np.arange(4).reshape(4, 1),
            np.asarray(["yes", "no", "red", "yes"], dtype=object),
            {"task_type": "vqa", "hf_task": "visual_question_answering"},
            "answer_vocab",
        ),
        (
            "text_image_retrieval",
            {"caption_lengths": np.asarray([2, 7, 3, 9], dtype="int64"), "attention_mask": np.ones((4, 2), dtype="int64")},
            np.zeros((4,), dtype="int64"),
            {"task_type": "retrieval", "hf_task": "text_image_retrieval"},
            "query_length_bucket",
        ),
    ]

    for hf_task, x, y, meta, expected_axis in cases:
        split = runner.ServiceRunner(
            {
                "service_id": f"svc_{hf_task}",
                "split_strategy": "dirichlet",
                "distribution_param": 0.2,
                "sample_size": 3,
                "sample_seed": 9,
                "task_type": meta["task_type"],
                "hf_task": hf_task,
            }
        )._resolve_service_split(x, y, meta)
        assert split["resolved"]["sample_strategy_effective"] == "dirichlet"
        assert split["resolved"]["effective_axis"] == expected_axis


def test_service_split_rejects_unsupported_local_strategy():
    with pytest.raises(runner.ServiceExecutionError):
        runner.ServiceRunner(
            {
                "service_id": "svc_bad_local_strategy",
                "split_strategy": "shard",
            }
        )._resolve_service_split(np.arange(4).reshape(4, 1), np.asarray([0, 1, 0, 1]), {"task_type": "classification"})
