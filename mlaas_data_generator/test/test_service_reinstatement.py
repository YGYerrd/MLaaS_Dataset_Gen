import sqlite3
import sys
import types

import numpy as np

from mlaas_data_generator.data.distributions import get_data_distribution
from mlaas_data_generator.services import runner


class DummyModel:
    device = "cpu"

    def __init__(self):
        self._weights = [np.asarray([1.0, 2.0])]

    def fit(self, x, y, epochs=1, batch_size=32, verbose=0):
        self._weights = [self._weights[0] + np.asarray([0.25, 0.0])]
        return None

    def evaluate(self, x, y, verbose=0):
        return [0.2, 0.75]

    def predict(self, x, verbose=0):
        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        p1 = 1.0 / (1.0 + np.exp(-arr[:, 0]))
        return np.stack([1.0 - p1, p1], axis=1)

    def count_params(self):
        return 256

    def get_weights(self):
        return [np.array(w, copy=True) for w in self._weights]


def test_restored_distribution_logic_summarises_detection_labels():
    y = [
        {"boxes": [[0, 0, 10, 10], [10, 10, 20, 20]], "labels": [1, 2]},
        {"boxes": [[0, 0, 8, 8]], "labels": [2]},
    ]

    distribution = get_data_distribution(y, num_classes=None)

    assert distribution["samples"] == 2
    assert distribution["total_boxes"] == 3
    assert distribution["class_counts"] == {1: 1, 2: 2}


def _fake_dataset(name, **kwargs):
    x_train = np.asarray([[8.0], [7.0], [-7.0], [-8.0], [6.0], [-6.0], [5.0], [-5.0]], dtype="float32")
    y_train = np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype="int64")
    x_test = np.asarray([[8.0], [-8.0], [6.0], [-6.0]], dtype="float32")
    y_test = np.asarray([1, 0, 1, 0], dtype="int64")
    meta = {"input_shape": (1,), "num_classes": 2, "task_type": "classification", "dataset_name": "demo-ds"}
    return (x_train, y_train), (x_test, y_test), meta


def test_service_reinstated_summary_distributions_metrics_and_flatten(monkeypatch, tmp_path, capsys):
    db_path = tmp_path / "services.db"
    monkeypatch.setattr(runner, "load_dataset", _fake_dataset)
    monkeypatch.setattr(runner, "create_model", lambda **kwargs: DummyModel())
    monkeypatch.setattr(runner, "capture_hardware_snapshot", lambda: {"platform": "test"})
    monkeypatch.setattr(
        runner,
        "_fetch_hf_metadata",
        lambda config, meta: {
            "hf_model_id": "org/model",
            "hf_dataset_id": "org/dataset",
            "downloads": 123,
            "likes": 7,
            "model_size": 256,
            "params_count": 256,
            "pipeline_tag": "text-classification",
            "library_name": "transformers",
            "license": "apache-2.0",
            "tags": ["unit-test"],
            "last_modified": "2026-01-01T00:00:00",
        },
    )

    result = runner.execute_service(
        {
            "service_id": "svc_reinstated",
            "db_path": str(db_path),
            "dataset": "synthetic",
            "dataset_name": "org/dataset",
            "model_type": "mlp",
            "hf_model_id": "org/model",
            "task_type": "classification",
            "training_regime": "generic",
            "split_strategy": "dirichlet",
            "distribution_param": 0.5,
            "sample_size": 4,
            "training_epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "explainability_enabled": True,
            "perturbation_sample_count": 1,
            "perturbation_candidate_units": 1,
            "perturbation_target_units": 1,
            "perturbation_trust_trials": 1,
        }
    )

    assert result.status == "success"
    out = capsys.readouterr().out
    assert "========== SERVICE RUN SUMMARY ==========" in out
    assert "split_strategy: dirichlet" in out
    assert "effective model input samples:" in out
    assert "device: cpu" in out
    assert "[Perturbation] service stage starts" in out
    assert "[Perturbation] service stage ends" in out

    with sqlite3.connect(db_path) as conn:
        metric_names = {
            row[0]
            for row in conn.execute(
                "SELECT metric_name FROM service_metrics WHERE service_id = 'svc_reinstated'"
            )
        }
        assert {"latency", "tail_latency", "throughput", "runtime_s", "compute_time_s"} <= metric_names
        assert {"split_strategy", "split_provenance_json", "dataset_distribution_json"} <= metric_names
        assert {"split_strategy_requested", "split_strategy_effective", "split_skew_axis_effective", "split_bucket_spec_json"} <= metric_names
        assert {"downloads", "likes", "model_size", "params_count"} <= metric_names
        assert "explainability_score" in metric_names
        assert "perturbation_duration_s" in metric_names
        assert {
            "update_signature_id",
            "signature_dim",
            "signature_norm",
            "update_signature_path",
            "update_signature_method",
        } <= metric_names
        signature_path = conn.execute(
            """
            SELECT value_text
            FROM service_metrics
            WHERE service_id = 'svc_reinstated'
              AND metric_name = 'update_signature_path'
            """
        ).fetchone()[0]
        assert signature_path
        assert conn.execute(
            "SELECT COUNT(*) FROM service_split_provenance WHERE service_id = 'svc_reinstated'"
        ).fetchone()[0] == 2
        assert conn.execute(
            """
            SELECT COUNT(*)
            FROM service_artifacts
            WHERE service_id = 'svc_reinstated'
              AND artifact_type = 'update_signature'
            """
        ).fetchone()[0] == 1


def test_hf_metadata_enrichment_uses_mocked_hf_api(monkeypatch):
    class Info:
        def __init__(self, payload):
            self._payload = payload

        def to_dict(self):
            return dict(self._payload)

    class HfApi:
        def model_info(self, model_id):
            assert model_id == "org/model"
            return Info(
                {
                    "downloads": 10,
                    "likes": 2,
                    "pipeline_tag": "text-classification",
                    "library_name": "transformers",
                    "tags": ["a"],
                    "last_modified": "2026-01-01T00:00:00",
                    "cardData": {"license": "mit"},
                    "safetensors": {"total": 999},
                }
            )

        def dataset_info(self, dataset_id):
            assert dataset_id == "org/dataset"
            return Info({"downloads": 20, "likes": 4, "tags": ["d"], "last_modified": "2026-01-02T00:00:00"})

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))

    meta = runner._fetch_hf_metadata({"hf_model_id": "org/model", "dataset_name": "org/dataset"}, {})

    assert meta["hf_model_id"] == "org/model"
    assert meta["hf_dataset_id"] == "org/dataset"
    assert meta["downloads"] == 10
    assert meta["likes"] == 2
    assert meta["model_size"] == 999
    assert meta["params_count"] == 999
    assert meta["pipeline_tag"] == "text-classification"
    assert meta["library_name"] == "transformers"
    assert meta["license"] == "mit"
