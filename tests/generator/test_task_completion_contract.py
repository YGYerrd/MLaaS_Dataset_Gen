import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator import shared_parent
from mlaas_data_generator.services import runner
from mlaas_data_generator.services.taxonomy import canonical_task_family
from mlaas_data_generator.provenance import label_semantics, sha256_file, sha256_json
from mlaas_data_generator.cli.manifest import preparation


@pytest.mark.parametrize("alias", ["text_classification", "text-classification", "sequence_classification", "classification"])
def test_preparation_and_service_resolve_same_classification_contract(alias):
    cfg = {"task_type": "classification", "hf_task": "sequence_classification", "training_regime": "finetune_transfer"}
    train = (np.ones((4, 2)), np.array([0, 1, 0, 1]))
    meta = {"task_type": alias, "num_labels": 2, "label_format": "single_index", "label_mapping": {"no": 0, "yes": 1}}
    model = SimpleNamespace(core=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_labels=2, problem_type=None))))
    calls=[]
    def build(config, loaded, task):
        calls.append((config["task_type"], loaded["task_type"], task))
        return model
    prepared = shared_parent._prepare_loaded(cfg, train, train, meta, build_model=build)
    assert calls == [("classification", "classification", "classification")]
    assert model.core.model.config.problem_type == "single_label_classification"
    loaded, task = prepared[2:4]
    actual = canonical_task_family(runner._resolved_task_type(cfg, loaded), cfg["hf_task"])
    assert label_semantics(loaded, task_family=task) == label_semantics(loaded, task_family=actual)


def test_unknown_prepared_task_is_rejected_before_model_build():
    with pytest.raises(ValueError, match="cannot freeze an unknown"):
        shared_parent._prepare_loaded({"task_type": "unresolved"}, ([], []), ([], []), {},
                                     build_model=lambda *args: pytest.fail("model constructed before task resolution"))


@pytest.mark.parametrize("task,primary,secondary", [("classification", "accuracy", "f1"), ("regression", "pearson", "spearman")])
def test_pair_metrics_pass_final_service_validation_and_require_correct_secondary(task, primary, secondary):
    args = dict(task_family=task, hf_task="sentence_similarity", primary_name=primary,
                primary_value=0.7, secondary_name=secondary, secondary_value=0.6, metric_score=0.7)
    runner._validate_service_metrics(**args)
    with pytest.raises(runner.ServiceExecutionError, match=secondary):
        runner._validate_service_metrics(**{**args, "secondary_value": None})


def test_invalid_historical_preparation_has_actionable_error(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"contracts": {"labels": {"task_family": "unknown"}}}))
    with pytest.raises(ValueError, match="new service ID and manifest"):
        preparation.read_preparation(path, sha256_file(path))


def test_new_text_cache_namespace_does_not_reuse_unknown_task_preparation():
    text = {"task_type": "classification", "hf_task": "sequence_classification"}
    image = {"task_type": "classification", "hf_task": "image_classification"}
    assert preparation.group_key(text) != sha256_json(preparation.contract_config(text))
    assert preparation.group_key(image) == sha256_json(preparation.contract_config(image))


def test_precision_summary_works_with_windows_redirected_output(monkeypatch):
    import io
    import sys
    from mlaas_data_generator.terminal import print_service_summary
    sink = io.BytesIO()
    stream = io.TextIOWrapper(sink, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    print_service_summary({"service_id": "mobilevit"}, {
        "precision_type_requested": "fp32",
        "precision_fallback_reason": "mobile_models_fp32_stability_policy_v2",
    })
    stream.flush()
    assert b"FP32 fallback -> mobile_models_fp32_stability_policy_v2" in sink.getvalue()
