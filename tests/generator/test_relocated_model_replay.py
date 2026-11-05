from copy import deepcopy
from types import SimpleNamespace
import sqlite3

import pytest

from mlaas_data_generator import client_reconstruction as replay
from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.provenance import model_identity, sha256_json
from tests.generator.test_client_reconstruction import raw_family


ORIGINAL_SOURCE = "/home/generator/.cache/huggingface/hub/models--albert--albert-base-v2/snapshots/abc123"
LOCAL_SOURCE = r"D:\huggingface\hub\models--albert--albert-base-v2\snapshots\abc123"


@pytest.fixture
def relocated_family(raw_family):
    cfg, model, _ = raw_family
    model_config = {"_name_or_path": ORIGINAL_SOURCE, "hidden_size": 768,
                    "_private_semantic": "preserve", "num_labels": 2}
    # Return the live dictionary to ensure identity verification never edits it.
    model.config = SimpleNamespace(to_dict=lambda: model_config)
    cfg.update(hf_model_id="albert/albert-base-v2", hf_model_revision="abc123")
    _, ref, *_ = preparation.prepare(cfg, build_model=lambda *a: model)
    doc = preparation.read_preparation(ref["document_ref"], ref["document_hash"])
    cfg["_family_reference"] = ref
    model_config["_name_or_path"] = LOCAL_SOURCE
    return cfg, model, model_config, doc


def test_relocated_family_recomputes_exact_original_hash_without_mutation(relocated_family):
    cfg, model, model_config, doc = relocated_family
    before_config, before_doc = deepcopy(model_config), deepcopy(doc)
    with pytest.raises(ValueError, match="frozen family: architecture_id"):
        preparation.prepare(cfg, build_model=lambda *a: model)
    loaded, _, loaded_cfg, *_ = preparation.prepare(
        {**cfg, "_recorded_model_source": ORIGINAL_SOURCE}, build_model=lambda *a: model)
    identity = model_identity(model, loaded_cfg)
    assert identity["architecture_id"] == doc["compatibility"]["architecture_id"]
    assert identity["architecture_id"] == sha256_json({k: v for k, v in identity.items() if k != "architecture_id"})
    assert loaded == before_doc and doc == before_doc
    assert model_config == before_config
    assert "_recorded_model_source" not in loaded["config"]


@pytest.mark.parametrize("change", ["recorded_path", "hidden_size", "private_semantic", "missing_path"])
def test_relocated_family_still_rejects_architecture_drift(relocated_family, change):
    cfg, model, model_config, doc = relocated_family
    cfg["_recorded_model_source"] = ORIGINAL_SOURCE
    if change == "recorded_path":
        cfg["_recorded_model_source"] += "-wrong"
    elif change == "hidden_size":
        model_config["hidden_size"] = 1024
    elif change == "private_semantic":
        model_config["_private_semantic"] = "changed"
    else:
        model_config.pop("_name_or_path")
    with pytest.raises(ValueError, match="frozen family: architecture_id"):
        preparation.prepare(cfg, build_model=lambda *a: model)


@pytest.mark.parametrize("consumer", ["member_replay", "aggregate_evaluation"])
def test_recipe_source_reaches_member_and_aggregate_preparation(relocated_family, monkeypatch, consumer):
    cfg, model, _, doc = relocated_family
    family = {**cfg["_family_reference"], "family_id": doc["family_id"],
              "initial_state_hash": doc["initial_state_hash"]}
    recipe = {"family": family, "training": {"observations": {
        "hf_model_id": ORIGINAL_SOURCE, "train_stopping_reason": "epochs_completed", "optimizer_step_count": 1}},
        "selection": {"ordered_source_row_ids": ["train:0", "train:1"]},
        "fingerprints": {"source": sha256_json(doc["raw_source"])}}
    original_prepare = preparation.prepare

    class PreparedSuccessfully(Exception):
        pass

    def prepare(*args, **kwargs):
        # Exercise the real frozen-family check, then stop before training/inference.
        original_prepare(*args, **kwargs, build_model=lambda *a: model)
        raise PreparedSuccessfully

    monkeypatch.setattr(preparation, "prepare", prepare)
    with pytest.raises(PreparedSuccessfully):
        if consumer == "member_replay":
            with sqlite3.connect(cfg["db_path"]) as conn:
                conn.execute("CREATE TABLE services(service_id, status)")
                conn.execute("INSERT INTO services VALUES (?, 'completed')", (cfg["service_id"],))
            monkeypatch.setattr(replay, "read_recipe", lambda *a: recipe)
            with replay.materialize_service(cfg["db_path"], cfg["service_id"]):
                pytest.fail("should stop before training")
        else:
            from mlaas_composition.regeneration import execute_raw_family_state
            member = {"parent_document": doc, "recipe": recipe, "service_id": cfg["service_id"]}
            execute_raw_family_state(member, "aggregate", None, None)
