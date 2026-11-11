"""Exercise continued-round initialization through the real recipe trainer."""

from pathlib import Path

import pytest

from mlaas_data_generator import client_reconstruction as replay
from mlaas_data_generator.checkpoint_store import TensorEntry, TensorState, capture_model_tensor_state
from mlaas_data_generator.services.runner import ServiceRunner
from tests.generator.test_weight_retention import trained_recipe


def test_recipe_training_starts_from_previous_global_state(trained_recipe, monkeypatch):
    db = trained_recipe["db_path"]
    before = Path(db).read_bytes()
    with replay.materialize_service(db, "recipe") as first:
        previous = capture_model_tensor_state(first["model"])
        original_hash = first["family"]["initial_state_hash"]
    assert previous.sha256 != original_hash
    starts = []
    train = ServiceRunner._train_model_effective
    def observe(self, model, *args, **kwargs):
        starts.append(capture_model_tensor_state(model).sha256)
        return train(self, model, *args, **kwargs)
    monkeypatch.setattr(ServiceRunner, "_train_model_effective", observe)
    with replay.materialize_service(db, "recipe", initial_state=previous) as second:
        assert second["initial"].sha256 == previous.sha256
        assert second["family"]["initial_state_hash"] == original_hash
        assert capture_model_tensor_state(second["model"]).sha256 != previous.sha256
    assert starts == [previous.sha256]
    assert Path(db).read_bytes() == before


def test_incompatible_round_tensor_schema_is_rejected_before_training(trained_recipe, monkeypatch):
    state = TensorState((TensorEntry("wrong", "float32", (1,), b"\x00" * 4),))
    monkeypatch.setattr(ServiceRunner, "_train_model_effective", lambda *a, **kw: pytest.fail("trained incompatible state"))
    with pytest.raises(ValueError, match="round initial state differs"):
        with replay.materialize_service(trained_recipe["db_path"], "recipe", initial_state=state):
            pytest.fail("accepted incompatible state")
