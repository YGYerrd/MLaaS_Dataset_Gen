"""New distribution rosters and compatibility with frozen partition artifacts."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mlaas_data_generator.clients import recipes, training
from mlaas_data_generator.clients.partitioning import client_partition
from mlaas_data_generator.clients.profiles import PROFILES, ROSTER_VERSION
from mlaas_data_generator.clients.quantities import stratified_sample_quantities
from mlaas_data_generator.tasks.contracts import bind_task_contract, resolve_task_contract, TaskContractError
from mlaas_data_generator.provenance import sha256_json


@pytest.mark.parametrize("task,axis,expected", [
    ("text_classification", "class_label", PROFILES),
    ("token_classification", "entity_present_sentence", PROFILES),
    ("fill_mask", None, PROFILES),
    ("image_segmentation", None, ("iid",)),
    ("text2text_generation", "supervised_token_bucket", ("iid",)),
])
def test_new_recipes_keep_quantities_independent_of_distribution(task, axis, expected):
    policy = training.build_training_policy({"config": {
        "model_type": "hf_finetune", "training_regime": "finetune_transfer",
        "task": task, "skew_axis": axis,
    }}, task=task)
    rows = recipes.sample_client_recipes(policy, count=12, seed="profiles", quantity_seed="quantities",
                                         average_samples=500)
    assert set(row["split_strategy"] for row in rows) == set(expected)
    assert set(Counter(row["split_strategy"] for row in rows).values()) == {12 // len(expected)}
    assert [row["sample_size"] for row in rows] == stratified_sample_quantities(500, count=12, seed="quantities")
    assert sum(row["sample_size"] for row in rows) == 6000
    assert all(row["skew_axis"] is None for row in rows if row["split_strategy"] == "iid")


@pytest.mark.parametrize("task,kind,tag", [
    ("fill_mask", "fill_mask", None),
    ("token_classification", "token_classification", None),
    ("image_segmentation", "segmentation", None),
    ("seq2seq_generation", "generation", "summarization"),
])
def test_frozen_task_profiles_replay_but_new_children_use_current_profiles(task, kind, tag):
    config = {"model_type": "hf_finetune", "hf_task": task, "task_type": kind,
              "num_labels": 3, "task_tag": tag, "training_regime": "finetune_transfer"}
    if task == "token_classification":
        config["skew_axis"] = "entity_present_sentence"
    current = resolve_task_contract(config).as_dict()
    frozen = deepcopy(current)
    frozen["sampling"]["profiles"].append("quantity_skew")
    original = deepcopy(frozen)
    cfg = {**config, "task_contract": frozen}
    assert bind_task_contract(cfg, {}, None).as_dict() == frozen == original
    assert resolve_task_contract(config).as_dict() == current
    policy = training.build_training_policy({"config": cfg}, task=task)
    assert policy["partition_profiles"] == current["sampling"]["profiles"]
    rows = recipes.sample_client_recipes(policy, count=6, seed="old-parent", average_samples=500)
    assert all(row["split_strategy"] in PROFILES for row in rows)
    changed = deepcopy(frozen)
    changed["sampling"]["profiles"].append("invented")
    with pytest.raises(TaskContractError, match="sampling.profiles"):
        bind_task_contract({**config, "task_contract": changed}, {}, None)


def test_saved_uniform_partition_keeps_source_membership_and_hash_metadata():
    config = {"sample_size": 7, "sample_seed": 42, "client_id": "saved", "split_strategy": "quantity_skew"}
    original = deepcopy(config)
    args = dict(source_ids=[f"train:{i}" for i in range(20)], meta={}, task_family="classification")
    saved = client_partition(None, None, config, **args)
    iid = client_partition(None, None, {**config, "split_strategy": "iid"}, **args)
    np.testing.assert_array_equal(saved["indices"], iid["indices"])
    assert saved["metadata"]["requested_strategy"] == saved["metadata"]["effective_strategy"] == "quantity_skew"
    assert saved["metadata"]["partition_hash"] != iid["metadata"]["partition_hash"]
    assert config == original


def fedavg_rows(*, legacy=False):
    from mlaas_data_generator.fedavg_manifest import expand_fedavg_run
    from mlaas_data_generator.fedavg_eligibility import fedavg_benchmark_sample_seed
    rows = expand_fedavg_run({"dataset": "digits", "model_type": "mlp", "task_type": "classification",
                             "training_regime": "generic", "sample_size": 500},
        run_index=0, clients_per_run=12, rounds=2, participation_rate=1.0, seed=42,
        service_id_fn=lambda prefix, data: prefix + sha256_json(data),
        service_config_fn=lambda row: {}, client_knobs_fn=lambda **kwargs: {})
    for row in rows:
        params = json.loads(row["requested_partition_params"])
        if legacy:
            profile = ("iid", "dirichlet", "shard", "quantity_skew")[row["client_partition_index"] % 4]
            row.update(split_strategy=profile, distribution_type=profile)
            params.update(profile=profile)
            params.pop("profile_roster_version")
        row["requested_partition_params"] = params
        # The coordinator still verifies the historical run-derived benchmark
        # seed. Isolate roster validation from the separate evaluation migration.
        row["benchmark_sample_seed"] = fedavg_benchmark_sample_seed(row["fl_run_id"])
    return rows


@pytest.mark.parametrize("legacy", [False, True])
def test_fedavg_accepts_current_and_frozen_balanced_rosters(legacy):
    from mlaas_data_generator.fedavg_run import validate_fedavg_run_entries
    rows = fedavg_rows(legacy=legacy)
    before = deepcopy(rows)
    assert validate_fedavg_run_entries([SimpleNamespace(resolved=row) for row in rows]) is None
    assert rows == before
    first = [row for row in rows if row["round_id"] == 1]
    if not legacy:
        assert Counter(row["split_strategy"] for row in first) == {profile: 4 for profile in PROFILES}
        assert {row["requested_partition_params"]["profile_roster_version"] for row in rows} == {ROSTER_VERSION}
    assert sorted(row["sample_size"] for row in first) == sorted(stratified_sample_quantities(500, count=12, seed="42:0"))
    # Change the same client in every round: each roster must reject this as
    # an unbalanced distribution, not silently fall back to another version.
    selected = next(row["client_id"] for row in first if row["split_strategy"] == "shard")
    for row in rows:
        if row["client_id"] == selected:
            row.update(split_strategy="iid", distribution_type="iid")
            row["requested_partition_params"]["profile"] = "iid"
    assert "balanced" in validate_fedavg_run_entries([SimpleNamespace(resolved=row) for row in rows])


def test_explicit_generic_variants_only_generate_current_profiles():
    from mlaas_data_generator.clients.legacy import _generic_split_knobs_for_variant
    variants = [_generic_split_knobs_for_variant({"task_type": "classification"}, index)
                for index in range(16)]
    assert all(strategy in PROFILES for strategy, *_ in variants)


def test_explicit_fill_mask_variants_use_raw_length_for_dirichlet():
    from mlaas_data_generator.clients.legacy import _training_knobs_for_variant

    for index in range(12):
        knobs = _training_knobs_for_variant(task_key="fill_mask",
            model={"model_id": "distilroberta-base"},
            training_regime="finetune_transfer", knob_variant=index)
        assert knobs["split_strategy"] == knobs["distribution_type"] == ("iid", "dirichlet")[index % 2]
        assert knobs["skew_axis"] == "raw_text_length_bucket"
        assert knobs["distribution_param"] == (None, .2)[index % 2]


@pytest.mark.parametrize("profiles", [["iid"], ["iid", "quantity_skew"]])
def test_frozen_iid_fill_mask_contracts_keep_their_capabilities(profiles):
    cfg = {"hf_task": "fill_mask", "task_type": "fill_mask",
           "model_type": "hf_finetune", "training_regime": "finetune_transfer"}
    frozen = resolve_task_contract(cfg).as_dict()
    frozen["sampling"]["profiles"] = profiles
    cfg["task_contract"] = deepcopy(frozen)
    assert bind_task_contract(cfg, {}, None).as_dict() == frozen
    assert training.build_training_policy({"config": cfg})["partition_profiles"] == ["iid"]
