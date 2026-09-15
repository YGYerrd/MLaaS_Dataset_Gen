import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from mlaas_data_generator.clients import calibration, recipes, training
from mlaas_data_generator.clients.partitioning import client_partition


def policy(model="bert-base-uncased", dataset="imdb", subset=None):
    return training.build_training_policy({"config": {"hf_model_id": model,
        "dataset_name": dataset, "dataset_config": subset, "max_length": 128,
        "training_regime": "finetune_transfer", "task": "text_classification"}},
        task="text_classification")


def sample(p=None, count=100):
    return recipes.sample_client_recipes(p or policy(), count=count, seed="test-calibration",
        average_samples=1000)


def test_calibrated_sampling_preserves_variation_and_recorded_joint_recipes():
    rows = sample()
    assert rows == sample()
    assert Counter(r["calibration_band"] for r in rows) == {"supported": 70, "challenging": 20, "stress": 10}
    assert set(r["split_strategy"] for r in rows) == set(recipes.PROFILES)
    assert set(r["optimizer"] for r in rows) == {"adam", "adamw", "sgd", "rmsprop"}
    assert len({r["learning_rate"] for r in rows}) > 2
    assert len({r["sample_size"] for r in rows}) == 100
    assert sum(r["sample_size"] for r in rows) == 100_000
    for r in rows:
        training.validate_training_recipe(r)
        template = calibration.selected_template(r)
        if template:
            assert training.planned_optimizer_steps(r, r["sample_size"]) >= template["minimum_steps"]
            assert {k: r[k] for k in calibration.KNOBS if k != "training_epochs"} == {
                k: v for k, v in template["knobs"].items() if k != "training_epochs"}


def test_supported_knob_mutation_is_rejected_even_when_in_marginal_range():
    r = next(r for r in sample() if r["calibration_band"] == "supported")
    r["learning_rate"] = next(v for v in r["training_policy"]["limits"]["learning_rate_by_optimizer"][r["optimizer"]] if v != r["learning_rate"])
    with pytest.raises(ValueError, match="joint template"):
        training.validate_training_recipe(r)


def test_small_quantities_change_epochs_not_joint_knobs_or_quantity():
    r = next(r for r in sample() if r["calibration_band"] == "supported")
    before = {k: r[k] for k in calibration.KNOBS if k != "training_epochs"}
    r["sample_size"] = 100
    training.constrain_training_opportunity(r, seed="small")
    assert r["sample_size"] == 100
    assert before == {k: r[k] for k in before}
    training.validate_training_recipe(r)
    with pytest.raises(ValueError, match="insufficient training"):
        training.validate_training_opportunity(r, sample_size=1)


def partition(r):
    labels = np.array([0]*300 + [1]*700)
    return client_partition(np.arange(1000)[:, None], labels,
        {**r, "sample_size": 200, "requested_quota": 200, "max_client_quantity": 200, "hf_task": "sequence_classification",
         "skew_axis": "class_label", "heterogeneity_intensity": 1.0, "shard_support_quantile": 0.9},
        source_ids=[str(i) for i in range(1000)], meta={}, task_family="classification")


def test_supported_shards_keep_two_classes_while_stress_keeps_pure_shards():
    rows = sample()
    good = next(r for r in rows if r["calibration_band"] == "supported" and r["split_strategy"] == "shard")
    bad = next(r for r in rows if r["calibration_band"] == "stress")
    bad = {**bad, "split_strategy": "shard"}
    assert partition(good)["metadata"]["represented_bucket_count"] == 2
    assert partition(bad)["metadata"]["represented_bucket_count"] == 1
    assert len(partition(good)["indices"]) == 200


def test_supported_dirichlet_blends_training_prior_not_benchmark_labels():
    r = next(r for r in sample() if r["calibration_band"] == "supported")
    r = {**r, "split_strategy": "dirichlet"}
    d = partition(r)["metadata"]
    assert min(d["realised_bucket_counts"].values()) >= 30
    assert d["classification_coverage"]["source_mix"] == 0.5
    assert d["effective_strategy"] == "dirichlet"


def test_previous_policy_replays_without_consulting_new_evidence(monkeypatch):
    from copy import deepcopy
    from mlaas_data_generator.provenance import sha256_json
    before = deepcopy(sample())
    for row in before:
        p = deepcopy(row["training_policy"])
        p.pop("calibration")
        p["version"] = training.PREVIOUS_SAMPLER_VERSION
        p["tiers"] = {"medium": p.pop("limits")}
        row.update(training_policy=p, training_policy_hash=sha256_json(p),
                   recipe_sampler_version=p["version"], resource_tier="medium")
        row.pop("calibration_band", None)
        row.pop("calibration_template_id", None)
    monkeypatch.setattr(calibration, "load_evidence", lambda: pytest.fail("legacy replay read calibration"))
    for row in before:
        training.validate_training_recipe(row)
    r = {**before[0], "split_strategy": "shard"}
    assert partition(r)["metadata"]["represented_bucket_count"] == 1
    r["sample_size"] = 1
    with pytest.raises(ValueError, match="insufficient training"):
        training.validate_training_opportunity(r)


def test_fallback_scope_and_unrelated_task_and_parent_are_explicit():
    assert policy()["calibration"]["scope"] == "model_dataset_task"
    assert policy(dataset="unseen")["calibration"]["scope"] == "model_task"
    assert policy(model="unseen")["calibration"]["scope"] == "task"
    for parent in ({"config": {"task": "object_detection"}},
                   {"config": {"task": "text_classification"}, "parent_state_hash": "trained"}):
        p = training.build_training_policy(parent, task=parent["config"]["task"])
        assert p["calibration"]["status"] == "uncalibrated"
        assert all("calibration_band" not in row for row in sample(p, 10))


def test_evidence_miner_excludes_invalid_runs_and_deduplicates_imports(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE services (service_id TEXT,model_id TEXT,dataset_name TEXT,dataset_config TEXT,
          service_config_json TEXT,status TEXT,hf_task TEXT,task_family TEXT,model_type TEXT,training_regime TEXT);
        CREATE TABLE service_metrics (service_id TEXT,metric_name TEXT,value_num REAL,value_int INTEGER,
          value_bool INTEGER,value_text TEXT,value_json TEXT);
    """)
    knobs = dict(optimizer="adamw",learning_rate=2e-5,batch_size=8,gradient_accumulation_steps=1,
                 training_epochs=2,weight_decay=0.01,warmup_ratio=0.03,max_length=128)
    for i in range(7):
        cfg = {**knobs, "sample_seed": i if i != 3 else 0}
        task = "fill_mask" if i == 6 else "sequence_classification"
        status = "failed" if i == 5 else "completed"
        con.execute("INSERT INTO services VALUES (?,?,?,?,?,?,?,?,?,?)",
            (str(i),"model","dataset",None,json.dumps(cfg),status,task,"classification","hf_finetune","finetune_transfer"))
        for key, num in {"accuracy": 0.8,"f1":0.8,"loss":0.4,"optimizer_step_count":100}.items():
            if i == 4 and key == "loss":
                num = float("inf")
            con.execute("INSERT INTO service_metrics VALUES (?,?,?,NULL,NULL,NULL,NULL)",(str(i),key,num))
        for key in ("train_distribution_json","benchmark_distribution_json"):
            con.execute("INSERT INTO service_metrics VALUES (?,?,NULL,NULL,NULL,NULL,?)",(str(i),key,'{"0":50,"1":50}'))
    con.commit(); con.close()
    first = calibration.build_evidence(db)
    assert first == calibration.build_evidence(db)
    assert first["eligible_classification_records"] == 5
    assert first["unique_observations"] == 3
    assert first["exclusions"] == {"duplicate_observation":1,"nonfinite or missing loss":1}
    assert first["templates"][0]["outcomes"] == {"healthy":3}


def test_truncated_manifest_roundtrip_freezes_final_mixture_and_evidence(tmp_path):
    from mlaas_data_generator.cli.manifest.hf_manifest_builder import build_hf_manifest, save_manifest
    from mlaas_data_generator.cli.manifest.review import expand_review_manifest
    from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row
    df = build_hf_manifest(task_keys=["text_classification"],models_per_task=1,datasets_per_model=1,
        clients_per_family=50,total_services=20,seed=42,
        training_regimes=["finetune_transfer"],avg_sample_size=1000)
    assert Counter(df["calibration_band"]) == {"supported":14,"challenging":4,"stress":2}
    path = tmp_path / "clients.csv"
    save_manifest(df,path)
    expanded, defaults = expand_review_manifest(pd.read_csv(path), path)
    for _, row in expanded.iterrows():
        resolved = _resolve_row(row,defaults)
        training.validate_training_recipe(resolved)
        validation = _validate_row(resolved)
        assert validation.ok, validation.error
        assert resolved["training_policy"]["calibration"]["evidence_hash"] == calibration.load_evidence()["evidence_hash"]
    assert Counter(expanded["calibration_band"]) == Counter(df["calibration_band"])


def test_macro_f1_and_token_histograms_have_separate_contracts():
    distribution = {"0": 30, "1": 70}
    macro = (2 * 0.7 / 1.7) / 2
    assert calibration.outcome(0.7, macro, distribution, average="macro") == "collapsed_signature"
    assert calibration.outcome(0.7, macro, distribution) == "weak"
    tokens = {"top_supervised_token_ids": distribution, "unique_supervised_token_ids": 2, "supervised_tokens": 100}
    assert calibration.token_distribution(tokens) == distribution
    assert calibration.token_distribution({**tokens, "unique_supervised_token_ids": 3}) is None
    assert calibration.token_distribution({**tokens, "supervised_tokens": 101}) is None


def test_pair_classification_aliases_are_not_regression_evidence():
    record = {"hf_task": "sentence_similarity", "pearson": 0.7, "spearman": 0.6,
              "metadata_json": json.dumps({"loader_meta": {"task_type": "classification"}})}
    with pytest.raises(ValueError, match="regression contract"):
        calibration.record_outcome(record, {"0": 5, "1": 5}, {"0": 5, "1": 5})
    record["metadata_json"] = json.dumps({"loader_meta": {"task_type": "regression"}})
    assert calibration.record_outcome(record, {"bin_0": 5}, {"bin_0": 5})[0] == "healthy"
    record.update(pearson=0, spearman=0)
    assert calibration.record_outcome(record, {"bin_0": 5}, {"bin_0": 5})[0] == "weak"


@pytest.mark.parametrize("task,model,dataset,subset", [
    ("image_classification", "google/vit-base-patch16-224", "ylecun/mnist", "mnist"),
    ("token_classification", "bert-base-uncased", "unimelb-nlp/wikiann", "en"),
    ("sentence_similarity", "bert-base-uncased", "glue", "stsb"),
])
def test_additional_composition_tasks_only_borrow_same_task_evidence(task, model, dataset, subset):
    p = training.build_training_policy({"config": {"hf_model_id": model, "dataset_name": dataset,
        "dataset_config": subset, "max_length": 128, "task": task}}, task=task)
    assert p["calibration"]["status"] == "available"
    by_id = {t["id"]: t for t in calibration.load_evidence()["templates"]}
    for t in p["calibration"]["templates"]:
        assert {by_id[i]["task"] for i in t["evidence_ids"]} == {task}
    rows = sample(p, 50)
    assert Counter(r["calibration_band"] for r in rows) == {"supported": 35, "challenging": 10, "stress": 5}
    for row in rows:
        training.validate_training_recipe(row)


def test_recipe_diversity_does_not_count_new_seeds_as_new_recipes():
    rows = [dict(optimizer="adam", learning_rate=1e-4, batch_size=8, training_epochs=2,
                 sample_size=100 + i, training_seed=i, sample_seed=i, split_strategy="iid") for i in range(10)]
    summary = recipes.recipe_summary(rows)
    assert summary["unique_training_recipes"] == 1
    assert summary["largest_recipe_share"] == 1
    assert summary["unique_sample_quantities"] == 10
