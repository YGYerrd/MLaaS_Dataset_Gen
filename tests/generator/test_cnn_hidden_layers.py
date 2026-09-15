import json
from collections import Counter

import pytest

from mlaas_data_generator.cli.main import build_parser
from mlaas_data_generator.cli.manifest import hf_manifest_builder as builder
from mlaas_data_generator.cli.manifest.preparation import group_key
from mlaas_data_generator.cli.run_manifest import load_manifest, _resolve_row
from mlaas_data_generator.services.runner import ServiceRunner


@pytest.mark.parametrize("manifest_format", ["review", "expanded"])
@pytest.mark.parametrize("legacy_variants", [False, True])
def test_cnn_draw_survives_manifest_and_separates_families(tmp_path, monkeypatch, manifest_format, legacy_variants):
    options = dict(task_keys=["keras_image_classification"], seed=42)
    if legacy_variants:
        options["knob_variants_per_pair"] = 3
    else:
        options["clients_per_family"] = 3
    families, services = [], []
    for units in (32, 256):
        monkeypatch.setattr(builder, "CNN_HIDDEN_LAYER_POOL", ((units,),))
        frame = builder.build_hf_manifest(**options)
        assert len(frame) == 3
        assert all(json.loads(value)["hidden_layers"] == [units] for value in frame.service_config)
        path = tmp_path / f"cnn_{units}.csv"
        builder.save_manifest(frame, path, manifest_format=manifest_format)
        # Reloading must use the recorded architecture, even if the pool changes.
        monkeypatch.setattr(builder, "CNN_HIDDEN_LAYER_POOL", ((64,),))
        loaded, defaults = load_manifest(path)
        resolved = [_resolve_row(row, defaults) for _, row in loaded.iterrows()]
        assert all(row["hidden_layers"] == [units] for row in resolved)
        keys = {group_key(row) for row in resolved}
        assert len(keys) == 1
        families.append(keys)
        services.append(set(frame.service_id))
    assert families[0].isdisjoint(families[1])
    assert services[0].isdisjoint(services[1])


def test_cli_automatically_draws_architecture_and_builds_actual_model(tmp_path):
    path = tmp_path / "cnn.csv"
    args = build_parser().parse_args([
        "hf-manifest", "--task-keys", "keras_image_classification",
        "--clients-per-family", "1", "--seed", "42", "--output", str(path),
    ])
    args._handler(args)
    frame, defaults = load_manifest(path)
    config = _resolve_row(frame.iloc[0], defaults)
    assert tuple(config["hidden_layers"]) in builder.CNN_HIDDEN_LAYER_POOL
    pytest.importorskip("tensorflow")
    model = ServiceRunner(config)._build_model(
        meta={"input_shape": (32, 32, 3), "num_classes": 10}, task_family="classification",
    )
    assert [layer.units for layer in model.layers if hasattr(layer, "units")] == config["hidden_layers"] + [10]
    assert model.count_params() == {32: 150826, 64: 282250, 128: 545098, 256: 1070794}[config["hidden_layers"][0]]


def test_architecture_is_not_a_cli_argument():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["hf-manifest", "--cnn-hidden-layers", "128"])


def test_draw_is_reproducible_and_independent_of_client_count_and_other_tasks():
    options = dict(task_keys=["keras_image_classification"], clients_per_family=1, seed=42)
    original = builder.build_hf_manifest(**options)
    replay = builder.build_hf_manifest(**options)
    assert original.service_id.tolist() == replay.service_id.tolist()
    assert original.hidden_layers.tolist() == replay.hidden_layers.tolist()
    widths = original.iloc[0].hidden_layers
    mixed = builder.build_hf_manifest(
        task_keys=["sklearn_image_classification", "keras_image_classification"],
        clients_per_family=8, seed=42, )
    assert all(value == widths for value in mixed.loc[mixed.model_type == "cnn", "hidden_layers"])
    assert mixed.loc[mixed.model_type != "cnn", "hidden_layers"].isna().all()
    assert all("hidden_layers" not in case for case in builder.GENERIC_MANIFEST_CASES)


def test_architecture_draw_covers_pool_uniformly_across_seeds():
    case = next(case for case in builder.GENERIC_MANIFEST_CASES if case["model_type"] == "cnn")
    counts = Counter(tuple(builder._sample_cnn_hidden_layers(case, seed=seed)) for seed in range(4096))
    assert set(counts) == {(32,), (64,), (128,), (256,)}
    # Fixed seeds make this deterministic; a broad tolerance catches biased or
    # constant selection without demanding perfectly balanced random draws.
    assert all(850 < count < 1200 for count in counts.values())
