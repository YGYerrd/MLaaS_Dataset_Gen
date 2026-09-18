import json
import sys

import pandas as pd
import pytest

from mlaas_data_generator.cli import main as cli, run_manifest as manifest
from mlaas_data_generator.config import CONFIG


@pytest.mark.parametrize("module,command", [(cli, ["run-manifest"]), (manifest, [])])
@pytest.mark.parametrize("disabled", [False, True])
def test_cli_dispatches_scoring_flag(monkeypatch, module, command, disabled):
    received = {}
    monkeypatch.setattr(module, "run_manifest", lambda **kwargs: received.update(kwargs))
    args = ["mlaas", *command, "--file", "clients.csv"]
    if disabled:
        args.append("--no-trust-explainability")
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    assert received["no_trust_explainability"] is disabled


@pytest.mark.parametrize("profile", [None, "benchmark", "diagnostic", "pilot", "production"])
def test_scoring_override_preserves_other_resolved_settings(profile):
    row = pd.Series({"service_id": "client", "enable_perturbation_metrics": True,
                     "explainability_enabled": True, "update_signature_enabled": True,
                     "_frozen_manifest_defaults": {**CONFIG, "explainability_enabled": True}})
    normal = manifest._resolve_execution_row(row, {}, profile)
    disabled = manifest._resolve_execution_row(row, {}, profile, no_trust_explainability=True)
    assert "_row_resolution_error" not in disabled
    assert normal["enable_perturbation_metrics"] and normal["explainability_enabled"]
    assert disabled == {**normal, "enable_perturbation_metrics": False, "explainability_enabled": False}


def test_failed_manifest_retains_disabled_scoring(monkeypatch, tmp_path):
    source = tmp_path / "clients.csv"
    pd.DataFrame([{"service_id": "client", "enable_perturbation_metrics": True,
                   "explainability_enabled": True}]).to_csv(source, index=False)
    monkeypatch.setattr(manifest, "_validate_row", lambda _: manifest.RowValidation(False, "test failure"))
    results = manifest.run_manifest(str(source), dry_run=True, no_trust_explainability=True,
                                    metric_evaluation_profile="benchmark")
    row = pd.read_csv(results).iloc[0]
    assert row.status == "failed"
    config = json.loads(row.resolved_config_json)
    assert not config["enable_perturbation_metrics"] and not config["explainability_enabled"]
    retry = pd.read_csv(manifest.FAILED_MANIFEST_PATH).iloc[0]
    resolved = manifest._resolve_execution_row(retry, {})
    assert not resolved["enable_perturbation_metrics"] and not resolved["explainability_enabled"]
    assert pd.read_csv(source).iloc[0].explainability_enabled
