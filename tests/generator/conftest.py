"""Keep manifest integration tests away from the user's run exports."""
import pytest


@pytest.fixture(autouse=True)
def isolated_manifest_exports(tmp_path, monkeypatch):
    from mlaas_data_generator import config
    from mlaas_data_generator.cli import run_manifest

    for name, filename in (
        ("MANIFEST_RESULTS_PATH", "service_manifest_results.csv"),
        ("FAILED_MANIFEST_PATH", "service_manifest_failed.csv"),
        ("FAILURE_LOG_PATH", "service_failures.log"),
    ):
        for module in (config, run_manifest):
            monkeypatch.setattr(module, name, tmp_path / filename)
