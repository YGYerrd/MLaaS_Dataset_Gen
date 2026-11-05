import json
from pathlib import Path

import pytest

from mlaas_data_generator.artifact_paths import (
    artifact_path_scope, current_artifact_prefixes, load_artifact_path_map, resolve_artifact_path)


def write_map(path, prefixes):
    path.write_text(json.dumps({"version": "composition_path_map_v1", "prefixes": prefixes}))
    return path


def test_cross_platform_prefixes_relative_targets_and_scope(tmp_path):
    path = write_map(tmp_path / "path_map.json", [
        {"recorded": "/home/generator/outputs/", "resolved": "local"},
        {"recorded": "/home/generator/outputs/special/", "resolved": "special"},
        {"recorded": "Z:\\original\\", "resolved": "windows"},
    ])
    prefixes = load_artifact_path_map(tmp_path / "services.db")
    with artifact_path_scope(prefixes):
        assert resolve_artifact_path("/home/generator/outputs") == tmp_path / "local"
        assert resolve_artifact_path(r"\home\generator\outputs\file.json") == tmp_path / "local/file.json"
        assert resolve_artifact_path("/home/generator/outputs/special/file") == tmp_path / "special/file"
        assert resolve_artifact_path("Z:/original/file") == tmp_path / "windows/file"
        assert resolve_artifact_path("/home/generator/outputs-other/file") == Path("/home/generator/outputs-other/file")
        with pytest.raises(ValueError, match="escapes"):
            resolve_artifact_path("/home/generator/outputs/../outside")
        with artifact_path_scope(()):
            assert current_artifact_prefixes() == ()
        assert current_artifact_prefixes() == prefixes
    assert current_artifact_prefixes() == ()
    assert load_artifact_path_map(tmp_path / "elsewhere/services.db", path) == prefixes


def test_missing_optional_map_and_missing_explicit_map(tmp_path):
    assert load_artifact_path_map(tmp_path / "services.db") == ()
    with pytest.raises(FileNotFoundError):
        load_artifact_path_map(tmp_path / "services.db", tmp_path / "missing.json")


@pytest.mark.parametrize("payload", [
    {}, {"version": "future", "prefixes": []},
    {"version": "composition_path_map_v1", "prefixes": "invalid"},
    {"version": "composition_path_map_v1", "prefixes": [{"recorded": "/old"}]},
])
def test_invalid_maps_fail_explicitly(tmp_path, payload):
    (tmp_path / "path_map.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_artifact_path_map(tmp_path / "services.db")
