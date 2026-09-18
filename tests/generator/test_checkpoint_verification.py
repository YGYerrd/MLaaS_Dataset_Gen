import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from mlaas_data_generator import checkpoint_verification as checks
from mlaas_data_generator.checkpoint_store import _identity_manifest_path, ContentAddressedTensorStore
from mlaas_data_generator.cli.manifest import preparation


@pytest.fixture(params=["reduced", "patch", "base_only"])
def saved(tmp_path, request, monkeypatch):
    # These fixtures isolate checkpoint byte verification; recipe integrity is
    # covered with real recipe artifacts in test_client_reconstruction.py.
    monkeypatch.setattr("mlaas_data_generator.client_reconstruction.read_recipe",
                        lambda *a: {"retention": {"policy": "checkpoint", "client_checkpoint_retained": True}})
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE services(service_id TEXT, status TEXT);
        CREATE TABLE service_checkpoint_storage(service_id TEXT, round_trip_status TEXT,
            round_trip_verified_at TEXT, parameter_schema_hash TEXT, base_state_sha256 TEXT,
            target_state_sha256 TEXT, reconstruction_artifact_ref TEXT, reconstruction_artifact_sha256 TEXT,
            full_artifact_ref TEXT, full_artifact_sha256 TEXT);
        CREATE TABLE service_artifacts(service_id TEXT, artifact_uri TEXT, metadata_json TEXT);
        CREATE TABLE service_generation_provenance(service_id TEXT, family_document_ref TEXT,
            family_document_hash TEXT, final_model_hash TEXT, initial_state_hash TEXT,
            family_id TEXT, compatibility_contract_hash TEXT, composition_compatible INTEGER);
        INSERT INTO services VALUES ('s', 'completed');
        INSERT INTO service_generation_provenance VALUES ('s','family','hash','target','base','f','c',1);
    ''')
    reconstruction = tmp_path / "reconstruction.json"
    reconstruction.write_text('{}')
    manifest = _identity_manifest_path(tmp_path, "services", "s")
    manifest.parent.mkdir(parents=True)
    doc = dict(service_id="s", parameter_schema_hash="schema", base_state_sha256="base")
    if request.param == "reduced":
        payload = tmp_path / "weights.bin"
        payload.write_bytes(b"stored reduced weights")
        doc.update(format=checks.REDUCED_FORMAT, source_state_sha256="target", reloaded_state_sha256="target",
                   payload_path=str(payload), payload_sha256=checks.sha256_file(payload), payload_bytes=payload.stat().st_size)
    else:
        store = ContentAddressedTensorStore(tmp_path)
        base = store.put(b"base weights", object_kind="tensor_state", semantic_format="test")
        patch = store.put(b"patch weights", object_kind="tensor_patch", semantic_format="test")
        doc.update(format=checks.SERVICE_PATCH_MANIFEST_FORMAT, target_state_sha256="target",
                   base_state=base.link(tmp_path), patch=patch.link(tmp_path) if request.param == "patch" else None)
        payload = patch.payload_path if request.param == "patch" else base.payload_path
    manifest.write_text(json.dumps(doc))
    conn.execute('INSERT INTO service_artifacts VALUES (?,?,?)', ('s', str(manifest), json.dumps(
        dict(sha256=checks.sha256_file(manifest), size_bytes=manifest.stat().st_size))))
    conn.execute('INSERT INTO service_checkpoint_storage VALUES (?,?,?,?,?,?,?,?,?,?)',
        ('s','passed','2026-09-09','schema','base','target', str(reconstruction), checks.sha256_file(reconstruction),
         str(payload), checks.sha256_file(payload)))
    conn.commit()
    row = dict(conn.execute('SELECT * FROM service_generation_provenance').fetchone())
    family = dict(store_root=str(tmp_path), initial_state_hash='base', family_id='f',
                  compatibility_contract_hash='c', compatibility={})
    yield conn, row, family, payload, manifest
    conn.close()


def test_verified_bytes_skip_reconstruction(saved, monkeypatch):
    conn, row, family, _, _ = saved
    monkeypatch.setattr(preparation, 'read_preparation', lambda *a, **k: family)
    deep = Mock(side_effect=AssertionError('must not reconstruct unchanged checkpoint'))
    monkeypatch.setattr(preparation, 'reconstruct_service_state', deep)
    db = conn.execute('PRAGMA database_list').fetchone()[2]
    preparation.verify_completed(db, 's', cache={})
    deep.assert_not_called()


@pytest.mark.parametrize('damage', ['missing', 'size', 'same_size', 'manifest', 'no_evidence', 'running'])
def test_suspect_checkpoint_falls_back(saved, monkeypatch, damage):
    conn, row, family, payload, manifest = saved
    cache = {}
    assert checks.checkpoint_bytes_verified(conn, 's', row, family, cache=cache)
    if damage == 'missing':
        payload.unlink()
    elif damage == 'size':
        payload.write_bytes(b'x')
    elif damage == 'same_size':
        payload.write_bytes(b'x' * payload.stat().st_size)
    elif damage == 'manifest':
        manifest.write_text('{}')
    elif damage == 'no_evidence':
        conn.execute("UPDATE service_checkpoint_storage SET round_trip_status='not_run'")
    else:
        conn.execute("UPDATE services SET status='running'")
    conn.commit()
    assert not checks.checkpoint_bytes_verified(conn, 's', row, family, cache=cache)
    monkeypatch.setattr(preparation, 'read_preparation', lambda *a, **k: family)
    deep = Mock(return_value=object())
    monkeypatch.setattr(preparation, 'reconstruct_service_state', deep)
    monkeypatch.setattr(preparation, 'source_state_hash', lambda *a: 'target')
    preparation.verify_completed(conn.execute('PRAGMA database_list').fetchone()[2], 's', cache=cache)
    deep.assert_called_once()


def test_failed_full_verification_invalidates(saved, monkeypatch):
    conn, row, family, payload, _ = saved
    payload.unlink()
    monkeypatch.setattr(preparation, 'read_preparation', lambda *a, **k: family)
    monkeypatch.setattr(preparation, 'reconstruct_service_state', Mock(side_effect=ValueError('corrupt')))
    with pytest.raises(ValueError, match='corrupt'):
        preparation.verify_completed(conn.execute('PRAGMA database_list').fetchone()[2], 's')
    assert conn.execute('SELECT composition_compatible FROM service_generation_provenance').fetchone()[0] == 0


def test_unchanged_shared_payload_hashed_once(saved, monkeypatch):
    conn, row, family, payload, _ = saved
    digest = Mock(wraps=checks.sha256_file)
    monkeypatch.setattr(checks, 'sha256_file', digest)
    cache = {}
    for _ in range(2):
        assert checks.checkpoint_bytes_verified(conn, 's', row, family, cache=cache)
    assert sum(Path(call.args[0]) == payload for call in digest.call_args_list) == 1
