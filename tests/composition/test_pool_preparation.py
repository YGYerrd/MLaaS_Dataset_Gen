import json
import sqlite3
from pathlib import Path

import pytest

from scripts.prepare_classification_pools import (
    ArtifactAudit, Files, choose_completed, merge_database, read_only, selected_occurrences,
)


def completed(source="a.db", **changes):
    return {"source_database": source, "service_id": "s1", "status": "completed", "family_id": "f1",
            "client_recipe_hash": "r", "family_document_hash": "f", "initial_state_hash": "i",
            "final_model_hash": "z", **changes}


def test_failure_does_not_override_completed_and_conflicting_success_is_rejected():
    failed = completed(status="failed", client_recipe_hash=None)
    good = completed("b.db")
    assert choose_completed([failed, good]) is good
    assert choose_completed([failed]) is None
    with pytest.raises(ValueError, match="conflicting"):
        choose_completed([good, completed("c.db", final_model_hash="different")])


def test_relocation_cannot_escape_the_extracted_root(tmp_path):
    audit = ArtifactAudit(tmp_path)
    assert audit.local("/home/joshua/MLaaS_Dataset_Gen/outputs/a.json") == tmp_path / "a.json"
    with pytest.raises(ValueError, match="outside"):
        audit.local("/home/joshua/MLaaS_Dataset_Gen/outputs/../secret")
    with pytest.raises(ValueError, match="unrecognised"):
        audit.local("/unrelated/file")


def test_dependency_inventory_rejects_bad_digest_and_changed_files(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"original")
    files = Files()
    files.check(path, "weights")
    with pytest.raises(ValueError, match="checksum"):
        files.check(path, "weights", "incorrect")
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="changed"):
        files.check(path, "weights")


def test_merge_preserves_sources_and_remaps_local_integer_ids(tmp_path):
    schema = (Path(__file__).resolve().parents[2] / "mlaas_data_generator/storage/schema.sql").read_text()
    connections = {}
    for index, name in enumerate(("validation.db", "validation.gpu0.db", "validation.gpu1.db")):
        path = tmp_path / name
        conn = sqlite3.connect(path)
        conn.executescript(schema)
        sid = "s1" if index < 2 else "s2"
        status = "failed" if index == 0 else "completed"
        conn.execute("INSERT INTO services (service_id,status) VALUES (?,?)", (sid,status))
        if index:
            conn.execute("INSERT INTO model_lineages (lineage_id,provenance_json) VALUES (?,?)", (sid,'{}'))
            conn.execute("INSERT INTO service_generation_provenance (service_id,provenance_version,family_id,lineage_id,provenance_json,client_recipe_hash,family_document_hash,initial_state_hash,final_model_hash) VALUES (?,1,'f1',?,'{}','r','f','i','z')", (sid,sid))
            conn.execute("INSERT INTO service_metrics (service_id,metric_name,domain,value_num) VALUES (?,'accuracy','quality',0.5)", (sid,))
        conn.commit(); conn.close()
        connections[name] = read_only(path)
    before = {name:(tmp_path/name).read_bytes() for name in connections}
    occurrences = selected_occurrences(connections, {"f1"})
    assert [r['status'] for r in occurrences['s1']] == ['failed', 'completed']
    selected = [choose_completed(r) for r in occurrences.values()]
    merged, counts = merge_database(tmp_path/'merged.db', connections, selected)
    assert merged.execute("SELECT COUNT(*) FROM services").fetchone()[0] == 2
    assert merged.execute("SELECT COUNT(DISTINCT metric_id) FROM service_metrics").fetchone()[0] == 2
    assert not merged.execute("PRAGMA foreign_key_check").fetchall()
    assert merged.execute("PRAGMA integrity_check").fetchone()[0] == 'ok'
    merged.close()
    for name, conn in connections.items():
        conn.close()
        assert (tmp_path/name).read_bytes() == before[name]
    with pytest.raises(FileExistsError):
        merge_database(tmp_path/'merged.db', {}, [])
