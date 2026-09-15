"""Resume checks of stored checkpoint bytes, without decoding model weights."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .checkpoint_store import (
    ContentAddressedTensorStore, IntegrityError, SERVICE_PATCH_MANIFEST_FORMAT, _identity_manifest_path,
)
from .provenance import sha256_file
from .reduced_checkpoint import FORMAT as REDUCED_FORMAT


def _matches(path, digest, size=None, *, cache=None):
    if not path or not digest:
        return False
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file() or (size is not None and stat.st_size != int(size)):
        return False
    key = ("checkpoint_bytes", str(path), digest, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if cache is not None and cache.get(key):
        return True
    if sha256_file(path) != digest:
        return False
    # Do not cache a file that changed during hashing.
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        return False
    if cache is not None:
        cache[key] = True
    return True


def checkpoint_bytes_verified(conn, service_id, provenance, family, *, cache=None):
    """Return False to request full verification for missing or suspect evidence."""
    try:
        service = conn.execute("SELECT status FROM services WHERE service_id=?", (service_id,)).fetchone()
        raw = conn.execute("SELECT * FROM service_checkpoint_storage WHERE service_id=?", (service_id,)).fetchone()
        if not service or service[0] != "completed" or not raw or not provenance.get("composition_compatible"):
            return False
        storage = dict(raw)
        if storage.get("round_trip_status") != "passed" or not storage.get("round_trip_verified_at"):
            return False
        manifest = _identity_manifest_path(Path(family["store_root"]).resolve(), "services", str(service_id))
        # The DB anchors the manifest hash; the manifest anchors every payload.
        artifacts = conn.execute("SELECT * FROM service_artifacts WHERE service_id=?", (service_id,)).fetchall()
        anchor = None
        for artifact in artifacts:
            item = dict(artifact)
            if Path(item["artifact_uri"]).resolve() == manifest:
                metadata = json.loads(item.get("metadata_json") or "{}")
                digest = item.get("artifact_sha256") or metadata.get("sha256")
                size = item.get("size_bytes") if item.get("size_bytes") is not None else metadata.get("size_bytes")
                if size is not None and _matches(manifest, digest, size, cache=cache):
                    anchor = item
                    break
        if anchor is None:
            return False
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if (document.get("service_id") != str(service_id)
                or document.get("parameter_schema_hash") != storage.get("parameter_schema_hash")
                or document.get("base_state_sha256") != storage.get("base_state_sha256")):
            return False
        if not _matches(storage.get("reconstruction_artifact_ref"), storage.get("reconstruction_artifact_sha256"), cache=cache):
            return False
        if document.get("format") == REDUCED_FORMAT:
            if (document.get("source_state_sha256") != provenance.get("final_model_hash")
                    or document.get("reloaded_state_sha256") != storage.get("target_state_sha256")
                    or document.get("payload_sha256") != storage.get("full_artifact_sha256")
                    or Path(document["payload_path"]).resolve() != Path(storage["full_artifact_ref"]).resolve()):
                return False
            return _matches(document["payload_path"], document["payload_sha256"], document["payload_bytes"], cache=cache)
        if (document.get("format") != SERVICE_PATCH_MANIFEST_FORMAT
                or document.get("target_state_sha256") != provenance.get("final_model_hash")
                or document.get("target_state_sha256") != storage.get("target_state_sha256")):
            return False
        store = ContentAddressedTensorStore(family["store_root"])
        for link in (document["base_state"], document.get("patch")):
            if link is not None:
                ref = store.inspect_link(link)
                if not _matches(ref.payload_path, ref.payload_sha256, ref.stored_payload_size_bytes, cache=cache):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IntegrityError, sqlite3.DatabaseError):
        return False
