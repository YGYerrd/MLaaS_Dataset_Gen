"""Convert saved exact checkpoints to INT8 or FP16 without training or catalogue writes.

Run ``python -m mlaas_data_generator.migrate_checkpoints --help``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from . import checkpoint_store as cs
from . import reduced_checkpoint as reduced


def _store_paths(source_root, destination_root):
    source, destination = Path(source_root).resolve(), Path(destination_root).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination stores must be separate, non-nested directories")
    if not (source / "references" / "services").is_dir():
        raise ValueError(f"source has no service checkpoint references: {source}")
    return source, destination


def _object_stamp(ref):
    """Invalidate run-local verification if either immutable file changes."""
    return tuple((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                 for p in (ref.manifest_path, ref.payload_path))


def _copy_base(destination, link, base_ref, base_payload, verified_bases):
    key = (str(destination), cs._canonical_json_bytes(link))
    cached = verified_bases.get(key)
    if cached is not None:
        ref, stamp = cached
        if _object_stamp(ref) == stamp:
            return ref
    store = cs.ContentAddressedTensorStore(destination)
    if (destination / link["object_manifest"]).exists():
        # Same CAS identity and codec: verify the existing copy, without encoding it.
        ref, _ = store.read_link(link)
    else:
        # Reuse the source's compressed bytes. CAS put verifies the copied payload
        # against the already-verified canonical base; no recompression is needed.
        ref = store.put(base_payload, object_kind=base_ref.object_kind,
                        semantic_format=base_ref.semantic_format, compression=base_ref.compression,
                        _encoded_payload=base_ref.payload_path.read_bytes())
    verified_bases[key] = (ref, _object_stamp(ref))
    return ref


def _verify_file(path, size, digest):
    path = Path(path)
    if path.stat().st_size != size or cs._sha256_file(path) != digest:
        raise cs.IntegrityError(f"stored checkpoint file failed integrity verification: {path}")


def _verify_existing_files(store_root, saved, verified_bases):
    """Verify encoded bytes only; avoid source reconstruction and dequantization."""
    if saved.get("format") != reduced.FORMAT:
        raise cs.IntegrityError("unsupported reduced checkpoint format")
    codec = reduced.REPRESENTATIONS.get(saved.get("representation"))
    if codec is None or codec.version != saved.get("representation_version"):
        raise cs.IntegrityError("unsupported checkpoint representation version")
    _verify_file(saved["payload_path"], saved["payload_bytes"], saved["payload_sha256"])
    ref = cs.ContentAddressedTensorStore(store_root).inspect_link(saved["base_state"])
    if ref.content_sha256 != saved["base_state_sha256"]:
        raise cs.IntegrityError("saved checkpoint base-state hash mismatch")
    key = (str(store_root), cs._canonical_json_bytes(saved["base_state"]))
    stamp = _object_stamp(ref)
    if verified_bases.get(key) != stamp:
        _verify_file(ref.payload_path, ref.stored_payload_size_bytes, ref.payload_sha256)
        verified_bases[key] = stamp


def _result_report(report, saved, status, verification):
    return {**report, "status": status,
            "reloaded_state_sha256": saved["reloaded_state_sha256"],
            "payload_path": saved["payload_path"], "payload_bytes": saved["payload_bytes"],
            "weight_error": saved["weight_error"], "quality_validation": saved["quality_validation"],
            "existing_verification": verification,
            "round_trip_verified": verification != "hash"}


def migrate_checkpoint(source_root, destination_root, service_id, *,
                       compression=cs.CompressionSpec("none"), dry_run=False, representation="fp16",
                       verify_existing="hash", _verified_bases=None, _verified_existing_bases=None):
    """Migrate one exact service; retain identity and copy its immutable base.

    Publication and decoding use the production reduced-checkpoint implementation.
    Existing matching outputs are verified and reused; conflicting outputs fail.
    This migrates tensor storage only, without changing catalogue/provenance rows.
    """
    source, destination = _store_paths(source_root, destination_root)
    if representation not in reduced.REPRESENTATIONS:
        raise ValueError(f"unsupported checkpoint representation: {representation}")
    if verify_existing not in {"hash", "full"}:
        raise ValueError("verify_existing must be hash or full")
    existing_bases = {} if _verified_existing_bases is None else _verified_existing_bases
    service_id = str(service_id)
    source_manifest = cs._identity_manifest_path(source, "services", service_id)
    document = cs._read_json(source_manifest)
    if document.get("service_id") != service_id:
        raise cs.IntegrityError("source checkpoint service identity mismatch")
    report = {"service_id": service_id, "source_manifest": str(source_manifest),
              "source_manifest_sha256": cs._sha256_file(source_manifest), "representation": representation}
    if document.get("format") == reduced.FORMAT:
        if document.get("representation") != representation:
            raise ValueError("convert from the original exact checkpoint store to avoid compounding quantization error")
        if verify_existing == "full":
            reduced.load_reduced_state(document)
            cs.reconstruct_service_base_state(source, service_id)
        else:
            _verify_existing_files(source, document, existing_bases)
        return _result_report(report, document, "already_reduced_source", verify_existing)
    if document.get("format") != cs.SERVICE_PATCH_MANIFEST_FORMAT:
        raise ValueError("migration requires a saved exact service checkpoint manifest")

    manifest_path = cs._identity_manifest_path(destination, "services", service_id)
    report.update(source_state_sha256=document["target_state_sha256"], destination_manifest=str(manifest_path))
    saved = None
    if manifest_path.exists():
        saved = cs._read_json(manifest_path)
        expected = {"format": reduced.FORMAT, "service_id": service_id,
                    "lineage_id": document["lineage_id"], "source_state_sha256": document["target_state_sha256"],
                    "base_state_sha256": document["base_state_sha256"],
                    "parameter_schema_hash": document.get("parameter_schema_hash"),
                    "representation": representation,
                    "representation_version": reduced.REPRESENTATIONS[representation].version,
                    "compression": compression.as_dict()}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise cs.ManifestConflictError(f"destination checkpoint conflicts with migration: {manifest_path}")
        if verify_existing == "hash":
            _verify_existing_files(destination, saved, existing_bases)
            return _result_report(report, saved, "already_migrated", "hash")

    # Verify the original base and exact final weights before any destination write.
    source_store = cs.ContentAddressedTensorStore(source)
    base_ref, base_payload = source_store.read_link(document["base_state"])
    base = cs.deserialize_tensor_state(base_payload)
    if base.sha256 != document.get("base_state_sha256"):
        raise cs.IntegrityError("source checkpoint base-state hash mismatch")
    if document.get("patch") is None:
        state = base
    else:
        _, patch_payload = source_store.read_link(document["patch"])
        state = cs.apply_tensor_patch(base, cs.deserialize_tensor_patch(patch_payload))
        del patch_payload
    if state.sha256 != document.get("target_state_sha256"):
        raise cs.IntegrityError("source checkpoint target-state hash mismatch")
    del base
    if saved is not None:
        reduced.load_reduced_state(saved)
        cs.reconstruct_service_base_state(destination, service_id)
        status = "already_migrated"
    elif dry_run:
        return {**report, "status": "would_migrate"}
    else:
        # Cache only small verified references, never model-sized tensor arrays.
        copied_base = _copy_base(destination, document["base_state"], base_ref, base_payload,
                                 {} if _verified_bases is None else _verified_bases)
        del base_payload
        result = reduced.persist_reduced_checkpoint(
            destination, service_id, document["lineage_id"], state,
            base_ref=copied_base, parameter_schema_hash=document.get("parameter_schema_hash"),
            representation=representation, compression=compression, measure_full=False)
        saved = result.document
        del state
        # Production persistence already decodes and hash-verifies the on-disk
        # representation before publishing its immutable manifest.
        status = "migrated"
    return _result_report(report, saved, status, "full")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True, help="Existing exact checkpoint store")
    parser.add_argument("--output-store", type=Path, required=True, help="Separate destination checkpoint store")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--service-id", action="append", help="Service to convert; repeat for a batch")
    selection.add_argument("--all", action="store_true", help="Visit all source service references")
    parser.add_argument("--weight-format", choices=tuple(reduced.REPRESENTATIONS), default="int8",
                        help="Stored representation (default: int8)")
    parser.add_argument("--compression", choices=("none", "zlib", "zstd"), default="none")
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--verify-existing", choices=("hash", "full"), default="hash",
                        help="Resume checks: stored SHA-256 hashes (default) or full reconstruction/decode")
    parser.add_argument("--dry-run", action="store_true", help="Verify sources and existing outputs without writing")
    args = parser.parse_args(argv)
    try:
        source, destination = _store_paths(args.store_root, args.output_store)
        compression = cs.CompressionSpec(args.compression, args.compression_level)
    except ValueError as exc:
        parser.error(str(exc))

    # Snapshot the selected references so a concurrent generator cannot extend this run.
    paths = (sorted((source / "references" / "services").glob("*/*.json")) if args.all else
             [cs._identity_manifest_path(source, "services", sid) for sid in dict.fromkeys(args.service_id)])
    if not paths:
        parser.error("source has no service checkpoint manifests")
    failed = False
    verified_bases = {}
    verified_existing_bases = {}
    pending_paths = {path for path in paths
                     if not (destination / "references/services" / path.parent.name / path.name).exists()}
    conversion_s = 0.0
    conversions = 0
    for index, path in enumerate(paths, 1):
        sid = None
        started = perf_counter()
        try:
            sid = cs._read_json(path)["service_id"]
            if path != cs._identity_manifest_path(source, "services", str(sid)):
                raise cs.IntegrityError("source service identity does not match its manifest path")
            existing = cs._identity_manifest_path(destination, "services", str(sid)).exists()
            action = "Checking saved checkpoint" if existing else "Checking" if args.dry_run else "Migrating"
            print(f"[{index}/{len(paths)}] {action} {sid}", file=sys.stderr, flush=True)
            report = migrate_checkpoint(source, destination, sid, compression=compression,
                                        dry_run=args.dry_run, representation=args.weight_format,
                                        verify_existing=args.verify_existing, _verified_bases=verified_bases,
                                        _verified_existing_bases=verified_existing_bases)
        except Exception as exc:
            failed = True
            report = {"service_id": sid, "source_manifest": str(path), "status": "failed",
                      "error_type": type(exc).__name__, "error": str(exc)}
        report.update(elapsed_s=perf_counter() - started, processed=index, total=len(paths))
        print(json.dumps(report), flush=True)
        pending_paths.discard(path)
        if report["status"] == "migrated":
            conversion_s += report["elapsed_s"]
            conversions += 1
        remaining = f"{len(pending_paths)} new conversions pending"
        if conversions and pending_paths:
            remaining += f" (~{conversion_s / conversions * len(pending_paths) / 60:.1f} min, excluding resume checks)"
        print(f"  {report['status']} in {report['elapsed_s']:.1f}s; "
              f"{remaining}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
