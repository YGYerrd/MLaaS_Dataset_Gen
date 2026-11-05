"""Checksum and provenance checks for the prepared service catalogue."""
from __future__ import annotations

import hashlib
import json
import math

from mlaas_data_generator.tasks.contracts import provenance_contract_values

SOURCE_PREFIX = "/home/joshua/MLaaS_Dataset_Gen/outputs/"
RULE_VERSION = "lambda_completed_record_audit_v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ArtifactAudit:
    def __init__(self, root):
        self.root = root.resolve()
        self.files = {}
        self.families = {}
        self.support = {}
        self.base_checks = set()

    def local(self, reference):
        require(isinstance(reference, str) and reference.startswith(SOURCE_PREFIX), "unrecognised_source_path")
        path = (self.root / reference[len(SOURCE_PREFIX):]).resolve()
        require(path.is_relative_to(self.root), "artifact_path_outside_selected_root")
        return path

    def file(self, path, expected=None, size=None):
        path = path.resolve()
        require(path.is_relative_to(self.root), "artifact_path_outside_selected_root")
        before = path.stat()
        key = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if key not in self.files:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "artifact_changed_during_read")
            self.files[key] = h.hexdigest()
        value = self.files[key]
        require(expected is None or value == expected, "artifact_sha256_mismatch")
        require(size is None or before.st_size == size, "artifact_size_mismatch")
        return value

    def document(self, reference, expected):
        path = self.local(reference)
        self.file(path, expected)
        return json.loads(path.read_text(encoding="utf-8"))

    def support_artifact(self, artifact):
        key = (artifact["artifact_uri"], artifact["metadata"]["sha256"])
        if key not in self.support:
            path = self.local(key[0])
            require(path.exists(), "support_artifact_missing")
            if path.is_dir():
                records = []
                for child in sorted(p for p in path.rglob("*") if p.is_file()):
                    records.append({"path": child.relative_to(path).as_posix(),
                        "sha256": self.file(child), "size_bytes": child.stat().st_size})
                require(canonical_hash(records) == key[1], "support_directory_hash_mismatch")
            else:
                self.file(path, key[1])
            self.support[key] = True

    def family(self, provenance):
        key = (provenance["family_document_ref"], provenance["family_document_hash"])
        if key not in self.families:
            d = self.document(*key)
            require(d["version"] in ("raw_source_family_v3", "raw_source_family_v4"), "unsupported_family_contract")
            c = d["family_contract"]
            require(d["family_id"] == "family_" + canonical_hash(c), "family_identity_mismatch")
            require(d["compatibility_contract_hash"] == canonical_hash(c), "family_compatibility_hash_mismatch")
            require(all(d[k] == c[k] for k in ("config", "initial_state_hash", "compatibility", "raw_source", "shared_fitted_state", "evaluation_data_hash")), "family_contract_content_mismatch")
            require(d.get("contracts", {}).get("labels", {}).get("task_family") != "unknown", "unknown_family_task")
            reservation = self.document(d["reservation_ref"], d["reservation_file_hash"])
            require(reservation == d["reservation"] and reservation["reservation_hash"] == c["reservation_hash"], "reservation_identity_mismatch")
            groups = [set(reservation[k]) for k in ("training_ids", "catalogue_metric_ids", "audit_ids")]
            require(not any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)), "family_split_overlap")
            for artifact in d["reconstruction_support"]:
                self.support_artifact(artifact)
            support_hashes = sorted([{"type": a["artifact_type"], "sha256": a["metadata"]["sha256"]} for a in d["reconstruction_support"]], key=lambda a: (a["type"], a["sha256"]))
            require(support_hashes == c["reconstruction_support_hashes"], "family_support_identity_mismatch")
            self.families[key] = d
        d = self.families[key]
        require(all(provenance[k] == d[k] for k in ("family_id", "initial_state_hash", "compatibility_contract_hash")), "service_family_identity_mismatch")
        require(all(provenance.get(k) == v for k, v in d["compatibility"].items()), "service_family_interface_mismatch")
        return d

    def base(self, storage, family):
        require(storage is not None, "missing_base_storage_record")
        key = (storage["base_artifact_ref"], storage["base_artifact_sha256"])
        if key in self.base_checks:
            return
        d = self.document(*key)
        require(d["lineage_id"] == family["lineage_id"], "base_lineage_mismatch")
        require(d["tensor_state_sha256"] == family["initial_state_hash"] == storage["base_state_sha256"], "base_state_identity_mismatch")
        require(d["parameter_schema_hash"] == family["compatibility"]["parameter_schema_hash"], "base_parameter_schema_mismatch")
        reference = self.local(family["store_root"] + "/" + d["base_state"]["object_manifest"])
        obj = json.loads(reference.read_text(encoding="utf-8"))
        require(obj["content_sha256"] == family["initial_state_hash"], "base_object_identity_mismatch")
        self.file(reference)
        self.file(reference.parent / obj["stored_payload"], obj["stored_payload_sha256"], obj["stored_payload_size_bytes"])
        self.base_checks.add(key)


def validate(audit, service, provenance, metrics, storage):
    require(provenance is not None, "missing_generation_provenance")
    provenance = provenance_contract_values(provenance)
    require(provenance["generation_mode"] == "independent", "unsupported_generation_mode")
    family = audit.family(provenance)
    audit.base(storage, family)
    recipe = audit.document(provenance["client_recipe_ref"], provenance["client_recipe_hash"])
    require(recipe["version"] == "client_reconstruction_recipe_v1", "unsupported_recipe_version")
    require(recipe["recipe_hash"] == canonical_hash({k:v for k,v in recipe.items() if k != "recipe_hash"}), "recipe_self_hash_mismatch")
    f = recipe["family"]
    require(recipe["service_id"] == service["service_id"] and f["family_id"] == provenance["family_id"]
        and f["document_hash"] == provenance["family_document_hash"] and f["initial_state_hash"] == provenance["initial_state_hash"]
        and recipe["fingerprints"]["original_model"] == provenance["final_model_hash"], "recipe_provenance_mismatch")
    ids = recipe["selection"]["ordered_source_row_ids"]
    require(len(ids) == recipe["selection"]["realised_count"] > 0 and len(set(ids)) == len(ids)
        and canonical_hash(ids) == recipe["fingerprints"]["ordered_source_rows"], "invalid_recipe_training_rows")
    require(not set(ids) & (set(family["reservation"]["catalogue_metric_ids"]) | set(family["reservation"]["audit_ids"])), "client_evaluation_overlap")
    require(recipe["fingerprints"]["source"] == canonical_hash(family["raw_source"]), "recipe_raw_source_mismatch")
    # The recipe fingerprints the client evaluation view; the family hash
    # covers its prepared evaluation pool. Their equality is not a contract.
    # Recomputing either from raw source data requires a separate replay audit.
    observations = recipe["training"]["observations"]
    memory = recipe["training"].get("memory_metadata") or {}
    attempted = observations.get("optimizer_step_count", memory.get("optimizer_step_count"))
    skipped = observations.get("skipped_optimizer_step_count", memory.get("skipped_optimizer_step_count", 0))
    successful = observations.get("successful_optimizer_step_count", memory.get("successful_optimizer_step_count"))
    require(attempted is not None and attempted > 0 and 0 <= skipped <= .2 * attempted
        and successful == attempted - skipped and successful > 0, "unsupported_optimizer_update_history")
    require(observations.get("train_stopping_reason") == "epochs_completed" and not observations.get("train_stopped_early")
        and not observations.get("oom_retry_count", memory.get("oom_retry_count", 0))
        and not observations.get("train_fp32_loss_retry_count", memory.get("train_fp32_loss_retry_count", 0)), "unsupported_training_replay_history")
    require(recipe["retention"]["policy"] == "recipe_only" and not recipe["retention"]["client_checkpoint_retained"], "retention_mode_needs_separate_validation")
    primary = metrics.get("primary_metric_name")
    require(isinstance(primary, str) and primary in metrics, "missing_named_primary_metric")
    for name in {primary, "metric_score", "loss", "train_loss"}:
        require(finite(metrics.get(name)), "nonfinite_or_missing_" + name)
        require(recipe["observed_metrics"].get(name, {}).get("value") == metrics[name], "recipe_database_metric_mismatch_" + name)
    require(0 <= metrics["metric_score"] <= 1 and metrics["loss"] >= 0, "invalid_metric_range")
    return recipe, family

