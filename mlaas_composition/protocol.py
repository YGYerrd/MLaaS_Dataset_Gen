"""Versioned request and run-provenance contracts for composition experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

LEGACY_REQUEST_SCHEMA_VERSION = "1.0.0"
V2_REQUEST_SCHEMA_VERSION = "2.0.0"
REQUEST_SCHEMA_VERSION = "3.0.0"
FAMILY_REQUEST_SCHEMA_VERSION = "4.0.0"
TASK_REQUEST_SCHEMA_VERSION = "5.0.0"
RECIPE_REQUEST_SCHEMA_VERSION = "6.0.0"
RECIPE_OBJECTIVE_VERSION = "collaborative_objective.v3"
LEGACY_OBJECTIVE_VERSION = "collaborative_objective.v1"
OBJECTIVE_VERSION = "collaborative_objective.v2"
DEFAULT_PROFILE_ID = "controlled_collaborative_v3"
DEFAULT_PRINCIPLE_ID = "feasible_reference_bounds_v1"

SUPPORTED_REQUEST_SCHEMA_VERSIONS = (LEGACY_REQUEST_SCHEMA_VERSION, V2_REQUEST_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, FAMILY_REQUEST_SCHEMA_VERSION, TASK_REQUEST_SCHEMA_VERSION, RECIPE_REQUEST_SCHEMA_VERSION)
OBJECTIVE_BY_REQUEST_SCHEMA = {
    RECIPE_REQUEST_SCHEMA_VERSION: RECIPE_OBJECTIVE_VERSION,
    TASK_REQUEST_SCHEMA_VERSION: OBJECTIVE_VERSION,
    FAMILY_REQUEST_SCHEMA_VERSION: OBJECTIVE_VERSION,
    LEGACY_REQUEST_SCHEMA_VERSION: LEGACY_OBJECTIVE_VERSION,
    V2_REQUEST_SCHEMA_VERSION: OBJECTIVE_VERSION,
    REQUEST_SCHEMA_VERSION: OBJECTIVE_VERSION,
}
PROTOCOL_ASSETS_BY_REQUEST_SCHEMA = {
    LEGACY_REQUEST_SCHEMA_VERSION: (
        "request_profiles.v1.json",
        "request_principles.v1.json",
        "request.schema.v1.json",
        "objective.v1.json",
    ),
    V2_REQUEST_SCHEMA_VERSION: (
        "request_profiles.v2.json",
        "request_principles.v1.json",
        "request.schema.v2.json",
        "objective.v2.json",
    ),
}

PROTOCOL_ASSETS_BY_REQUEST_SCHEMA[REQUEST_SCHEMA_VERSION] = (
    "request_profiles.v3.json", "request_principles.v1.json", "request.schema.v3.json", "objective.v2.json",
)
PROTOCOL_ASSETS_BY_REQUEST_SCHEMA[FAMILY_REQUEST_SCHEMA_VERSION] = (
    "request_profiles.v4.json", "request_principles.v1.json", "request.schema.v4.json", "objective.v2.json",
)
PROTOCOL_ASSETS_BY_REQUEST_SCHEMA[TASK_REQUEST_SCHEMA_VERSION] = (
    "request_profiles.v5.json", "request_principles.v1.json", "request.schema.v5.json", "objective.v2.json",
)
PROTOCOL_ASSETS_BY_REQUEST_SCHEMA[RECIPE_REQUEST_SCHEMA_VERSION] = (
    "request_profiles.v6.json", "request_principles.v1.json", "request.schema.v6.json", "objective.v3.json",
)

_STRICT_COMPATIBILITY_FIELDS = {
    "compatibility_contract_hash",
    "fl_run_id",
    "parent_global_round",
    "parent_state_hash",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def protocol_asset_path(name: str) -> Path:
    """Return the installed path for a checked-in protocol asset."""
    return Path(str(resources.files("mlaas_composition").joinpath("assets", name)))


def load_protocol_asset(name: str) -> dict[str, Any]:
    with protocol_asset_path(name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_request_profile(profile_id: str = DEFAULT_PROFILE_ID) -> dict[str, Any]:
    for asset in ("request_profiles.v6.json", "request_profiles.v5.json", "request_profiles.v4.json", "request_profiles.v3.json", "request_profiles.v2.json", "request_profiles.v1.json"):
        library = load_protocol_asset(asset)
        if profile_id in library["profiles"]:
            profile = dict(library["profiles"][profile_id])
            profile["profile_id"] = profile_id
            profile["library_id"] = library["library_id"]
            return profile
    raise ValueError(f"Unknown request profile '{profile_id}'")


def load_request_principle(principle_id: str = DEFAULT_PRINCIPLE_ID) -> dict[str, Any]:
    library = load_protocol_asset("request_principles.v1.json")
    try:
        principle = dict(library["principles"][principle_id])
    except KeyError as error:
        raise ValueError(f"Unknown request principle '{principle_id}'") from error
    principle["principle_id"] = principle_id
    principle["library_id"] = library["library_id"]
    return principle


def read_request_file(path: str | Path) -> pd.DataFrame:
    """Read an immutable CSV or JSON request export and validate its contract."""
    request_path = Path(path)
    if request_path.suffix.lower() == ".csv":
        frame = pd.read_csv(request_path)
    elif request_path.suffix.lower() == ".json":
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        records = payload.get("requests") if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise ValueError("JSON request files must contain a list or an object with a 'requests' list")
        frame = pd.DataFrame(records)
        if isinstance(payload, Mapping) and payload.get("request_schema_version") is not None:
            outer_version = str(payload["request_schema_version"])
            row_versions = set(frame.get("request_schema_version", pd.Series(dtype=str)).astype(str))
            if row_versions and row_versions != {outer_version}:
                raise ValueError("JSON request_schema_version does not match every request record")
    else:
        raise ValueError("Request file must be .csv or .json")
    validate_request_frame(frame)
    return frame


def write_request_file(frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write generated requests as an immutable CSV or JSON contract."""
    validate_request_frame(frame)
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace immutable request file '{output}'. Pass overwrite=True to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    elif output.suffix.lower() == ".json":
        records = [_json_record(row) for row in frame.to_dict(orient="records")]
        schema_version = str(frame["request_schema_version"].iloc[0])
        output.write_text(json.dumps({"request_schema_version": schema_version, "requests": records}, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError("Request output must be .csv or .json")
    return output


def validate_request_frame(frame: pd.DataFrame) -> None:
    required = {
        "request_id", "request_schema_version", "composition_type", "functional_requirements",
        "qos_requirements", "hard_constraints", "composition_size", "preference_weights",
        "constraint_penalties",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Request file is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Request file must contain at least one request")
    if frame["request_id"].isna().any() or frame["request_id"].astype(str).duplicated().any():
        raise ValueError("request_id values must be present and unique")
    versions = set(frame["request_schema_version"].astype(str))
    if len(versions) != 1:
        raise ValueError("A request file must contain exactly one request_schema_version")
    for _, row in frame.iterrows():
        request = row.to_dict()
        schema_version = str(request["request_schema_version"])
        if schema_version not in SUPPORTED_REQUEST_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported request_schema_version '{schema_version}'")
        if str(request["composition_type"]).lower() != "collaborative":
            raise ValueError("Only collaborative composition requests are supported")
        for field in ("functional_requirements", "qos_requirements", "hard_constraints", "preference_weights", "constraint_penalties"):
            _request_object(request, field)
        from .evidence import requested_evaluation_dimensions
        requested_evaluation_dimensions(request)
        if int(request["composition_size"]) < 1:
            raise ValueError("composition_size must be positive")
        if schema_version in {V2_REQUEST_SCHEMA_VERSION, REQUEST_SCHEMA_VERSION, FAMILY_REQUEST_SCHEMA_VERSION, TASK_REQUEST_SCHEMA_VERSION, RECIPE_REQUEST_SCHEMA_VERSION}:
            _validate_v2_request(request)
        elif _present(request.get("compatibility_mode")) and str(request["compatibility_mode"]).lower() != "functional":
            raise ValueError("Request schema 1.0.0 supports functional compatibility only")


def request_schema_version(request: Mapping[str, Any]) -> str:
    """Return the request schema, treating legacy in-memory requests as v1."""
    value = request.get("request_schema_version")
    if not _present(value):
        return LEGACY_REQUEST_SCHEMA_VERSION
    version = str(value)
    if version not in SUPPORTED_REQUEST_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported request_schema_version '{version}'")
    return version


def objective_version_for_request(request: Mapping[str, Any]) -> str:
    return OBJECTIVE_BY_REQUEST_SCHEMA[request_schema_version(request)]


def objective_version_for_schema(schema_version: str) -> str:
    try:
        return OBJECTIVE_BY_REQUEST_SCHEMA[str(schema_version)]
    except KeyError as error:
        raise ValueError(f"Unsupported request_schema_version '{schema_version}'") from error


def _validate_v2_request(request: Mapping[str, Any]) -> None:
    mode = str(request.get("compatibility_mode") or "").lower()
    if request_schema_version(request) in {FAMILY_REQUEST_SCHEMA_VERSION, TASK_REQUEST_SCHEMA_VERSION, RECIPE_REQUEST_SCHEMA_VERSION} and mode != "strict":
        raise ValueError("family requests require strict compatibility")
    if mode not in {"functional", "strict"}:
        raise ValueError("compatibility_mode must be 'functional' or 'strict'")
    requirements = _request_object(request, "compatibility_requirements")
    hard_constraints = _request_object(request, "hard_constraints")
    qos_requirements = _request_object(request, "qos_requirements")
    constraint_penalties = _request_object(request, "constraint_penalties")
    if (
        "max_resource_cost_score" in hard_constraints
        or "max_resource_cost" in qos_requirements
        or "max_resource_cost" in constraint_penalties
    ):
        raise ValueError("Request schema 2.0.0 uses min_resource_cost_score")
    if mode == "functional":
        if requirements:
            raise ValueError("functional compatibility_requirements must be empty")
        if _present(request.get("eligible_pool_snapshot_hash")):
            raise ValueError("eligible_pool_snapshot_hash is reserved for strict requests")
        return
    fields = ({"cohort_id", "parent_state_hash", "compatibility_contract_hash"}
              if request_schema_version(request) == REQUEST_SCHEMA_VERSION else _STRICT_COMPATIBILITY_FIELDS)
    if request_schema_version(request) in {FAMILY_REQUEST_SCHEMA_VERSION, RECIPE_REQUEST_SCHEMA_VERSION}:
        fields = {"family_id", "initial_state_hash", "compatibility_contract_hash"}
    if request_schema_version(request) == RECIPE_REQUEST_SCHEMA_VERSION and int(request["composition_size"]) < 2:
        raise ValueError("recipe composition requires at least two services")
    if request_schema_version(request) == TASK_REQUEST_SCHEMA_VERSION:
        fields = ({"family_id", "initial_state_hash"} if "family_id" in requirements else {"cohort_id", "parent_state_hash"}) | {
            "compatibility_contract_hash", "task_semantics_hash", "evaluation_protocol_hash"}
        if _request_object(request, "required_capabilities") != {"parameter_aggregation": True, "measured_composition_validation": True}:
            raise ValueError("task requests must require parameter aggregation and measured composition validation")
        for key in ("task_semantics_hash", "evaluation_protocol_hash"):
            if not _SHA256_PATTERN.fullmatch(str(requirements.get(key, ""))):
                raise ValueError(f"task requests require a SHA-256 {key}")
    if set(requirements) != fields:
        raise ValueError(
            "strict compatibility_requirements must contain exactly: "
            + ", ".join(sorted(fields))
        )
    for field in fields - {"parent_global_round"}:
        if not _present(requirements.get(field)):
            raise ValueError(f"strict compatibility requirement '{field}' must be non-empty")
    if "parent_global_round" in fields:
        try:
            parent_round = int(requirements["parent_global_round"])
        except (TypeError, ValueError) as error:
            raise ValueError("parent_global_round must be a non-negative integer") from error
        if parent_round < 0:
            raise ValueError("parent_global_round must be a non-negative integer")
    pool_hash = str(request.get("eligible_pool_snapshot_hash") or "").lower()
    if not _SHA256_PATTERN.fullmatch(pool_hash):
        raise ValueError("strict requests require a lowercase SHA-256 eligible_pool_snapshot_hash")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_run_manifest(
    *,
    db_path: str | Path,
    request_path: str | Path,
    output_dir: str | Path,
    output_files: Sequence[Path],
    seed: int,
    command: Sequence[str] | None = None,
    request_schema_version: str = REQUEST_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Describe exact code, inputs, protocol assets, and outputs of one run."""
    root = Path(__file__).resolve().parents[1]
    schema_version = str(request_schema_version)
    try:
        assets = PROTOCOL_ASSETS_BY_REQUEST_SCHEMA[schema_version]
    except KeyError as error:
        raise ValueError(f"Unsupported request_schema_version '{schema_version}'") from error
    return {
        "run_manifest_version": "1.0.0",
        "request_schema_version": schema_version,
        "objective_version": objective_version_for_schema(schema_version),
        "seed": int(seed),
        "command": list(command or sys.argv),
        "code": _git_metadata(root),
        "dependency_versions": _dependency_versions(root / "requirements.txt"),
        "inputs": {
            "service_catalogue": {"path": str(Path(db_path).resolve()), "sha256": sha256_file(db_path)},
            "requests": {"path": str(Path(request_path).resolve()), "sha256": sha256_file(request_path)},
            "protocol_assets": {
                name: {"path": str(protocol_asset_path(name)), "sha256": sha256_file(protocol_asset_path(name))}
                for name in assets
            },
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in output_files
        },
        "output_directory": str(Path(output_dir).resolve()),
    }


def write_run_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _git_metadata(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _dependency_versions(requirements_path: Path) -> dict[str, str | None]:
    names = ["python"]
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", line)
        if match and not line.lstrip().startswith("#"):
            names.append(match.group(1))
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in sorted(set(names) - {"python"}):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and pd.isna(value):
            value = None
        if key in {"functional_requirements", "compatibility_requirements", "qos_requirements", "hard_constraints", "preference_weights", "constraint_penalties", "evaluation_requirements"} and isinstance(value, str):
            value = json.loads(value)
        result[key] = value
    return result


def _request_object(request: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = request.get(field)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} must be a JSON object") from error
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError(f"{field} must be a JSON object")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan"}
