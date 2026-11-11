"""Reproducible random selections from already-prepared family pools."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sqlite3
from typing import Any


COMPOSITION_SIZES = (2, 3, 5, 7, 10)
GENERATOR_VERSION = "random_composition_manifest_v1"
_COLUMNS = ("combination_id", "family_id", "composition_size", "service_ids")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_roster(path: Path, key: str, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        records = {}
        for row in reader:
            if any(not row.get(field) or not row[field].strip() for field in required):
                raise ValueError(f"{path.name} contains an empty required field")
            identity = row[key]
            if identity in records:
                raise ValueError(f"Duplicate {key} in {path.name}: {identity}")
            records[identity] = row
    return records


def _runtime(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return (result or 0.0) if math.isfinite(result) and result >= 0 else None


def _load_pools(prepared_dir: Path, sizes=COMPOSITION_SIZES) -> tuple[dict[str, list[tuple[str, float | None]]], list[dict]]:
    families = _read_roster(prepared_dir / "pools.csv", "family_id", {"family_id", "dependency_status"})
    roster = _read_roster(prepared_dir / "services.csv", "service_id", {"service_id", "family_id", "status"})
    pools: dict[str, list[tuple[str, float | None]]] = {
        fid: [] for fid in sorted(families) if families[fid]["dependency_status"] == "verified"
    }
    skipped = [
        {"family_id": fid, "composition_size": size, "reason": "pool_not_verified"}
        for fid in sorted(families) if fid not in pools
        for size in sizes
    ]
    # Read only the small identity/timing projection, never model artifacts or
    # the catalogue loader's recipe validation and scoring fields.
    uri = (prepared_dir / "services.db").resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        query = """
            SELECT s.service_id, s.family_id, s.status, g.family_id,
                   COALESCE(t.value_num, t.value_int), COALESCE(a.value_num, a.value_int)
            FROM services s
            LEFT JOIN service_generation_provenance g ON g.service_id = s.service_id
            LEFT JOIN service_metrics t ON t.service_id = s.service_id AND t.metric_name = 'training_runtime_s'
            LEFT JOIN service_metrics a ON a.service_id = s.service_id AND a.metric_name = 'train_runtime_s'
        """
        seen = set()
        for sid, service_family, status, family, primary, fallback in conn.execute(query):
            if sid not in roster:
                continue
            if sid in seen:
                raise ValueError(f"Duplicate service_id in database projection: {sid}")
            seen.add(sid)
            row = roster[sid]
            if family != row["family_id"] or family not in families or (service_family and service_family != family):
                raise ValueError(f"Family membership mismatch for service {sid}")
            if family in pools and row["status"] == "verified" and status == "completed":
                timing = _runtime(primary)
                pools[family].append((sid, timing if timing is not None else _runtime(fallback)))
        missing = set(roster) - seen
        if missing:
            raise ValueError(f"Roster services missing from database: {sorted(missing)[:5]}")
    finally:
        conn.close()
    for members in pools.values():
        members.sort()
    return pools, skipped


def _sampling_weights(times: list[float | None]) -> list[float]:
    valid = [value for value in times if value is not None]
    if not valid or min(valid) == max(valid):
        return [1.0] * len(times)
    low, high = min(valid), max(valid)
    return [1.0 if value is None else 1.0 + 0.2 * (high - value) / (high - low) for value in times]


def _draw_members(rng: random.Random, ids: list[str], weights: list[float], size: int) -> tuple[str, ...]:
    remaining_ids, remaining_weights = ids.copy(), weights.copy()
    selected = []
    for _ in range(size):
        index = rng.choices(range(len(remaining_ids)), weights=remaining_weights, k=1)[0]
        selected.append(remaining_ids.pop(index))
        remaining_weights.pop(index)
    return tuple(sorted(selected))


def _sample_compositions(pools: dict, skipped: list[dict], compositions: int, seed: int,
                         *, sizes=COMPOSITION_SIZES, eligible_pairs=None) -> tuple[list[dict], dict]:
    skipped = list(skipped)
    pairs = []
    for family, members in sorted(pools.items()):
        for size in sizes:
            if eligible_pairs is not None and (family, size) not in eligible_pairs:
                skipped.append({"family_id": family, "composition_size": size, "reason": "no_round_timing_baseline"})
                continue
            if len(members) >= size:
                pairs.append((family, size))
            else:
                skipped.append({"family_id": family, "composition_size": size, "reason": "insufficient_services"})
    if not pairs:
        raise ValueError("No eligible pool/size pairs in the prepared inputs")
    # Integer arithmetic implements half-up rounding without float precision loss.
    quota = max(1, (2 * compositions + len(pairs)) // (2 * len(pairs)))
    rng = random.Random(seed)
    rows, shortfalls = [], []
    for family, size in pairs:
        members = pools[family]
        ids = [sid for sid, _ in members]
        weights = _sampling_weights([timing for _, timing in members])
        capacity = math.comb(len(ids), size)
        target = min(quota, capacity)
        selected: set[tuple[str, ...]] = set()
        for _ in range(max(100, 20 * target)):
            selected.add(_draw_members(rng, ids, weights, size))
            if len(selected) == target:
                break
        for service_ids in sorted(selected):
            rows.append({
                "combination_id": "comb_" + _digest([family, service_ids]),
                "family_id": family,
                "composition_size": size,
                "service_ids": _canonical(service_ids),
            })
        if len(selected) < quota:
            shortfalls.append({
                "family_id": family, "composition_size": size, "requested_count": quota,
                "actual_count": len(selected), "available_combinations": capacity,
                "reason": "sampling_attempt_limit" if len(selected) < target else "combination_space_exhausted",
            })
    rows.sort(key=lambda row: (row["family_id"], row["composition_size"], row["combination_id"]))
    skipped.sort(key=lambda row: (row["family_id"], row["composition_size"]))
    return rows, {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "requested_count": compositions,
        "actual_count": len(rows),
        "composition_sizes": list(sizes),
        "eligible_pair_count": len(pairs),
        "per_pair_quota": quota,
        "selection_inputs_sha256": _digest({"pools": pools, "skipped_pairs": skipped}),
        "skipped_pairs": skipped,
        "shortfalls": shortfalls,
    }


def generate_composition_manifest(
    prepared_dir: str | Path,
    output: str | Path,
    *,
    compositions: int,
    seed: int = 42,
    overwrite: bool = False,
    rounds: int | None = None,
    timing_db: str | Path | None = None,
    deadline_profile: str = "moderate",
    availability_probability: float = 1.0,
    family_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Write a CSV and deterministic metadata JSON; return the saved metadata.

    ``compositions`` is an approximate total, rounded to full pool/size batches.
    Selection is reproducible for identical normalized inputs, count, seed and
    generator version. No training or aggregation is performed.
    """
    if isinstance(compositions, bool) or not isinstance(compositions, int) or compositions <= 0:
        raise ValueError("compositions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if rounds is not None:
        from .round_manifest import validate_options
        validate_options(rounds, timing_db, deadline_profile, availability_probability)
    elif timing_db is not None or deadline_profile != "moderate" or availability_probability != 1.0:
        raise ValueError("Round timing and availability options require --rounds")
    prepared_dir, output = Path(prepared_dir), Path(output)
    if output.suffix.lower() != ".csv":
        raise ValueError("Manifest output must be .csv")
    metadata_path = output.with_suffix(".metadata.json")
    sources = {(prepared_dir / name).resolve() for name in ("pools.csv", "services.csv", "services.db")}
    if timing_db is not None:
        sources.add(Path(timing_db).resolve())
    for destination in (output, metadata_path):
        if destination.resolve() in sources or any(
            destination.exists() and source.exists() and destination.samefile(source) for source in sources
        ):
            raise ValueError("Manifest output must not replace a prepared input file")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace '{destination}'. Pass --overwrite to replace it.")
        if destination.is_dir():
            raise IsADirectoryError(destination)
    sizes = (2, 3, 5) if rounds is not None else COMPOSITION_SIZES
    pools, skipped = _load_pools(prepared_dir, sizes)
    if family_ids:
        unknown = set(family_ids) - set(pools)
        if unknown:
            raise ValueError(f"Unknown or unverified family IDs: {sorted(unknown)}")
        pools = {fid: members for fid, members in pools.items() if fid in family_ids}
        skipped = [row for row in skipped if row["family_id"] in family_ids]
    baselines = None
    if rounds is not None:
        from .round_manifest import load_baselines, extend_manifest
        baselines = load_baselines(timing_db)
    rows, metadata = _sample_compositions(pools, skipped, compositions, seed, sizes=sizes,
                                          eligible_pairs=baselines)
    columns = _COLUMNS
    if rounds is not None:
        extend_manifest(rows, metadata, baselines, rounds=rounds, seed=seed,
                        deadline_profile=deadline_profile, availability_probability=availability_probability)
        columns = (*columns, "rounds", "round_config")
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8", newline="") as stream:
        stream.write(csv_text.getvalue())
    with metadata_path.open(mode, encoding="utf-8", newline="") as stream:
        stream.write(metadata_text)
    return metadata
