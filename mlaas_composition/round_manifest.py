"""Frozen local-round schedules and deadlines derived from completed executions."""

from collections import defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics

from .manifest import _canonical, _digest


VERSION = "local_composition_rounds_v1"
DEADLINE_MULTIPLIERS = {"tight": 0.75, "moderate": 1.0, "relaxed": 1.5}


def validate_options(rounds, timing_db, profile, availability):
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if timing_db is None:
        raise ValueError("--rounds requires --timing-db with completed composition executions")
    if profile not in DEADLINE_MULTIPLIERS:
        raise ValueError("deadline_profile must be tight, moderate or relaxed")
    if isinstance(availability, bool) or not isinstance(availability, (int, float)) or not 0 <= availability <= 1:
        raise ValueError("availability_probability must be a finite number in [0,1]")


def _duration(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def load_baselines(db):
    """Median local execution time, excluding optional client-loss measurement.

    Use only completed independent executions, never preceding round studies.
    Each family/size needs its own evidence; absent pairs are skipped explicitly.
    """
    samples = defaultdict(list)
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        query = """SELECT c.run_id,c.combination_id,c.family_id,c.composition_size,
                          COALESCE(t.value_num,t.value_int),COALESCE(l.value_num,l.value_int)
            FROM compositions c JOIN composition_runs r USING(run_id)
            JOIN composition_metrics t USING(run_id,combination_id)
            LEFT JOIN composition_metrics l ON l.run_id=c.run_id AND l.combination_id=c.combination_id
                 AND l.metric_name='client_loss_measurement_wall_s'
            WHERE c.status='completed' AND c.composition_size IN (2,3,5)
                 AND r.executor_version='composition_batch_v1' AND t.metric_name='total_wall_s'
            ORDER BY c.run_id,c.combination_id"""
        for run, cid, family, size, total, losses in conn.execute(query):
            total = _duration(total, "historical total_wall_s", positive=True)
            losses = 0.0 if losses is None else _duration(losses, "historical client loss time")
            duration = _duration(total - losses, "historical round baseline", positive=True)
            samples[family, size].append([run, cid, duration])
    if not samples:
        raise ValueError("No completed size 2, 3 or 5 composition timings in --timing-db")
    return {key: {"baseline_round_time_s": statistics.median(row[2] for row in rows),
                  "sample_count": len(rows), "min_s": min(row[2] for row in rows),
                  "max_s": max(row[2] for row in rows), "observations_sha256": _digest(rows),
                  "basis": "median_total_wall_s_minus_client_loss_measurement_wall_s"}
            for key, rows in samples.items()}


def extend_manifest(rows, metadata, baselines, *, rounds, seed, deadline_profile, availability_probability):
    multiplier = DEADLINE_MULTIPLIERS[deadline_profile]
    for row in rows:
        ids = json.loads(row["service_ids"])
        baseline = baselines[row["family_id"], row["composition_size"]]
        rng = random.Random(_digest([seed, row["combination_id"], "availability"]))
        config = {"version": VERSION, "timing_scope": "local_sequential_wall",
                  "communication": "in_process_no_network", "baseline": baseline,
                  "deadline_profile": deadline_profile, "deadline_multiplier": multiplier,
                  "round_deadline_s": baseline["baseline_round_time_s"] * multiplier,
                  "availability_probability": availability_probability,
                  "availability_basis": "seeded_bernoulli_schedule",
                  "client_availability": [{sid: rng.random() < availability_probability for sid in ids}
                                          for _ in range(rounds)]}
        row.update(rounds=rounds, round_config=_canonical(config))
    metadata.update(generator_version=VERSION, rounds=rounds, deadline_profile=deadline_profile,
                    deadline_multiplier=multiplier, availability_probability=availability_probability,
                    timing_scope="local_sequential_wall", communication="in_process_no_network",
                    round_inputs_sha256=_digest(rows))


def validate_config(config, rounds, size, ids):
    validate_options(rounds, "recorded_in_manifest", config.get("deadline_profile"),
                     config.get("availability_probability"))
    if size not in (2, 3, 5):
        raise ValueError("round experiments support composition sizes 2, 3 and 5 only")
    if (config.get("version") != VERSION or config.get("timing_scope") != "local_sequential_wall"
            or config.get("communication") != "in_process_no_network"
            or config.get("availability_basis") != "seeded_bernoulli_schedule"):
        raise ValueError("unsupported round configuration or timing scope")
    baseline = config.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("round configuration requires baseline evidence")
    baseline_s = _duration(baseline.get("baseline_round_time_s"), "baseline_round_time_s", positive=True)
    count = baseline.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("baseline requires a positive sample_count")
    digest = baseline.get("observations_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("baseline requires an observations_sha256")
    multiplier = DEADLINE_MULTIPLIERS[config["deadline_profile"]]
    deadline = _duration(config.get("round_deadline_s"), "round_deadline_s", positive=True)
    if config.get("deadline_multiplier") != multiplier or not math.isclose(deadline, baseline_s * multiplier, rel_tol=1e-12):
        raise ValueError("round deadline differs from its baseline and profile")
    availability = config.get("client_availability")
    if not isinstance(availability, list) or len(availability) != rounds:
        raise ValueError("client_availability must contain one schedule per round")
    for schedule in availability:
        if (not isinstance(schedule, dict) or set(schedule) != set(ids)
                or any(type(value) is not bool for value in schedule.values())):
            raise ValueError("each round must declare boolean availability for every composition client")
    _canonical(config)  # Reject non-finite values anywhere in frozen metadata.
    return config
