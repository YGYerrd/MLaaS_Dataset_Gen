from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_TOLERANCE = 1e-12


def clone_weights(weights: Any) -> dict[str, np.ndarray] | None:
    """Return numeric weights as a named array mapping, copied for later comparison."""
    named = _as_named_arrays(weights)
    if not named:
        return None
    return {key: np.array(value, copy=True) for key, value in named.items()}


def snapshot_model_weights(model: Any) -> dict[str, np.ndarray] | None:
    """Best-effort snapshot for models/adapters with a get_weights method."""
    if model is None or not hasattr(model, "get_weights"):
        return None
    try:
        return clone_weights(model.get_weights())
    except Exception:
        return None


def weight_delta(
    before: Any,
    after: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare two weight payloads and return scalar diagnostics."""
    left = _as_named_arrays(before)
    right = _as_named_arrays(after)
    if not left or not right:
        return _empty_delta()

    common = [key for key in left.keys() if key in right and left[key].shape == right[key].shape]
    if not common:
        return _empty_delta()

    squared_sum = 0.0
    max_abs = 0.0
    element_count = 0
    for key in common:
        diff = np.asarray(right[key], dtype=np.float64) - np.asarray(left[key], dtype=np.float64)
        if diff.size == 0:
            continue
        squared_sum += float(np.sum(diff * diff))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        element_count += int(diff.size)

    if element_count <= 0:
        return _empty_delta()

    l2 = float(math.sqrt(squared_sum))
    changed = bool(max_abs > float(tolerance))
    return {
        "available": True,
        "l2": l2,
        "max_abs": max_abs,
        "changed": changed,
        "keys_compared": int(len(common)),
        "element_count": int(element_count),
    }


def client_update_metrics(
    round_start_weights: Any,
    client_payload: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    delta = weight_delta(round_start_weights, client_payload, tolerance=tolerance)
    if not delta["available"]:
        return {}
    return {
        "client_update_l2": delta["l2"],
        "client_update_max_abs": delta["max_abs"],
        "client_update_changed_flag": delta["changed"],
        "client_update_layer_count": delta["keys_compared"],
    }


def global_update_metrics(
    before: Any,
    after: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    delta = weight_delta(before, after, tolerance=tolerance)
    if not delta["available"]:
        return {}
    return {
        "round_global_weight_delta_l2": delta["l2"],
        "round_global_weight_delta_max_abs": delta["max_abs"],
        "round_global_weight_changed_flag": delta["changed"],
        "round_global_weight_layer_count": delta["keys_compared"],
    }


def carry_forward_metrics(
    previous_round_weights: Any,
    current_round_weights: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    delta = weight_delta(previous_round_weights, current_round_weights, tolerance=tolerance)
    if not delta["available"]:
        return {}
    return {
        "round_start_global_delta_l2": delta["l2"],
        "round_start_global_delta_max_abs": delta["max_abs"],
        "global_weights_carried_forward_flag": not delta["changed"],
    }


def repeated_round_metrics(
    previous_metrics: Mapping[str, Any] | None,
    current_metrics: Mapping[str, Any],
    *,
    expected_update: bool,
    global_weights_changed: bool | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    if previous_metrics is None:
        return {
            "round_repeated_global_metrics_flag": False,
            "round_repetition_expected_flag": False,
            "round_redundant_flag": False,
        }

    repeated = _metrics_equal(previous_metrics, current_metrics, tolerance=tolerance)
    weights_static = global_weights_changed is False
    return {
        "round_repeated_global_metrics_flag": repeated,
        "round_repetition_expected_flag": bool(repeated and not expected_update),
        "round_redundant_flag": bool(repeated and expected_update and weights_static),
    }


def evaluate_run_dynamics(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, Any]]:
    """Read recorded FL dynamics from a run DB and summarize likely issues."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        runs = _load_runs(conn, run_id=run_id)
        summaries = []
        for run in runs:
            rid = str(run["run_id"])
            params = _load_run_params(conn, rid)
            round_metrics = _load_round_measurements(conn, rid)
            client_metrics = _load_client_measurements(conn, rid)
            summaries.append(
                _summarize_run(
                    run=dict(run),
                    params=params,
                    round_metrics=round_metrics,
                    client_metrics=client_metrics,
                    tolerance=tolerance,
                )
            )
        return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate federated learning dynamics recorded in an MLaaS DB")
    parser.add_argument("--db", required=True, help="Path to the SQLite run database")
    parser.add_argument("--run-id", default=None, help="Optional run_id to inspect")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Numeric equality tolerance")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args(argv)

    summaries = evaluate_run_dynamics(args.db, run_id=args.run_id, tolerance=args.tolerance)
    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
        return

    for summary in summaries:
        print(f"run_id: {summary['run_id']}")
        print(f"  task/model: {summary.get('task_type')} / {summary.get('model_type')}")
        print(f"  rounds: {summary.get('num_rounds')}")
        print(f"  update_expected_rounds: {summary.get('update_expected_rounds')}")
        print(f"  redundant_rounds: {summary.get('redundant_rounds')}")
        print(f"  expected_repeated_rounds: {summary.get('expected_repeated_rounds')}")
        if summary["issues"]:
            print("  issues:")
            for issue in summary["issues"]:
                print(f"    - {issue}")
        else:
            print("  issues: none")


def _as_named_arrays(weights: Any) -> dict[str, np.ndarray] | None:
    if weights is None:
        return None

    if isinstance(weights, Mapping):
        out: dict[str, np.ndarray] = {}
        for key, value in weights.items():
            arr = _coerce_numeric_array(value)
            if arr is not None:
                out[str(key)] = arr
        return out or None

    if isinstance(weights, (list, tuple)):
        out = {}
        for idx, value in enumerate(weights):
            arr = _coerce_numeric_array(value)
            if arr is not None:
                out[f"layer_{idx}"] = arr
        return out or None

    arr = _coerce_numeric_array(weights)
    if arr is None:
        return None
    return {"value": arr}


def _coerce_numeric_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    try:
        arr = arr.astype(np.float64, copy=False)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _empty_delta() -> dict[str, Any]:
    return {
        "available": False,
        "l2": None,
        "max_abs": None,
        "changed": None,
        "keys_compared": 0,
        "element_count": 0,
    }


def _metrics_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float,
) -> bool:
    keys = ("loss", "metric", "score", "extra")
    for key in keys:
        if not _value_equal(left.get(key), right.get(key), tolerance=tolerance):
            return False
    return True


def _value_equal(left: Any, right: Any, *, tolerance: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        lf = float(left)
        rf = float(right)
        if np.isfinite(lf) and np.isfinite(rf):
            return abs(lf - rf) <= tolerance
    except Exception:
        pass
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)


def _load_runs(conn: sqlite3.Connection, *, run_id: str | None) -> list[sqlite3.Row]:
    if run_id:
        rows = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY rowid").fetchall()
    return list(rows)


def _load_run_params(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    params = {}
    rows = conn.execute(
        """
        SELECT scope, key, value_text, value_num, value_int, value_bool, value_json
        FROM run_params
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    for row in rows:
        params[f"{row['scope']}.{row['key']}"] = _measurement_value(row)
    return params


def _load_round_measurements(conn: sqlite3.Connection, run_id: str) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.round AS round_idx, md.name, m.value_text, m.value_num, m.value_int, m.value_bool, m.value_json
        FROM measurements m
        JOIN metrics md ON md.metric_id = m.metric_id
        WHERE m.run_id = ? AND m.client_id IS NULL AND m.round IS NOT NULL
        ORDER BY m.round, md.name
        """,
        (run_id,),
    ).fetchall()
    by_round: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_round.setdefault(int(row["round_idx"]), {})[str(row["name"])] = _measurement_value(row)
    return by_round


def _load_client_measurements(conn: sqlite3.Connection, run_id: str) -> dict[tuple[int, str], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.round AS round_idx, m.client_id, md.name,
               m.value_text, m.value_num, m.value_int, m.value_bool, m.value_json
        FROM measurements m
        JOIN metrics md ON md.metric_id = m.metric_id
        WHERE m.run_id = ? AND m.client_id IS NOT NULL AND m.round IS NOT NULL
        ORDER BY m.round, m.client_id, md.name
        """,
        (run_id,),
    ).fetchall()
    by_client: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["round_idx"]), str(row["client_id"]))
        by_client.setdefault(key, {})[str(row["name"])] = _measurement_value(row)
    return by_client


def _measurement_value(row: sqlite3.Row) -> Any:
    if row["value_num"] is not None:
        return float(row["value_num"])
    if row["value_int"] is not None:
        return int(row["value_int"])
    if row["value_bool"] is not None:
        return bool(row["value_bool"])
    if row["value_text"] is not None:
        return row["value_text"]
    if row["value_json"] is not None:
        try:
            return json.loads(row["value_json"])
        except Exception:
            return row["value_json"]
    return None


def _summarize_run(
    *,
    run: dict[str, Any],
    params: dict[str, Any],
    round_metrics: dict[int, dict[str, Any]],
    client_metrics: dict[tuple[int, str], dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    redundant_rounds = []
    expected_repeated_rounds = []
    update_expected_rounds = []
    issues = []

    for round_idx, values in sorted(round_metrics.items()):
        if values.get("federated_update_expected_flag"):
            update_expected_rounds.append(round_idx)
        if values.get("round_redundant_flag"):
            redundant_rounds.append(round_idx)
        if values.get("round_repetition_expected_flag"):
            expected_repeated_rounds.append(round_idx)

    if not any("round_redundant_flag" in values for values in round_metrics.values()):
        inferred = _infer_redundant_rounds(round_metrics, tolerance=tolerance)
        redundant_rounds.extend(idx for idx in inferred if idx not in redundant_rounds)

    if redundant_rounds:
        issues.append(
            "consecutive rounds repeated global metrics while weight-update diagnostics did not show a global change"
        )

    expected_updates = bool(update_expected_rounds)
    if expected_updates:
        changed_rounds = [
            idx for idx, values in round_metrics.items()
            if values.get("round_global_weight_changed_flag") is True
        ]
        if not changed_rounds:
            issues.append("no recorded round changed global weights despite update-expected rounds")

        client_updates = [
            values.get("client_update_changed_flag")
            for values in client_metrics.values()
            if "client_update_changed_flag" in values
        ]
        if client_updates and not any(client_updates):
            issues.append("no participating client recorded a local model update")

    for round_idx, values in sorted(round_metrics.items()):
        if round_idx <= 1:
            continue
        if values.get("global_weights_carried_forward_flag") is False:
            issues.append(f"round {round_idx} did not start from the previous round's global weights")

    return {
        "run_id": run.get("run_id"),
        "task_type": run.get("task_type"),
        "model_type": run.get("model_type"),
        "num_rounds": run.get("num_rounds"),
        "dataset": run.get("dataset"),
        "inference_only": params.get("adapter.inference_only"),
        "aggregation_weight_unit": params.get("adapter.aggregation_weight_unit")
        or params.get("aggregator.aggregation_weight_unit"),
        "update_expected_rounds": update_expected_rounds,
        "redundant_rounds": sorted(set(redundant_rounds)),
        "expected_repeated_rounds": sorted(set(expected_repeated_rounds)),
        "issues": issues,
    }


def _infer_redundant_rounds(
    round_metrics: dict[int, dict[str, Any]],
    *,
    tolerance: float,
) -> list[int]:
    redundant = []
    previous = None
    for round_idx, values in sorted(round_metrics.items()):
        current = _extract_global_metric_signature(values)
        if previous is not None and _metrics_equal(previous, current, tolerance=tolerance):
            if values.get("round_global_weight_changed_flag") is False:
                redundant.append(round_idx)
        previous = current
    return redundant


def _extract_global_metric_signature(values: Mapping[str, Any]) -> dict[str, Any]:
    metric_value = None
    for key, value in values.items():
        if key.startswith("global_") and key not in {"global_loss", "global_metric_score", "global_aux_metric"}:
            metric_value = value
            break
    return {
        "loss": values.get("global_loss"),
        "metric": metric_value,
        "score": values.get("global_metric_score"),
        "extra": values.get("global_aux_metric"),
    }


if __name__ == "__main__":
    main()
