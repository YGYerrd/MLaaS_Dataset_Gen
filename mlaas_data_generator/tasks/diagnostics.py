"""Task diagnostics with one example from each client in memory during pairs."""
from __future__ import annotations

import csv
import io
import json
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from ..provenance import sha256_json, sha256_file
from .runtime import evaluate_task_model


def diagnose_tasks(model, contract, x, y, ids, clients, restore, output_dir, context):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    services, excluded, pairs, per_class = [], [], [], []
    for index, client in enumerate(clients):
        sid = client["service_id"]
        path = root / f"predictions_{sha256_json(sid)[:20]}.npz"
        try:
            state_hash = restore(client)
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                def sink(position, pred, truth):
                    for prefix, value in (("prediction", pred), ("target", truth)):
                        buffer = io.BytesIO()
                        np.save(buffer, np.asarray(value), allow_pickle=False)
                        archive.writestr(f"{prefix}_{position}.npy", buffer.getvalue())
                evaluation = evaluate_task_model(model, x, y, contract, ids, prediction_sink=sink)
            for label, values in evaluation.metrics.get("per_class", {}).items():
                per_class.append({"service_id": sid, "label": label, **values})
            services.append({"service_id": sid, "state_hash": state_hash, "prediction_artifact": path.name,
                "prediction_hash": sha256_file(path), **{k: v for k, v in evaluation.metrics.items() if k != "per_class"}})
        except Exception as exc:
            excluded.append({"service_id": sid, "reason_code": getattr(exc, "code", "checkpoint_or_inference_failed"), "reason": str(exc)})
        print(f"Task diagnostics: {index + 1}/{len(clients)} clients checked", flush=True)
    for left, right in combinations(services, 2):
        disagreement = overlap = union = count = 0
        with np.load(root / left["prediction_artifact"], allow_pickle=False) as a, np.load(root / right["prediction_artifact"], allow_pickle=False) as b:
            for position in range(len(ids)):
                p, q, truth = a[f"prediction_{position}"], b[f"prediction_{position}"], a[f"target_{position}"]
                if not np.array_equal(truth, b[f"target_{position}"]):
                    raise ValueError("diagnostic target alignment changed between clients")
                if contract.family == "generation":
                    disagreement += int(str(p) != str(q))
                    count += 1
                else:
                    mask = truth != contract.semantics["ignore_index"]
                    disagreement += int(np.count_nonzero((p != q) & mask))
                    wrong_a, wrong_b = (p != truth) & mask, (q != truth) & mask
                    overlap += int(np.count_nonzero(wrong_a & wrong_b))
                    union += int(np.count_nonzero(wrong_a | wrong_b))
                    count += int(mask.sum())
        pair = {"service_a": left["service_id"], "service_b": right["service_id"], "unit_count": count,
                "disagreement": disagreement / count, "identical_predictions": disagreement == 0}
        if contract.family != "generation":
            pair.update(error_overlap=overlap / count, error_jaccard=overlap / union if union else None)
        pairs.append(pair)
    report = {"version": "task_diagnostics_v1", **context, "task_contract": contract.as_dict(),
              "evaluation_subset": "catalogue_metric", "evaluation_ids": list(ids),
              "recorded_clients": len(clients), "diagnosed_clients": len(services),
              "status": "insufficient_clients" if len(services) < 2 else "partial" if excluded else "completed",
              "services": services, "pairs": pairs, "per_class": per_class, "excluded": excluded}
    report["report_hash"] = sha256_json(report)
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for key in ("services", "pairs", "per_class", "excluded"):
        rows = report[key]
        with (root / f"{key}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    return report
