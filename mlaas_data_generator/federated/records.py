# records_metrics.py

import json

def save_global_metrics_json(metric_key: str, records: list[dict], task_type: str, save: bool):
    if not save:
        return
    with open("weights/global_metrics.json", "w") as f:
        json.dump({"metric": metric_key, "records": records}, f, indent=4)
    if task_type == "classification":
        with open("weights/global_accuracies.json", "w") as f:
            json.dump(records, f, indent=4)