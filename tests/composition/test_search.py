import json

import pandas as pd

from mlaas_composition import run_search_algorithms


def test_all_search_methods_share_the_same_feasible_instance_and_evaluator():
    services = pd.DataFrame([_service(f"svc_{index}", 1.0 - index / 10) for index in range(5)])
    request = {
        "request_id": "req_1",
        "composition_type": "collaborative",
        "functional_requirements": json.dumps({"task_family": "classification"}),
        "hard_constraints": json.dumps({"min_metric_score": 0.0, "max_latency": 1.0, "max_resource_cost_score": 1.0, "min_reliability_score": 0.0}),
        "composition_size": 2,
        "preference_weights": json.dumps({"quality": 1.0, "latency": 0.0, "resource_cost": 0.0, "reliability": 0.0, "diversity": 0.0}),
    }

    results = run_search_algorithms(
        services,
        pd.DataFrame([request], index=["canonical-request"]),
        budget=12,
        seed=9,
    )

    assert set(results["method"]) == {"mab", "ga", "greedy", "random_search"}
    assert set(results["search_status"]) == {"completed"}
    assert set(results["evaluator_name"]) == {"paper_informed_collaborative_heuristic"}
    assert set(results["candidate_space_size"]) == {10}
    assert results["evaluations"].between(1, 12).all()
    assert results["utility_score"].between(0.0, 1.0).all()


def _service(service_id, metric_score):
    return {
        "service_id": service_id,
        "task_family": "classification",
        "metric_score": metric_score,
        "latency": 0.1,
        "resource_cost_score": 0.1,
        "reliability_score": 1.0,
    }
