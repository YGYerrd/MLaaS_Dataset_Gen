"""Collaborative search baselines connected to one common evaluator."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .pool import CompositionInstance, build_composition_instance, request_object
from .protocol import OBJECTIVE_VERSION, objective_version_for_request
from .sampler import calculate_utility, evaluate_provisional_composition


SEARCH_METHODS = ("mab", "ga", "greedy", "random_search")


@dataclass(frozen=True)
class CandidateEvaluation:
    """One evaluation of a valid candidate from an instance's implicit Omega_R."""

    indices: tuple[int, ...]
    service_ids: tuple[str, ...]
    scores: Mapping[str, Any]
    utility_score: float


class CommonCompositionEvaluator:
    """Evaluate feasible groups through the same objective and call counter.

    The default evaluator is the paper-informed collaborative heuristic. Replacing the
    callable with a cached-prediction ensemble evaluator later leaves every
    search algorithm unchanged.
    """

    def __init__(
        self,
        instance: CompositionInstance,
        preference_weights: Mapping[str, Any],
        *,
        evaluator: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
        qos_requirements: Mapping[str, Any] | None = None,
        constraint_penalties: Mapping[str, Any] | None = None,
        evaluator_name: str = "paper_informed_collaborative_heuristic",
        objective_version: str | None = None,
    ) -> None:
        self.instance = instance
        self.preference_weights = dict(preference_weights)
        self.qos_requirements = dict(qos_requirements or {})
        self.constraint_penalties = dict(constraint_penalties or {})
        self.objective_version = objective_version or instance.objective_version or OBJECTIVE_VERSION
        self.evaluator = evaluator or (
            lambda selected: evaluate_provisional_composition(
                selected, objective_version=self.objective_version
            )
        )
        self.evaluator_name = evaluator_name
        self.evaluations = 0

    def evaluate(self, indices: Sequence[int]) -> CandidateEvaluation:
        """Score one complete, feasible group and increment its call count."""
        candidate = tuple(sorted(int(index) for index in indices))
        if len(candidate) != self.instance.composition_size or len(set(candidate)) != len(candidate):
            raise ValueError("Candidate must contain exactly composition_size distinct services")
        if not candidate or candidate[0] < 0 or candidate[-1] >= len(self.instance.C_R):
            raise ValueError("Candidate index is outside C_R")
        selected = self.instance.C_R.iloc[list(candidate)].reset_index(drop=True)
        scores = self.evaluator(selected)
        self.evaluations += 1
        return CandidateEvaluation(
            indices=candidate,
            service_ids=tuple(sorted(selected["service_id"].astype(str).tolist())),
            scores=scores,
            utility_score=calculate_utility(
                scores,
                self.preference_weights,
                self.qos_requirements,
                self.constraint_penalties,
                objective_version=self.objective_version,
            ),
        )


@dataclass(frozen=True)
class SearchResult:
    """Comparable outcome emitted by every search baseline."""

    method: str
    status: str
    service_ids: tuple[str, ...]
    utility_score: float | None
    evaluations: int
    runtime_s: float
    candidate_space_size: int
    evaluator_name: str
    scores: Mapping[str, Any]


def run_search(
    method: str,
    evaluator: CommonCompositionEvaluator,
    *,
    budget: int,
    seed: int = 42,
) -> SearchResult:
    """Run one baseline against the supplied common evaluator."""
    method = str(method).lower()
    if method not in SEARCH_METHODS:
        raise ValueError(f"Unsupported search method '{method}'; choose from {SEARCH_METHODS}")
    if budget < 1:
        raise ValueError("budget must be at least one")

    started = time.perf_counter()
    if evaluator.instance.omega_size == 0:
        return _result(method, "infeasible", None, evaluator, started)

    rng = np.random.default_rng(seed)
    algorithms = {
        "random_search": _run_random_search,
        "greedy": _run_greedy,
        "mab": _run_mab,
        "ga": _run_ga,
    }
    best = algorithms[method](evaluator, budget, rng)
    return _result(method, "completed", best, evaluator, started)


def run_search_algorithms(
    services: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    methods: Iterable[str] = SEARCH_METHODS,
    budget: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all requested baselines against identical instances and objectives."""
    records: list[dict[str, Any]] = []
    for request_position, (_, request_row) in enumerate(requests.iterrows()):
        request = request_row.to_dict()
        instance = build_composition_instance(services, request)
        objective_version = objective_version_for_request(request)
        weights = request_object(request, "preference_weights")
        qos_requirements = _optional_request_object(request, "qos_requirements")
        constraint_penalties = _optional_request_object(request, "constraint_penalties")
        for method_index, method in enumerate(methods):
            evaluator = CommonCompositionEvaluator(
                instance,
                weights,
                evaluator=lambda selected, qos=qos_requirements: evaluate_provisional_composition(
                    selected,
                    qos_requirements=qos,
                    objective_version=objective_version,
                ),
                qos_requirements=qos_requirements,
                constraint_penalties=constraint_penalties,
                objective_version=objective_version,
            )
            result = run_search(
                method,
                evaluator,
                budget=budget,
                seed=seed + request_position * 1_000 + method_index,
            )
            records.append({
                "request_id": str(request.get("request_id")),
                "composition_type": "collaborative",
                "compatibility_mode": instance.compatibility_mode,
                "requested_pool_snapshot_hash": instance.requested_pool_snapshot_hash,
                "computed_pool_snapshot_hash": instance.computed_pool_snapshot_hash,
                "objective_version": objective_version,
                "method": result.method,
                "search_status": result.status,
                "service_ids": json.dumps(result.service_ids),
                "utility_score": result.utility_score,
                "evaluations": result.evaluations,
                "runtime_s": result.runtime_s,
                "eligible_pool_size": len(instance.C_R),
                "candidate_space_size": result.candidate_space_size,
                "evaluator_name": result.evaluator_name,
                **result.scores,
            })
    return pd.DataFrame(records)


def _run_random_search(
    evaluator: CommonCompositionEvaluator, budget: int, rng: np.random.Generator
) -> CandidateEvaluation:
    best: CandidateEvaluation | None = None
    for _ in range(budget):
        best = _best(best, evaluator.evaluate(_random_candidate(evaluator.instance, rng)))
    assert best is not None
    return best


def _run_greedy(
    evaluator: CommonCompositionEvaluator, budget: int, rng: np.random.Generator
) -> CandidateEvaluation:
    """Construct a group one service at a time using common-evaluator lookahead."""
    instance = evaluator.instance
    selected: list[int] = []
    best: CandidateEvaluation | None = None
    while len(selected) < instance.composition_size and evaluator.evaluations < budget:
        remaining = [index for index in rng.permutation(len(instance.C_R)) if int(index) not in selected]
        step_best: tuple[int, CandidateEvaluation] | None = None
        for candidate in remaining:
            if evaluator.evaluations >= budget:
                break
            completion = _complete_candidate(selected + [int(candidate)], instance)
            evaluated = evaluator.evaluate(completion)
            best = _best(best, evaluated)
            if step_best is None or _is_better(evaluated, step_best[1]):
                step_best = (int(candidate), evaluated)
        if step_best is None:
            break
        selected.append(step_best[0])
    assert best is not None
    return best


def _run_mab(
    evaluator: CommonCompositionEvaluator, budget: int, rng: np.random.Generator
) -> CandidateEvaluation:
    """UCB1 over a sampled set of valid composition arms."""
    # Composition counts exceed NumPy integer ranges even for 100 services.
    arm_count = min(budget, max(2, min(20, math.isqrt(evaluator.instance.omega_size) + 1)))
    arms = _unique_random_candidates(evaluator.instance, arm_count, rng)
    rewards = np.zeros(len(arms), dtype=float)
    pulls = np.zeros(len(arms), dtype=int)
    best: CandidateEvaluation | None = None
    for step in range(budget):
        untried = np.flatnonzero(pulls == 0)
        if len(untried):
            arm_index = int(untried[0])
        else:
            ucb = rewards / pulls + np.sqrt(2.0 * np.log(step + 1) / pulls)
            arm_index = int(np.argmax(ucb))
        evaluated = evaluator.evaluate(arms[arm_index])
        pulls[arm_index] += 1
        rewards[arm_index] += evaluated.utility_score
        best = _best(best, evaluated)
    assert best is not None
    return best


def _run_ga(
    evaluator: CommonCompositionEvaluator, budget: int, rng: np.random.Generator
) -> CandidateEvaluation:
    """Simple genetic search whose every fitness call uses the common evaluator."""
    instance = evaluator.instance
    population_size = min(budget, max(2, min(20, len(instance.C_R) * 2)))
    population = _unique_random_candidates(instance, population_size, rng)
    best: CandidateEvaluation | None = None
    while evaluator.evaluations < budget:
        scored: list[CandidateEvaluation] = []
        for candidate in population:
            if evaluator.evaluations >= budget:
                break
            evaluated = evaluator.evaluate(candidate)
            scored.append(evaluated)
            best = _best(best, evaluated)
        if not scored:
            break
        ranked = sorted(scored, key=lambda item: (item.utility_score, item.service_ids), reverse=True)
        elites = [item.indices for item in ranked[: max(1, len(ranked) // 4)]]
        population = elites[:]
        while len(population) < population_size:
            first = elites[int(rng.integers(len(elites)))]
            second = elites[int(rng.integers(len(elites)))]
            population.append(_mutate(_crossover(first, second, instance, rng), instance, rng))
    assert best is not None
    return best


def _result(
    method: str,
    status: str,
    best: CandidateEvaluation | None,
    evaluator: CommonCompositionEvaluator,
    started: float,
) -> SearchResult:
    return SearchResult(
        method=method,
        status=status,
        service_ids=best.service_ids if best else tuple(),
        utility_score=best.utility_score if best else None,
        evaluations=evaluator.evaluations,
        runtime_s=time.perf_counter() - started,
        candidate_space_size=evaluator.instance.omega_size,
        evaluator_name=evaluator.evaluator_name,
        scores=best.scores if best else {},
    )


def _random_candidate(instance: CompositionInstance, rng: np.random.Generator) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in rng.choice(len(instance.C_R), size=instance.composition_size, replace=False)))


def _unique_random_candidates(
    instance: CompositionInstance, count: int, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    candidates: set[tuple[int, ...]] = set()
    attempts = 0
    while len(candidates) < min(count, instance.omega_size) and attempts < count * 30:
        candidates.add(_random_candidate(instance, rng))
        attempts += 1
    if len(candidates) < min(count, instance.omega_size):
        for candidate in instance.iter_candidate_indices():
            candidates.add(candidate)
            if len(candidates) == min(count, instance.omega_size):
                break
    return sorted(candidates)


def _complete_candidate(selected: Sequence[int], instance: CompositionInstance) -> tuple[int, ...]:
    values = list(dict.fromkeys(selected))
    values.extend(index for index in range(len(instance.C_R)) if index not in values)
    return tuple(sorted(values[: instance.composition_size]))


def _crossover(
    first: Sequence[int], second: Sequence[int], instance: CompositionInstance, rng: np.random.Generator
) -> tuple[int, ...]:
    values = list(dict.fromkeys(list(first[: len(first) // 2]) + list(second)))
    rng.shuffle(values)
    return _complete_candidate(values, instance)


def _mutate(candidate: Sequence[int], instance: CompositionInstance, rng: np.random.Generator) -> tuple[int, ...]:
    values = list(candidate)
    position = int(rng.integers(len(values)))
    alternatives = [index for index in range(len(instance.C_R)) if index not in values]
    if alternatives:
        values[position] = int(rng.choice(alternatives))
    return tuple(sorted(values))


def _optional_request_object(request: Mapping[str, Any], field: str) -> dict[str, Any]:
    return request_object(request, field) if request.get(field) is not None else {}


def _best(current: CandidateEvaluation | None, candidate: CandidateEvaluation) -> CandidateEvaluation:
    return candidate if current is None or _is_better(candidate, current) else current


def _is_better(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    return left.utility_score > right.utility_score or (
        left.utility_score == right.utility_score and left.service_ids < right.service_ids
    )
