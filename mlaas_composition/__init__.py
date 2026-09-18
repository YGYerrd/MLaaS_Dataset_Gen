"""Post-processing tools for synthetic MLaaS composition experiments.

This package searches completed service catalogues. Validation materialises
selected models from checkpoints or replays authoritative training recipes,
then aggregates their parameters for held-out inference. It creates no services.
"""

from .catalogue import load_services_from_db, normalise_service_rows
from .pool import (
    CompositionInstance,
    PoolSnapshotMismatch,
    ProvenanceIntegrityError,
    StrictCompatibilityUnsupported,
    build_composition_instance,
    eligible_pool_snapshot_hash,
)
from .protocol import (
    LEGACY_OBJECTIVE_VERSION,
    LEGACY_REQUEST_SCHEMA_VERSION,
    OBJECTIVE_VERSION,
    REQUEST_SCHEMA_VERSION,
    RECIPE_OBJECTIVE_VERSION,
    RECIPE_REQUEST_SCHEMA_VERSION,
    read_request_file,
    validate_request_frame,
    write_request_file,
)
from .reference import ReferenceSolution, find_exhaustive_reference, generate_exhaustive_references
from .requirements import generate_service_requests
from .sampler import calculate_utility, evaluate_provisional_composition, generate_compositions
from .search import CommonCompositionEvaluator, SearchResult, run_search, run_search_algorithms
from .fedavg_validation import (
    FedAvgCompositionExecution,
    FedAvgValidationError,
    execute_selected_fedavg_composition,
)
from .validation import (
    CohortIsolationError,
    CohortPlan,
    ServicePrediction,
    build_cohort_plan,
    evidence_summary,
    measure_model_outputs,
)

__all__ = [
    "generate_compositions",
    "generate_service_requests",
    "CompositionInstance",
    "build_composition_instance",
    "eligible_pool_snapshot_hash",
    "PoolSnapshotMismatch",
    "ProvenanceIntegrityError",
    "StrictCompatibilityUnsupported",
    "calculate_utility",
    "evaluate_provisional_composition",
    "find_exhaustive_reference",
    "generate_exhaustive_references",
    "load_services_from_db",
    "normalise_service_rows",
    "OBJECTIVE_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RECIPE_OBJECTIVE_VERSION",
    "RECIPE_REQUEST_SCHEMA_VERSION",
    "LEGACY_OBJECTIVE_VERSION",
    "LEGACY_REQUEST_SCHEMA_VERSION",
    "read_request_file",
    "validate_request_frame",
    "write_request_file",
    "ReferenceSolution",
    "CommonCompositionEvaluator",
    "SearchResult",
    "run_search",
    "run_search_algorithms",
    "CohortIsolationError",
    "CohortPlan",
    "ServicePrediction",
    "build_cohort_plan",
    "evidence_summary",
    "FedAvgCompositionExecution",
    "FedAvgValidationError",
    "execute_selected_fedavg_composition",
    "measure_model_outputs",
]
