"""Command-line entry points for frozen composition experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .catalogue import load_services_from_db
from .protocol import (
    DEFAULT_PRINCIPLE_ID,
    DEFAULT_PROFILE_ID,
    build_run_manifest,
    read_request_file,
    write_request_file,
    write_run_manifest,
)
from .reference import DEFAULT_MAX_CANDIDATE_SPACE, generate_exhaustive_references
from .requirements import generate_service_requests
from .sampler import generate_compositions
from .search import run_search_algorithms
from .validation_cli import validate_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate immutable requests or evaluate a frozen collaborative-composition experiment"
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    generate = subparsers.add_parser("generate-requests", help="Generate and save versioned synthetic requests")
    generate.add_argument("--db", required=True, help="SQLite database produced by mlaas_data_generator")
    generate.add_argument("--output", required=True, help="Immutable request .csv or .json output")
    generate.add_argument("--requests", type=int, default=25, help="Number of controlled synthetic requests")
    generate.add_argument("--seed", type=int, default=42, help="Request-generation seed")
    generate.add_argument("--profile", default=DEFAULT_PROFILE_ID, help="Checked-in request profile identifier")
    generate.add_argument("--principle", default=DEFAULT_PRINCIPLE_ID, help="Checked-in request principle identifier")
    generate.add_argument(
        "--compatibility-mode",
        choices=("functional", "strict"),
        default="functional",
        help="Construct historical functional requests or strict requests within a frozen family, parent cohort, or FedAvg snapshot",
    )
    generate.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing request file")
    generate.add_argument(
        "--include-legacy-fedavg-partitions",
        action="store_true",
        help="Include sequential_depletion_v1 FedAvg observations",
    )
    generate.set_defaults(handler=_generate_requests)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an immutable request file")
    evaluate.add_argument("--db", required=True, help="SQLite database produced by mlaas_data_generator")
    evaluate.add_argument("--requests-file", required=True, help="Versioned immutable request .csv or .json file")
    evaluate.add_argument("--output-dir", required=True, help="Directory for result CSVs and run_manifest.json")
    evaluate.add_argument("--candidates-per-request", type=int, default=10, help="Candidate groups per request")
    evaluate.add_argument("--reference-max-candidates", type=int, default=DEFAULT_MAX_CANDIDATE_SPACE, help="Exhaustive-reference candidate-space limit")
    evaluate.add_argument("--search-budget", type=int, default=100, help="Common evaluator-call budget for every search method")
    evaluate.add_argument("--seed", type=int, default=42, help="Search seed")
    evaluate.add_argument(
        "--include-legacy-fedavg-partitions",
        action="store_true",
        help="Include sequential_depletion_v1 FedAvg observations",
    )
    evaluate.set_defaults(handler=_evaluate)

    validate = subparsers.add_parser("validate", help="Execute and validate one already-selected composition")
    validate.add_argument("--db", required=True, help="SQLite catalogue database")
    validate.add_argument("--requests-file", required=True, help="Immutable request .csv or .json file")
    validate.add_argument("--evaluation-dir", required=True, help="Directory created by 'evaluate'")
    validate.add_argument("--request-id", required=True, help="Request ID to validate")
    validate.add_argument(
        "--selection-source", choices=("evaluator", "search"), default="evaluator",
        help="Load the selected composition from compositions.csv or search_results.csv",
    )
    validate.add_argument(
        "--search-method",
        help="Search method whose selected composition to validate; required with --selection-source search",
    )
    validate.add_argument(
        "--candidate-id",
        help="Explicit evaluator-selected candidate ID; mutually exclusive with search selection",
    )
    validate.add_argument("--cohort-file", required=True, help="Immutable raw-ID cohort source JSON")
    validate.add_argument("--predictions-file", help=argparse.SUPPRESS)
    validate.add_argument("--output-dir", required=True, help="Validation result directory")
    validate.add_argument("--audit-seed", type=int, default=42, help="Deterministic audit partition seed")
    validate.add_argument("--audit-samples", type=int, default=512, help="Requested held-out audit examples")
    validate.add_argument("--min-audit-samples", type=int, default=64, help="Minimum strict audit examples")
    validate.add_argument("--max-parallelism", type=int, help=argparse.SUPPRESS)
    validate.add_argument("--backfill-missing", action="store_true", help=argparse.SUPPRESS)
    validate.add_argument("--stability-seeds", type=int, default=0, help=argparse.SUPPRESS)
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args, argv=list(argv) if argv is not None else list(sys.argv))


def _generate_requests(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    services = load_services_from_db(
        args.db,
        include_legacy_fedavg_partitions=args.include_legacy_fedavg_partitions,
    )
    requests = generate_service_requests(
        services,
        request_count=args.requests,
        seed=args.seed,
        profile_id=args.profile,
        principle_id=args.principle,
        compatibility_mode=args.compatibility_mode,
    )
    output = write_request_file(requests, args.output, overwrite=args.overwrite)
    print(f"Wrote {len(requests)} immutable requests to {output}")


def _evaluate(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    services = load_services_from_db(
        args.db,
        include_legacy_fedavg_partitions=args.include_legacy_fedavg_partitions,
    )
    requests = read_request_file(args.requests_file)
    compositions = generate_compositions(services, requests, candidates_per_request=args.candidates_per_request, seed=args.seed)
    references = generate_exhaustive_references(services, requests, max_candidate_space=args.reference_max_candidates)
    search_results = run_search_algorithms(services, requests, budget=args.search_budget, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        output_dir / "services.csv",
        output_dir / "requests.csv",
        output_dir / "compositions.csv",
        output_dir / "references.csv",
        output_dir / "search_results.csv",
    ]
    for frame, path in zip((services, requests, compositions, references, search_results), output_files):
        frame.to_csv(path, index=False)
    manifest = build_run_manifest(
        db_path=args.db,
        request_path=args.requests_file,
        output_dir=output_dir,
        output_files=output_files,
        seed=args.seed,
        command=argv,
        request_schema_version=str(requests["request_schema_version"].iloc[0]),
    )
    manifest_path = write_run_manifest(manifest, output_dir)
    print(f"Wrote composition experiment outputs to {output_dir}")
    print(f"Wrote run manifest to {manifest_path}")


def _validate(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    retired = []
    if args.predictions_file is not None:
        retired.append("--predictions-file")
    if args.max_parallelism is not None:
        retired.append("--max-parallelism")
    if args.backfill_missing:
        retired.append("--backfill-missing")
    if int(args.stability_seeds) != 0:
        retired.append("--stability-seeds")
    if retired:
        raise ValueError(
            ", ".join(retired)
            + " cannot be used for FedAvg parameter validation; exact stored client "
            "checkpoints are aggregated and one materialized model is executed"
        )
    requests = read_request_file(args.requests_file)
    results, summary = validate_from_files(
        db_path=args.db,
        requests=requests,
        evaluation_dir=args.evaluation_dir,
        request_id=args.request_id,
        output_dir=args.output_dir,
        cohort_file=args.cohort_file,
        audit_seed=args.audit_seed,
        audit_samples=args.audit_samples,
        min_audit_samples=args.min_audit_samples,
        selection_source=args.selection_source,
        search_method=args.search_method,
        candidate_id=args.candidate_id,
    )
    print(f"Wrote selected composition validation to {args.output_dir}")
    print(json.dumps(summary, sort_keys=True, default=str))
