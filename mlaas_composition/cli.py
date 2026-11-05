"""Command-line entry points for frozen composition experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .catalogue import load_services_from_db
from .manifest import generate_composition_manifest
from .protocol import (
    DEFAULT_PRINCIPLE_ID,
    DEFAULT_PROFILE_ID,
    build_run_manifest,
    read_request_file,
    write_request_file,
    write_run_manifest,
    sha256_file,
)
from .reference import DEFAULT_MAX_CANDIDATE_SPACE, generate_exhaustive_references
from .requirements import generate_service_requests
from .sampler import generate_compositions
from .search import run_search_algorithms
from .validation_cli import validate_from_files, validate_search_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate composition manifests or requests, or evaluate a frozen collaborative-composition experiment"
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    manifest = subparsers.add_parser("generate-manifest", help="Sample reproducible compositions from prepared family pools")
    manifest.add_argument("--prepared-dir", required=True, help="Directory containing pools.csv, services.csv and services.db")
    manifest.add_argument("--compositions", type=_positive_count, required=True, help="Approximate total, rounded to full pool/size batches")
    manifest.add_argument("--seed", type=int, default=42, help="Composition sampling seed")
    manifest.add_argument("--output", required=True, help="Output .csv; a companion .metadata.json is also written")
    manifest.add_argument("--overwrite", action="store_true", help="Explicitly replace existing manifest output files")
    manifest.set_defaults(handler=_generate_manifest)

    run = subparsers.add_parser("run-manifest", help="Retrain and measure every composition, one job per GPU")
    run.add_argument("--file", required=True, help="Composition CSV created by generate-manifest")
    run.add_argument("--source-db", required=True, help="Read-only source service catalogue")
    run.add_argument("--db", required=True, help="Separate composition results SQLite database")
    run.add_argument("--device", choices=("auto", "cpu"), default="auto")
    run.add_argument("--workers", type=_positive_count, help="Maximum GPU workers; defaults to every visible GPU")
    run.add_argument("--dry-run", action="store_true", help="Verify metadata without training or writing results")
    run.add_argument("--path-map", help="Artifact relocation JSON; defaults to path_map.json beside --source-db")
    run.set_defaults(handler=_run_manifest)

    generate = subparsers.add_parser("generate-requests", help="Generate and save versioned synthetic requests")
    generate.add_argument("--db", required=True, help="SQLite database produced by mlaas_data_generator")
    generate.add_argument("--output", required=True, help="Immutable request .csv or .json output")
    generate.add_argument("--requests", type=int, default=25, help="Number of controlled synthetic requests")
    generate.add_argument("--seed", type=int, default=42, help="Request-generation seed")
    generate.add_argument("--profile", default=DEFAULT_PROFILE_ID, help="Checked-in request profile identifier")
    generate.add_argument("--family-id", help="Limit request generation to one family ID")
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
    validate.add_argument("--cohort-file", help="Immutable cohort JSON; inferred from raw-source family when omitted")
    validate.add_argument("--device", default="cpu", help="Recipe reconstruction and aggregate evaluation device (cpu, cuda, auto)")
    validate.add_argument("--predictions-file", help=argparse.SUPPRESS)
    validate.add_argument("--output-dir", required=True, help="Validation result directory")
    validate.add_argument("--audit-seed", type=int, default=42, help="Deterministic audit partition seed")
    validate.add_argument("--audit-samples", type=int, default=512, help="Requested held-out audit examples")
    validate.add_argument("--min-audit-samples", type=int, default=64, help="Minimum strict audit examples")
    validate.add_argument("--max-parallelism", type=int, help=argparse.SUPPRESS)
    validate.add_argument("--backfill-missing", action="store_true", help=argparse.SUPPRESS)
    validate.add_argument("--stability-seeds", type=int, default=0, help=argparse.SUPPRESS)
    validate.set_defaults(handler=_validate)

    families = subparsers.add_parser("list-families", help="Show verified raw-source families and eligibility failures")
    families.add_argument("--db", required=True)
    families.add_argument("--output", help="Optional CSV with service eligibility details")
    families.set_defaults(handler=_list_families)

    batch = subparsers.add_parser("validate-search", help="Reconstruct and measure every algorithm's saved selection")
    batch.add_argument("--db", required=True)
    batch.add_argument("--requests-file", required=True)
    batch.add_argument("--evaluation-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--request-id", help="Validate only this request; default is every request")
    batch.add_argument("--cohort-file", help="Optional override; raw-source families supply their own reservation")
    batch.add_argument("--device", default="cpu")
    batch.add_argument("--min-audit-samples", type=int, default=64)
    batch.add_argument("--audit-seed", type=int, default=42)
    batch.add_argument("--audit-samples", type=int, default=512)
    batch.set_defaults(handler=_validate_search)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args, argv=list(argv) if argv is not None else list(sys.argv))


def _run_manifest(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    from .run_manifest import run_composition_manifest
    summary = run_composition_manifest(args.file, args.source_db, args.db,
        device=args.device, workers=args.workers, dry_run=args.dry_run, path_map=args.path_map)
    print(json.dumps(summary, indent=2))
    if summary.get("failed"):
        raise SystemExit(1)


def _positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def _generate_manifest(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    metadata = generate_composition_manifest(
        args.prepared_dir, args.output, compositions=args.compositions,
        seed=args.seed, overwrite=args.overwrite,
    )
    print(f"Wrote {metadata['actual_count']} compositions (requested approximately {metadata['requested_count']}; "
          f"quota {metadata['per_pair_quota']} per eligible pool/size pair) to {args.output}")
    print(f"Metadata: {Path(args.output).with_suffix('.metadata.json')}")
    if metadata["skipped_pairs"] or metadata["shortfalls"]:
        print(f"Skipped pairs: {len(metadata['skipped_pairs'])}; pairs with shortfalls: {len(metadata['shortfalls'])}")


def _generate_requests(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    services = load_services_from_db(
        args.db,
        include_legacy_fedavg_partitions=args.include_legacy_fedavg_partitions,
    )
    if args.family_id:
        services = services[services.family_id.eq(args.family_id)]
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
    evidence = {}
    if str(requests.request_schema_version.iloc[0]) == "6.0.0":
        for row in services[services.composition_compatible.eq(1)].to_dict("records"):
            artifacts = []
            for field in ("family_document_ref", "client_recipe_ref", "update_signature_path"):
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    continue
                path = Path(value).resolve()
                artifacts.append({"field": field, "path": str(path),
                                  "sha256": sha256_file(path) if path.is_file() else None})
            evidence[row["service_id"]] = artifacts
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
    if str(requests.request_schema_version.iloc[0]) == "6.0.0":
        for artifacts in evidence.values():
            for artifact in artifacts:
                path = Path(artifact["path"])
                if (sha256_file(path) if path.is_file() else None) != artifact["sha256"]:
                    raise ValueError("composition evidence changed during search")
        evidence_path = output_dir / "evidence_hashes.json"
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        output_files.append(evidence_path)
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
            + " cannot be used for parameter validation; selected models are "
            "materialized from checkpoints or recipes and aggregated before execution"
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
        device=args.device,
    )
    print(f"Wrote selected composition validation to {args.output_dir}")
    print(json.dumps(summary, sort_keys=True, default=str))
    if summary["validation_status"] != "completed":
        raise SystemExit(1)


def _list_families(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    from .raw_family import RAW_FAMILY_VERSIONS
    services = load_services_from_db(args.db)
    raw = services[services.family_contract_version.isin(RAW_FAMILY_VERSIONS)]
    for family_id, frame in raw.groupby("family_id", sort=True):
        print(json.dumps({"family_id": family_id, "completed_services": len(frame),
            "eligible_services": int(frame.composition_compatible.eq(1).sum()),
            "task_family": sorted(frame.task_family.unique()),
            "exclusion_reasons": frame.loc[frame.composition_compatible.ne(1), "composition_eligibility_reason"].value_counts().to_dict()}))
    if raw.empty:
        print("No completed raw-source family services found.")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(destination, index=False)


def _validate_search(args: argparse.Namespace, *, argv: Sequence[str]) -> None:
    _, summary = validate_search_from_files(db_path=args.db, requests=read_request_file(args.requests_file),
        evaluation_dir=args.evaluation_dir, output_dir=args.output_dir, request_id=args.request_id,
        cohort_file=args.cohort_file, device=args.device, min_audit_samples=args.min_audit_samples,
        audit_seed=args.audit_seed, audit_samples=args.audit_samples)
    print(json.dumps(summary, sort_keys=True))
    if summary["failed_count"]:
        raise SystemExit(1)
