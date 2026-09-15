"""Public CLI for historical selection, parent training, and client recipes."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..shared_parent import DEFAULT_COHORT_DB, train_parents
from ..parent_selection import select_parent_configs
from ..cohort_manifest import build_cohort_manifest


def _client_options(args) -> dict:
    return {key: getattr(args, key) for key in (
        "clients_per_parent", "seed", "heterogeneity_intensity",
        "average_samples", "regression_support_policy", "min_target_bin_fraction")}


def _register_client_options(parser) -> None:
    parser.add_argument("--clients-per-parent", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heterogeneity-intensity", type=float, default=0.5)
    parser.add_argument("--regression-support-policy", choices=("general", "specialist"), default="general")
    parser.add_argument("--min-target-bin-fraction", type=float, default=0.6)
    parser.add_argument("--average-samples", "--base-client-quantity", dest="average_samples", type=int, default=600,
                        help="Average client sample count with a fixed +/-75% stratified spread")


def register_shared_parent_commands(subparsers) -> None:
    selection = subparsers.add_parser("select-parent-configs", help="Automatically select historical parent training recipes")
    selection.add_argument("--history-db", default="outputs/services.db")
    selection.add_argument("--output", required=True)
    selection.add_argument("--limit", type=int, default=5, help="Maximum parents to select per eligible family (default: 5)")

    def select(args):
        result = select_parent_configs(args.history_db, args.output, limit=args.limit)
        print(f"Selected {result['parent_count']} parents across {result['family_count']} families; "
              f"wrote execution-ready selections and informational CSV evidence to {args.output}")
    selection.set_defaults(_handler=select)

    parents = subparsers.add_parser("train-parents", help="Train full-task parents from selected historical recipes")
    parents.add_argument("--file", required=True)
    parents.add_argument("--db", default=DEFAULT_COHORT_DB)
    parents.add_argument("--output")
    parents.add_argument("--dry-run", action="store_true")
    parents.add_argument("--train-clients", action="store_true",
                         help="After parent training, save and execute clients of successful parents even if other parents fail")
    parents.add_argument("--client-manifest",
                         help="Save client recipes as CSV/XLSX; without --train-clients, stop for review. "
                              "With --train-clients, defaults to <parents output stem>.clients.csv")
    parents.add_argument("--log-mode", choices=("concise", "debug"), default=None)
    _register_client_options(parents)

    def train(args):
        client_file = None
        parent_file = Path(args.output) if args.output else Path(args.file).with_suffix(".parents.json")
        if args.train_clients or args.client_manifest:
            client_file = Path(args.client_manifest) if args.client_manifest else parent_file.with_suffix(".clients.csv")
            if client_file.suffix.lower() not in {".csv", ".xlsx"}:
                parents.error("--client-manifest must be CSV or XLSX")
            if min(args.clients_per_parent, args.average_samples) < 1:
                parents.error("client count and quantities must be positive")
            if not 0 <= args.heterogeneity_intensity <= 1:
                parents.error("--heterogeneity-intensity must be in [0,1]")
            paths = [Path(args.file), Path(args.db), parent_file, client_file]
            if len({p.resolve() for p in paths}) != len(paths):
                parents.error("selection, database, parent output, and client manifest must use distinct paths")
        result = train_parents(args.file, db=args.db, output=args.output, dry_run=args.dry_run)
        if args.dry_run:
            if client_file:
                result["client_plan"] = {"manifest": str(client_file), "train_clients": args.train_clients,
                                         "client_count": len(result["parents"]) * args.clients_per_parent,
                                         **_client_options(args)}
            print(json.dumps(result, indent=2))
        else:
            print(f"Parents: {len(result['parents'])} completed, {len(result['failures'])} failed")
            if client_file:
                if not result["parents"]:
                    if result["failures"]:
                        print("No parents completed; cannot create or execute a client manifest", flush=True)
                        raise SystemExit(1)
                    parents.error("no parents selected; cannot create a client manifest")
                if result["failures"]:
                    print("Continuing with clients of completed parents; failed parents have no clients in this run", flush=True)
                frame = build_cohort_manifest(parent_file, file=client_file, db=args.db,
                                              reuse_existing=True, **_client_options(args))
                print(f"Client manifest saved: {client_file} ({len(frame)} frozen recipes)", flush=True)
                if args.train_clients:
                    from .run_manifest import run_manifest
                    results_file = run_manifest(str(client_file), db_path=args.db, log_mode=args.log_mode)
                    outcomes = pd.read_csv(results_file)
                    if (len(outcomes) != len(frame) or "status" not in outcomes
                            or not outcomes["status"].isin(["success", "skipped"]).all()):
                        print(f"Client execution incomplete; inspect {results_file} and resume with run-manifest --file {client_file}")
                        raise SystemExit(1)
            if result["failures"]:
                print("Parent failures remain; rerun train-parents to retry them and reuse completed work", flush=True)
                raise SystemExit(1)
    parents.set_defaults(_handler=train)

    cohort = subparsers.add_parser("cohort-manifest", help="Freeze independent heterogeneous clients of completed parents")
    cohort.add_argument("--parents", required=True)
    cohort.add_argument("--file", required=True)
    _register_client_options(cohort)
    cohort.add_argument("--db", default=DEFAULT_COHORT_DB)

    def manifest(args):
        frame = build_cohort_manifest(args.parents, file=args.file, db=args.db, **_client_options(args))
        print(f"Wrote {len(frame)} frozen client recipes to {args.file}")
    cohort.set_defaults(_handler=manifest)

    diagnostics = subparsers.add_parser("cohort-diagnostics", help="Measure post-generation behavioural diversity on the frozen catalogue benchmark")
    diagnostics.add_argument("--db", default=DEFAULT_COHORT_DB)
    diagnostics.add_argument("--cohort-id", required=True)
    diagnostics.add_argument("--output-dir", required=True)
    diagnostics.add_argument("--batch-size", type=int)

    def diagnose(args):
        from ..behavioural_diagnostics import cohort_diagnostics
        report = cohort_diagnostics(db=args.db, cohort_id=args.cohort_id,
                                    output_dir=args.output_dir, batch_size=args.batch_size)
        print(f"Diagnostics {report['status']}: {report['diagnosed_clients']} clients, {report['pair_count']} pairs; {args.output_dir}")
    diagnostics.set_defaults(_handler=diagnose)

    family = subparsers.add_parser("family-diagnostics", help="Compare independent family recipes and predictions on the frozen catalogue subset")
    family.add_argument("--db", required=True)
    family.add_argument("--family-id", required=True)
    family.add_argument("--output-dir", required=True)
    family.add_argument("--batch-size", type=int)

    def diagnose_family(args):
        from ..behavioural_diagnostics import family_diagnostics
        report = family_diagnostics(db=args.db, family_id=args.family_id, output_dir=args.output_dir, batch_size=args.batch_size)
        print(f"Family diagnostics {report['status']}: {report['diagnosed_clients']} clients; {args.output_dir}")
    family.set_defaults(_handler=diagnose_family)
