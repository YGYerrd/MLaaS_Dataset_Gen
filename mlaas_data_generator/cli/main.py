from __future__ import annotations

import argparse

from .manifest.cmd_hf_manifest import register_hf_manifest
from .run_manifest import run_manifest
from .shared_parent_commands import register_shared_parent_commands


def _register_run_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run-manifest", help="Execute reviewed service manifest rows")
    p.add_argument("--file", required=True)
    p.add_argument("--sheet", default="services")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    p.add_argument("--db", "--db-path", dest="db_path", default=None)
    p.add_argument("--log-mode", choices=("concise", "debug"), default=None, help="Terminal output detail (default: concise)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--rerun-existing", action="store_true")
    p.add_argument("--no-save-weights", action="store_true", help="Independent debugging: train/evaluate without saving weight checkpoints")
    p.add_argument("--smoke-test", action="store_true", help="Skip perturbation and update signatures; use separate smoke-test service IDs")
    p.add_argument("--no-grouped-hf", dest="grouped_hf", action="store_false")
    p.set_defaults(grouped_hf=True)

    def _run(args: argparse.Namespace) -> None:
        run_manifest(
            file=args.file,
            sheet=args.sheet,
            dry_run=args.dry_run,
            db_path=args.db_path,
            grouped_hf=args.grouped_hf,
            workers=args.workers,
            rerun_existing=args.rerun_existing,
            log_mode=args.log_mode,
            no_save_weights=args.no_save_weights,
            smoke_test=args.smoke_test,
        )

    p.set_defaults(_handler=_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MLaaS service records from reviewed manifests")
    subparsers = parser.add_subparsers(dest="command")
    register_hf_manifest(subparsers)
    _register_run_manifest(subparsers)
    register_shared_parent_commands(subparsers)
    capabilities = subparsers.add_parser("task-capabilities", help="Report offline task/backend capabilities as JSON")
    capabilities.add_argument("--task", default=None)
    capabilities.add_argument("--backend", default=None)
    def report(args):
        import json
        from ..task_capabilities import capability_matrix
        rows = [row for row in capability_matrix() if (args.task is None or row["task"] == args.task)
                and (args.backend is None or row["backend"] == args.backend)]
        print(json.dumps(rows, indent=2))
    capabilities.set_defaults(_handler=report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "_handler"):
        parser.print_help()
        return
    args._handler(args)


if __name__ == "__main__":
    main()
