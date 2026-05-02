from __future__ import annotations

import argparse

from .manifest.cmd_hf_manifest import register_hf_manifest
from .cmd_export_compositions import register_export_compositions
from .run_manifest import run_manifest


def _register_run_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("run-manifest", help="Execute reviewed service manifest rows")
    p.add_argument("--file", required=True)
    p.add_argument("--sheet", default="services")
    p.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    p.add_argument("--db", "--db-path", dest="db_path", default=None)
    p.add_argument("--workers", type=int, default=1)
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
        )

    p.set_defaults(_handler=_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MLaaS service records from reviewed manifests")
    subparsers = parser.add_subparsers(dest="command")
    register_hf_manifest(subparsers)
    register_export_compositions(subparsers)
    _register_run_manifest(subparsers)
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
