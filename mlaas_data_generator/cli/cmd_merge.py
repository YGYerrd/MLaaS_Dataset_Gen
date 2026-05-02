# cli/cmd_merge.py
from __future__ import annotations
import argparse
from ..path_resolver import resolve_output_path
from ..files import combine_data_files
from .utils import expand_inputs

def _handle(args: argparse.Namespace) -> None:
    inputs = expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No files match the provided paths")
    out_path = resolve_output_path(args.output, kind="merged")
    combined = combine_data_files(
        paths=inputs, output_path=out_path,
        id_col=args.id_col, start_id=args.start_id, dedupe=args.dedupe,
    )
    print(f"Merged {len(inputs)} files into {out_path} ({len(combined)} rows).")

def register_merge(subparsers):
    p = subparsers.add_parser("merge", help="Merge CSVs into one file")
    p.add_argument("inputs", nargs="+", help="Input CSVs or globs (e.g. data/*.csv)")
    p.add_argument("--output", type=str, default="merged.csv", help="Destination CSV")
    p.add_argument("--id-col", default="MLaaS_ID", help="Sequential ID column name")
    p.add_argument("--start-id", type=int, default=1, help="First ID value")
    p.add_argument("--dedupe", action="store_true", help="Drop duplicates before assigning IDs")
    p.set_defaults(_handler=_handle)
