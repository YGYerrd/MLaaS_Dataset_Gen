from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlaas_data_generator.config import DEFAULT_MANIFEST_PATH

from .hf_manifest_builder import MANIFEST_PROFILES, RESOURCE_TIERS, build_hf_manifest, save_manifest


def _parse_csv_arg(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    raw_values = value if isinstance(value, list) else [value]
    items: list[str] = []
    for raw_value in raw_values:
        items.extend(item.strip() for item in str(raw_value).split(",") if item.strip())
    return items or None


def _normalised_failure_keys(df: pd.DataFrame) -> pd.Series:
    parts = []
    for column in ("model_type", "hf_model_id", "dataset_name", "hf_task"):
        values = df[column] if column in df.columns else pd.Series([""] * len(df), index=df.index)
        parts.append(values.fillna("").astype(str).str.strip().str.lower())
    return parts[0] + "\t" + parts[1] + "\t" + parts[2] + "\t" + parts[3]


def drop_known_failure_rows(df: pd.DataFrame, failures_csv: str | None) -> pd.DataFrame:
    if not failures_csv:
        return df
    path = Path(failures_csv)
    if not path.exists() or df.empty:
        return df
    failures = pd.read_csv(path)
    required = {"model_type", "hf_model_id", "dataset_name", "hf_task"}
    if not required.issubset(failures.columns):
        return df
    failure_keys = set(_normalised_failure_keys(failures))
    keep = ~_normalised_failure_keys(df).isin(failure_keys)
    return df.loc[keep].reset_index(drop=True)


def register_hf_manifest(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("hf-manifest", help="Generate reviewed MLaaS service manifest rows")
    p.add_argument("--input-json", help="Optional HF audit JSON used only for metadata enrichment")
    p.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH), help="Output .csv or .xlsx path")
    p.add_argument("--sheet", default="services", help="Sheet name for xlsx output")
    p.add_argument(
        "--task-keys",
        nargs="+",
        help="Registry task keys. Accepts comma-separated values, space-separated values, or both.",
    )
    p.add_argument("--models-per-task", type=int, default=10)
    p.add_argument("--max-models-per-family", type=int, help="Optional cap on models selected from the same family within each task")
    p.add_argument("--datasets-per-model", type=int, default=1)
    p.add_argument(
        "--training-regimes",
        nargs="+",
        help="Training regimes. Accepts comma-separated values, space-separated values, or both.",
    )
    p.add_argument("--dataset-variants-per-pair", type=int, default=1)
    p.add_argument("--split-variants-per-pair", type=int, default=1)
    p.add_argument("--knob-variants-per-pair", type=int, default=1)
    p.add_argument("--total-services", type=int, help="Total service rows to emit")
    p.add_argument("--manifest-profile", choices=sorted(MANIFEST_PROFILES), default="balanced")
    p.add_argument("--resource-tier", choices=sorted(RESOURCE_TIERS), help="Workload budget. Defaults from --manifest-profile.")
    p.add_argument("--avg-sample-size", type=int, help="Target average max_samples across emitted service rows")
    p.add_argument("--exclude-failures-csv", help="Drop manifest rows matching known failures from this CSV")
    p.add_argument("--seed", type=int, default=42)

    def _run(args: argparse.Namespace) -> None:
        df = build_hf_manifest(
            json_path=args.input_json,
            task_keys=_parse_csv_arg(args.task_keys),
            models_per_task=args.models_per_task,
            datasets_per_model=args.datasets_per_model,
            training_regimes=_parse_csv_arg(args.training_regimes),
            dataset_variants_per_pair=args.dataset_variants_per_pair,
            split_variants_per_pair=args.split_variants_per_pair,
            knob_variants_per_pair=args.knob_variants_per_pair,
            total_services=args.total_services,
            seed=args.seed,
            manifest_profile=args.manifest_profile,
            resource_tier=args.resource_tier,
            avg_sample_size=args.avg_sample_size,
            max_models_per_family=args.max_models_per_family,
        )
        before_filter = len(df)
        df = drop_known_failure_rows(df, args.exclude_failures_csv)
        output_path = Path(args.output)
        save_manifest(df, output_path, sheet_name=args.sheet)
        dropped = before_filter - len(df)
        suffix = f", dropped {dropped} known-failure rows" if dropped else ""
        print(f"Wrote {len(df)} service rows to {output_path}{suffix}")

    p.set_defaults(_handler=_run)
