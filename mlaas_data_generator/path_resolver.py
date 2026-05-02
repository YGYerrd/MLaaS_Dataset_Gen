"""Path resolver logic for saving outputs"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Dict
from .config import RUNS_DIR, MERGED_DIR

def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

def default_filename(stem: str, ext: str = ".csv") -> str:
    return f"{stem}{ext}"

def resolve_output_path(
    filename: str = "clients",
    kind: Literal["run", "merged"] = "run",) -> Path:
    ensure_dirs()
    base = RUNS_DIR if kind == "run" else MERGED_DIR
    fname = Path(filename).name
    if not fname.lower().endswith(".csv"):
        fname += ".csv"
    return base / fname

def resolve_output_stem(
    filename: str = "clients",
    kind: Literal["run", "merged"] = "run",
) -> Path:
    """Return the base path (without suffix) for multi-table outputs."""
    ensure_dirs()
    base = RUNS_DIR if kind == "run" else MERGED_DIR
    fname = Path(filename).name
    if fname.lower().endswith(".csv"):
        fname = Path(fname).stem
    return base / fname

def resolve_table_output_paths(
    filename: str = "clients",
    kind: Literal["run", "merged"] = "run",
) -> Dict[str, Path]:
    """Return file paths for the runs/rounds/client_rounds tables."""
    stem_path = resolve_output_stem(filename, kind=kind)
    stem_name = stem_path.name
    return {
        "runs": stem_path.with_name(f"{stem_name}_runs.csv"),
        "rounds": stem_path.with_name(f"{stem_name}_rounds.csv"),
        "client_rounds": stem_path.with_name(f"{stem_name}_client_rounds.csv"),
    }

