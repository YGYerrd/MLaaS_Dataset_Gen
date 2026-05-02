from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HF_TOKEN_FILE = Path(__file__).resolve().parent.parent / ".hf_token"


def resolve_hf_token_file() -> Path:
    configured = os.getenv("MLAAS_HF_TOKEN_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_HF_TOKEN_FILE


def load_hf_token_from_file(*, quiet: bool = False) -> str | None:
    if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"):
        return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    token_path = resolve_hf_token_file()
    if not token_path.exists():
        return None

    token = _parse_token_file(token_path)
    if not token:
        return None

    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    if not quiet:
        print(f"Loaded Hugging Face token from {token_path}")
    return token


def _parse_token_file(token_path: Path) -> str | None:
    for raw_line in token_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                continue
            line = value.strip()
        return line.strip().strip("\"'")
    return None
