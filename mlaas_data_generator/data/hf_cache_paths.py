import io
import os
from pathlib import Path


def _candidate_dataset_cache_roots():
    seen = set()
    roots = []

    def _add(value):
        if not value:
            return
        path = Path(value).expanduser()
        key = str(path).lower()
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    _add(os.getenv("HF_DATASETS_CACHE"))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        _add(Path(hf_home) / "datasets")
    _add(Path.home() / ".cache" / "huggingface" / "datasets")
    return roots


def resolve_hf_cache_path(path):
    """Resolve stale HF dataset-cache paths after the cache root was moved."""
    raw = Path(path).expanduser()
    if raw.exists():
        return raw

    parts = raw.parts
    try:
        marker = max(i for i, part in enumerate(parts) if part.lower() == "datasets")
    except ValueError:
        path_parts = tuple(part for part in str(path).replace("\\", "/").split("/") if part)
        try:
            marker = max(i for i, part in enumerate(path_parts) if part.lower() == "datasets")
        except ValueError:
            return raw
        parts = path_parts

    suffix = Path(*parts[marker + 1 :])
    if not suffix.parts:
        return raw

    for root in _candidate_dataset_cache_roots():
        candidate = root / suffix
        if candidate.exists():
            return candidate
    return raw


def with_hf_columns_decode_disabled(ds, *columns):
    """Return a dataset whose HF Image-like columns yield path/bytes dicts."""
    current = ds
    for column in columns:
        if not column:
            continue
        features = getattr(current, "features", None)
        feature = None
        try:
            feature = features.get(column) if features is not None else None
        except Exception:
            feature = None

        if feature is None or not hasattr(feature, "decode"):
            continue
        if getattr(feature, "decode", True) is False:
            continue

        try:
            from datasets import Image
        except Exception:
            return current

        try:
            current = current.cast_column(column, Image(decode=False))
        except Exception:
            continue
    return current


def with_hf_image_decode_disabled(ds, image_column):
    """Return a dataset whose HF Image column yields path/bytes dicts."""
    return with_hf_columns_decode_disabled(ds, image_column)


def load_image_from_path(path):
    resolved = resolve_hf_cache_path(path)
    if not resolved.exists():
        raise FileNotFoundError(str(path))
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from path requires Pillow") from e
    with Image.open(resolved) as im:
        return im.convert("RGB")


def load_image_from_bytes(data):
    try:
        from PIL import Image
    except Exception as e:
        raise ImportError("Image decoding from bytes requires Pillow") from e
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")
