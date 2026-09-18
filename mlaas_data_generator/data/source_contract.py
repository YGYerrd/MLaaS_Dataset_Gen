"""Source identity before client selection or expensive preparation."""
from ..provenance import sha256_json


def prepared_fingerprint(pair, meta):
    """Hash selected content, excluding batch padding that carries no input."""
    import numpy as np
    from ..shared_parent import prepared_data_hash
    x, y = pair
    if meta.get("dynamic_padding") and isinstance(x, dict) and "attention_mask" in x:
        masks = np.asarray(x["attention_mask"]).astype(bool)
        x = {key: [np.asarray(row)[mask] if np.asarray(row).shape == mask.shape else row
                   for row, mask in zip(values, masks)] for key, values in x.items()}
        y = unpadded_targets(y, meta)
    return prepared_data_hash((x, y))


def unpadded_targets(y, meta):
    import numpy as np
    if meta.get("dynamic_padding") and np.ndim(y) == 2 and np.asarray(y).dtype.kind in "iu":
        ignore = int(meta.get("ignore_index", meta.get("label_pad_value", -100)))
        result = []
        for row in y:
            row = np.asarray(row)
            valid = np.flatnonzero(row != ignore)
            result.append(row[:int(valid[-1]) + 1] if len(valid) else row[:0])
        return result
    return y


def source_contract(name, train, test, meta, args):
    from ..shared_parent import prepared_data_hash
    splits = {}
    for split, (x, y) in (("train", train), ("test", test)):
        count = len(y) if y is not None else len(x)
        # Arrow datasets already fingerprint their raw tables and row views.
        # Numeric sources have no such identity, so hash their raw bytes.
        fingerprint = getattr(x, "_fingerprint", None) or prepared_data_hash((x, y))
        splits[split] = {
            "name": meta.get(f"{split}_split") or split,
            "count": count, "fingerprint": fingerprint,
            "ordered_row_ids_hash": sha256_json(meta.get(f"selected_{split}_rows", list(range(count)))),
        }
    return {"version": "raw_source_v1", "dataset": name,
            "dataset_name": args.get("dataset_name") or meta.get("dataset_name"),
            "dataset_config": args.get("dataset_config"),
            "revision": args.get("hf_dataset_revision") or args.get("dataset_revision"),
            "row_id_scheme": "split_name:source_position", "splits": splits}
