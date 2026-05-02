import numpy as np
from numpy.random import Generator, default_rng

from .skew_axes import axis_supports_strategy, bucket_distribution, resolve_skew_axis


def _take(x, idx):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: _take(v, idx) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, (list, tuple)):
        idx_list = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
        return [x[i] for i in idx_list]
    return np.asarray(x, dtype=object)[idx]


def _num_samples(x):
    if isinstance(x, dict):
        if not x:
            return 0
        return int(len(next(iter(x.values()))))
    return int(len(x))


def _is_scalar_label_vector(y):
    arr = np.asarray(y, dtype=object)
    if arr.ndim != 1:
        return False
    if arr.size == 0:
        return True
    return not isinstance(arr[0], (list, tuple, dict, np.ndarray))


def _build_clients_from_indices(x, y, indices_by_client: dict):
    return {cid: {"x": _take(x, idx), "y": _take(y, idx)} for cid, idx in indices_by_client.items()}


def _seed(rng):
    return rng if isinstance(rng, Generator) else default_rng()


def _resize_clients_to_sample_size(clients, x, y, sample_size, rng=None):
    if sample_size is None:
        return clients
    target = max(0, int(sample_size))
    seed = _seed(rng)
    global_n = _num_samples(x)
    resized = {}
    for cid, data in clients.items():
        source_x = data.get("x")
        source_y = data.get("y")
        source_n = _num_samples(source_x)
        if source_n == 0 and global_n > 0:
            source_x, source_y, source_n = x, y, global_n
        if target == 0 or source_n == 0:
            idx = np.asarray([], dtype=int)
        elif source_n == target:
            idx = np.arange(source_n, dtype=int)
        else:
            idx = seed.choice(source_n, size=target, replace=source_n < target)
        resized[cid] = {"x": _take(source_x, idx), "y": _take(source_y, idx)}
    return resized


def _fix_counts(counts: np.ndarray, target_total: int, *, scores: np.ndarray | None = None) -> np.ndarray:
    diff = int(target_total - counts.sum())
    if diff == 0 or len(counts) == 0:
        return counts
    order = np.arange(len(counts), dtype=int)
    if scores is not None:
        order = np.argsort(scores)[::-1]
    for i in range(abs(diff)):
        idx = int(order[i % len(order)])
        counts[idx] += 1 if diff > 0 else -1
    return counts


def _split_iid(x, y, num_clients, rng=None):
    seed = _seed(rng)
    idx = seed.permutation(_num_samples(x))
    splits = np.array_split(idx, num_clients)
    return _build_clients_from_indices(x, y, {f"client_{i+1}": split for i, split in enumerate(splits)})


def _split_quantity_skew(x, y, num_clients, alpha, rng=None, bucket_ids=None):
    n = _num_samples(x)
    seed = _seed(rng)
    proportions = seed.dirichlet([alpha] * num_clients)
    counts = _fix_counts((proportions * n).astype(int), n)
    by_client = {f"client_{i+1}": [] for i in range(num_clients)}
    if bucket_ids is None:
        idx = seed.permutation(n)
        start = 0
        for i, count in enumerate(counts):
            end = start + int(count)
            by_client[f"client_{i+1}"] = idx[start:end]
            start = end
        return _build_clients_from_indices(x, y, by_client)

    bucket_arr = np.asarray(bucket_ids, dtype=np.int64).reshape(-1)
    size_mix = counts.astype(np.float64) / max(1, counts.sum())
    for bucket in np.unique(bucket_arr):
        bucket_idx = np.where(bucket_arr == bucket)[0]
        seed.shuffle(bucket_idx)
        raw = size_mix * float(len(bucket_idx))
        bucket_counts = _fix_counts(np.floor(raw).astype(int), len(bucket_idx), scores=raw - np.floor(raw))
        start = 0
        for i, count in enumerate(bucket_counts):
            end = start + max(int(count), 0)
            if end > start:
                by_client[f"client_{i+1}"].extend(bucket_idx[start:end].tolist())
            start = end
    return _build_clients_from_indices(x, y, {cid: np.asarray(idxs, dtype=int) for cid, idxs in by_client.items()})


def _split_dirichlet_label_skew(x, y, num_clients, alpha, rng=None):
    seed = _seed(rng)
    labels = np.asarray(y).reshape(-1)
    unique_labels = np.unique(labels)
    by_client = {f"client_{i+1}": [] for i in range(num_clients)}
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        seed.shuffle(idxs)
        raw = seed.dirichlet([alpha] * num_clients) * float(len(idxs))
        counts = _fix_counts(np.floor(raw).astype(int), len(idxs), scores=raw - np.floor(raw))
        start = 0
        for i, count in enumerate(counts):
            end = start + max(int(count), 0)
            if end > start:
                by_client[f"client_{i+1}"].extend(idxs[start:end].tolist())
            start = end
    return _build_clients_from_indices(x, y, {cid: np.asarray(idxs, dtype=int) for cid, idxs in by_client.items()})


def _split_shard_based(x, y, num_clients, shards_per_client, rng=None):
    seed = _seed(rng)
    labels = np.asarray(y).reshape(-1)
    num_shards = num_clients * shards_per_client
    idx_sorted = np.argsort(labels, kind="stable")
    shards = np.array_split(idx_sorted, num_shards)
    seed.shuffle(shards)
    by_client = {}
    for i in range(num_clients):
        by_client[f"client_{i+1}"] = np.concatenate(shards[i * shards_per_client : (i + 1) * shards_per_client])
    return _build_clients_from_indices(x, y, by_client)


def _split_label_per_client(x, y, num_clients, k, rng=None):
    seed = _seed(rng)
    labels = np.asarray(y).reshape(-1)
    unique_labels = np.unique(labels)
    by_label = {label: np.where(labels == label)[0] for label in unique_labels}
    clients_labels = {i: seed.choice(unique_labels, k, replace=False) for i in range(num_clients)}
    by_client = {f"client_{i+1}": [] for i in range(num_clients)}
    for label, idxs in by_label.items():
        recipients = [cid for cid, allowed in clients_labels.items() if label in allowed]
        if not recipients:
            continue
        seed.shuffle(idxs)
        splits = np.array_split(idxs, len(recipients))
        for cid, split in zip(recipients, splits):
            if len(split) > 0:
                by_client[f"client_{cid+1}"].extend(split.tolist())
    return _build_clients_from_indices(x, y, {cid: np.asarray(idxs, dtype=int) for cid, idxs in by_client.items()})


def _split_custom_data(x, y, client_distributions: dict, rng=None):
    seed = _seed(rng)
    labels = np.asarray(y).reshape(-1)
    unique_labels = np.unique(labels)
    pools = {}
    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        seed.shuffle(idxs)
        pools[int(label)] = idxs
    by_client = {cid: [] for cid in client_distributions.keys()}
    for cid, dist in client_distributions.items():
        for label_raw, count in dist.items():
            label = int(label_raw)
            pool = pools.get(label)
            if pool is None or len(pool) == 0 or count <= 0:
                continue
            take = min(int(count), len(pool))
            chosen, pools[label] = pool[:take], pool[take:]
            by_client[cid].extend(chosen.tolist())
    return _build_clients_from_indices(x, y, {cid: np.asarray(idxs, dtype=int) for cid, idxs in by_client.items()})


def _shrink_dataset(x, y, sample_size=None, sample_frac=None, rng=None):
    seed = _seed(rng)
    n = _num_samples(x)
    if sample_size is None and sample_frac is None:
        return x, y
    requested_size = None if sample_size is None else int(sample_size)
    if sample_frac is not None:
        frac_size = int(round(n * float(sample_frac)))
        sample_size = frac_size if requested_size is None else min(requested_size, frac_size)
    sample_size = max(0, min(n, int(sample_size)))
    idx = seed.choice(n, size=sample_size, replace=False)
    return _take(x, idx), (_take(y, idx) if isinstance(y, np.ndarray) else [y[i] for i in idx])


def split_data(
    x,
    y,
    num_clients,
    strategy="iid",
    distribution_param=None,
    custom_distributions=None,
    sample_size=None,
    sample_frac=None,
    rng=None,
    *,
    meta=None,
    task_family=None,
    hf_task=None,
    skew_axis=None,
    skew_axis_config=None,
):
    strategy = str(strategy or "iid").lower()
    num_clients = int(num_clients)
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")

    resolved = {
        "requested_strategy": strategy,
        "strategy": strategy,
        "distribution_param": None,
        "requested_skew_axis": skew_axis,
    }
    if sample_size is not None or sample_frac is not None:
        requested_per_client = None if sample_size is None else int(sample_size)
        effective_sample_size = None
        effective_sample_frac = sample_frac
        if requested_per_client is not None:
            effective_sample_size = max(0, requested_per_client) * num_clients
            effective_sample_frac = None
            resolved["requested_sample_size_per_client"] = requested_per_client
            resolved["effective_sample_size_total"] = effective_sample_size
            if sample_frac is not None:
                resolved["ignored_sample_frac"] = sample_frac
        x, y = _shrink_dataset(x=x, y=y, sample_frac=effective_sample_frac, sample_size=effective_sample_size, rng=rng)

    axis = resolve_skew_axis(
        x,
        y,
        meta or {},
        split_name="train",
        task_family=task_family,
        hf_task=hf_task,
        requested_axis=skew_axis,
        axis_config=skew_axis_config,
    )
    resolved.update(
        {
            "skew_axis": axis.effective_axis,
            "axis_family": axis.axis_family,
            "bucket_spec": axis.bucket_spec,
            "source_fields": axis.source_fields,
            "bucket_distribution": bucket_distribution(axis.bucket_ids, axis.bucket_labels),
        }
    )
    if axis.fallback_reason:
        resolved["axis_fallback_reason"] = axis.fallback_reason

    if strategy == "iid":
        clients = _split_iid(x, y, num_clients, rng=rng)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    if strategy == "quantity_skew":
        alpha = float(distribution_param) if distribution_param is not None else 1.0
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for quantity_skew.")
        resolved["distribution_param"] = alpha
        clients = _split_quantity_skew(x, y, num_clients, alpha, rng=rng, bucket_ids=axis.bucket_ids)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    if strategy == "custom":
        from .distributions import prepare_client_distributions

        if not custom_distributions:
            raise ValueError("custom_distributions must be provided for 'custom' strategy.'")
        if not _is_scalar_label_vector(y):
            raise ValueError("custom strategy requires scalar class labels.")
        adjusted = prepare_client_distributions(custom_distributions, num_clients)
        clients = _split_custom_data(x, y, adjusted, rng=rng)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    compatible, reason = axis_supports_strategy(axis, strategy)
    if not compatible:
        if axis.bucket_ids is None:
            resolved["strategy"] = "iid"
            resolved["fallback_reason"] = reason or f"strategy='{strategy}' requires a resolvable skew axis"
            clients = _split_iid(x, y, num_clients, rng=rng)
            return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved
        raise ValueError(reason or f"strategy '{strategy}' is not compatible with skew axis '{axis.effective_axis}'")

    if strategy == "dirichlet":
        alpha = float(distribution_param) if distribution_param is not None else 0.5
        if alpha <= 0:
            raise ValueError("alpha must be > 0 for dirichlet.")
        resolved["distribution_param"] = alpha
        clients = _split_dirichlet_label_skew(x, axis.bucket_ids, num_clients, alpha, rng=rng)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    if strategy == "shard":
        shards_per_client = int(distribution_param) if distribution_param is not None else 2
        if shards_per_client <= 0:
            raise ValueError("shards_per_client must be > 0 for shard.")
        resolved["distribution_param"] = shards_per_client
        clients = _split_shard_based(x, axis.bucket_ids, num_clients, shards_per_client, rng=rng)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    if strategy == "label_per_client":
        k = int(distribution_param) if distribution_param is not None else 1
        cardinality = int(axis.bucket_spec.get("cardinality") or len(np.unique(axis.bucket_ids)))
        if not (1 <= k <= cardinality):
            raise ValueError("k must be in [1, num_buckets] for label_per_client.")
        resolved["distribution_param"] = k
        clients = _split_label_per_client(x, axis.bucket_ids, num_clients, k, rng=rng)
        return _resize_clients_to_sample_size(clients, x, y, sample_size, rng=rng), resolved

    raise ValueError(f"Unknown data split strategy: {strategy}")
