"""Group-preserving reservations shared by token, image and text protocols."""
from .contracts import TaskContractError


def grouped_audit_positions(train_ids, test_ids, groups, *, count, seed):
    import numpy as np
    train, test = groups.get("train"), groups.get("test")
    if train is None or test is None or len(train) != len(train_ids) or len(test) != len(test_ids):
        raise TaskContractError("invalid_source_groups", "both group lists must align with their complete sources")
    if set(map(str, train)) & set(map(str, test)):
        raise TaskContractError("audit_group_overlap", "a source group occurs in training and evaluation")
    buckets = {}
    for i, group in enumerate(test):
        buckets.setdefault(str(group), []).append(i)
    keys = sorted(buckets)
    if len(keys) < 2:
        raise TaskContractError("insufficient_audit_support", "at least two evaluation source groups are required")
    np.random.default_rng(seed).shuffle(keys)
    positions = []
    for key in keys[:-1]:
        positions.extend(buckets[key])
        if len(positions) >= count:
            break
    return sorted(positions)
