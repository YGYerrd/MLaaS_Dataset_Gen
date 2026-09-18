"""Read compatibility for frozen partition contracts; never generate these profiles.

Historical contracts and partition hashes include the original strategy names.
Older multi-client splitters also give the retired profile different semantics,
so changing persisted names to IID would invalidate replay.
"""

UNIFORM_PROFILE = "quantity_skew"
PROFILES = ("iid", "dirichlet", "shard", UNIFORM_PROFILE)
ALIASES = {"quantity": UNIFORM_PROFILE, "quantity_skewed": UNIFORM_PROFILE}


def restored_task_profiles(current, recorded):
    """Retain the exact old capability list when rebuilding a frozen task."""
    if recorded == [*current, UNIFORM_PROFILE]:
        return list(recorded)
    return list(current)
