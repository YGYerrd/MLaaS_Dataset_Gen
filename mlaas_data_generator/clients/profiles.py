"""Distribution profiles for newly generated clients; quantities are independent."""

PROFILES = ("iid", "dirichlet", "shard")
ROSTER_VERSION = "distribution_profiles_v2"


def current_profiles(profiles):
    """Use an existing parent's capabilities without emitting retired profiles."""
    from .legacy_profiles import UNIFORM_PROFILE

    result = tuple(dict.fromkeys("iid" if profile == UNIFORM_PROFILE else profile for profile in profiles))
    if not result or any(profile not in PROFILES for profile in result):
        raise ValueError("unsupported client partition profiles")
    return result
