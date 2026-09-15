"""Explicit metrics-only execution for independent generator debugging."""


def weights_disabled(config):
    return str(config.get("no_save_weights", False)).strip().lower() in {"true", "1", "yes", "on"}


def apply_debug_mode(config):
    smoke = str(config.get("smoke_test", False)).strip().lower() in {"true", "1", "yes", "on"}
    if smoke:
        config.update(smoke_test=True, enable_perturbation_metrics=False,
                      explainability_enabled=False, update_signature_enabled=False)
    if weights_disabled(config):
        config.update(no_save_weights=True, save_weights=False)
    if (smoke or weights_disabled(config)) and config.get("service_id"):
        suffix = ("__debug_no_weights" if weights_disabled(config) else "") + ("__smoke_test" if smoke else "")
        sid = str(config["service_id"])
        config["service_id"] = sid if sid.endswith(suffix) else sid + suffix
    return config


def debug_mode_error(config):
    if weights_disabled(config) and (config.get("fl_run_id") or config.get("generation_mode") in {"fedavg", "shared_parent"}):
        return "no_save_weights supports independent debugging; FedAvg and shared-parent execution require checkpoints"
    return None
