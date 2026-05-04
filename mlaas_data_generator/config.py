"""Configuration for the manifest-only MLaaS service dataset generator."""

from __future__ import annotations

import os
from pathlib import Path

BASE_OUTPUT_DIR = Path(os.getenv("MLAAS_OUTDIR") or "outputs")
DEFAULT_MANIFEST_PATH = BASE_OUTPUT_DIR / "service_manifest.xlsx"
MANIFEST_RESULTS_PATH = BASE_OUTPUT_DIR / "service_manifest_results.csv"
FAILED_MANIFEST_PATH = BASE_OUTPUT_DIR / "service_manifest_failed.csv"
FAILURE_LOG_PATH = BASE_OUTPUT_DIR / "service_failures.log"

SQL_DB_PATH = Path(os.getenv("MLAAS_SQL_DB_PATH") or os.getenv("MLAAS_DB_PATH") or BASE_OUTPUT_DIR / "services2.db")

CONFIG = {
    "db_path": str(SQL_DB_PATH),
    "seed": 42,
    "batch_size": 32,
    "learning_rate": 0.001,
    "training_epochs": 1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "momentum": 0.0,
    "warmup_ratio": 0.0,
    "gradient_accumulation_steps": 1,
    "mixed_precision": False,
    "precision_type": "fp16",
    "sample_seed": None,
    "hidden_layers": [64],
    "activation": "relu",
    "device": "auto",
    "sample_size": None,
    "max_samples": 200,
    "dataset_args": None,
    "measure_system_metrics": True,
    "explainability_enabled": False,
    "enable_perturbation_metrics": False,
    "perturbation_stage_logging": True,
    "perturbation_progress_logging": False,
    "perturbation_progress_sample_interval": 1,
    "perturbation_sample_count": 1,
    "perturbation_candidate_units": 4,
    "perturbation_target_units": 1,
    "perturbation_trust_trials": 2,
    "perturbation_random_strength": 0.02,
    "explainability_random_trials": 8,
    "explainability_budget_fractions": [0.1, 0.2, 0.3],
    "explainability_meaningful_drop_threshold": 0.2,
    "explainability_selectivity_floor": 0.5,
    "save_weights": False,
    "train_progress_log_interval": 10,
    "update_signature_enabled": True,
    "update_signature_dim": 256,
    "update_signature_dir": None,
    "update_signature_max_source_elements": None,
    "perturbation_max_duration_s": 30,
    "perturbation_detection_max_duration_s": 15,
    "perturbation_detection_candidate_units_cap": 2,
    "perturbation_detection_budget_count_cap": 1,
    "perturbation_detection_random_trials_cap": 2,
    "perturbation_detection_trust_trials_cap": 1,
}
