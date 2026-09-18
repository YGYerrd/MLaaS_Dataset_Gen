"""Configuration for the manifest-only MLaaS service dataset generator."""

from __future__ import annotations

import os
from pathlib import Path

BASE_OUTPUT_DIR = Path(os.getenv("MLAAS_OUTDIR") or "outputs")
DEFAULT_MANIFEST_PATH = BASE_OUTPUT_DIR / "service_manifest.xlsx"
MANIFEST_RESULTS_PATH = BASE_OUTPUT_DIR / "service_manifest_results.csv"
FAILED_MANIFEST_PATH = BASE_OUTPUT_DIR / "service_manifest_failed.csv"
FAILURE_LOG_PATH = BASE_OUTPUT_DIR / "service_failures.log"

SQL_DB_PATH = Path(os.getenv("MLAAS_SQL_DB_PATH") or os.getenv("MLAAS_DB_PATH") or BASE_OUTPUT_DIR / "services.db")

# Uniform architecture pool for new independent CNN families. Draw once per
# model/dataset contract; clients in that family share the selected shape.
CNN_HIDDEN_LAYER_POOL = ((32,), (64,), (128,), (256,))

CONFIG = {
    "db_path": str(SQL_DB_PATH),
    "log_mode": os.getenv("MLAAS_LOG_MODE", "concise"),
    "seed": 42,
    "benchmark_sample_size": 1024,
    "evaluation_seed": 42,
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
    "dataset_args": None,
    "measure_system_metrics": True,
    "explainability_enabled": False,
    "enable_perturbation_metrics": True,
    "evaluation_protocol_version": "benchmark_evaluation_v1",
    "metric_evaluation_profile": "benchmark",
    "perturbation_stage_logging": True,
    "perturbation_progress_logging": False,
    "perturbation_progress_sample_interval": 1,
    # Historical budgets apply only to explicit legacy_perturbation_v1 runs.
    # Current benchmark/diagnostic/pilot/production budgets live in evaluation_protocol.py.
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
    "no_save_weights": False,
    # Resolve after model/mode selection: recipe_only for supported independent
    # services, checkpoint otherwise. Explicit policies always take precedence.
    "weight_retention": None,
    "retain_validation_checkpoint": False,
    "smoke_test": False,
    "provenance_artifact_dir": None,
    "checkpoint_patch_compression": "zlib",
    "checkpoint_patch_operation_policy": "fixed_xor",
    "checkpoint_measure_full": False,
    "checkpoint_weight_format": "int8",
    "checkpoint_reduced_compression": "none",
    "checkpoint_reduced_compression_level": 1,
    "checkpoint_validation_samples": 256,
    "checkpoint_quality_tolerance": 0.01,
    # Level 1 keeps checkpoint persistence bounded for transformer-sized
    # states. Callers can still explicitly request a denser level.
    "checkpoint_patch_compression_level": 1,
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
