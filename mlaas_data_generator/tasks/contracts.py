"""Serializable task declarations; runtime operations live in runtime.py."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from ..hf_tasks import normalize_hf_task
from ..provenance import sha256_json

VERSION = "task_contract_v1"
EXTENDED_TASKS = frozenset({"token_classification", "image_segmentation", "seq2seq_generation"})


def uses_task_protocol(config):
    task = normalize_hf_task(config.get("hf_task"), default="unknown")
    return task in EXTENDED_TASKS and (task != "seq2seq_generation" or
        str(config.get("task_tag") or "").lower() == "summarization")


class TaskContractError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _declared_fields(values: Mapping) -> dict:
    # Heterogeneous manifest tables fill absent scalar fields with NaN. Treat
    # those like None without inspecting or serializing runtime arrays/models.
    return {str(key): value for key, value in values.items()
            if not str(key).startswith("_") and value is not None
            and not (isinstance(value, (float, np.floating)) and np.isnan(value))}


@dataclass(frozen=True)
class TaskContract:
    task: str
    family: str
    semantics: dict[str, Any]
    sampling: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    version: str = VERSION

    @property
    def semantics_hash(self):
        return sha256_json({"version": self.version, "task": self.task, "semantics": self.semantics})

    @property
    def protocol_hash(self):
        return sha256_json({"semantics_hash": self.semantics_hash, "evaluation": self.evaluation})

    def as_dict(self):
        return plain(asdict(self)) | {"task_semantics_hash": self.semantics_hash,
                                     "evaluation_protocol_hash": self.protocol_hash}

    @classmethod
    def from_dict(cls, value):
        if value.get("version") != VERSION:
            raise TaskContractError("unknown_task_contract", "unsupported task contract version")
        result = cls(**{k: value[k] for k in ("task", "family", "semantics", "sampling", "training", "evaluation", "version")})
        if result.as_dict() != value:
            raise TaskContractError("target_semantics_mismatch", "task contract hash or fields changed")
        return result

    def supervision_count(self, targets):
        if self.family in {"classification", "regression"}:
            values = np.asarray(targets)
            if not np.isfinite(values).all():
                raise TaskContractError("invalid_targets", "targets must be finite")
            return len(values)
        count = 0
        for row in targets:
            values = np.asarray(row)
            if values.dtype.kind not in "iu":
                raise TaskContractError("invalid_targets", "structured targets must be integer IDs")
            mask = values != self.semantics["ignore_index"]
            valid = values[mask]
            if np.any(valid < 0):
                raise TaskContractError("invalid_targets", "negative supervised label ID")
            limit = self.semantics.get("vocab_size") if self.family in {"generation", "fill_mask"} else self.semantics.get("num_labels")
            if limit and np.any(valid >= limit):
                raise TaskContractError("invalid_targets", "supervised label is outside the frozen output space")
            count += int(mask.sum())
        if not count:
            raise TaskContractError("insufficient_supervision", "no supervised positions remain")
        return count


def resolve_task_contract(config: Mapping, meta: Mapping | None = None, model=None) -> TaskContract:
    from ..task_contract import resolve_sentence_pair_contract
    from ..services.taxonomy import canonical_task_family
    # Runtime configs also contain prepared datasets, checkpoints and live models.
    # Normalize only the fields selected into the contract below, rather than
    # recursively materializing every image/mask or tensor into Python lists.
    cfg = _declared_fields(config)
    loaded = _declared_fields(meta or {})
    if isinstance(cfg.get("dataset_args"), Mapping):
        cfg["dataset_args"] = dict(cfg["dataset_args"])
    for values in (cfg, loaded):
        if "label_mapping" in values:
            values["label_mapping"] = plain(values["label_mapping"])
    resolve_sentence_pair_contract(cfg, loaded)
    task = normalize_hf_task(cfg.get("hf_task") or cfg.get("task"), default="unknown", unknown="unknown")
    observed_task = normalize_hf_task(loaded.get("hf_task"), default="unknown", unknown="unknown")
    if task != "unknown" and observed_task != "unknown" and task != observed_task:
        raise TaskContractError("target_semantics_mismatch", "loaded dataset and declared HF task disagree")
    if isinstance(cfg.get("label_mapping"), dict) and isinstance(loaded.get("label_mapping"), dict) and cfg["label_mapping"] != loaded["label_mapping"]:
        raise TaskContractError("target_semantics_mismatch", "loaded and declared label mappings disagree")
    family = canonical_task_family(cfg.get("task_type") or loaded.get("task_type"), task)
    if task == "unknown" and family in {"classification", "regression"} and cfg.get("model_type") in {"mlp", "cnn", "logreg", "mobilenetv2"}:
        task = family
    expected = {"sequence_classification": "classification", "image_classification": "classification",
                "token_classification": "token_classification", "image_segmentation": "segmentation",
                "seq2seq_generation": "generation"}.get(task)
    if family == "unknown" and expected:
        family = expected
    if expected and family != expected:
        raise TaskContractError("target_semantics_mismatch", f"{task} disagrees with task family {family}")
    if loaded.get("task_type"):
        observed = canonical_task_family(loaded["task_type"], task)
        if observed != "unknown" and observed != family:
            raise TaskContractError("target_semantics_mismatch", "loaded dataset and declared task disagree")
    if task not in {"classification", "regression", "sequence_classification", "image_classification",
                    "sentence_similarity", "fill_mask", *EXTENDED_TASKS} or family == "unknown":
        raise TaskContractError("unknown_task_contract", f"no protocol implemented for {task}")
    if task == "seq2seq_generation" and str(cfg.get("task_tag") or loaded.get("task_tag") or "").lower() != "summarization":
        raise TaskContractError("unknown_task_contract", "seq2seq validation currently requires the summarization profile")
    if str(loaded.get("label_format") or cfg.get("label_format") or "").lower() in {"multilabel", "multihot"}:
        raise TaskContractError("target_semantics_mismatch", "multilabel validation is not implemented")
    core = getattr(model, "core", None)
    tokenizer = getattr(core, "tokenizer", None)
    model_config = getattr(getattr(core, "model", None), "config", None)
    ignore = int(getattr(core, "label_pad_value", loaded.get("label_pad_value", cfg.get("label_pad_value", -100))))
    names = loaded.get("label_mapping") or cfg.get("label_mapping") or getattr(model_config, "id2label", None)
    semantics = {"output_kind": {"classification": "class_probabilities", "regression": "scalar",
                 "token_classification": "aligned_token_ids", "fill_mask": "masked_token_ids",
                 "segmentation": "spatial_class_ids", "generation": "decoded_text"}[family],
                 "label_mapping": plain(names), "num_labels": loaded.get("num_labels") or loaded.get("num_classes") or cfg.get("num_labels") or cfg.get("num_classes"),
                 "ignore_index": ignore,
                 "model_revision": cfg.get("hf_model_revision"), "model_id": cfg.get("hf_model_id"),
                 "max_length": int(getattr(core, "max_length", cfg.get("max_length", 128)) or 128),
                 "input_contract": plain({k: loaded.get(k, cfg.get(k)) for k in (
                     "preprocessing", "preprocessing_identity", "column_mapping", "input_schema", "image_size",
                     "normalization", "label_format", "processor_config", "source_max_length", "target_max_length",
                     "dynamic_padding", "padding_mode", "training_augmentations", "eval_augmentations",
                     "segmentation_reduce_labels", "source_group_column") if k in loaded or k in cfg}),
                 "task_tag": cfg.get("task_tag") or loaded.get("task_tag")}
    if tokenizer is not None:
        semantics.update(tokenizer_class=type(tokenizer).__name__,
                         vocabulary_hash=sha256_json(tokenizer.get_vocab()),
                         vocab_size=len(tokenizer), special_tokens=plain(tokenizer.special_tokens_map),
                         padding_side=tokenizer.padding_side, truncation_side=tokenizer.truncation_side)
    if family == "token_classification":
        semantics["alignment"] = "first_subword_special_and_padding_ignored_v1"
    if family == "fill_mask":
        semantics["masking"] = {
            "algorithm": loaded.get("masking_policy", "source_row_mask_v1"),
            "probability": float(loaded.get("mlm_probability", cfg.get("mlm_probability", 0.15))),
            "seed": int(loaded.get("masking_seed", cfg.get("initialization_seed", 42))),
            "candidates": "attended_non_special_tokens",
            "replacement": "80_percent_mask_10_percent_random_10_percent_unchanged",
        }
    if family == "segmentation":
        semantics["target_resize"] = "nearest_to_logits_v1"
    primary, secondary = {"classification": ("accuracy", "f1_macro"), "regression": ("rmse", "mae"),
                          "token_classification": ("f1", "accuracy"), "segmentation": ("iou", "dice"),
                          "generation": ("rouge1", "rouge2"),
                          "fill_mask": ("masked_accuracy", "perplexity_proxy")}[family]
    if task == "sentence_similarity" and family == "regression":
        primary, secondary = "pearson", "spearman"
    evaluation = {"profile": f"{task}_v1", "primary": primary, "secondary": secondary,
                  "higher_is_better": primary != "rmse", "quality_transform": "inverse_one_plus" if primary == "rmse" else "correlation_unit" if primary == "pearson" else "identity",
                  "unit": "supervised_token" if family == "token_classification" else "valid_pixel" if family == "segmentation" else "example",
                  "reduction": "corpus_confusion_weighted_f1" if family == "token_classification" else "corpus_macro_iou_dice_union_present" if family == "segmentation" else "dataset",
                  "minimum_supervision": 1}
    if family == "fill_mask":
        evaluation.update(unit="masked_token", reduction="corpus_masked_tokens",
                          metric_implementation="masked_accuracy_exp_token_mean_cross_entropy_v1",
                          perplexity_loss_clip=[-50.0, 50.0])
    if family == "generation":
        if cfg.get("prompt_prefix") or cfg.get("task_prefix"):
            raise TaskContractError("unknown_task_contract", "summarisation v1 accepts source text as stored; prompt prefixes require a preprocessing profile")
        generation = dict(getattr(core, "generation_config", {}) or {})
        if cfg.get("task_contract"):
            generation.update(cfg["task_contract"]["evaluation"]["decoding"])
        else:
            generation.update(cfg.get("generation_config") or {})
        generation.update({k: cfg[k] for k in ("max_new_tokens", "num_beams", "do_sample") if cfg.get(k) is not None})
        if generation.get("do_sample"):
            raise TaskContractError("evaluation_protocol_mismatch", "summarisation v1 requires deterministic decoding")
        decoding = {"max_new_tokens": int(generation.get("max_new_tokens", 64)),
                    "num_beams": int(generation.get("num_beams", 1)), "do_sample": False}
        if decoding["max_new_tokens"] < 1 or decoding["num_beams"] < 1:
            raise TaskContractError("evaluation_protocol_mismatch", "decoding length and beam count must be positive")
        defaults = getattr(getattr(core, "model", None), "generation_config", None)
        for key, fallback in {"length_penalty": 1.0, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                              "min_length": 0, "min_new_tokens": None, "early_stopping": False,
                              "eos_token_id": None, "pad_token_id": None, "decoder_start_token_id": None,
                              "forced_bos_token_id": None, "forced_eos_token_id": None}.items():
            value = generation.get(key, getattr(defaults, key, fallback))
            if value is not None:
                decoding[key] = plain(value)
        evaluation.update(decoding=decoding,
                          metric_implementation="mlaas_decoded_rouge_f1_v1", reduction="example_mean",
                          text_normalization="lowercase_regex_word_punctuation_tokens_v1",
                          reference_policy="one_decoded_masked_truncated_target_per_source",
                          empty_output_policy="score_emitted_eos_as_empty_text; missing_output_is_error")
        semantics.update(decoder_start_token_id=getattr(model_config, "decoder_start_token_id", None),
                         eos_token_id=getattr(model_config, "eos_token_id", None),
                         pad_token_id=getattr(model_config, "pad_token_id", None),
                         source_prefix="", prepared_padding="right_with_attention_mask; target_padding_ignored")
    from ..clients.profiles import PROFILES
    profiles = ["iid"] if family in {"segmentation", "generation"} else list(PROFILES)
    if cfg.get("task_contract"):
        from ..clients.legacy_profiles import restored_task_profiles
        previous = TaskContract.from_dict(cfg["task_contract"])
        profiles = restored_task_profiles(profiles, previous.sampling["profiles"])
        # Expanding new fill-mask capabilities must not rewrite frozen recipes.
        if family == "fill_mask" and previous.sampling["profiles"] in (["iid"], ["iid", "quantity_skew"]):
            profiles = list(previous.sampling["profiles"])
    return TaskContract(task, family, semantics,
        {"unit": "image" if family == "segmentation" else "sequence" if family in {"token_classification", "generation", "fill_mask"} else "example",
         "profiles": profiles, "grouping": "declared_source_group", "aggregation_weight_unit": "training_examples"},
        {"objective": "masked_cross_entropy" if family in {"token_classification", "segmentation", "generation", "fill_mask"} else "regression" if family == "regression" else "cross_entropy",
         "budget_unit": "optimizer_updates", "minimum_successful_updates": 1}, evaluation)


def target_fingerprints(targets):
    """Bounded-memory references to targets in the immutable loaded source."""
    from ..shared_parent import prepared_data_hash
    return [{"target_sha256": prepared_data_hash(row)} for row in targets]


def validate_partition(contract, config):
    profile = config.get("split_strategy", "iid")
    if profile not in contract.sampling["profiles"]:
        raise TaskContractError("partition_axis_unavailable", f"{contract.task} does not implement {profile}")


def bind_task_contract(config, meta, model):
    contract = resolve_task_contract(config, meta, model)
    previous = config.get("task_contract")
    if previous and TaskContract.from_dict(previous).as_dict() != contract.as_dict():
        code = "target_semantics_mismatch" if previous["task_semantics_hash"] != contract.semantics_hash else "evaluation_protocol_mismatch"
        differences = []
        def compare(left, right, path):
            if isinstance(left, dict) and isinstance(right, dict):
                for key in sorted(left.keys() | right.keys()):
                    compare(left.get(key), right.get(key), f"{path}.{key}" if path else key)
            elif left != right:
                differences.append(path)
        compare(previous, contract.as_dict(), "")
        raise TaskContractError(code, "loaded task contract differs from its frozen declaration: " + ", ".join(differences))
    config["task_contract"] = contract.as_dict()
    return contract


def provenance_contract_values(row):
    """Read new semantic fields from their authoritative versioned JSON envelope."""
    import json
    result = dict(row)
    provenance = result.get("provenance_json") or {}
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    if provenance.get("task_contract"):
        contract = TaskContract.from_dict(provenance["task_contract"])
        for key, value in (("task_semantics_hash", contract.semantics_hash), ("evaluation_protocol_hash", contract.protocol_hash)):
            if result.get(key) is not None and result[key] != value:
                raise TaskContractError("target_semantics_mismatch", "provenance columns differ from task contract")
            result[key] = value
    return result
