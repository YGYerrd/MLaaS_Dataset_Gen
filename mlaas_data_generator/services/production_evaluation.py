"""Versioned, resumable measurements of robustness and explanation faithfulness.

The independent statistical unit is an input (or an explicitly supplied group),
never an individual perturbation. Scores from an incomplete run remain useful
diagnostics, but cannot make a service eligible for production composition.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from collections import defaultdict

import numpy as np

from ..evaluation_protocol import PROTOCOL_VERSION, BENCHMARK_PROTOCOL_VERSION, resolve_evaluation_plan
from . import perturbation as legacy
from . import evaluation_outputs as outputs


def _progress_logging_enabled(config):
    value = (config or {}).get("perturbation_stage_logging", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _progress_log(config, state, message, *, force=False):
    """Emit compact scoring progress, including during long individual inputs."""
    if not _progress_logging_enabled(config):
        return
    now = time.monotonic()
    if force or now - state[0] >= 10.0:
        print(f"[metric evaluation] {message}", flush=True)
        state[0] = now


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _hash_update(digest, value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            value = value.numpy()
        except TypeError:
            value = value.float().numpy()
    if isinstance(value, np.ndarray) and value.dtype.kind != "O":
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode())
        for entry in value:
            _hash_update(digest, entry)
    else:
        digest.update(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False).encode())
    digest.update(b"\x00")


def _digest(value):
    digest = hashlib.sha256()
    _hash_update(digest, value)
    return digest.hexdigest()


def _evaluation_definitions(config, meta):
    prefixes = ("feature_", "categorical_", "immutable_", "integer_", "one_hot_", "image_", "detection_",
                "segmentation_", "retrieval_", "regression_", "evaluation_", "explainability_expected_",
                "label_", "ignore_index", "generation_special_token_ids")
    definitions = {key: value for key, value in {**meta, **config}.items() if key.startswith(prefixes)}
    definitions["output_adapter_version"] = "task_output_v2"
    definitions["perturbation_implementation"] = "calibrated_operating_operators_v1"
    for key, value in list(definitions.items()):
        if callable(value):
            code = getattr(value, "__code__", None)
            definitions[key] = {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}",
                                "code_hash": _digest([code.co_code.hex(), code.co_consts]) if code else None}
    return _jsonable(definitions)


def _seed(*parts):
    return legacy._stable_seed(PROTOCOL_VERSION, *parts)


def _effective_runtime(model):
    core = getattr(model, "core", None)
    underlying = getattr(core, "model", model)
    runtime = {"python_version": platform.python_version(), "numpy_version": np.__version__,
               "torch_version": getattr(getattr(core, "torch", None), "__version__", None),
               "transformers_version": getattr(sys.modules.get("transformers"), "__version__", None),
               "device": str(getattr(core, "device", getattr(underlying, "device", "generic_predict"))),
               "model_dtype": str(getattr(underlying, "dtype", "unspecified"))}
    if core is not None:
        runtime.update(max_length=getattr(core, "max_length", None), label_pad_value=getattr(core, "label_pad_value", -100))
        if bool(getattr(getattr(core, "task_spec", None), "supports_generation", False)):
            # The output adapter explicitly disables sampling; all other
            # generation settings come from the live core, not caller aliases.
            runtime["generation_config"] = {**dict(getattr(core, "generation_config", {}) or {}), "do_sample": False}
        parameters = getattr(underlying, "parameters", None)
        if callable(parameters):
            runtime["parameter_dtypes"] = sorted({str(value.dtype) for value in parameters()})
    return _jsonable(runtime)


def _model_identity(model, config, meta):
    underlying = getattr(getattr(model, "core", None), "model", model)
    identity = {"class": f"{type(underlying).__module__}.{type(underlying).__qualname__}"}
    declared = next((str(source[key]) for source in (config, meta)
                     for key in ("metric_evaluation_model_hash", "final_model_hash", "model_hash") if source.get(key)), None)
    if declared:
        identity["declared_model_hash"] = declared
    try:
        if callable(getattr(underlying, "state_dict", None)):
            state = underlying.state_dict()
            identity.update(sha256=_digest([identity["class"], state]), source="state_dict", verified_weights=bool(state))
        elif callable(getattr(underlying, "get_weights", None)):
            state = underlying.get_weights()
            identity.update(sha256=_digest([identity["class"], state]), source="get_weights", verified_weights=bool(state))
        else:
            learned = {k: v for k, v in vars(underlying).items()
                       if k.endswith("_") and isinstance(v, (np.ndarray, int, float, str, list, tuple))}
            identity.update(sha256=_digest([identity, learned]), source="estimator_parameters" if learned else "class_only",
                            verified_weights=bool(learned))
    except Exception as exc:
        identity.update(sha256=_digest(identity), source="class_only", verified_weights=False, error=type(exc).__name__)
    return identity


def _indexed_values(config, meta, key, count):
    value = config.get(key)
    if value is None:
        value = meta.get(key)
    if value is None:
        return None
    if len(value) != count:
        raise ValueError(f"{key} must have one entry per held-out input")
    return value


def _strata(y_eval, family, count, config, meta):
    supplied = _indexed_values(config, meta, "metric_evaluation_subgroups", count)
    labels = []
    for idx in range(count):
        parts = {}
        if family == "classification" and y_eval is not None:
            target = np.asarray(legacy._get_sample(y_eval, idx))
            if target.size == 1:
                parts["class"] = str(target.reshape(-1)[0])
            elif target.ndim == 1 and np.isclose(target.sum(), 1):
                parts["class"] = str(int(np.argmax(target)))
        if supplied is not None:
            subgroup = supplied[idx]
            parts.update({str(k): str(v) for k, v in subgroup.items()} if isinstance(subgroup, dict)
                         else {"subgroup": str(subgroup)})
        labels.append(parts)
    return labels


def frozen_sample_indices(count, limit, strata, seed):
    """A nested class/subgroup-stratified sample, shared by comparable services."""
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for index in range(count):
        buckets[json.dumps(strata[index], sort_keys=True)].append(index)
    queues = [list(rng.permutation(buckets[key])) for key in sorted(buckets)]
    selected = []
    # First preserve coverage, then sample in proportion to the remaining pool.
    for bucket in queues:
        if bucket and len(selected) < limit:
            selected.append(int(bucket.pop()))
    remainder = [int(i) for bucket in queues for i in bucket]
    if remainder and len(selected) < limit:
        selected.extend(int(i) for i in rng.choice(remainder, min(limit - len(selected), len(remainder)), replace=False))
    return selected


def independent_sample_indices(count, limit, strata, seed, groups, source_ids=None):
    """Select one row per independent component, without replacement.

    Shared source IDs and explicit document/patient IDs both establish
    dependence, including when a repeated source bridges two supplied groups.
    Eligibility is frozen before inference; failures never shrink the target.
    """
    parents = list(range(count))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for values in (groups, source_ids):
        seen = {}
        for index, value in enumerate(values if values is not None else []):
            key = str(value)
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
    order = frozen_sample_indices(count, count, strata, seed)
    representatives = []
    seen = set()
    for index in order:
        component = root(index)
        if component not in seen:
            seen.add(component)
            representatives.append(index)
    return representatives[:limit], len(representatives)


def summarize_inputs(records, *, seed=42, bootstrap_resamples=1000):
    """Cluster bootstrap; replicating trials or group members cannot inflate n."""
    grouped = defaultdict(list)
    for record in records:
        score = record.get("score")
        if score is not None and np.isfinite(score):
            grouped[str(record.get("group_id", record["sample_index"]))].append(float(score))
    values = np.asarray([np.mean(v) for v in grouped.values()], dtype=float)
    if not len(values):
        return {"mean": None, "ci95": None, "p10": None, "minimum": None, "independent_inputs": 0}
    ci = None
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        # Keep peak memory independent of the requested bootstrap budget.
        boot = [float(rng.choice(values, len(values), replace=True).mean())
                for _ in range(max(100, int(bootstrap_resamples)))]
        ci = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return {"mean": float(values.mean()), "ci95": ci, "p10": float(np.percentile(values, 10)),
            "minimum": float(values.min()), "independent_inputs": len(values)}


def _metric_summary(records, stage_name, **kwargs):
    summary = summarize_inputs(records, **kwargs)
    if stage_name == "explainability":
        # Preserve negative controls through aggregation. Clipping per input
        # would reward random rankings merely for chance-positive effects.
        raw = dict(summary)
        for key in ("mean", "p10", "minimum"):
            if summary[key] is not None:
                summary[key] = float(np.clip(summary[key], 0, 1))
        if summary["ci95"] is not None:
            summary["ci95"] = [float(np.clip(value, 0, 1)) for value in summary["ci95"]]
        summary["signed_control_gain"] = raw
    return summary


def _save_progress(path, progress):
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(progress), sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, target)


def _remaining(deadline):
    return deadline is None or time.monotonic() < deadline


def _deadline(plan, stage):
    duration = plan.get(f"{stage}_max_duration_s")
    return None if duration is None else time.monotonic() + max(0, float(duration))


def _quality(output, target, scale):
    try:
        return outputs.task_output_quality(output, target, target_scale=scale)
    except (ValueError, TypeError, outputs.UnsupportedTaskOutput) as exc:
        return {"score": None, "metric": None, "reason": str(exc)}


def _predict(model, sample, family, hf_task, config, meta):
    return outputs.predict_task_output(model, sample, task_family=family, hf_task=hf_task, config=config, meta=meta)


def _predict_many(model, samples, family, hf_task, config, meta):
    """Use one inference batch where the adapter supports it; isolate failures."""
    batch_fn = getattr(outputs, "predict_task_outputs", None)
    if batch_fn is not None:
        try:
            return batch_fn(model, samples, task_family=family, hf_task=hf_task, config=config, meta=meta)
        except Exception:
            pass
    predictions = []
    for sample in samples:
        try:
            predictions.append(_predict(model, sample, family, hf_task, config, meta))
        except Exception as exc:
            predictions.append(exc)
    return predictions


def _trial_error(exc):
    unsupported = isinstance(exc, outputs.UnsupportedTaskOutput)
    return {"status": "excluded" if unsupported else "failed", "reason": str(exc), "unsupported": unsupported}


def _record_for(stage, idx, groups, strata):
    return stage["samples"].setdefault(str(idx), {"sample_index": idx, "group_id": str(groups[idx]),
                                                "strata": strata[idx], "trials": [], "score": None})


def _mean(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def _trust_stage(model, x_eval, y_eval, selected, groups, strata, config, meta, plan, progress, save,
                 baseline_cache=None):
    stage = progress["trust"]
    baseline_cache = {} if baseline_cache is None else baseline_cache
    deadline = _deadline(plan, "trust")
    family, hf_task, scale = progress["task_family"], progress["hf_task"], config.get("regression_target_scale")
    batch_size = plan.get("batch_size", 32)
    progress_log_at = [float("-inf")]
    stage["truncated"] = False
    completed = sum(bool(stage["samples"].get(str(idx), {}).get("done")) for idx in selected)
    _progress_log(config, progress_log_at,
                  f"trust starts | completed_inputs={completed}/{len(selected)} | trials_per_input={plan['trust_trials']}",
                  force=True)
    for input_number, idx in enumerate(selected, start=1):
        record = _record_for(stage, idx, groups, strata)
        if record.get("done"):
            continue
        if not _remaining(deadline):
            stage["truncated"] = True
            break
        sample = legacy._get_sample(x_eval, idx)
        target = legacy._get_sample(y_eval, idx) if y_eval is not None else None
        record["attempted"] = True
        record.setdefault("inference_attempted", bool(record.get("clean_quality")) or any(
            row.get("score") is not None for row in record["trials"]))

        # Construct the operating interventions before the clean prediction.
        # In particular, text datasets without a validated transformation
        # contract should be reported as unsupported without paying for model
        # inference that cannot contribute to the trust score.
        new_rows, pending = [], []
        next_trial = len(record["trials"])
        while next_trial < plan["trust_trials"] and _remaining(deadline):
            stop = min(plan["trust_trials"], next_trial + batch_size)
            for trial in range(next_trial, stop):
                trial_seed = _seed(plan["seed"], progress["input_hash"], "trust", idx, trial)
                row = {"trial_index": trial, "seed": trial_seed}
                try:
                    changed = outputs.perturb_input(sample, np.random.default_rng(trial_seed), task_family=family,
                                trial_index=trial, config=config, meta=meta, model=model)
                    row.update(operator=changed.operator, strength=changed.strength, changed=bool(changed.changed),
                               regime=changed.regime, metadata=changed.metadata)
                    if changed.status not in {"ok", "complete", "supported", "success"} or not changed.changed or changed.regime != "operating":
                        row.update(status="excluded", reason=changed.reason or ("unchanged_input" if not changed.changed else "stress_test"),
                                   unsupported=changed.status == "unsupported")
                    else:
                        pending.append((row, changed.sample))
                except Exception as exc:
                    row.update(_trial_error(exc))
                new_rows.append(row)
            next_trial = stop

        baseline = None
        baseline_failed = False
        if pending:
            record["inference_attempted"] = True
            try:
                if idx not in baseline_cache:
                    baseline_cache[idx] = _predict(model, sample, family, hf_task, config, meta)
                baseline = baseline_cache[idx]
                record["clean_quality"] = _quality(baseline, target, scale)
            except Exception as exc:
                baseline_failed = True
                error = _trial_error(exc)
                record.update(error, done=True)
                for row, _ in pending:
                    row.update(error)

        record["trials"].extend(new_rows)
        if baseline is not None:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                predictions = _predict_many(model, [sample for _, sample in batch], family, hf_task, config, meta)
                for (row, _), prediction in zip(batch, predictions):
                    try:
                        if isinstance(prediction, Exception):
                            raise prediction
                        comparison = outputs.compare_task_outputs(baseline, prediction, target_scale=scale)
                        row.update(status="successful", score=comparison["stability_score"], components=comparison,
                                   perturbed_quality=_quality(prediction, target, scale))
                    except Exception as exc:
                        row.update(_trial_error(exc))
        record["score"] = _mean(row.get("score") for row in record["trials"])
        save()
        _progress_log(config, progress_log_at,
                      f"trust | input={input_number}/{len(selected)} | trials={len(record['trials'])}/{plan['trust_trials']}")
        record["done"] = baseline_failed or len(record["trials"]) == plan["trust_trials"]
        if not record["done"]:
            stage["truncated"] = True
            save()
            break
        save()
        completed += 1
        _progress_log(config, progress_log_at,
                      f"trust | completed_inputs={completed}/{len(selected)} | current_input={input_number}/{len(selected)}",
                      force=True)
        if legacy._progress_enabled(config) and (len([r for r in stage["samples"].values() if r.get("done")]) % 10 == 0):
            legacy._progress_log(config, f"trust | input={idx} | completed={sum(bool(r.get('done')) for r in stage['samples'].values())}/{len(selected)}")
    stage["finished"] = all(stage["samples"].get(str(idx), {}).get("done") for idx in selected)
    save()
    _progress_log(config, progress_log_at,
                  f"trust ends | completed_inputs={sum(bool(stage['samples'].get(str(idx), {}).get('done')) for idx in selected)}/{len(selected)} | truncated={stage['truncated']}",
                  force=True)


def _replacement(sample, units, reference, method, model):
    """Replace whole feature groups, preserving positions and unselected input."""
    result = legacy._copy_sample(sample)
    if not units:
        return result
    if units[0][0] == "word":
        words = str(sample).split()
        donor_words = str(reference).split()
        token = legacy._mask_token_text(model)
        # Position-preserving unknown tokens avoid shifting later candidate units.
        for unit in units:
            pos = int(unit[1])
            if pos < len(words):
                words[pos] = donor_words[pos % len(donor_words)] if method == "empirical" and donor_words else (token or "[UNK]")
        return " ".join(words)
    for unit in units:
        kind = unit[0]
        if kind in {"feature", "feature_group"}:
            arr, donor = np.asarray(result).copy(), np.asarray(reference)
            positions = [int(unit[1])] if kind == "feature" else list(unit[1])
            arr.reshape(-1)[positions] = donor.reshape(-1)[positions]
            result = arr
        elif kind == "patch":
            arr = np.asarray(legacy._image_array_view(result)).copy()
            donor = np.asarray(legacy._image_array_view(reference))
            _, y0, y1, x0, x1, channel_first = unit
            selection = (slice(None), slice(y0, y1), slice(x0, x1)) if channel_first else (slice(y0, y1), slice(x0, x1), slice(None))
            if arr.shape == donor.shape:
                arr[selection] = donor[selection]
            else:
                arr[selection] = np.mean(donor)
            result = legacy._set_image_array(result, arr)
        elif kind in {"token", "span"}:
            positions = [int(unit[1])] if kind == "token" else range(int(unit[1]), int(unit[2]))
            for pos in positions:
                if method == "empirical" and isinstance(reference, dict) and "input_ids" in reference:
                    ids = np.asarray(result["input_ids"]).copy()
                    donor = np.asarray(reference["input_ids"]).reshape(-1)
                    if pos < ids.size and pos < donor.size:
                        ids.reshape(-1)[pos] = donor[pos]
                        result["input_ids"] = ids
                else:
                    result = legacy._mask_token(result, pos, model=model)
        else:
            raise outputs.UnsupportedTaskOutput(f"unsupported_explanation_unit:{kind}")
    return result


def _reference(x_eval, sample, idx, method, rng, selected, settings=None):
    pool = sorted(index for index in selected if index != idx)
    if method == "empirical":
        if not pool:
            raise outputs.UnsupportedTaskOutput("empirical_replacement_requires_another_input")
        return legacy._get_sample(x_eval, int(rng.choice(pool)))
    if method not in {"mean", "mask"}:
        raise outputs.UnsupportedTaskOutput(f"unvalidated_replacement_method:{method}")
    modality = legacy._modality(sample)
    if modality in {"tokens", "text"}:
        return sample
    if modality == "numeric":
        references = [np.asarray(legacy._get_sample(x_eval, int(i))) for i in pool[:64]]
        if not references:
            return np.asarray(sample).copy()
        reference = np.mean(references, axis=0)
        cfg = settings or {}
        matrix = np.asarray(references).reshape(len(references), -1)
        # Means of category codes and one-hot columns are not category values.
        for index in cfg.get("categorical_feature_indices", []):
            values, counts = np.unique(matrix[:, int(index)], return_counts=True)
            reference.reshape(-1)[int(index)] = values[np.argmax(counts)]
        for group in cfg.get("one_hot_feature_groups", []):
            values, counts = np.unique(matrix[:, group], axis=0, return_counts=True)
            reference.reshape(-1)[list(group)] = values[np.argmax(counts)]
        for index in cfg.get("integer_feature_indices", []):
            reference.reshape(-1)[int(index)] = np.round(reference.reshape(-1)[int(index)])
        return reference.astype(np.asarray(sample).dtype)
    if modality == "image":
        arr = np.asarray(legacy._image_array_view(sample))
        return legacy._set_image_array(sample, np.full_like(arr, np.mean(arr)))
    raise outputs.UnsupportedTaskOutput("unsupported_replacement_modality")


def _explanation_input(model, sample, units, reference, method, family, config, meta):
    changed = _replacement(sample, units, reference, method, model)
    cfg = {**meta, **config}
    if legacy._modality(changed, task_family=family) == "numeric":
        flat = np.asarray(changed).reshape(-1)
        for constraint in cfg.get("feature_constraints", []):
            value = float(np.asarray(constraint["coefficients"], dtype=float) @ flat)
            if not float(constraint.get("min", -np.inf)) <= value <= float(constraint.get("max", np.inf)):
                raise outputs.UnsupportedTaskOutput("explanation_replacement_violates_feature_relationship")
        validator = cfg.get("evaluation_feature_validator")
        if callable(validator) and not bool(validator(changed)):
            raise outputs.UnsupportedTaskOutput("explanation_replacement_violates_feature_relationship")
    return changed


def _explanation_effect(model, baseline, sample, units, reference, method, family, hf_task, config, meta):
    changed = _explanation_input(model, sample, units, reference, method, family, config, meta)
    prediction = _predict(model, changed, family, hf_task, config, meta)
    comparison = outputs.compare_task_outputs(baseline, prediction, target_scale=config.get("regression_target_scale"))
    return float(1.0 - comparison["stability_score"])


def _explanation_trial_specs(plan):
    methods = plan.get("explanation_replacement_methods", ["empirical", "mean"])
    for method in methods:
        for fraction in plan["explainability_budget_fractions"]:
            for replicate in range(plan["explainability_random_trials"]):
                yield method, float(fraction), replicate


def _explanation_stage(model, x_eval, y_eval, selected, groups, strata, config, meta, plan, progress, save,
                       baseline_cache=None):
    stage = progress["explainability"]
    baseline_cache = {} if baseline_cache is None else baseline_cache
    deadline = _deadline(plan, "explainability")
    family, hf_task = progress["task_family"], progress["hf_task"]
    method_name = str(config.get("metric_evaluation_explanation_method") or "occlusion")
    trial_specs = list(_explanation_trial_specs(plan))
    progress_log_at = [float("-inf")]
    stage["truncated"] = False
    completed = sum(bool(stage["samples"].get(str(idx), {}).get("done")) for idx in selected)
    _progress_log(config, progress_log_at,
                  f"explainability starts | completed_inputs={completed}/{len(selected)} | trials_per_input={len(trial_specs)}",
                  force=True)
    for input_number, idx in enumerate(selected, start=1):
        record = _record_for(stage, idx, groups, strata)
        if record.get("done"):
            continue
        if not _remaining(deadline):
            stage["truncated"] = True
            break
        sample = legacy._get_sample(x_eval, idx)
        target = legacy._get_sample(y_eval, idx) if y_eval is not None else None
        record["attempted"] = True
        try:
            # Candidate construction and ranking never see relevance annotations.
            units = outputs.candidate_units(sample, task_family=family, meta={**meta, **config}, limit=plan["explainability_candidate_units"])
            if len(units) < 2:
                raise outputs.UnsupportedTaskOutput("fewer_than_two_candidate_groups")
            if idx not in baseline_cache:
                baseline_cache[idx] = _predict(model, sample, family, hf_task, config, meta)
            baseline = baseline_cache[idx]
            record.update(candidate_count=len(units), explanation_method=method_name)
            ranking = record.setdefault("ranking_evidence", [])
            while len(ranking) < len(units) and _remaining(deadline):
                start = len(ranking)
                stop = min(len(units), start + plan.get("batch_size", 32))
                interventions = []
                for unit_index in range(start, stop):
                    seed = _seed(plan["seed"], progress["input_hash"], "ranking", idx, unit_index)
                    rng = np.random.default_rng(seed)
                    reference = _reference(x_eval, sample, idx, "empirical", rng,
                                           progress["eligible_reference_indices"], {**meta, **config})
                    changed = _explanation_input(model, sample, [units[unit_index]], reference,
                                                 "empirical", family, config, meta)
                    interventions.append((unit_index, seed, changed))
                predictions = _predict_many(model, [changed for _, _, changed in interventions],
                                            family, hf_task, config, meta)
                for (unit_index, seed, _), prediction in zip(interventions, predictions):
                    if isinstance(prediction, Exception):
                        raise prediction
                    comparison = outputs.compare_task_outputs(
                        baseline, prediction, target_scale=config.get("regression_target_scale"))
                    ranking.append({"unit_index": unit_index,
                                    "effect": float(1.0 - comparison["stability_score"]), "seed": seed})
                save()
                _progress_log(config, progress_log_at,
                              f"explainability | input={input_number}/{len(selected)} | phase=ranking | units={len(ranking)}/{len(units)}")
            if len(ranking) < len(units):
                stage["truncated"] = True
                break
            ordering = sorted(range(len(units)), key=lambda i: (-ranking[i]["effect"], i))
            if method_name == "random":
                ordering = np.random.default_rng(_seed(plan["seed"], "random_explanation", idx)).permutation(len(units)).tolist()
            elif method_name == "reverse_occlusion":
                ordering.reverse()
            elif method_name != "occlusion":
                raise outputs.UnsupportedTaskOutput(f"unknown_explanation_method:{method_name}")
            record["ranking"] = [units[i] for i in ordering]
            expected, provenance = legacy._expected_units(sample, target, task_family=family, hf_task=hf_task,
                                    meta=meta, model=model, config=config, candidate_units=units, limit=len(units))
            record["semantic_alignment"] = {"score": legacy._alignment_score(record["ranking"], expected) if expected else None,
                "provenance": provenance, "interpretation": "relevance_proxy_only", "included_in_faithfulness": False}
            variants_per_trial = 6
            validation_batch_size = plan.get("batch_size", 32)
            trials_per_batch = max(1, validation_batch_size // variants_per_trial)
            while len(record["trials"]) < len(trial_specs) and _remaining(deadline):
                start = len(record["trials"])
                stop = min(len(trial_specs), start + trials_per_batch)
                rows, pending_trials = [], []
                for trial_index in range(start, stop):
                    if not _remaining(deadline):
                        break
                    replacement, fraction, replicate = trial_specs[trial_index]
                    seed = _seed(plan["seed"], progress["input_hash"], "faithfulness_validation", idx, trial_index)
                    row = {"trial_index": trial_index, "replacement_method": replacement,
                           "mask_fraction": fraction, "random_replicate": replicate, "seed": seed}
                    rows.append(row)
                    try:
                        rng = np.random.default_rng(seed)
                        reference = _reference(x_eval, sample, idx, replacement, rng,
                                               progress["eligible_reference_indices"], {**meta, **config})
                        count = min(len(units) - 1, max(1, int(math.ceil(len(units) * fraction))))
                        selections = {"ranked": ordering[:count], "reversed": ordering[-count:],
                                      "random": rng.choice(len(units), size=count, replace=False).tolist()}
                        variants = []
                        for control, chosen in selections.items():
                            for operation in ("delete", "retain"):
                                # Retaining means replace the complement of the selected units.
                                masked = chosen if operation == "delete" else [
                                    i for i in range(len(units)) if i not in chosen]
                                variants.append((f"{control}_{operation}", _explanation_input(
                                    model, sample, [units[i] for i in masked], reference, replacement,
                                    family, config, meta)))
                        pending_trials.append((row, count, variants))
                    except Exception as exc:
                        row.update(_trial_error(exc))

                flattened = [(key, value) for _, _, variants in pending_trials for key, value in variants]
                predictions = []
                for batch_start in range(0, len(flattened), validation_batch_size):
                    batch = flattened[batch_start:batch_start + validation_batch_size]
                    predictions.extend(_predict_many(model, [value for _, value in batch],
                                                     family, hf_task, config, meta))
                cursor = 0
                for row, count, variants in pending_trials:
                    trial_predictions = predictions[cursor:cursor + len(variants)]
                    cursor += len(variants)
                    try:
                        if len(trial_predictions) != len(variants):
                            raise ValueError("prediction_batch_length_mismatch")
                        effects = {}
                        for (key, _), prediction in zip(variants, trial_predictions):
                            if isinstance(prediction, Exception):
                                raise prediction
                            effects[key] = float(1. - outputs.compare_task_outputs(
                                baseline, prediction,
                                target_scale=config.get("regression_target_scale"))["stability_score"])
                        gain = np.mean([effects["ranked_delete"] - effects["random_delete"],
                                        effects["ranked_delete"] - effects["reversed_delete"],
                                        effects["random_retain"] - effects["ranked_retain"],
                                        effects["reversed_retain"] - effects["ranked_retain"]])
                        row.update(status="successful", score=float(gain), unbounded_control_gain=float(gain),
                                   effects=effects, selected_unit_count=count)
                    except Exception as exc:
                        row.update(_trial_error(exc))
                record["trials"].extend(rows)
                record["score"] = _mean(row.get("score") for row in record["trials"])
                save()
                _progress_log(config, progress_log_at,
                              f"explainability | input={input_number}/{len(selected)} | phase=validation | trials={len(record['trials'])}/{len(trial_specs)}")
            record["done"] = len(record["trials"]) == len(trial_specs)
        except Exception as exc:
            record.update(_trial_error(exc), done=True)
        save()
        if record.get("done"):
            completed += 1
        _progress_log(config, progress_log_at,
                      f"explainability | completed_inputs={completed}/{len(selected)} | current_input={input_number}/{len(selected)}",
                      force=True)
        if legacy._progress_enabled(config) and (len([r for r in stage["samples"].values() if r.get("done")]) % 10 == 0):
            legacy._progress_log(config, f"explainability | input={idx} | completed={sum(bool(r.get('done')) for r in stage['samples'].values())}/{len(selected)}")
        if not record.get("done"):
            stage["truncated"] = True
            break
    stage["finished"] = all(stage["samples"].get(str(idx), {}).get("done") for idx in selected)
    save()
    _progress_log(config, progress_log_at,
                  f"explainability ends | completed_inputs={sum(bool(stage['samples'].get(str(idx), {}).get('done')) for idx in selected)}/{len(selected)} | truncated={stage['truncated']}",
                  force=True)


def _stage_summary(stage, requested, trials_per_sample, plan, provenance, stage_name):
    eligible = provenance.get("eligible_independent_inputs", requested)
    effective = min(requested, eligible) if plan["protocol_version"] == BENCHMARK_PROTOCOL_VERSION else requested
    records = list(stage["samples"].values())
    trial_rows = [trial for record in records for trial in record["trials"]]
    valid = [record for record in records if record.get("score") is not None and np.isfinite(record["score"])]
    sample_failed = lambda r: r.get("score") is None and (r.get("status") == "failed" or any(t.get("status") == "failed" for t in r["trials"]))
    sample_excluded = lambda r: bool(r.get("score") is None and not sample_failed(r) and (r.get("status") == "excluded" or (r.get("done") and bool(r["trials"]))))
    sample_counts = {"requested": requested, "attempted": sum(bool(r.get("attempted")) for r in records),
                     "configured_target": requested, "eligible_independent_inputs": eligible,
                     "effective_target": effective,
                     "successful": len(valid), "failed": sum(sample_failed(r) for r in records),
                     "excluded": sum(sample_excluded(r) for r in records),
                     "completed": sum(bool(r.get("done")) for r in records)}
    trial_counts = {"requested": effective * trials_per_sample, "attempted": len(trial_rows),
                    **{key: sum(t.get("status") == key for t in trial_rows) for key in ("successful", "failed", "excluded")}}
    summary = _metric_summary(valid, stage_name, seed=_seed(plan["seed"], stage_name, "bootstrap"),
                               bootstrap_resamples=plan.get("bootstrap_resamples", 1000))
    ci = summary.get("signed_control_gain", summary)["ci95"]
    ci_width = None if ci is None else ci[1] - ci[0]
    minimum_coverage = plan.get("min_coverage", .8)
    sample_coverage = summary["independent_inputs"] / effective if effective else 0.
    trial_coverage = trial_counts["successful"] / trial_counts["requested"] if trial_counts["requested"] else 0.
    reasons = []
    details = {}
    def insufficient(reason, **detail):
        reasons.append(reason)
        details[reason] = detail

    if plan["protocol_version"] == BENCHMARK_PROTOCOL_VERSION:
        # Budgets are ceilings, including when a deadline ends an otherwise
        # sufficiently covered run. Failures never shrink either denominator.
        if stage.get("stage_error"):
            insufficient("stage_error", error=stage["stage_error"])
        if sample_coverage < minimum_coverage:
            insufficient("sample_coverage_below_minimum", observed=sample_coverage, required=minimum_coverage,
                         successful_independent_inputs=summary["independent_inputs"], denominator=effective)
    else:
        if not stage.get("finished") or stage.get("truncated"):
            insufficient("stage_incomplete", time_budget_exhausted=bool(stage.get("truncated")))
        if sample_counts["successful"] < effective:
            insufficient("requested_sample_budget_not_met", observed=sample_counts["successful"], required=effective)
    minimum = plan.get(f"{stage_name}_min_independent_inputs", plan["min_independent_inputs"])
    if summary["independent_inputs"] < minimum:
        insufficient("too_few_independent_inputs", observed=summary["independent_inputs"], required=minimum)
    if ci_width is None or ci_width > plan["max_ci_width"]:
        insufficient("precision_requirement_not_met", observed_ci_width=ci_width, maximum_ci_width=plan["max_ci_width"],
                     cause="confidence_interval_unavailable" if ci_width is None else "confidence_interval_too_wide")
    if trial_coverage < minimum_coverage:
        insufficient("trial_coverage_below_minimum", observed=trial_coverage, required=minimum_coverage,
                     successful_trials=trial_counts["successful"], denominator=trial_counts["requested"])
    if not provenance["model_identity"].get("verified_weights"):
        insufficient("model_identity_unverified")
    sufficient = not reasons
    if summary["mean"] is not None:
        status = "complete" if sufficient else "insufficient_evidence"
    elif stage.get("stage_error") or any(r.get("status") == "failed" for r in records) or any(t.get("status") == "failed" for t in trial_rows):
        status = "failed"
    elif stage.get("finished") and records:
        status = "unsupported"
    else:
        status = "insufficient_evidence"
    by_stratum = defaultdict(list)
    for record in records:
        for key, value in record["strata"].items():
            by_stratum[f"{key}={value}"].append(record)
    by_trial = defaultdict(lambda: defaultdict(list))
    trial_strata = defaultdict(list)
    for record in records:
        for trial in record["trials"]:
            if stage_name == "trust":
                keys = [f"operator={trial.get('operator')}", f"operator={trial.get('operator')};strength={trial.get('strength')}"]
            else:
                keys = [f"replacement={trial.get('replacement_method')}", f"mask_fraction={trial.get('mask_fraction')}"]
            for key in keys:
                trial_strata[key].append(trial)
                if trial.get("score") is not None:
                    by_trial[key][record["sample_index"]].append(trial["score"])
    record_map = {r["sample_index"]: r for r in records}
    for key in trial_strata:
        entries = by_trial[key]
        by_stratum[key] = [{**record_map[idx], "score": _mean(values)} for idx, values in entries.items()]
    stratified = {key: _metric_summary(values, stage_name, seed=_seed(plan["seed"], stage_name, key),
                          bootstrap_resamples=plan.get("bootstrap_resamples", 1000)) for key, values in by_stratum.items()}
    for key, trials in trial_strata.items():
        stratified[key]["trial_counts"] = {"attempted": len(trials),
            **{status: sum(t.get("status") == status for t in trials) for status in ("successful", "failed", "excluded")}}
    exclusion_reasons, failure_reasons = defaultdict(int), defaultdict(int)
    for row in records + trial_rows:
        if row.get("reason") and row.get("status") in {"excluded", "failed"}:
            (exclusion_reasons if row["status"] == "excluded" else failure_reasons)[row["reason"]] += 1
    summary.update(status=status, evidence_sufficient=sufficient, insufficiency_reasons=reasons,
                   insufficiency_details=details, exclusion_reasons=dict(exclusion_reasons), failure_reasons=dict(failure_reasons),
                   minimum_independent_inputs=minimum,
                   sample_coverage=sample_coverage, trial_coverage=trial_coverage, minimum_coverage=minimum_coverage,
                   ci_width=ci_width, maximum_ci_width=plan["max_ci_width"],
                   sample_counts=sample_counts, trial_counts=trial_counts,
                   trial_definition="one_operating_perturbation" if stage_name == "trust" else "six_paired_delete_retain_control_conditions",
                   by_stratum=stratified,
                   clean_quality_mean=_mean(r.get("clean_quality", {}).get("score") for r in records),
                   perturbed_quality_mean=_mean(t.get("perturbed_quality", {}).get("score") for t in trial_rows),
                   uncertainty_unit="group" if provenance["explicit_grouping"] else "input",
                   time_budget_exhausted=bool(stage.get("truncated")), duration_s=stage.get("duration_s", 0),
                   stage_error=stage.get("stage_error"))
    return summary


def run_production_evaluation(model, x_eval, y_eval=None, *, task_family=None, hf_task=None,
                              config=None, meta=None, service_id=None):
    """Run trust and faithfulness independently with frozen, resumable evidence."""
    config, meta = dict(config or {}), dict(meta or {})
    plan = resolve_evaluation_plan(config)
    version = plan["protocol_version"]
    if not legacy._enabled(config):
        return {"perturbation_enabled_flag": False, "evaluation_protocol_version": version,
                "trust_score": None, "explainability_score": None, "trust_status": "unsupported",
                "explainability_status": "unsupported", "trust_evidence_sufficient": False,
                "explainability_evidence_sufficient": False, "metric_evaluation_disabled": True}
    start = time.monotonic()
    count = legacy._sample_count(x_eval, y_eval)
    family = str(task_family or "classification")
    tokenizer = getattr(getattr(model, "core", None), "tokenizer", None)
    if tokenizer is not None and config.get("evaluation_boundary_token_ids") is None:
        config["evaluation_boundary_token_ids"] = sorted({int(value) for name in ("cls", "sep", "bos", "eos", "pad")
            if (value := getattr(tokenizer, f"{name}_token_id", None)) is not None})
    original_x_eval = x_eval
    specification = getattr(getattr(getattr(model, "core", None), "task_spec", None), "name", None)
    prompt_preparation = hf_task == "causal_lm_generation" or specification == "causal_lm_generation"
    preparation_errors = {}
    if prompt_preparation:
        # Target values never reach ranking/inference; only the stored prompt
        # boundary (-100 label mask) determines which input tokens are visible.
        prepared = []
        for idx in range(count):
            try:
                prepared.append(outputs.prepare_task_input(legacy._get_sample(x_eval, idx),
                    legacy._get_sample(y_eval, idx) if y_eval is not None else None,
                    task_family=family, hf_task=hf_task, model=model))
            except Exception as exc:
                preparation_errors[idx] = _trial_error(exc)
                prepared.append(None)
        x_eval = prepared
    if family == "fill_mask":
        mask_id = getattr(getattr(getattr(model, "core", None), "tokenizer", None), "mask_token_id", None)
        if mask_id is not None and config.get("evaluation_mask_token_id") is None:
            config["evaluation_mask_token_id"] = int(mask_id)
    strata = _strata(y_eval, family, count, config, meta)
    groups = _indexed_values(config, meta, "metric_evaluation_group_ids", count)
    source_ids = _indexed_values(config, meta, "metric_evaluation_source_ids", count)
    explicit_grouping = groups is not None
    groups = groups if groups is not None else source_ids if source_ids is not None else list(range(count))
    limit = max(plan["trust_sample_count"], plan["explainability_sample_count"])
    selection_seed = _seed(plan["seed"], family, hf_task, "sample_selection")
    independent_indices, eligible_count = independent_sample_indices(count, limit, strata, selection_seed, groups, source_ids)
    indices = independent_indices if version == BENCHMARK_PROTOCOL_VERSION else frozen_sample_indices(
        count, min(count, limit), strata, selection_seed)
    sample_hash = _digest([(idx, legacy._get_sample(x_eval, idx), legacy._get_sample(y_eval, idx) if y_eval is not None else None)
                          for idx in indices])
    input_hash = _digest([(idx, legacy._get_sample(x_eval, idx)) for idx in sorted(indices)])
    model_identity = _model_identity(model, config, meta)
    dataset_identity = {"name": str(config.get("dataset_name") or meta.get("dataset_name") or meta.get("name") or "held_out"),
                        "input_count": count, "sha256": str(config.get("metric_evaluation_dataset_hash") or meta.get("dataset_hash") or sample_hash),
                        "split": config.get("metric_evaluation_dataset_split") or meta.get("dataset_split") or "held_out",
                        "revision": config.get("metric_evaluation_dataset_revision") or meta.get("dataset_revision"),
                        "hash_scope": "declared_dataset" if config.get("metric_evaluation_dataset_hash") or meta.get("dataset_hash") else "selected_inputs_and_targets"}
    provenance = {"protocol_version": version, "plan": plan, "dataset_identity": dataset_identity,
                  "eligible_independent_inputs": eligible_count,
                  "sampling_policy": "one_row_per_independent_component" if version == BENCHMARK_PROTOCOL_VERSION else "stratified_rows",
                  "sample_hash": sample_hash, "input_hash": input_hash, "sample_indices": indices, "model_identity": model_identity,
                  "source_ids": [str(source_ids[idx]) for idx in indices] if source_ids is not None else None,
                  "source_ids_hash": _digest([source_ids[idx] for idx in indices]) if source_ids is not None else None,
                  "seed": plan["seed"], "seed_scope": "dataset_sample_task_stage_input_trial_excludes_service_id",
                  "explicit_grouping": explicit_grouping, "explanation_method": config.get("metric_evaluation_explanation_method") or "occlusion",
                  "grouping_source": "explicit_group_ids" if explicit_grouping else "source_ids" if source_ids is not None else "input_index",
                  "explanation_score_definition": "positive_population_mean_delete_retain_gain_over_random_and_reversed_controls",
                  "semantic_alignment_role": "diagnostic_relevance_proxy_excluded_from_score"}
    provenance["runtime"] = _effective_runtime(model)
    if prompt_preparation:
        provenance["input_preparation"] = {"method": "causal_prompt_boundary_v1",
            "original_encoded_input_hash": _digest([(idx, legacy._get_sample(original_x_eval, idx)) for idx in indices]),
            "target_tokens_visible_to_model": False, "sample_errors": preparation_errors}
    # Feature calibration is shared across services and explicitly persisted.
    image_calibration_fn = getattr(outputs, "resolve_image_calibration", None)
    first_sample = next((legacy._get_sample(x_eval, idx) for idx in indices if idx not in preparation_errors), None)
    if image_calibration_fn is not None and first_sample is not None and legacy._modality(first_sample, task_family=family, meta=meta) == "image":
        try:
            calibration = image_calibration_fn(model, config=config, meta=meta)
            config.update({key: value for key, value in calibration.items() if value is not None})
            provenance["image_input_calibration"] = _jsonable(calibration)
        except (ValueError, TypeError, outputs.UnsupportedTaskOutput) as exc:
            provenance["image_input_calibration_error"] = str(exc)
    calibration_fn = getattr(outputs, "derive_input_calibration", None)
    if calibration_fn is not None and count:
        try:
            calibration = calibration_fn(x_eval, task_family=family, meta=meta)
            for key, value in calibration.items():
                if config.get(key) is None and meta.get(key) is None:
                    config[key] = value
            provenance["input_calibration"] = _jsonable(calibration)
        except (ValueError, TypeError, outputs.UnsupportedTaskOutput) as exc:
            provenance["input_calibration_error"] = str(exc)
    if family == "regression" and not config.get("regression_target_scale") and y_eval is not None:
        targets = np.asarray(y_eval, dtype=float)
        scale = float(np.nanstd(targets))
        if np.isfinite(scale) and scale > 0:
            config["regression_target_scale"] = scale
            provenance["regression_target_scale"] = {"value": scale, "source": "frozen_held_out_target_std"}
    provenance["evaluation_definitions"] = _evaluation_definitions(config, meta)
    progress_path = config.get("metric_evaluation_progress_path")
    # Time budgets are operational and may be increased to resume the frozen plan.
    identity_plan = {key: value for key, value in plan.items() if not key.endswith("_max_duration_s")}
    run_key = _digest([version, identity_plan, sample_hash, model_identity, family, hf_task,
                       groups, strata, dataset_identity, provenance["source_ids_hash"],
                       provenance["explanation_method"], provenance["explanation_score_definition"], provenance["evaluation_definitions"],
                       provenance.get("input_preparation"), provenance["runtime"]])
    progress = {"protocol_version": version, "run_key": run_key, "task_family": family, "hf_task": hf_task,
                "sample_hash": sample_hash, "input_hash": input_hash, "selected_indices": indices, "provenance": provenance,
                "eligible_reference_indices": [idx for idx in indices if idx not in preparation_errors],
                "trust": {"samples": {}}, "explainability": {"samples": {}}}
    resumed = False
    if progress_path and Path(progress_path).exists():
        saved = json.loads(Path(progress_path).read_text(encoding="utf-8"))
        if saved.get("run_key") != run_key:
            raise ValueError("Evaluation progress does not match frozen samples, model, grouping, definitions, or protocol plan")
        progress, resumed = saved, True
        progress["provenance"] = provenance
    for stage_name, requested in (("trust", plan["trust_sample_count"]), ("explainability", plan["explainability_sample_count"])):
        for idx in indices[:requested]:
            if idx in preparation_errors:
                record = _record_for(progress[stage_name], idx, groups, strata)
                record.update(preparation_errors[idx], done=True, attempted=True, exclusion_phase="input_preparation", inference_attempted=False)
    last_save = [float("-inf")]
    def save(force=False):
        now = time.monotonic()
        if force or now - last_save[0] >= 2.0:
            _save_progress(progress_path, progress)
            last_save[0] = now
    # Clean outputs are compatible across these two stages because both use the
    # same frozen input, model, task adapter, and inference settings.
    baseline_cache = {}
    # A failure, unsupported adapter, or timeout in either stage never skips the other.
    for stage_name, stage_fn, requested in (
        ("trust", _trust_stage, plan["trust_sample_count"]),
        ("explainability", _explanation_stage, plan["explainability_sample_count"]),
    ):
        stage_start = time.monotonic()
        try:
            stage_fn(model, x_eval, y_eval, indices[:requested], groups, strata, config, meta, plan, progress, save,
                     baseline_cache)
        except Exception as exc:
            progress[stage_name]["stage_error"] = f"{type(exc).__name__}: {exc}"
            progress[stage_name]["finished"] = False
        progress[stage_name]["duration_s"] = progress[stage_name].get("duration_s", 0) + time.monotonic() - stage_start
        save(force=True)
    trust = _stage_summary(progress["trust"], plan["trust_sample_count"], plan["trust_trials"], plan, provenance, "trust")
    explainability = _stage_summary(progress["explainability"], plan["explainability_sample_count"],
                                   len(list(_explanation_trial_specs(plan))), plan, provenance, "explainability")
    records = []
    for idx in indices:
        records.append({"sample_index": idx, "evaluation_protocol_version": version,
                        "source_id": str(source_ids[idx]) if source_ids is not None else None,
                        "trust": progress["trust"]["samples"].get(str(idx)),
                        "explainability": progress["explainability"]["samples"].get(str(idx))})
    result = {"evaluation_protocol_version": version, "metric_evaluation_profile": plan["profile"],
              "perturbation_enabled_flag": True, "perturbation_supported_flag": trust["mean"] is not None,
              "explainability_supported_flag": explainability["mean"] is not None,
              "trust_score": trust["mean"], "explainability_score": explainability["mean"],
              "trust_score_name": "task_output_operating_stability", "explainability_score_name": "independent_deletion_retention_faithfulness",
              "trust_status": trust["status"], "explainability_status": explainability["status"],
              "trust_evidence_sufficient": trust["evidence_sufficient"], "explainability_evidence_sufficient": explainability["evidence_sufficient"],
              "trust_evaluation": trust, "explainability_evaluation": explainability,
              "trust_independent_inputs": trust["independent_inputs"],
              "explainability_independent_inputs": explainability["independent_inputs"],
              "trust_duration_s": trust["duration_s"], "explainability_duration_s": explainability["duration_s"],
              "trust_insufficiency_reasons": trust["insufficiency_reasons"],
              "explainability_insufficiency_reasons": explainability["insufficiency_reasons"],
              "metric_evaluation_provenance": provenance, "metric_evaluation_resumed": resumed,
              "perturbation_duration_s": time.monotonic() - start, "perturbation_sample_count": len(records),
              "perturbation_truncated_flag": bool(trust["time_budget_exhausted"] or explainability["time_budget_exhausted"]),
              "perturbation_samples": records}
    return _jsonable(result)
