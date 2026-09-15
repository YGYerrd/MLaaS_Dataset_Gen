"""Select raw client candidates, prepare their union, then restore client order."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

import numpy as np

from .accounting import append_accounting_stage, update_accounting
from .fedavg_partition import _take_rows

_ACTIVE = ContextVar("training_row_selection", default=None)
RAW_AXES = {"class_label", "score_bin", "question_type", "domain_source", "text_domain",
            "image_domain", "source_domain", "query_domain", "document_domain", "pair_type",
            "relevance_label", "task_format"}
PREPARED_AXES = {"entity_present_sentence", "entity_token_label", "supervised_token_bucket",
                 "prompt_length_bucket", "query_length_bucket", "target_length_bucket", "masked_token_id", "answer_vocab"}


def selection_blocker(config):
    if config.get("generation_mode", "independent") != "independent" or config.get("fl_run_id"):
        return "shared-parent/FedAvg uses its existing prepared partition contract"
    if config.get("_prepared_train_partition") is not None:
        return "the training partition is already prepared"
    if config.get("training_regime") in {"inference", "inference_only"}:
        return "inference-only execution has no client training selection"
    strategy = config.get("split_strategy") or config.get("distribution_type") or "iid"
    axis = config.get("skew_axis")
    if strategy in {"dirichlet", "shard"}:
        if not axis:
            raise ValueError(f"{strategy} requires an explicit skew_axis")
        if axis not in RAW_AXES | PREPARED_AXES:
            raise ValueError(f"unknown explicit skew_axis: {axis}")
        if axis not in RAW_AXES:
            return f"explicit prepared-feature skew axis: {axis}"
    return None


def active_selection():
    active = _ACTIVE.get()
    return active if active is not None and not any(selection_blocker(c) for c in active.configs) else None


def take_rows(value, indices):
    return value.select(list(map(int, indices))) if hasattr(value, "select") else _take_rows(value, np.asarray(indices, dtype=int))


def select_metadata(meta, indices, total, *, split="train"):
    result = dict(meta)
    rows = meta.get(f"selected_{split}_rows", range(total))
    if len(rows) != total:
        raise ValueError("raw source IDs do not align with the client selection pool")
    result[f"selected_{split}_rows"] = [rows[int(i)] for i in indices]
    if meta.get(f"{split}_group_ids") is not None:
        result[f"{split}_group_ids"] = [meta[f"{split}_group_ids"][int(i)] for i in indices]
    if isinstance(meta.get("split_sidecars"), dict) and meta["split_sidecars"].get(split):
        result["split_sidecars"] = {**meta["split_sidecars"], split: {
            key: [values[int(i)] for i in indices] for key, values in meta["split_sidecars"][split].items()}}
    return result


def _concat(values, *, fill=0):
    if len(values) == 1:
        return values[0]
    if isinstance(values[0], dict):
        return {key: _concat([v[key] for v in values], fill=fill if key == "input_ids" else 0) for key in values[0]}
    from .image_store import DiskArrayStore, close_image_stores
    template = next((v for v in values if isinstance(v, DiskArrayStore)), None)
    if template is not None:
        combined = type(template)(dtype=template.dtype, sample_ndim=template.ndim - 1)
        try:
            for value in values:
                for row in value:
                    combined.append(row)
        except BaseException:
            combined.close()
            raise
        close_image_stores(values)
        return combined
    if all(isinstance(v, np.ndarray) for v in values):
        shapes = [v.shape[1:] for v in values]
        if len(set(shapes)) > 1:
            shape = np.max(np.asarray(shapes), axis=0)
            values = [np.pad(v, [(0, 0)] + [(0, int(n - old)) for old, n in zip(v.shape[1:], shape)],
                             constant_values=fill) for v in values]
        return np.concatenate(values)
    return [row for value in values for row in value]


class TrainingRowSelection:
    def __init__(self, configs):
        self.configs, self.selections = configs, {}
        self.indices = self.meta = None

    def _plan(self, x, y, meta):
        from ..clients.partitioning import client_partition
        from ..services.runner import _stable_sample_seed, _resolved_task_type
        from ..services.taxonomy import canonical_task_family

        total = len(y) if y is not None else len(x)
        rows = meta.get("selected_train_rows", range(total))
        source_ids = [f"{meta.get('train_split') or 'train'}:{i}" for i in rows]
        for original in self.configs:
            config = dict(original)
            if config.get("sample_seed") is None:
                config["sample_seed"] = _stable_sample_seed(config)
            if config.get("sample_size") is None:
                config["sample_size"] = max(1, round(total * float(config.get("sample_frac") or 1)))
            config.setdefault("client_partition_index", 0)
            config["split_strategy"] = config.get("split_strategy") or config.get("distribution_type") or "iid"
            task = canonical_task_family(_resolved_task_type(config, meta), meta.get("hf_task") or config.get("hf_task"))
            partition = client_partition(x, y, config, source_ids=source_ids, meta=meta, task_family=task)
            self.selections[original["service_id"]] = dict(config=config, partition=partition,
                order=partition["candidate_indices"], target=min(int(config["sample_size"]), total),
                cursor=0, accepted=[], attempted=[], source_ids=source_ids)
        return total

    def _next(self, accepted, rejected):
        needed = set()
        for item in self.selections.values():
            while len(item["accepted"]) < item["target"] and item["cursor"] < len(item["order"]):
                row = int(item["order"][item["cursor"]])
                if row in accepted:
                    item["accepted"].append(row)
                elif row not in rejected:
                    deficit = item["target"] - len(item["accepted"])
                    # Stop at the deficit; do not read/tokenize the rest of the source.
                    pending = []
                    for i in item["order"][item["cursor"]:]:
                        if int(i) not in rejected:
                            pending.append(int(i))
                        if len(pending) == deficit:
                            break
                    needed.update(i for i in pending if i not in accepted)
                    break
                item["attempted"].append(row)
                item["cursor"] += 1
        return np.asarray(sorted(needed), dtype=int)

    def _finish(self, meta, raw_meta, total, attempted):
        result = select_metadata(raw_meta, self.indices, total)
        result.update({k: v for k, v in meta.items() if k not in {"selected_train_rows", "train_group_ids", "split_sidecars"}})
        if meta.get("split_sidecars"):
            result["split_sidecars"] = {**meta["split_sidecars"], **{
                "train": result.get("split_sidecars", {}).get("train", {})}}
        result.pop("accepted_train_positions", None)
        result.pop("_preparation_cache", None)
        result.pop("_vqa_source_answers", None)
        result["tokenization_selection"] = dict(mode="client_rows_first", source_pool_count=total,
            selected_training_count=len(self.indices), candidate_training_count=attempted,
            rejected_training_count=attempted - len(self.indices), client_count=len(self.selections),
            selection_semantics="raw_source_candidates_v1", evaluation_selection="unchanged",
            padding_scope="selected_train_and_evaluation" if meta.get("dynamic_padding") else "fixed_max_length")
        result = append_accounting_stage(result, stage="client_row_selection", split="train",
                                         input_record_count=total, emitted_record_count=len(self.indices))
        result = update_accounting(result, tokenized_record_count=attempted, post_filter_record_count=len(self.indices),
                                   sequence_count=len(self.indices), metric_instance_count=len(self.indices))
        self.meta = result
        return result

    def select(self, ds, labels, meta, *, dynamic_padding=False):
        """Row-preserving preprocessors call this after fitting shared labels."""
        total = self._plan(ds, labels, meta)
        for item in self.selections.values():
            item["accepted"] = item["order"][:item["target"]].tolist()
            item["attempted"] = list(item["accepted"])
        self.indices = np.unique([i for item in self.selections.values() for i in item["accepted"]])
        meta = self._finish({**meta, "dynamic_padding": dynamic_padding}, meta, total, len(self.indices))
        print(f"Preparation: selected {len(self.indices):,}/{total:,} training rows for {len(self.selections)} client(s)", flush=True)
        return take_rows(ds, self.indices), take_rows(labels, self.indices), meta

    def prepare(self, train, test, meta, prepare):
        """Prepare candidate unions, advancing only by each client's deficit."""
        x, y = train
        total = self._plan(x, y, meta)
        accepted, rejected, chunks, chunk_indices = set(), set(), [], []
        preparation_cache = {}
        final_test = final_meta = None
        while True:
            indices = self._next(accepted, rejected)
            if not len(indices):
                break
            print(f"Preparation: selected {len(indices):,} candidate training rows from {total:,}", flush=True)
            batch_meta = select_metadata(meta, indices, total)
            batch_meta["client_candidate_preparation"] = True
            batch_meta["_preparation_cache"] = preparation_cache
            try:
                with suspend_selection():
                    prepared, prepared_test, prepared_meta = prepare(
                        (take_rows(x, indices), take_rows(y, indices) if y is not None else None), test, batch_meta)
            except ValueError as exc:
                # Only explicit row-eligibility failures permit replacement.
                if not str(exc).startswith(("insufficient_supervision: generation train", "all samples failed preprocessing for split='train'",
                                            "no samples survived preprocessing for split='train'",
                                            "HF multimodal split 'train' produced zero valid examples")):
                    raise
                rejected.update(indices.tolist())
                continue
            positions = prepared_meta.get("accepted_train_positions")
            if positions is None:
                if len(prepared[1]) != len(indices):
                    raise ValueError("filtering preprocessor must report accepted_train_positions")
                positions = np.arange(len(indices))
            positions = np.asarray(positions, dtype=int)
            if len(positions) != len(prepared[1]) or len(np.unique(positions)) != len(positions) or np.any((positions < 0) | (positions >= len(indices))):
                raise ValueError("invalid accepted_train_positions from preprocessor")
            kept = indices[positions]
            accepted.update(kept.tolist())
            rejected.update(set(indices.tolist()) - set(kept.tolist()))
            chunks.append(prepared)
            chunk_indices.extend(kept.tolist())
            final_test, final_meta = prepared_test, prepared_meta
        if not accepted:
            raise ValueError("insufficient_supervision: source pool exhausted without valid training examples")
        self.indices = np.asarray(sorted(accepted), dtype=int)
        order = np.argsort(chunk_indices)
        result = (take_rows(_concat([c[0] for c in chunks], fill=int(final_meta.get("pad_token_id") or 0)), order),
                  take_rows(_concat([c[1] for c in chunks], fill=int(final_meta.get("ignore_index", -100))), order))
        if isinstance(result[0], dict) and "input_ids" in result[0]:
            # Replacement batches may have a different dynamic padding width.
            final_meta["input_shape"] = tuple(result[0]["input_ids"].shape[1:])
            if np.asarray(result[1]).ndim == 2:
                final_meta["target_shape"] = tuple(result[1].shape[1:])
        final_meta = self._finish(final_meta, meta, total, len(accepted) + len(rejected))
        if final_meta.get("label_granularity") == "token" and isinstance(result[1], np.ndarray):
            final_meta = update_accounting(final_meta,
                supervised_token_count=int(np.count_nonzero(result[1] != int(final_meta.get("ignore_index", -100)))))
        final_meta.pop("client_candidate_preparation", None)
        self.meta = final_meta
        return result, final_test, final_meta

    def materialize(self, service_id, train):
        if service_id not in self.selections:
            return None
        if len(train[1]) != len(self.indices):
            raise ValueError("prepared training rows no longer align with the recorded selection")
        from ..services.runner import ServiceRunner
        from ..provenance import sha256_json
        item = self.selections[service_id]
        rows = item["accepted"]
        positions = np.searchsorted(self.indices, rows)
        doc = deepcopy(item["partition"]["metadata"])
        doc.update(source_ids=[item["source_ids"][i] for i in rows], effective_quota=len(rows),
                   capacity_reduction_count=doc["requested_quota"] - len(rows),
                   accepted_source_row_ids=[self.meta["selected_train_rows"][i] for i in positions],
                   attempted_candidate_count=len(item["attempted"]), rejected_candidate_count=len(item["attempted"]) - len(rows),
                   resolution_reason="requested_target_met" if len(rows) == doc["requested_quota"] else "source_pool_exhausted")
        if "bucket_ids" in item["partition"]:
            from ..clients.partitioning import update_bucket_evidence
            update_bucket_evidence(doc, item["partition"]["bucket_ids"][rows], item["partition"]["bucket_labels"])
        doc.pop("partition_hash", None)
        doc["partition_hash"] = sha256_json(doc)
        split = ServiceRunner(item["config"])._split_info_from_prepared_partition({
            "x_train": take_rows(train[0], positions), "y_train": take_rows(train[1], positions),
            "partition_meta": doc, "split_resolution": {"strategy": doc["effective_strategy"]}}, meta=self.meta)
        split["resolved"].update({key: doc[key] for key in ("accepted_source_row_ids", "attempted_candidate_count",
                                                         "rejected_candidate_count", "resolution_reason")})
        return split


@contextmanager
def training_row_selection(configs):
    selection = TrainingRowSelection(configs)
    token = _ACTIVE.set(selection)
    try:
        yield selection
    finally:
        _ACTIVE.reset(token)


@contextmanager
def suspend_selection():
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def report_tokenization_policy(meta, dataset_args):
    if meta.get("client_candidate_preparation"):
        return
    active = _ACTIVE.get()
    reason = next((r for c in active.configs if (r := selection_blocker(c))), None) if active else "no client selection context"
    if reason:
        meta["tokenization_selection"] = {"mode": "full_pool", "reason": reason}
        print(f"Preparation: full training pool | {reason}", flush=True)


def select_training_rows(ds, labels, meta, *, dynamic_padding=False):
    active = active_selection()
    return active.select(ds, labels, meta, dynamic_padding=dynamic_padding) if active else (ds, labels, meta)


def finalize_selected_accounting(meta, train):
    if meta.get("tokenization_selection", {}).get("mode") == "client_rows_first":
        meta = update_accounting(meta, tokenized_record_count=meta["tokenization_selection"]["candidate_training_count"],
                                 sequence_count=len(train[1]), metric_instance_count=len(train[1]))
        active = active_selection()
        if active is not None:
            active.meta = meta
    return meta
