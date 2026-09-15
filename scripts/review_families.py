"""Patch-aware, read-only catalogue screening and isolated family review artifacts.

Run with the workspace Python. Historical SQLite inputs are always opened mode=ro.
This utility never promotes a historical score to a current-contract clearance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import importlib.metadata
import platform
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/family_review_20260913'
METRICS = {'accuracy', 'f1', 'macro_f1', 'pearson', 'spearman', 'loss', 'train_loss',
    'rmse', 'mse', 'r2', 'iou', 'mean_iou', 'map', 'rouge1', 'rougeL', 'metric_score',
    'metric_name', 'masked_accuracy', 'train_distribution_json', 'benchmark_distribution_json',
    'optimizer_step_count', 'successful_optimizer_step_count', 'skipped_optimizer_step_count',
    'train_stopped_early', 'train_stop_reason', 'training_runtime_s', 'evaluation_runtime_s',
    'service_runtime_s', 'checkpoint_runtime_s', 'training_epochs', 'learning_rate', 'optimizer',
    'batch_size', 'gradient_accumulation_steps', 'precision_type', 'max_grad_norm',
    'gradient_clipped_step_count', 'training_budget_adjustment'}
METRICS.update({'baseline_local_mean_rmse','baseline_ridge_rmse','baseline_linear_rmse','metric_stat_empty_outputs',
                'eval_sample_count','prediction_std','prediction_variance','r@1','r@5','mrr',
                'eval_loss_candidate_count','contrastive_candidates_min','contrastive_candidates_max'})
METRICS.update({'logical_batch_size','micro_batch_size_effective','effective_optimizer_batch_size',
                'precision_type_effective','mixed_precision_effective','contrastive_batch_adjusted'})
METRICS.update({'checkpoint_s','checkpoint_model_restore_verify_s','checkpoint_state_capture_s',
                'checkpoint_state_hash_s','checkpoint_compression_s','checkpoint_publish_s',
                'checkpoint_provenance_write_s','checkpoint_reduced_quality_s','post_checkpoint_s'})
METRICS.update({'task_semantics_hash','evaluation_protocol_hash'})
KNOBS = ('optimizer', 'learning_rate', 'batch_size', 'gradient_accumulation_steps',
         'training_epochs', 'weight_decay', 'warmup_ratio')
RUNTIME = ('precision_type','mixed_precision','max_grad_norm','mobilebert_head_normalization')
PROV = ('family_id', 'model_revision', 'dataset_revision', 'label_map_hash',
    'preprocessing_contract_hash', 'dataset_contract_hash', 'logical_dataset_contract_hash', 'parameter_schema_hash',
    'initial_state_hash', 'family_document_ref', 'family_document_hash',
    'compatibility_contract_hash', 'generation_mode', 'composition_compatible')


def decode(v, default=None):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            pass
    return default if v is None else v


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')


def write_csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


@lru_cache(maxsize=1)
def intended_runtime():
    return {'requirements_sha256':hashlib.sha256((ROOT/'requirements.txt').read_bytes()).hexdigest(),
            'versions':{'python':platform.python_version(),'platform':platform.platform(),
                        **{name:importlib.metadata.version(name) for name in ('torch','transformers','datasets')}}}


def runtime_matches(r):
    expected=intended_runtime()
    actual=r.get('runtime') or {}
    keys=('python','platform','torch','transformers','datasets') if r.get('hf_task') else ('python','platform','torch')
    return (actual.get('requirements_sha256')==expected['requirements_sha256']
            and all(actual.get('versions',{}).get(k)==expected['versions'][k] for k in keys))


def distribution(v):
    v = decode(v, {})
    if not isinstance(v, dict):
        return {}
    if 'top_supervised_token_ids' in v:
        counts = v['top_supervised_token_ids']
        return counts if len(counts) == v.get('unique_supervised_token_ids') and sum(counts.values()) == v.get('supervised_tokens') else {}
    return v if v and all(finite(n) and n >= 0 for n in v.values()) else {}


def quality(r):
    from mlaas_data_generator.clients.calibration import outcome
    m = r['metrics']
    task = r.get('hf_task')
    if task == 'sentence_similarity' and r.get('dataset_config') not in {'mrpc', 'wnli', 'qqp', 'rte', 'qnli'}:
        # A pair-input task alone is not proof of a continuous target contract.
        if r['loader'].get('task_type') != 'regression':
            return 'unverified_metric_contract'
        return ('healthy' if min(m['pearson'], m['spearman']) >= .3 else 'weak') if all(finite(m.get(k)) for k in ('pearson', 'spearman')) else 'missing_quality'
    if task in {'sequence_classification', 'image_classification', 'token_classification', 'sentence_similarity'} or r.get('task_type') == 'classification':
        average = 'macro' if task == 'image_classification' else 'weighted'
        return outcome(m.get('accuracy'), m.get('macro_f1' if average == 'macro' else 'f1'),
                       distribution(m.get('benchmark_distribution_json')), average=average) or 'missing_quality'
    return 'needs_initial_baseline'


def patch_ids(r):
    model, dataset, subset = (r.get('model_id') or '').lower(), (r.get('dataset_name') or '').lower(), r.get('dataset_config')
    patches = []
    if 'mobilebert' in model:
        patches += ['mobilebert_stability', 'fp32_mobile_models']
    elif any(x in model for x in ('mobilevit', 'mobilenet', 'efficientnet')):
        patches += ['fp32_mobile_models']
    if 'ade20k-mini' in dataset:
        patches += ['ade20k_three_labels']
    if 'dpt' in model or 'upernet-convnext' in model:
        patches += ['rocm_conv_layout']
    if r.get('hf_task') == 'text_image_retrieval':
        patches += ['contrastive_batch']
    if r.get('hf_task') == 'sequence_classification' or subset in {'mrpc', 'wnli'}:
        patches += ['task_label_contract']
    if r.get('hf_task') in {'token_classification','image_segmentation','seq2seq_generation'}:
        patches += ['task_contract_payload']
    return patches


PATCHES = {
 'mobilebert_stability': {'source': 'docs/mobilebert-stability.md', 'activation': 'mobilebert_head_normalization=layer_norm_v1; max_grad_norm=1.0', 'evidence_limit': 'Six full-checkpoint retries; MRPC is at majority baseline; old/new evaluation membership differs.'},
 'fp32_mobile_models': {'source': 'docs/generation-policies.md', 'activation': 'Runtime precision policy selects FP32 and disables AMP', 'evidence_limit': 'Finite canaries are not quality certification.'},
 'ade20k_three_labels': {'source': 'docs/ade20k-mini-mask-audit.md', 'activation': 'New frozen preparation; declared three-label head with valid background 0', 'evidence_limit': 'Old 150-label metrics cannot validate corrected task semantics.'},
 'rocm_conv_layout': {'source': 'docs/upernet-layout-diagnostic.md', 'activation': 'New Windows ROCm process; scoped contiguous-convolution guard', 'evidence_limit': 'Controlled two-image timings, not end-to-end dataset quality.'},
 'contrastive_batch': {'source': 'docs/clip-contrastive-batch-diagnostic.md', 'activation': 'Physical training batches >=2 pairs and pooled evaluation candidates', 'evidence_limit': 'Synthetic update/memory test, not retrieval quality.'},
 'task_label_contract': {'source': 'outputs/checks/run_failure_fixes_20260910/report.md', 'activation': 'Current shared task resolver and fresh preparation with known label task', 'evidence_limit': 'Old unknown-label preparations and binary regression aliases are incompatible.'},
 'checkpoint_roundtrip': {'source': 'outputs/checks/run_failure_fixes_20260910/report.md', 'activation': 'Current full checkpoint path with persisted verification', 'evidence_limit': 'No-weights retries cannot verify saved artifacts.'},
 'task_contract_payload': {'source': 'docs/family-review-fixes.md', 'activation': 'Restart worker with selective task-contract normalization', 'evidence_limit': 'Fixes runtime-data traversal and retains prediction counts; does not establish model quality.'},
}

_DOCS = {}


def checked_document(r):
    p = r['provenance']
    ref, expected = p.get('family_document_ref'), p.get('family_document_hash')
    if not ref or not expected:
        return None
    try:
        if (ref, expected) not in _DOCS:
            payload = Path(ref).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                return None
            doc = json.loads(payload)
            _DOCS[(ref, expected)] = doc
        doc = _DOCS[(ref, expected)]
        # A cached document is not cached proof for another service's provenance.
        if doc['family_id'] != p['family_id'] or doc['initial_state_hash'] != p['initial_state_hash']:
            return None
        for key in ('parameter_schema_hash','label_map_hash','preprocessing_contract_hash','logical_dataset_contract_hash'):
            if not p.get(key) or doc.get('compatibility',{}).get(key)!=p[key]:
                return None
        for public, recorded in [('hf_model_revision','model_revision'),('hf_dataset_revision','dataset_revision')]:
            if (r.get('hf_task') and not p.get(recorded)) or doc['config'].get(public)!=p.get(recorded):
                return None
        if doc.get('contracts', {}).get('labels', {}).get('task_family') in {None,'unknown'}:
            return None
        reservation = doc['reservation']
        if set(reservation['catalogue_metric_ids']) & set(reservation['audit_ids']):
            return None
        return doc
    except (OSError, ValueError, KeyError):
        return None


def active_patch_gaps(r, doc):
    gaps = []
    cfg = r['config']
    for patch in r['patches']:
        if patch == 'mobilebert_stability' and (cfg.get('mobilebert_head_normalization') != 'layer_norm_v1' or cfg.get('max_grad_norm') != 1.0
                or (doc or {}).get('config', {}).get('mobilebert_head_normalization') != 'layer_norm_v1'):
            gaps.append('mobilebert_patch_not_enabled')
        if patch == 'fp32_mobile_models' and (cfg.get('precision_type') != 'fp32' or cfg.get('mixed_precision')):
            gaps.append('fp32_policy_not_demonstrated')
        if patch == 'ade20k_three_labels':
            labels = (doc or {}).get('contracts', {}).get('labels', {})
            if labels.get('num_labels', labels.get('num_classes')) != 3:
                gaps.append('corrected_three_label_contract_unverified')
        if patch in {'rocm_conv_layout', 'contrastive_batch','task_contract_payload'} and not any('family_review_20260913' in s for s in r['sources']):
            gaps.append('production_patch_activation_unverified')
    return gaps


def representatives(members):
    eligible = []
    for r in members:
        if r['engineering_gaps']:
            continue
        doc = checked_document(r)
        if doc is None or active_patch_gaps(r, doc):
            continue
        eligible.append(r)
    groups = defaultdict(list)
    for r in eligible:
        groups[(r['provenance']['family_id'],r['provenance'].get('family_document_hash'))].append(r)
    for family, rows in sorted(groups.items()):
        healthy = [r for r in rows if r['quality'] in {'healthy','weak'} and r['config'].get('split_strategy') == 'iid'
                   and training_support(r)]
        pairs = defaultdict(list)
        for r in healthy:
            pairs[digest({k:r['config'].get(k) for k in (*KNOBS,*RUNTIME)})].append(r)
        for recipe, candidates in sorted(pairs.items(), key=lambda item: (
                not any(r['quality']=='healthy' for r in item[1]),-len(item[1]),item[0])):
            seeds = {}
            for r in sorted(candidates,key=lambda r:(r['quality']!='healthy',r['service_id'])):
                seeds[(r['config'].get('sample_seed'),r['config'].get('training_seed'))] = r
            if len(seeds) >= 2 and len(rows) >= 3:
                picked = list(seeds.values())[:2]
                third = sorted((r for r in rows if r not in picked), key=lambda r:(r['quality']!='healthy',r['config'].get('optimizer')==picked[0]['config'].get('optimizer'),r['service_id']))[0]
                if any(r['quality']=='healthy' and r['config'].get('split_strategy')=='iid' for r in picked+[third]):
                    return picked + [third]
    return []


def training_support(r):
    if r.get('hf_task')=='text_image_retrieval':
        return r['metrics'].get('contrastive_candidates_min',0)>=2
    return sum(v>0 for v in distribution(r['metrics'].get('train_distribution_json')).values())>=2


def pilot_quality(r):
    label=quality(r)
    if label!='needs_initial_baseline':return label
    doc=checked_document(r)
    if not doc:return 'missing_initial_baseline'
    path=OUT/'initial_baselines'/f"{doc['family_id']}.json"
    if not path.is_file():return 'missing_initial_baseline'
    initial=json.loads(path.read_text())
    if (initial.get('status')!='completed' or initial.get('reservation_hash')!=doc['reservation']['reservation_hash']
            or initial.get('document_hash')!=r['provenance']['family_document_hash']
            or initial.get('initial_state_hash')!=doc['initial_state_hash'] or not finite(initial.get('primary'))):
        return 'incompatible_initial_baseline'
    m=r['metrics']
    if r.get('task_type')=='regression' and not r.get('hf_task'):
        values=doc['reservation']['metric_source_labels']
        selected=[values[i] for i in doc['reservation']['catalogue_positions']]
        selected=[v[0] if isinstance(v,list) and len(v)==1 else v for v in selected]
        if not selected or not all(finite(v) for v in selected):return 'unverified_prediction_spread'
        constant_rmse=statistics.pstdev(selected)
        # Beating the optimal constant prediction proves outputs are nonconstant.
        return 'healthy' if finite(m.get('rmse')) and m['rmse']<min(initial['primary'],constant_rmse) else 'weak'
    if r.get('hf_task')=='image_segmentation':
        if not initial.get('runtime_matched') or any(initial.get(k)!=r['config'].get(k) for k in ('batch_size','precision_type','mixed_precision')):return 'unverified_initial_runtime'
        if any(not m.get(k) or m[k]!=initial.get('metrics',{}).get(k) for k in ('task_semantics_hash','evaluation_protocol_hash')):
            return 'incompatible_evaluation_protocol'
        predicted=[v for k,v in m.items() if k.startswith('metric_stat_class_') and k.endswith('_pred_total') and finite(v) and v>0]
        return 'healthy' if finite(m.get('iou')) and m['iou']>initial['primary'] and len(predicted)>1 else 'weak_or_unverified_outputs'
    if r.get('hf_task')=='text_image_retrieval':
        n=m.get('eval_loss_candidate_count')
        if (not initial.get('runtime_matched') or any(initial.get(k)!=r['config'].get(k) for k in ('batch_size','precision_type','mixed_precision'))
                or not finite(n) or n<=1 or n!=initial.get('metrics',{}).get('eval_loss_candidate_count')):
            return 'incompatible_retrieval_baseline'
        if finite(m.get('r@1')) and finite(m.get('loss')) and m['r@1']>initial['primary'] and m['loss']<math.log(n):
            return 'healthy'
        return 'weak'
    return 'unverified_native_quality'


def load_observations():
    old=json.loads((OUT/'observations.json').read_text())
    # Supplement the initial compact scan with the stored logical-data binding.
    supplemental={}
    for source in json.loads((OUT/'source_inventory.json').read_text()):
        con=sqlite3.connect((ROOT/source).resolve().as_uri()+'?mode=ro',uri=True)
        con.execute('PRAGMA query_only=ON')
        for sid,document_hash,logical,runtime in con.execute("select service_id,family_document_hash,logical_dataset_contract_hash,json_extract(provenance_json,'$.runtime') from service_generation_provenance"):
            supplemental[(source,sid,document_hash)]={'logical':logical,'runtime':decode(runtime,{})}
        con.close()
    for r in old:
        p=r['provenance']
        extra=next((supplemental.get((s,r['service_id'],p.get('family_document_hash'))) for s in r['sources'] if supplemental.get((s,r['service_id'],p.get('family_document_hash')))),{})
        if not p.get('logical_dataset_contract_hash'):p['logical_dataset_contract_hash']=extra.get('logical')
        r['runtime']=extra.get('runtime',{})
    for r in old:
        r['patches']=patch_ids(r)
        r['engineering_gaps']=engineering(r)
    return old


def finalize():
    import pandas as pd
    from mlaas_data_generator.cli.manifest.review import save_review_manifest
    from mlaas_data_generator.cli.run_manifest import load_manifest,_resolve_row,_validate_row
    from mlaas_data_generator.family_preflight import manifest_family_error
    old=load_observations()
    inventory=json.loads((OUT/'family_inventory.json').read_text())
    mappings=json.loads((OUT/'pilot_mapping.json').read_text())
    mapping={m['service_id']:m for m in mappings}
    expanded=json.loads((OUT/'pilot_expanded.json').read_text())
    requested={r['service_id']:r for r in expanded}
    runs=[]
    if (OUT/'pilot.db').is_file():
        runs=read_catalogue(OUT/'pilot.db')
    if (OUT/'followup_mapping.json').is_file():
        mapping.update({m['service_id']:m for m in json.loads((OUT/'followup_mapping.json').read_text())})
        requested.update({r['service_id']:r for r in json.loads((OUT/'followup_expanded.json').read_text())})
    terminal_ids={r['service_id'] for r in runs}
    holds=[]
    for path in [OUT/'upernet_preparation_limit.json',OUT/'dpt_preparation_limit.json']:
        if path.is_file():holds.append(json.loads(path.read_text(encoding='utf-8-sig')))
    held_ids=sorted(sid for sid,m in mapping.items() if sid not in terminal_ids
                    and any(m['review_family_id']==h['review_family_id'] for h in holds))
    superseded_ids=sorted(m['previous_pilot_id'] for m in mapping.values()
                          if m.get('previous_pilot_id') and m['service_id'] in terminal_ids and m['previous_pilot_id'] not in terminal_ids)
    pending_ids=sorted(set(mapping)-terminal_ids-set(superseded_ids)-set(held_ids))
    dump(OUT/'review_progress.json',{'planned_services':len(mapping),'terminal_services':len(runs),
         'superseded_unexecuted_service_ids':superseded_ids,'pending_service_ids':pending_ids,
         'inconclusive_unexecuted_service_ids':held_ids,'preparation_holds':holds,
         'all_planned_services_resolved':not pending_ids})
    service_results=[]
    for r in runs:
        if r['service_id'] not in mapping:continue
        r['review_family_id']=mapping[r['service_id']]['review_family_id']
        r['observation_id']=digest({'source':r['sources'],'service_id':r['service_id']})
        r['engineering_gaps']=engineering(r)
        r['patches']=patch_ids(r)
        r['quality']=pilot_quality(r)
        cfg=requested[r['service_id']]
        executed=r.get('executed_training_knobs') or r['config']
        mismatches={k:{'requested':cfg.get(k),'executed':executed.get(k)} for k in KNOBS
                    if cfg.get(k) is not None and cfg.get(k)!=executed.get(k)}
        unexplained=dict(mismatches)
        adjustment=r.get('execution_adjustment') or {}
        if adjustment:
            # Preserve the exact persisted explanation; only budget-derived knobs can be explained by it.
            for key in ('training_epochs','gradient_accumulation_steps'):
                if key in unexplained and adjustment.get('requested',{}).get(key)==unexplained[key]['requested'] and adjustment.get('effective',{}).get(key)==unexplained[key]['executed']:
                    unexplained.pop(key,None)
        if unexplained:r['engineering_gaps'].append('recipe_mismatch_requires_execution_evidence')
        service_results.append(dict(service_id=r['service_id'],review_family_id=r['review_family_id'],status=r['status'],
            quality=r['quality'],engineering_gaps=r['engineering_gaps'],family_id=r['provenance'].get('family_id'),
            checkpoint=r['checkpoint'],signature_files=[a['artifact_uri'] for a in r['artifacts'] if a['artifact_type']=='update_signature'],
            metrics=r['metrics'],failures=r['failures'],requested_runtime_mismatches=mismatches,
            recorded_budget_adjustment=adjustment,unexplained_mismatches=unexplained))
    dump(OUT/'pilot_observations.json',runs)
    frozen_contracts={}
    for r in runs:
        doc=checked_document(r)
        if not doc:continue
        p=r['provenance']
        frozen_contracts[p['family_id']]={'family_id':p['family_id'],'family_document_ref':p['family_document_ref'],
            'family_document_hash':p['family_document_hash'],'initial_state_hash':p['initial_state_hash'],
            'compatibility':doc['compatibility'],'reservation_hash':doc['reservation']['reservation_hash'],
            'catalogue_count':len(doc['reservation']['catalogue_metric_ids']),
            'audit_count':len(doc['reservation']['audit_ids']),
            'service_ids':[x['service_id'] for x in runs if x['provenance'].get('family_id')==p['family_id']]}
    dump(OUT/'frozen_pilot_contracts.json',list(frozen_contracts.values()))
    dump(OUT/'service_results.json',service_results)
    write_csv(OUT/'service_results.csv',[{k:v for k,v in r.items() if k not in {'metrics','recorded_budget_adjustment'}} |
        {k:r['metrics'].get(k) for k in ('accuracy','f1','macro_f1','pearson','spearman','rmse','iou','loss','train_loss',
         'successful_optimizer_step_count','optimizer_step_count','training_runtime_s','evaluation_runtime_s','service_runtime_s')}
        for r in service_results])
    all_records={r['observation_id']:r for r in old+runs if r.get('observation_id')}
    reused=json.loads((OUT/'reused_evidence.json').read_text())
    decisions={e['review_family_id']:e for e in reused
               if representatives([all_records[key] for key in e['observation_ids']])}
    for f in inventory:
        f['patches']=sorted(set(f['patches'])|{p for r in old+runs if r.get('review_family_id')==f['review_family_id'] for p in r['patches']})
        rows=[r for r in runs if r.get('review_family_id')==f['review_family_id']]
        hold=next((h for h in holds if h['review_family_id']==f['review_family_id']),None)
        if hold:
            f.update(decision='held for investigation',reason=hold['cause'],pilot_completed=0,pilot_failed=0,
                     pilot_inconclusive=3)
            decisions.pop(f['review_family_id'],None)
            continue
        if not rows:continue
        picked=representatives(rows)
        if picked:
            decisions[f['review_family_id']]={'review_family_id':f['review_family_id'],'family_id':picked[0]['provenance']['family_id'],
                'observation_ids':[r['observation_id'] for r in picked],'service_ids':[r['service_id'] for r in picked]}
            f.update(decision='cleared with recipe restrictions',reason='Current pilot has an IID seed repeat, at least one healthy IID service and verified same-contract representatives; use reviewed healthy IID recipes only.')
        else:
            f.update(decision='held for investigation',reason='Pilot does not establish eligible same-contract representatives with an IID seed repeat and at least one healthy service: '+str(dict(Counter(r['quality'] for r in rows))))
        f['pilot_completed']=sum(r['status']=='completed' for r in rows)
        f['pilot_failed']=sum(r['status']=='failed' for r in rows)
        f['pilot_quality_values']={k:[r['metrics'][k] for r in rows if finite(r['metrics'].get(k))]
                                  for k in ('accuracy','f1','macro_f1','pearson','spearman','rmse','iou','r@1')
                                  if any(finite(r['metrics'].get(k)) for r in rows)}
    # Produce the future large-client roster; do not execute it.
    final,clearance,validation=[],[],[]
    for fid,e in sorted(decisions.items()):
        selected=[all_records[key] for key in e['observation_ids']]
        allowed=[r for r in selected if r['quality']=='healthy' and r['config'].get('split_strategy')=='iid']
        # Do not hide a same-contract, same-recipe execution failure behind a
        # successful subset of representatives. A changed/fixed recipe is scoped
        # separately; unresolved failures under an identical recipe stay held.
        failures=[r for r in all_records.values() if r['status']=='failed'
                  and r['provenance'].get('family_id')==e['family_id']]
        allowed=[candidate for candidate in allowed if not any(
            all(failed['config'].get(k)==candidate['config'].get(k) for k in (*KNOBS,*RUNTIME))
            for failed in failures)]
        if not allowed:
            next(f for f in inventory if f['review_family_id']==fid).update(
                decision='held for investigation',reason='No healthy recipe remains free of unexplained same-contract execution failures.')
            continue
        assert allowed and all(not r['engineering_gaps'] for r in selected)
        assert len({r['provenance']['family_id'] for r in selected})==1
        assert len({r['provenance']['family_document_hash'] for r in selected})==1
        template=allowed[0]
        templates=list({digest({**{k:r['config'].get(k) for k in (*KNOBS,*RUNTIME)},'sample_size':r['config'].get('sample_size')}):r for r in allowed}.values())
        template=templates[0]
        configs=[]
        for source in templates:
            cfg=recipe_from_record(source)
            cfg.update(family_document_ref=source['provenance']['family_document_ref'],
                       family_document_hash=source['provenance']['family_document_hash'],
                       db_path=str(OUT/'final_services.db'))
            configs.append(cfg)
        for i in range(100):
            row=dict(configs[i%len(configs)])
            row.update(service_id=f'{fid}__final_{i+1:03d}',sample_seed=1500000+i,training_seed=1600000+i)
            row['effective_batch_size']=int(row.get('batch_size',16))*int(row.get('gradient_accumulation_steps',1))
            check=_validate_row(_resolve_row(pd.Series(row),{}))
            error=manifest_family_error(row)
            if not check.ok or error:raise ValueError(f'Invalid final row {row["service_id"]}: {check} {error}')
            final.append(row)
        clearance.append({**e,'model':template['model_id'] or template['model_type'],'dataset':template['dataset_name'],
            'subset':template['dataset_config'],'family_document_ref':cfg['family_document_ref'],
            'family_document_hash':cfg['family_document_hash'],
            'restriction':'Only the listed healthy IID joint recipes and their sample quantities; vary sample/training seeds only.',
            'template_service_id':template['service_id'],'knobs':{k:configs[0].get(k) for k in KNOBS},
            'sample_size':configs[0].get('sample_size'),'final_clients':100,
            'allowed_recipes':[{'source_service_id':r['service_id'],'knobs':{k:r['config'].get(k) for k in KNOBS},
                                'runtime':{k:r['config'].get(k) for k in RUNTIME},
                                'sample_size':r['config'].get('sample_size')} for r in templates]})
    save_review_manifest(pd.DataFrame(final),OUT/'final_manifest.csv')
    frame,defaults=load_manifest(OUT/'final_manifest.csv')
    by_family={c['review_family_id']:c for c in clearance}
    for _,row in frame.iterrows():
        cfg=_resolve_row(row,defaults)
        cleared=by_family[cfg['service_id'].split('__final_')[0]]
        recipe_matches=cfg.get('split_strategy')=='iid' and any(
            all(cfg.get(k)==recipe['knobs'].get(k) for k in KNOBS)
            and all(cfg.get(k)==recipe['runtime'].get(k) for k in RUNTIME)
            and cfg.get('sample_size')==recipe['sample_size'] for recipe in cleared['allowed_recipes'])
        validation.append({'service_id':cfg['service_id'],'valid':_validate_row(cfg).ok,
                           'recipe_allowed':recipe_matches,
                           'clearance_bound':cfg.get('family_document_hash')==cleared['family_document_hash'],
                           'family_document_hash':cfg.get('family_document_hash')})
    assert all(v['valid'] and v['recipe_allowed'] and v['clearance_bound'] for v in validation)
    assert len(frame)==100*len(clearance)
    dump(OUT/'cleared_families.json',clearance)
    evidence_ids={key for c in clearance for key in c['observation_ids']}
    dump(OUT/'runtime_comparison.json',{'intended_runtime':intended_runtime(),
        'all_clearance_evidence_matches':all(runtime_matches(all_records[key]) for key in evidence_ids),
        'evidence':[{'service_id':all_records[key]['service_id'],'runtime':all_records[key].get('runtime'),
                     'matches':runtime_matches(all_records[key])} for key in sorted(evidence_ids)]})
    dump(OUT/'patch_evidence.json',PATCHES)
    patch_status=[]
    for f in inventory:
        observations=[r for r in old+runs if r.get('review_family_id')==f['review_family_id'] and r['provenance'].get('family_id')]
        for patch in f['patches']:
            enabled=[r for r in observations if checked_document(r) and not active_patch_gaps({**r,'patches':[patch]},checked_document(r))]
            patch_status.append({'review_family_id':f['review_family_id'],'patch':patch,'implementation_source':PATCHES[patch]['source'],
                'patch_present':(ROOT/PATCHES[patch]['source']).is_file(),
                'enabled_frozen_contracts':len({r['provenance']['family_id'] for r in enabled}),
                'verified_completed_services':sum(not r.get('engineering_gaps') for r in enabled),
                'validated_service_ids':[r['service_id'] for r in enabled if not r.get('engineering_gaps')],
                'healthy_service_ids':[r['service_id'] for r in enabled if not r.get('engineering_gaps') and r['quality']=='healthy'],
                'family_decision':f['decision'],'evidence_limit':PATCHES[patch]['evidence_limit']})
    write_csv(OUT/'family_patch_status.csv',patch_status)
    dump(OUT/'final_manifest_validation.json',{'rows':len(frame),'cleared_families':len(clearance),'all_rows_valid':True,
         'all_rows_bound_to_clearance':all(v['clearance_bound'] for v in validation),
         'all_recipes_within_clearance':all(v['recipe_allowed'] for v in validation)})
    dump(OUT/'family_inventory.json',inventory)
    write_csv(OUT/'family_inventory.csv',inventory)
    lines=['# Family review before client generation','',
       f'Screened {len(old):,} stored observations across {len(inventory)} historical candidate groups. Candidate groups are not frozen composition families.',
       f'Cleared {len(clearance)} specific frozen families with recipe restrictions. Final manifest: {len(final):,} new clients (100 per cleared family); not executed.',
       f'Pilot outcomes: {dict(Counter(r["status"] for r in runs))}. Historical-only candidates are held out as requested.','',
       f'Review execution: {len(pending_ids)} pending checks; {len(superseded_ids)} uncompleted original checks superseded by completed follow-ups. Superseded attempts are not training failures and were not inserted as services.', '',
       '## Clearance and limits','',
       'Clearance requires three verified representatives under one frozen family, an IID recipe repeated across two seeds, and at least one healthy IID service. The final roster cycles through the explicitly reviewed healthy IID recipes and their quantities, with new sample/training seeds. Broader optimizer or skew ranges are not cleared by this bundle.',
       'Classification uses accuracy and task-specific F1 relative to their own majority baselines (+0.05 each). Sentence-score regression requires Pearson and Spearman >=0.3. Other tasks require a matching initial-model baseline and evidence of nonconstant outputs.',
       'Original databases were opened read-only. Old failures remain historical; successful smoke tests do not rewrite them. Reserved audit data was not used for recipe selection. Prospective quality is not guaranteed.',
       '', '## Current pilot findings','', '| Family | Completed | Failed | Stored quality ranges | Decision | Reason |','|---|---:|---:|---|---|---|']
    for f in inventory:
        if 'pilot_completed' in f:
            ranges='; '.join(f'{k}: {min(v):.4f}–{max(v):.4f}' for k,v in f.get('pilot_quality_values',{}).items()) or 'No completed quality measurement'
            lines.append(f'| {f["model"]} / {f["dataset"]} / {f["subset"] or ""} | {f["pilot_completed"]} | {f["pilot_failed"]} | {ranges} | {f["decision"]} | {f["reason"]} |')
    from scripts.family_review_report import supplement
    lines+=supplement(OUT,old,runs,clearance,inventory,holds)
    lines+=['','## Files','',
       '- `family_inventory.csv`: all candidates, historical quality, failures, recipe/data profiles, timings, and decisions.',
       '- `patch_evidence.json`: fixes, activation requirements, and limits of earlier diagnostics.',
       '- `reused_evidence.json`: exact historical representative IDs reused for clearance.',
       '- `pilot.csv` and `pilot.families.json`: frozen representative recipes.',
       '- `followup.csv` and `followup.families.json`: the one bounded follow-up per selected family; `followup_mapping.json` records each diagnosis/intervention.',
       '- `frozen_pilot_contracts.json`: actual preparation, initialization and evaluation bindings for the executed pilots.',
       '- `pilot.db`, `service_results.csv`, and `initial_baselines/`: isolated outcomes and matched initial evaluations.',
       '- `cleared_families.json`: exact frozen families, evidence IDs, and recipe restrictions.',
       '- `final_manifest.csv` and `final_manifest.families.json`: validated future roster; keep the pair together.',
       '- `final_manifest_validation.json`: row and clearance-binding checks.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Final manifest:',len(final),'clients;',len(clearance),'cleared frozen families')


def recipe_from_record(r):
    doc = checked_document(r)
    cfg = dict(doc['config'] if doc else r['loader']['dataset_args'])
    cfg.update({k:r[k] for k in ('model_type','dataset','dataset_name','dataset_config','hf_task','training_regime','task_type','modality','input_schema','output_schema') if r.get(k) is not None})
    cfg.update(r['config'])
    cfg['hf_model_id'] = r['model_id']
    cfg['dataset_args'] = dict((doc or {}).get('config', {}).get('dataset_args') or r['loader']['dataset_args'])
    for key, prov in [('hf_model_revision','model_revision'),('hf_dataset_revision','dataset_revision')]:
        cfg[key] = cfg.get(key) or r['provenance'].get(prov) or r['loader'].get(prov) or cfg['dataset_args'].get(key)
    cfg.update(no_save_weights=False,save_weights=True,enabled=True,generation_mode='independent',
        provenance_artifact_dir=str(OUT/'service_artifacts'),db_path=str(OUT/'pilot.db'),
        checkpoint_weight_format='int8')
    for key in list(cfg):
        if key.startswith('_') or key in {'family_id','family_contract_version','family_document_ref','family_document_hash','manifest_group_id','resource_tier','model_role','initial_state_hash'}:
            cfg.pop(key, None)
    return cfg


PRIORITY = {
 ('google/vit-base-patch16-224','ufldl-stanford/svhn','cropped_digits'),
 ('nvidia/segformer-b0-finetuned-ade-512-512','qubvel-hf/ade20k-mini',None),
 ('Intel/dpt-large-ade','EduardoPacheco/FoodSeg103',None),
 ('openmmlab/upernet-convnext-tiny','AllanK24/apple-dms-materials-v2',None),
 ('openai/clip-vit-large-patch14','jxie/flickr8k',None),
}


def build():
    import pandas as pd
    from mlaas_data_generator.cli.manifest.review import save_review_manifest
    from mlaas_data_generator.cli.run_manifest import _resolve_row, _validate_row
    from mlaas_data_generator.family_preflight import manifest_family_error
    records = load_observations()
    inventory = json.loads((OUT/'family_inventory.json').read_text())
    groups = defaultdict(list)
    for r in records: groups[r['review_family_id']].append(r)
    evidence, pilot, mapping, preflight = [], [], [], []
    for f in inventory:
        members = groups[f['review_family_id']]
        picked = representatives(members)
        if picked:
            f.update(decision='cleared with recipe restrictions',reason='Three verified same-contract representatives, an IID seed repeat and at least one healthy IID service; restrict expansion to reviewed healthy IID joint recipes.')
            evidence.append({'review_family_id':f['review_family_id'],'family_id':picked[0]['provenance']['family_id'],
                             'observation_ids':[r['observation_id'] for r in picked],
                             'service_ids':[r['service_id'] for r in picked]})
            continue
        current = bool(f['frozen_contracts'])
        priority = (f['model'],f['dataset'],f['subset']) in PRIORITY and f['regime']=='finetune_transfer'
        if not current and not priority:
            f['reason'] = 'Historical-only candidate held out per review scope; no current verified frozen family selected.'
            continue
        available = [r for r in members if r['status']=='completed' and r['model_type'] in {'hf_finetune','cnn','mlp'}
                     and any('services_merged' in s or 'mobilebert_stability' in s for s in r['sources'])]
        if not available:
            f['reason'] = 'No reconstructable current recipe; held without inventing historical settings.'
            continue
        available.sort(key=lambda r:(r['quality']!='healthy', not bool(r['config'].get('mobilebert_head_normalization')),r['config'].get('split_strategy')!='iid',r['metrics'].get('training_runtime_s') or 1e9))
        base = recipe_from_record(available[0])
        # Pin revisions from the current recorded recipe even when an older recipe is used for its knobs.
        for key in ('hf_model_revision','hf_dataset_revision'):
            if not base.get(key):
                for r in available:
                    value = r['loader']['dataset_args'].get(key) or r['provenance'].get(key.replace('hf_',''))
                    if value: base[key]=value; break
        if base.get('dataset')=='hf' and not all(base.get(k) for k in ('hf_model_revision','hf_dataset_revision')):
            f['reason']='Current source revision pins missing; held without silently upgrading sources.'
            continue
        base.update(split_strategy='iid',distribution_type='iid')
        if base.get('training_policy',{}).get('calibration',{}).get('status') == 'available':
            base['calibration_band']='challenging'
        else:
            base.pop('calibration_band',None)
        base.pop('calibration_template_id',None)
        if 'mobilebert' in f['model']:
            base.update(mobilebert_head_normalization='layer_norm_v1',max_grad_norm=1.0,
                        precision_type='fp32',mixed_precision=False)
            if f['subset']=='mrpc':
                base.update(optimizer='adamw',learning_rate=5e-5,training_epochs=5,
                            batch_size=16,gradient_accumulation_steps=1,weight_decay=.01,warmup_ratio=.1,
                            sample_size=1245,requested_quota=1245,skew_axis='class_label')
        base['initialization_seed'] = int(base.get('initialization_seed',42))
        base['evaluation_seed'] = int(base.get('evaluation_seed',42))
        for i in range(3):
            cfg = dict(base)
            cfg['dataset_args'] = dict(base['dataset_args'])
            cfg.update(service_id=f"{f['review_family_id']}__pilot_{i+1}",sample_seed=913000+i,training_seed=914000+i)
            role = 'healthy_candidate_iid' if i==0 else 'iid_seed_repeat' if i==1 else 'distinct_optimizer'
            if i==2:
                alternatives=[r for r in available if r['config'].get('optimizer')!=base.get('optimizer')]
                if alternatives:
                    for key in KNOBS:
                        if alternatives[0]['config'].get(key) is not None: cfg[key]=alternatives[0]['config'][key]
                else:
                    cfg['optimizer']='adam' if base.get('optimizer')=='adamw' else 'adamw'
            cfg['effective_batch_size']=int(cfg.get('batch_size',16))*int(cfg.get('gradient_accumulation_steps',1))
            resolved = _resolve_row(pd.Series(cfg),{})
            check = _validate_row(resolved)
            error = manifest_family_error(resolved)
            if i==2 and not check.ok:
                for key in KNOBS:
                    if base.get(key) is not None: cfg[key]=base[key]
                cfg['effective_batch_size']=int(cfg.get('batch_size',16))*int(cfg.get('gradient_accumulation_steps',1))
                role='additional_seed_no_valid_alternative'
                resolved=_resolve_row(pd.Series(cfg),{})
                check=_validate_row(resolved)
                error=manifest_family_error(resolved)
            preflight.append({'service_id':cfg['service_id'],'review_family_id':f['review_family_id'],'row_valid':check.ok,'family_error':error,'validation':str(check)})
            if not check.ok or error:
                f['reason']='Current preflight excludes this contract: '+str(error or check)
                break
            pilot.append(cfg)
            mapping.append({'service_id':cfg['service_id'],'review_family_id':f['review_family_id'],'role':role,
                'source_observation_id':available[0]['observation_id'],'source_service_id':available[0]['service_id'],
                'patches':f['patches'],'requested_knobs':{k:cfg.get(k) for k in KNOBS},
                'source_knobs':{k:available[0]['config'].get(k) for k in KNOBS}})
            f['reason']='Scheduled current-contract representative pilot.'
    dump(OUT/'reused_evidence.json',evidence)
    dump(OUT/'pilot_mapping.json',mapping)
    dump(OUT/'pilot_preflight.json',preflight)
    dump(OUT/'pilot_expanded.json',pilot)
    save_review_manifest(pd.DataFrame(pilot),OUT/'pilot.csv')
    dump(OUT/'family_inventory.json',inventory)
    write_csv(OUT/'family_inventory.csv',inventory)
    print('Reused families:',len(evidence),'Pilot services:',len(pilot))


def read_catalogue(path):
    con = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    con.execute('BEGIN')
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    rows = {}
    # JSON extraction keeps selected-row arrays and hardware dumps out of memory.
    sql = """select service_id,status,model_id,model_type,hf_task,task_type,training_regime,
      dataset,dataset_name,dataset_config,train_split,benchmark_split,modality,input_schema,output_schema,
      service_config_json, json_extract(metadata_json,'$.loader_meta.dataset_args') as args,
      json_extract(metadata_json,'$.loader_meta.task_type') as loader_task,
      json_extract(metadata_json,'$.loader_meta.hf_model_revision') as model_revision,
      json_extract(metadata_json,'$.loader_meta.hf_dataset_revision') as dataset_revision
      from services"""
    for raw in con.execute(sql):
        r = dict(raw)
        r['config'] = decode(r.pop('service_config_json'), {})
        r['loader'] = {'task_type': r.pop('loader_task'), 'dataset_args': decode(r.pop('args'), {}),
                       'model_revision': r.pop('model_revision'), 'dataset_revision': r.pop('dataset_revision')}
        r.update(metrics={}, provenance={}, checkpoint={}, artifacts=[], failures=[], sources=[str(path.relative_to(ROOT))])
        rows[r['service_id']] = r
    placeholders = ','.join('?' for _ in METRICS)
    for raw in con.execute(f"select service_id,metric_name,value_num,value_int,value_bool,value_text,value_json from service_metrics where metric_name in ({placeholders}) or metric_name like 'metric_stat_class_%_pred_total'", tuple(METRICS)):
        sid, name, *values = raw
        if sid in rows:
            value = next((v for v in values if v is not None), None)
            rows[sid]['metrics'][name] = value if not isinstance(value, float) or math.isfinite(value) else None
    if 'service_generation_provenance' in tables:
        for raw in con.execute('select service_id,' + ','.join(PROV) + ",json_extract(provenance_json,'$.runtime') as runtime from service_generation_provenance"):
            if raw['service_id'] in rows:
                p=dict(raw)
                rows[raw['service_id']]['runtime']=decode(p.pop('runtime'),{})
                rows[raw['service_id']]['provenance'] = p
    if 'service_checkpoint_storage' in tables:
        for raw in con.execute("select service_id,round_trip_status,full_artifact_ref,patch_artifact_ref,reconstruction_artifact_ref,target_state_sha256,base_state_sha256,json_extract(metadata_json,'$.representation') as representation,json_extract(metadata_json,'$.quality_validation.status') as reduced_quality_status from service_checkpoint_storage"):
            if raw['service_id'] in rows:
                rows[raw['service_id']]['checkpoint'] = dict(raw)
    if 'service_artifacts' in tables:
        for raw in con.execute("select service_id,artifact_type,artifact_uri,metadata_json from service_artifacts where artifact_type in ('update_signature','weights','checkpoint')"):
            if raw['service_id'] in rows:
                a = dict(raw)
                a['metadata'] = decode(a.pop('metadata_json'), {})
                rows[raw['service_id']]['artifacts'].append(a)
    if 'service_failures' in tables:
        for raw in con.execute('select service_id,failure_stage,substr(error_message,1,1200) as error_message from service_failures'):
            if raw['service_id'] in rows:
                rows[raw['service_id']]['failures'].append(dict(raw))
    if 'service_execution_provenance' in tables:
        for raw in con.execute("select service_id,json_extract(document_json,'$.effective_config') as cfg,json_extract(document_json,'$.training_config') as training,json_extract(document_json,'$.training_budget_adjustment') as adjustment from service_execution_provenance order by started_at"):
            if raw['service_id'] in rows:
                r=rows[raw['service_id']]
                effective=decode(raw['cfg'],{})
                for key in (*KNOBS,'split_strategy','sample_seed','training_seed','precision_type','mixed_precision',
                            'mobilebert_head_normalization','max_grad_norm'):
                    if key in effective:r['config'][key]=effective[key]
                r['execution_adjustment']=decode(raw['adjustment'],{})
                training=decode(raw['training'],{})
                r['executed_training_knobs']={key:training.get(key,r['config'].get(key)) for key in KNOBS}
    con.close()
    return list(rows.values())


def family_key(r):
    # Candidate grouping only. Exact contracts remain separate inside each group.
    return (r.get('model_id') or r.get('model_type'), r.get('dataset_name') or r.get('dataset'),
            r.get('dataset_config'), r.get('hf_task') or r.get('task_type'), r.get('training_regime'))


def engineering(r):
    reasons = []
    p, c, m = r['provenance'], r['checkpoint'], r['metrics']
    if not runtime_matches(r):reasons.append('runtime_provenance_missing_or_incompatible')
    if r['status'] != 'completed': reasons.append('not_completed')
    if not p.get('family_id') or not p.get('family_document_hash'): reasons.append('no_frozen_family')
    if not p.get('composition_compatible'): reasons.append('not_composition_verified')
    if c.get('round_trip_status') != 'passed': reasons.append('checkpoint_not_verified')
    refs = [c.get(k) for k in ('full_artifact_ref', 'patch_artifact_ref', 'reconstruction_artifact_ref') if c.get(k)]
    if not refs or any(not Path(ref).is_file() or Path(ref).stat().st_size==0 for ref in refs): reasons.append('missing_checkpoint_files')
    if not any(a['artifact_type'] == 'update_signature' and Path(a['artifact_uri']).is_file()
               and Path(a['artifact_uri']).stat().st_size>0 and finite(a['metadata'].get('signature_norm'))
               and a['metadata']['signature_norm']>0 for a in r['artifacts']): reasons.append('missing_or_zero_signature')
    steps = m.get('successful_optimizer_step_count', m.get('optimizer_step_count'))
    if not finite(steps) or steps <= 0: reasons.append('no_successful_updates_evidence')
    if not finite(m.get('loss')): reasons.append('missing_or_nonfinite_loss')
    if m.get('train_stopped_early'): reasons.append('training_budget_limited')
    return reasons


def screen():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'outputs/services_merged.db', ROOT/'outputs/services.db',
               ROOT/'outputs/checks/mobilebert_stability_20260913/final/services.db',
               ROOT/'outputs/checks/run_failure_fixes_20260910/canary.db']
    records, seen, source_counts = [], {}, {}
    for path in sources:
        if not path.is_file(): continue
        current = read_catalogue(path)
        source_counts[str(path.relative_to(ROOT))] = dict(Counter(r['status'] for r in current))
        for r in current:
            # Identity plus stored recipe/outcomes avoids merging conflicting ID reuse.
            fp = digest({k: r[k] for k in ('service_id', 'status', 'config', 'metrics', 'provenance')})
            if fp in seen:
                seen[fp]['sources'].extend(r['sources'])
                continue
            r['observation_id'] = fp
            seen[fp] = r
            r['quality'] = quality(r)
            r['engineering_gaps'] = engineering(r)
            r['patches'] = patch_ids(r)
            records.append(r)
        print(f'Read {path.name}: {len(current)} observations', flush=True)
    groups = defaultdict(list)
    for r in records: groups[family_key(r)].append(r)
    inventory = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        fid = 'review_' + digest(key)[:12]
        for r in members: r['review_family_id'] = fid
        counts = Counter(r['quality'] for r in members if r['status'] == 'completed')
        patches = sorted({p for r in members for p in r['patches']})
        errors = Counter(f['failure_stage'] + ': ' + f['error_message'][:150] for r in members for f in r['failures'])
        if any('round-trip' in e or 'round trip' in e for e in errors): patches.append('checkpoint_roundtrip')
        train = [r['metrics']['training_runtime_s'] for r in members if finite(r['metrics'].get('training_runtime_s'))]
        triggers = []
        if any(r['status'] == 'failed' for r in members): triggers.append('historical_failure')
        if patches: triggers.append('patch_requires_contract_match')
        if counts['collapsed_signature']: triggers.append('constant_class_signature')
        measured = counts['healthy'] + counts['weak'] + counts['collapsed_signature']
        if measured and counts['healthy']/measured < .5: triggers.append('low_healthy_fraction')
        if counts['missing_quality'] or counts['unverified_metric_contract']: triggers.append('unverified_quality')
        if counts['needs_initial_baseline']: triggers.append('missing_matched_initial_baseline')
        if train and max(train) > 600: triggers.append('long_training')
        inventory.append(dict(review_family_id=fid,model=key[0],dataset=key[1],subset=key[2],task=key[3],regime=key[4],
            observations=len(members),completed=sum(r['status']=='completed' for r in members),failed=sum(r['status']=='failed' for r in members),
            healthy=counts['healthy'],weak=counts['weak'],constant_signatures=counts['collapsed_signature'],quality_counts=dict(counts),
            frozen_contracts=len({r['provenance'].get('family_id') for r in members if r['provenance'].get('family_id')}),
            engineering_verified=sum(not r['engineering_gaps'] for r in members),
            joint_recipes=len({digest({k:r['config'].get(k) for k in KNOBS}) for r in members}),
            optimizers=dict(Counter(r['config'].get('optimizer') or 'unknown' for r in members)),
            data_profiles=dict(Counter(r['config'].get('split_strategy') or 'unknown' for r in members)),
            train_seconds_sum=sum(train),train_seconds_median=statistics.median(train) if train else None,
            triggers=triggers,patches=patches,failure_causes=dict(errors),
            decision='held for investigation',reason='Current-contract representative review pending'))
    dump(OUT/'observations.json', records)
    dump(OUT/'source_inventory.json', source_counts)
    dump(OUT/'family_inventory.json', inventory)
    write_csv(OUT/'family_inventory.csv', inventory)
    dump(OUT/'patch_evidence.json', PATCHES)
    print(json.dumps({'families':len(inventory),'observations':len(records),'suspicious':sum(bool(f['triggers']) for f in inventory),'engineering_verified':sum(not r['engineering_gaps'] for r in records)},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['screen','build','finalize'])
    args = parser.parse_args()
    {'screen':screen,'build':build,'finalize':finalize}[args.command]()
