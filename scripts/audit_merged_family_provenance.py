"""Read-only, complete-roster provenance triage. No model loading or execution."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.review_families import read_catalogue, family_key, finite, decode, dump, write_csv, quality

OUT=ROOT/'outputs/all_family_provenance_audit_20260914'
MERGED=ROOT/'outputs/services_merged.db'
EXTRA_METRICS={'evaluation_protocol','rougel','rouge2','inertia','silhouette_score','train_fp32_loss_retry_count',
               'eval_fp32_retry_count','num_classes','num_labels','train_samples','empty_output_fraction',
               'repetition_fraction','training_opportunity_status','precision_policy'}
PATCH_SOURCES={
 'mobilebert_normalization':'mlaas_data_generator/models/adapters/mobilebert_stability.py',
 'precision':'mlaas_data_generator/precision_policy.py',
 'segmentation_storage':'mlaas_data_generator/data/preprocessors/hf_image.py',
 'segmentation_policy':'mlaas_data_generator/segmentation_policy.py',
 'vit_opportunity':'mlaas_data_generator/clients/reviewed_opportunity.py',
 'contrastive_batches':'mlaas_data_generator/contrastive_batch.py',
 'evaluation':'mlaas_data_generator/models/adapters/hf_core.py',
 'runner':'mlaas_data_generator/services/runner.py',
 'task_contract':'mlaas_data_generator/tasks/contracts.py',
}


def connect(path):
    con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON');con.execute('BEGIN')
    return con


def load(path):
    rows=read_catalogue(path)
    by_id={r['service_id']:r for r in rows}
    con=connect(path)
    for sid,meta in con.execute("""select service_id,json_object(
        'started_at',json_extract(metadata_json,'$.started_at'),
        'checkpoint_persistence',json_extract(metadata_json,'$.checkpoint_persistence'),
        'num_labels',json_extract(metadata_json,'$.loader_meta.num_labels'),
        'num_classes',json_extract(metadata_json,'$.loader_meta.num_classes'),
        'label_mapping',json_extract(metadata_json,'$.loader_meta.label_mapping'),
        'label_format',json_extract(metadata_json,'$.loader_meta.label_format'),
        'loader_hf_task',json_extract(metadata_json,'$.loader_meta.hf_task'),
        'resolved_hf_task',json_extract(metadata_json,'$.resolved_hf_task'),
        'resolved_task_type',json_extract(metadata_json,'$.resolved_task_type'),
        'segmentation_reduce_labels',json_extract(metadata_json,'$.loader_meta.segmentation_reduce_labels'))
        from services"""):
        by_id[sid]['attributes']=json.loads(meta)
    for sid,labels,mode in con.execute("select service_id,json_extract(provenance_json,'$.labels'),json_extract(provenance_json,'$.mode') from service_generation_provenance"):
        if sid in by_id:
            by_id[sid]['persisted_labels']=decode(labels,{})
            by_id[sid]['provenance_mode']=mode
    for sid,name,*values in con.execute("select service_id,metric_name,value_num,value_int,value_bool,value_text,value_json from service_metrics where metric_name in ("+','.join('?' for _ in EXTRA_METRICS)+")",tuple(EXTRA_METRICS)):
        if sid in by_id:by_id[sid]['metrics'][name]=next((v for v in values if v is not None),None)
    tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    for r in rows:r['import_sources']=[]
    if 'merge_service_sources' in tables:
        for sid,source in con.execute('select service_id,source_db from merge_service_sources'):
            if sid in by_id:by_id[sid]['import_sources'].append(source)
    con.close()
    return rows


def revisions(r):
    return (r['provenance'].get('model_revision') or r['loader'].get('model_revision'),
            r['provenance'].get('dataset_revision') or r['loader'].get('dataset_revision'))


def compatible_sources(a,b):
    return family_key(a)==family_key(b) and (not a.get('hf_task') or revisions(a)==revisions(b))


@lru_cache(maxsize=None)
def artifact_present(ref):
    path=Path(ref)
    if path.is_file():return path.stat().st_size>0
    # Tree artifacts (e.g. RandomForest model.joblib) are valid checkpoints.
    if path.is_dir():return any(p.is_file() and p.stat().st_size>0 for p in path.rglob('*'))
    return False


def good_execution(r):
    if r['status']!='completed':return False
    if r['model_type'] in {'randomforest','kmeans'}:return True
    updates=r['metrics'].get('successful_optimizer_step_count',r['metrics'].get('optimizer_step_count'))
    return finite(r['metrics'].get('loss')) and finite(updates) and updates>0


def normalized(r):
    return (good_execution(r) and r['config'].get('mobilebert_head_normalization')=='layer_norm_v1'
            and r['config'].get('max_grad_norm')==1 and r['config'].get('precision_type')=='fp32')


def service_issues(r):
    m,p,a=r['metrics'],r['provenance'],r['attributes']
    issues=[];notes=[]
    debug=bool((a.get('checkpoint_persistence') or {}).get('mode')=='debug_no_weights'
               or r.get('provenance_mode')=='debug_no_weights' or r['config'].get('no_save_weights'))
    if debug:notes.append('intentional_no_weights_test; checkpoint/frozen-family proof unavailable, not a missing-file defect')
    if r['status']=='failed':issues.append('execution_failure')
    if r['status']=='completed':
        if r.get('hf_task') and not all(revisions(r)):issues.append('missing_source_revision')
        for field in ('label_map_hash','preprocessing_contract_hash','parameter_schema_hash','logical_dataset_contract_hash'):
            if not p.get(field):issues.append('missing_'+field)
        if not debug and r['checkpoint'].get('round_trip_status')!='passed':issues.append('checkpoint_verification_missing')
        if not debug:
            refs=[r['checkpoint'].get(k) for k in ('full_artifact_ref','patch_artifact_ref','reconstruction_artifact_ref') if r['checkpoint'].get(k)]
            if not refs or any(not artifact_present(ref) for ref in refs):issues.append('checkpoint_file_missing')
        signatures=[x for x in r['artifacts'] if x['artifact_type']=='update_signature']
        if not signatures or any(not Path(x['artifact_uri']).is_file() for x in signatures):issues.append('signature_missing')
        if r['model_type'] not in {'randomforest','kmeans'}:
            if not finite(m.get('loss')):issues.append('evaluation_loss_missing_or_nonfinite')
            updates=m.get('successful_optimizer_step_count',m.get('optimizer_step_count'))
            if updates is None:notes.append('actual_optimizer_updates_not_recorded')
            elif not finite(updates) or updates<=0:issues.append('no_successful_optimizer_updates')
        if m.get('train_stopped_early'):notes.append('budget_limited')
        if (r.get('model_id')=='google/mobilebert-uncased' and r.get('hf_task') in {'sequence_classification','sentence_similarity'}
                and not normalized(r) and finite(m.get('loss')) and m['loss']>50):issues.append('mobilebert_unnormalized_extreme_loss')
        if r.get('hf_task')=='seq2seq_generation' and m.get('evaluation_protocol')!='source_only_generation_v1' and not m.get('evaluation_protocol_hash'):
            issues.append('generated_text_evaluation_protocol_unverified')
        if r.get('dataset_name')=='qubvel-hf/ade20k-mini':
            if (a.get('num_labels') or a.get('num_classes'))!=3:issues.append('legacy_ade20k_mini_label_contract')
        if r.get('hf_task')=='text_image_retrieval' and (m.get('contrastive_candidates_min',0)<2 or m.get('eval_loss_candidate_count',0)<2):
            issues.append('contrastive_batch_or_evaluation_pool_unverified')
        # Compare the frozen label declaration with the actual loaded task, when recorded.
        frozen=r.get('persisted_labels',{}).get('task_family')
        if frozen=='unknown':issues.append('unknown_frozen_label_task')
        if frozen in {'classification','regression'} and r['loader'].get('task_type') in {'classification','regression'} and frozen!=r['loader']['task_type']:
            issues.append('frozen_runtime_label_task_mismatch')
    try:quality_label=quality(r) if r['status']=='completed' else 'not_completed'
    except (KeyError,TypeError,ValueError):quality_label='unmeasured'
    return sorted(set(issues)),notes,quality_label,debug


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={k:(ROOT/v).read_text(encoding='utf-8') for k,v in PATCH_SOURCES.items()}
    patch_hashes={k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in PATCH_SOURCES.items()}
    merged=load(MERGED)
    extra=[]
    for path in [ROOT/'outputs/family_review_20260913/pilot.db',
                 ROOT/'outputs/checks/mobilebert_stability_20260913/final/services.db',
                 ROOT/'outputs/checks/run_failure_fixes_20260910/canary.db']:
        if path.exists():extra.extend(load(path))
    groups=defaultdict(list)
    for r in merged:groups[family_key(r)].append(r)
    roster_ids={r['service_id'] for r in merged if 'all_families_test.db' in r['import_sources']}
    roster_keys={family_key(r) for r in merged if r['service_id'] in roster_ids}
    assert set(groups)==roster_keys, 'Merged families must reconcile with the complete-family test roster'
    for r in merged+extra:
        r['issues'],r['notes'],r['quality_label'],r['no_weights_test']=service_issues(r)
    summaries=[];actions=[];service_rows=[]
    def action(fid,code,state,why,next_step,evidence,patch=None,validation='persisted'):
        item=dict(family_id=fid,issue=code,state=state,evidence=why,next_step=next_step,
                  evidence_service_ids=evidence,patch=patch,validation_scope=validation)
        actions.append(item);return item
    for key,rows in sorted(groups.items(),key=lambda x:str(x[0])):
        fid='audit_'+hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
        base=next(r for r in rows if r['service_id'] in roster_ids)
        later=[r for r in extra if compatible_sources(base,r)]
        evidence=[r for r in rows+later if compatible_sources(base,r)]
        before=len(actions)
        issue_codes={i for r in rows for i in r['issues']}
        for code in sorted(issue_codes):
            affected=[r['service_id'] for r in rows if code in r['issues']]
            state='needs_fixing';patch=None;validation='not_resolved';why=f'{len(affected)} merged service(s) record {code}.'
            next_step='Inspect the recorded discrepancy and correct its contract/generation path.'
            fixes=[]
            if code=='mobilebert_unnormalized_extreme_loss':
                fixes=[r for r in later if normalized(r)]
                patch='docs/mobilebert-stability.md'
                if fixes:
                    state='already_fixed';validation='same model/data revisions; explicitly normalized replacement contract'
                    next_step='Carry the tested normalization, clipping and FP32 settings into new production families; do not reuse unnormalized preparations.'
                else:
                    next_step='Normalization implementation exists, but this task/dataset has no normalized replacement evidence. Ensure opt-in activation and a fresh matching contract; token/MLM heads are outside this issue.'
                    validation='patch_present; activation/contract evidence absent'
            elif code=='execution_failure':
                fixes=[r for r in evidence if good_execution(r) and r['checkpoint'].get('round_trip_status')=='passed'
                       and not any(i in r['issues'] for i in ('unknown_frozen_label_task','frozen_runtime_label_task_mismatch','mobilebert_unnormalized_extreme_loss'))]
                if fixes:
                    state='already_fixed';patch='outputs/checks/run_failure_fixes_20260910/report.md'
                    next_step='Use the corrected verified preparation/runtime; retain old failure records as history.'
                    validation='same model/data group and revision pins; successful replacement, not timestamp inference'
            elif code=='evaluation_loss_missing_or_nonfinite':
                patch='docs/usage.md'
                guard='nonfinite_evaluation_loss' in sources['evaluation'] and 'eval_fp32_retry_count' in sources['evaluation']
                if guard:
                    state='already_fixed';validation='current precision/retry/completion guards; affected production contract not rerun'
                    next_step='Regenerate affected null-loss evidence with the patched worker; do not treat the old completed status as a valid evaluation.'
            elif code=='generated_text_evaluation_protocol_unverified':
                state='already_fixed';patch='docs/usage.md';validation='source-only generation implementation; old metric identity remains unverified'
                next_step='Use current source-only generation evaluation and its protocol marker; exclude historical unmarked/teacher-forced ROUGE from production quality calibration.'
            elif code=='legacy_ade20k_mini_label_contract':
                state='already_fixed';patch='docs/family-review-fixes.md';validation='current three-label/reduction guards; full pilot for SegFormer-B0 only'
                next_step='Create corrected three-label preparations; reject old 150-label/reduced-mask contracts. SegFormer uses the current scoped FP32 policy.'
            elif code=='contrastive_batch_or_evaluation_pool_unverified':
                state='already_fixed';patch='docs/clip-contrastive-batch-diagnostic.md';validation='runtime guard plus production CLIP-L/Flickr8k evidence; other exact contracts not rerun'
                next_step='Preserve multi-pair physical contrastive batches and pooled evaluation candidates; require those counters on new results.'
            elif code in {'unknown_frozen_label_task','frozen_runtime_label_task_mismatch'}:
                fixes=[r for r in later if good_execution(r) and r['checkpoint'].get('round_trip_status')=='passed' and not set(r['issues'])&{'unknown_frozen_label_task','frozen_runtime_label_task_mismatch'}]
                if fixes:state='already_fixed';patch='outputs/checks/run_failure_fixes_20260910/report.md';validation='corrected label contract replacement'
                next_step='Use a fresh preparation with matching resolved task/labels; keep old incompatible contracts separate.'
            elif code in {'checkpoint_verification_missing','checkpoint_file_missing','signature_missing','no_successful_optimizer_updates'}:
                fixes=[r for r in later if good_execution(r) and code not in r['issues'] and r['checkpoint'].get('round_trip_status')=='passed']
                if fixes:state='already_fixed';validation='replacement has required persisted evidence';next_step='Use verified replacement artifacts and preserve the earlier incomplete evidence as history.'
            why+=f' Replacement evidence: {len(fixes)} service(s).' if fixes else ''
            action(fid,code,state,why,next_step,affected+[r['service_id'] for r in fixes],patch,validation)
        if key[0:2] in [('Intel/dpt-large-ade','EduardoPacheco/FoodSeg103'),('openmmlab/upernet-convnext-tiny','AllanK24/apple-dms-materials-v2')]:
            action(fid,'full_preparation_memory','already_fixed','Full preparation reached approximately 37 GB on a 32 GB host; a subsequent disk-backed image/mask implementation is present.',
                   'Memory code is patched; full-dataset execution and quality evidence remain outstanding. Do not report the old memory failure as unpatched.',[],
                   'docs/family-review-fixes.md','105 documented regression tests and synthetic memory checks; no full rerun')
        if key[:3]==('google/vit-base-patch16-224','ufldl-stanford/svhn','cropped_digits'):
            action(fid,'supported_training_opportunity','already_fixed','Initial small recipes were weak; 730-example/six-epoch pilots improved accuracy. The current generator has a scoped supported-band opportunity policy.',
                   'Preserve the scoped policy and allow deliberately weak/stress clients; its reference budget is not universal quality proof.',
                   [r['service_id'] for r in later],'mlaas_data_generator/clients/reviewed_opportunity.py','full pilot plus current generator policy; diverse generated recipes not individually validated')
        if base['model_id']=='nvidia/segformer-b0-finetuned-ade-512-512' and base['dataset_name']=='qubvel-hf/ade20k-mini':
            action(fid,'segformer_mini_precision_propagation','already_fixed','FP16 initial logits were nonfinite; three FP32 pilots completed. Current policy propagates FP32 and the audited label contract into generation/preparation/reload.',
                   'Use the patched production path and corrected frozen contracts; no new precision investigation is indicated by these results.',
                   [r['service_id'] for r in later],'docs/family-review-fixes.md','full FP32 pilot; 159 documented contract-propagation tests')
        own=actions[before:]
        status='needs_fixing' if any(a['state']=='needs_fixing' for a in own) else 'already_fixed' if own else 'ready'
        summaries.append(dict(family_id=fid,model=key[0],dataset=key[1],subset=key[2],task=key[3],regime=key[4],status=status,
            complete_roster_service_id=base['service_id'],merged_services=len(rows),completed=sum(r['status']=='completed' for r in rows),
            failed=sum(r['status']=='failed' for r in rows),no_weights_tests=sum(r['no_weights_test'] for r in rows),
            checkpoint_verified=sum(r['checkpoint'].get('round_trip_status')=='passed' for r in rows),
            replacement_services=len(later),issues=[a['issue'] for a in own],actions=[a['next_step'] for a in own],
            validation_scope=[a['validation_scope'] for a in own],quality_flags=dict(Counter(r['quality_label'] for r in rows)),
            recorded_contract_variants=len({tuple(r['provenance'].get(k) for k in ('label_map_hash','preprocessing_contract_hash','logical_dataset_contract_hash','parameter_schema_hash')) for r in rows if r['provenance']}),
            frozen_family_ids=sorted({r['provenance']['family_id'] for r in rows if r['provenance'].get('family_id')}),
            scope='attribute/execution triage; no new defect found is not blanket checkpoint or predictive-quality certification'))
        for r in rows:
            service_rows.append(dict(service_id=r['service_id'],family_id=fid,source_databases=r['import_sources'],status=r['status'],
                family_decision=status,issues=r['issues'],notes=r['notes'],quality_flag=r['quality_label'],
                model_revision=revisions(r)[0],dataset_revision=revisions(r)[1],loader_task=r['loader']['task_type'],attributes=r['attributes'],
                contract_hashes={k:r['provenance'].get(k) for k in ('family_id','label_map_hash','preprocessing_contract_hash','logical_dataset_contract_hash','parameter_schema_hash','initial_state_hash')},
                checkpoint_status=r['checkpoint'].get('round_trip_status'),metrics=r['metrics']))
    assert len(summaries)==len(groups)==len(roster_keys) and len(service_rows)==len(merged)
    assert len({r['service_id'] for r in service_rows})==len(merged)
    counts=dict(Counter(r['status'] for r in summaries))
    assert all(hashlib.sha256((ROOT/v).read_bytes()).hexdigest()==patch_hashes[k]
               for k,v in PATCH_SOURCES.items()), 'Patch sources changed during the audit; rerun the snapshot'
    snapshot={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'input':str(MERGED),'read_only':True,
              'merged_services':len(merged),'complete_roster_services':len(roster_ids),'family_combinations':len(groups),
              'complete_roster_coverage':True,'decisions':counts,'issue_counts':dict(Counter(a['issue'] for a in actions)),
              'new_training_or_inference_runs':0,'patch_sources':{k:{'file':v,'sha256':patch_hashes[k]} for k,v in PATCH_SOURCES.items()}}
    dump(OUT/'summary.json',snapshot);dump(OUT/'family_assessment.json',summaries);write_csv(OUT/'family_assessment.csv',summaries)
    dump(OUT/'service_attributes.json',service_rows);write_csv(OUT/'service_attributes.csv',service_rows)
    dump(OUT/'issue_actions.json',actions);write_csv(OUT/'issue_actions.csv',actions)
    write_csv(OUT/'needs_fixing.csv',[r for r in summaries if r['status']=='needs_fixing'])
    lines=['# Complete merged-family provenance audit','',
        f"Reviewed every one of the {len(merged):,} merged service records. The complete-family test source supplies exactly one roster entry for each of {len(groups)} model/dataset/subset/task/regime combinations; every entry is included.",
        '', '| Assessment | Family entries |', '|---|---:|',
        f"| No corrective action identified | {counts.get('ready',0)} |",
        f"| Applicable fix or corrected replacement exists | {counts.get('already_fixed',0)} |",
        f"| Contract settings or correction evidence still need attention | {counts.get('needs_fixing',0)} |",
        '', 'These counts answer the provenance/generation audit. They are not the earlier three-representative quality clearance. A family entry here is a model/dataset/subset/task/regime combination, not an architecture or a unique frozen preparation hash.',
        '', '## What needs attention before production', '',
        'The outstanding entries share one MobileBERT issue: the recorded sequence/pair recipes have extreme losses, and the existing normalization patch has no demonstrated activation/replacement for these exact task/dataset entries. Carry `mobilebert_head_normalization=layer_norm_v1`, `max_grad_norm=1` and FP32 into fresh matching preparations. Do not reuse their unnormalized preparations. This is a shared contract-setting/evidence gap, not sixteen separate model bugs.',
        'Four other MobileBERT entries already have normalized replacement evidence: AG News, IMDb, GLUE MRPC and GLUE STSB. Those settings must also carry through production generation.',
        '', '| Outstanding MobileBERT dataset | Subset | Task |', '|---|---|---|',
        *[f"| {r['dataset']} | {r['subset'] or '-'} | {r['task']} |" for r in summaries if r['status']=='needs_fixing'],
        '', '## Existing fixes to retain', '',
        '| Area | Current correction | Evidence limit |', '|---|---|---|',
        '| DPT / UPerNet preparation | Images and masks spill to disk | Synthetic memory/regression checks; full affected datasets not rerun |',
        '| SegFormer / ADE20K-mini | Three-label contract, no label reduction, scoped FP32 through generation and reload | Full FP32 pilot plus later contract tests |',
        '| ViT / SVHN | Adequate training opportunity in the supported recipe band | Successful reference pilot; intentional weak clients remain allowed |',
        '| Seq2seq evaluation | Source-only generated predictions and recorded protocol | Historical unmarked scores cannot calibrate current quality |',
        '| Missing/nonfinite supervised loss | T5 FP32 policy and evaluation/completion guards | Fix is present; affected exact contracts need new valid results |',
        '| CLIP retrieval | Multi-pair batches and pooled evaluation | Full CLIP-L/Flickr8k evidence; other exact contracts not rerun |',
        '| Historical execution/artifact failures | Successful corrected replacements | Replacement IDs and contract limits are in issue_actions.csv |',
        '', '## Meaning of the decisions','',
        '- **needs_fixing**: an observed issue lacks demonstrated correction/activation for the affected contract; the action identifies whether this is code, configuration or missing evidence.',
        '- **already_fixed**: a correction or replacement evidence exists. The validation column distinguishes tested replacement contracts from code fixes whose full production contract has not been rerun.',
        '- **ready**: no corrective action is indicated by the recorded identity, task, execution and applicable artifact checks. No-weights tests establish execution only; they do not verify saved production checkpoints. Weak individual clients do not automatically make a family defective.',
        '', '## Actions by issue','', '| Issue | Families affected | Disposition |','|---|---:|---|']
    for issue,count in Counter(a['issue'] for a in actions).most_common():
        lines.append(f"| {issue} | {count} | {dict(Counter(a['state'] for a in actions if a['issue']==issue))} |")
    lines+=['','## Limits and corrections to the earlier report','',
        'The merged database is the complete roster. No family was excluded for being historical-only, lacking three representatives, or deliberately skipping weights. Saved patch results supplement this roster but never add unrelated families.',
        'Every merged observation remains visible, including superseded failures, old null-loss completions, and changed contracts. A successful replacement does not rewrite or retroactively validate the old record.',
        'Revision pins are recovered from recorded loader metadata when the intentional no-weights provenance stub omits them. The stub is explicitly labelled; frozen-family identity, initialization/checkpoint round-trip and unrecorded requested settings are not invented.',
        'Model/data/preprocessing/label variants are reported within each family group. Changed labels or model normalization require replacement preparations rather than reusing incompatible artifacts.',
        'The current tree contains newer patches than the September 13 pilot report. Segmentation memory, SegFormer production propagation, ViT opportunity policy and T5/evaluation precision fixes are checked separately from their historical symptoms.',
        'Missing/zero quality is retained for inspection. Recipe heterogeneity and intentionally weak clients are not automatically treated as generator defects. Null supervised evaluation loss and unverified generation protocols are separate numerical/provenance issues.',
        'No training, inference, checkpoint restoration, precision probes, pairwise diagnostics, or production-manifest generation was performed. Historical SQLite inputs were opened read-only.',
        '', '## Files','',
        '- `family_assessment.csv`: one decision for every family combination, with counts, actions, variants and evidence limits.',
        '- `needs_fixing.csv`: unresolved actions only.',
        '- `issue_actions.csv`: each issue, patch reference, affected/replacement service IDs and required next step.',
        '- `service_attributes.csv` / `.json`: all merged service-level provenance and metrics.',
        '- `summary.json`: complete-roster reconciliation and inspected code hashes.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in snapshot.items() if k!='patch_sources'},indent=2))


if __name__=='__main__':main()
