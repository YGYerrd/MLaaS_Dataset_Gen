"""Clearance must never borrow proof from an incompatible or unverified run."""
import copy
import hashlib
import json

from scripts import review_families as review


def observation(sid, *, family='family_a', healthy=True):
    return {'service_id':sid,'quality':'healthy' if healthy else 'weak','engineering_gaps':[],
        'config':{'split_strategy':'iid','optimizer':'adamw','learning_rate':2e-5,
                  'batch_size':8,'gradient_accumulation_steps':1,'training_epochs':2,
                  'weight_decay':0,'warmup_ratio':0,'sample_seed':sid,'training_seed':sid},
        'metrics':{'train_distribution_json':'{"0": 20,"1": 20}'},
        'provenance':{'family_id':family},'patches':[],'sources':['fixture']}


def test_reuse_requires_repeated_healthy_recipe_in_one_contract(monkeypatch):
    monkeypatch.setattr(review,'checked_document',lambda r:{'config':{}})
    rows=[observation(str(i)) for i in range(3)]
    assert len(review.representatives(rows))==3
    rows[1]['provenance']['family_id']='family_b'
    rows[2]['provenance']['family_id']='family_c'
    assert review.representatives(rows)==[]
    rows=[observation(str(i)) for i in range(3)]
    for i,row in enumerate(rows):row['provenance']['family_document_hash']=f'doc_{i}'
    assert review.representatives(rows)==[]


def test_family_needs_at_least_one_healthy_representative(monkeypatch):
    monkeypatch.setattr(review,'checked_document',lambda r:{'config':{}})
    rows=[observation('1',healthy=False),observation('2',healthy=False),observation('3',healthy=False)]
    assert review.representatives(rows)==[]
    rows[0]['quality']='healthy'
    assert len(review.representatives(rows))==3


def test_missing_checkpoint_evidence_excludes_representative(monkeypatch):
    monkeypatch.setattr(review,'checked_document',lambda r:{'config':{}})
    rows=[observation(str(i)) for i in range(3)]
    rows[2]['engineering_gaps']=['missing_checkpoint_files']
    assert review.representatives(rows)==[]


def test_mobilebert_requires_config_and_frozen_head_marker():
    r=observation('1');r['patches']=['mobilebert_stability']
    r['config'].update(mobilebert_head_normalization='layer_norm_v1',max_grad_norm=1.)
    assert review.active_patch_gaps(r,{'config':{}})==['mobilebert_patch_not_enabled']
    assert review.active_patch_gaps(r,{'config':{'mobilebert_head_normalization':'layer_norm_v1'}})==[]
    r['config'].pop('max_grad_norm')
    assert review.active_patch_gaps(r,{'config':{'mobilebert_head_normalization':'layer_norm_v1'}})


def test_legacy_segmentation_head_cannot_validate_patch():
    r=observation('1');r['patches']=['ade20k_three_labels']
    assert review.active_patch_gaps(r,{'contracts':{'labels':{'num_labels':150}}})
    assert not review.active_patch_gaps(r,{'contracts':{'labels':{'num_labels':3}}})


def test_unknown_labels_and_eval_overlap_rejected(tmp_path):
    r=observation('1')
    hashes={k:k+'_value' for k in ('parameter_schema_hash','label_map_hash','preprocessing_contract_hash','logical_dataset_contract_hash')}
    r['provenance'].update(hashes)
    def bind(doc):
        payload=json.dumps(doc).encode();path=tmp_path/(hashlib.sha256(payload).hexdigest()+'.json');path.write_bytes(payload)
        r['provenance'].update(family_document_ref=str(path),family_document_hash=hashlib.sha256(payload).hexdigest(),initial_state_hash='init')
    doc={'family_id':'family_a','initial_state_hash':'init','config':{},'compatibility':hashes,
         'contracts':{'labels':{'task_family':'unknown'}},'reservation':{'catalogue_metric_ids':['a'],'audit_ids':['b']}}
    bind(doc);assert review.checked_document(r) is None
    doc['contracts']['labels']['task_family']='classification'
    doc['reservation']['audit_ids']=['a']
    bind(doc);assert review.checked_document(r) is None
    doc['reservation']['audit_ids']=['b']
    bind(doc);assert review.checked_document(r)==doc
    incompatible=copy.deepcopy(r)
    incompatible['provenance']['label_map_hash']='different-labels'
    assert review.checked_document(incompatible) is None
    incompatible['provenance'].pop('label_map_hash')
    assert review.checked_document(incompatible) is None


def test_majority_accuracy_is_not_healthy():
    r={'hf_task':'sequence_classification','metrics':{'accuracy':.7,'f1':2*.7*.7/1.7,
       'benchmark_distribution_json':'{"0":30,"1":70}'}}
    assert review.quality(r)=='collapsed_signature'


def test_sentence_pair_requires_regression_semantics():
    r={'hf_task':'sentence_similarity','dataset_config':'stsb','loader':{'task_type':'classification'},
       'metrics':{'pearson':.8,'spearman':.8}}
    assert review.quality(r)=='unverified_metric_contract'


def test_native_baseline_requires_same_evaluation_initialization_and_runtime(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'OUT',tmp_path)
    doc={'family_id':'family_a','initial_state_hash':'init','reservation':{'reservation_hash':'eval'}}
    monkeypatch.setattr(review,'checked_document',lambda r:doc)
    hashes={'task_semantics_hash':'semantics','evaluation_protocol_hash':'protocol'}
    r={'hf_task':'image_segmentation','metrics':{**hashes,'iou':.4,'metric_stat_class_0_pred_total':80,
       'metric_stat_class_1_pred_total':20},'config':{'batch_size':4,'precision_type':'fp32','mixed_precision':False},
       'provenance':{'family_document_hash':'doc'}}
    initial={'status':'completed','primary':.2,'document_hash':'doc','initial_state_hash':'init',
             'reservation_hash':'eval','runtime_matched':True,'metrics':hashes,**r['config']}
    directory=tmp_path/'initial_baselines';directory.mkdir()
    path=directory/'family_a.json'
    path.write_text(json.dumps(initial))
    assert review.pilot_quality(r)=='healthy'
    for key,value in [('reservation_hash','different-eval'),('initial_state_hash','different-init'),('document_hash','different-doc')]:
        path.write_text(json.dumps({**initial,key:value}))
        assert review.pilot_quality(r)=='incompatible_initial_baseline'
    path.write_text(json.dumps({**initial,'precision_type':'fp16'}))
    assert review.pilot_quality(r)=='unverified_initial_runtime'
    path.write_text(json.dumps(initial))
    r['metrics']['metric_stat_class_1_pred_total']=0
    assert review.pilot_quality(r)=='weak_or_unverified_outputs'


def test_checkpoint_file_must_still_exist_and_have_content(tmp_path):
    weights=tmp_path/'weights';weights.write_bytes(b'checkpoint')
    signature=tmp_path/'signature';signature.write_bytes(b'update')
    r={'status':'completed','provenance':{'family_id':'f','family_document_hash':'h','composition_compatible':True},
       'runtime':review.intended_runtime(),
       'checkpoint':{'round_trip_status':'passed','full_artifact_ref':str(weights)},
       'metrics':{'successful_optimizer_step_count':1,'loss':.2},
       'artifacts':[{'artifact_type':'update_signature','artifact_uri':str(signature),'metadata':{'signature_norm':1}}]}
    assert not review.engineering(r)
    weights.write_bytes(b'')
    assert 'missing_checkpoint_files' in review.engineering(r)
    weights.unlink()
    assert 'missing_checkpoint_files' in review.engineering(r)
    signature.write_bytes(b'')
    assert 'missing_or_zero_signature' in review.engineering(r)
    r['runtime']={}
    assert 'runtime_provenance_missing_or_incompatible' in review.engineering(r)
