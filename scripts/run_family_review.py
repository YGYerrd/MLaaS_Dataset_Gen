"""Execute an isolated review manifest and measure its frozen initial baseline.

The hook only observes an already-prepared initial model. It does not alter the
production training path, stored checkpoint verification, or catalogue/audit split.
"""
import os
os.environ['ROCM_SDK_TARGET_FAMILY']='custom'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import argparse
import json
from pathlib import Path
import random
import sys
import time
import threading
import traceback
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mlaas_data_generator.cli import run_manifest as manifest
from mlaas_data_generator.cli.manifest import preparation
from mlaas_data_generator.data.fedavg_partition import _take_rows
from mlaas_data_generator.execution_provenance import evidence_value
from mlaas_data_generator.services.runner import ServiceRunner, _apply_runtime_knobs_to_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=Path,default=ROOT/'outputs/family_review_20260913/pilot.csv')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--family-id',help='Run only this review family, retaining the original frozen manifest binding')
    args=parser.parse_args()
    out=args.file.resolve().parent
    evidence_dir=out/'worker_evidence'
    evidence_dir.mkdir(exist_ok=True)
    sources=['mlaas_data_generator/tasks/contracts.py','mlaas_data_generator/services/runner.py',
             'mlaas_data_generator/models/adapters/hf_core.py','mlaas_data_generator/data/preprocessors/hf_image.py',
             'mlaas_data_generator/models/adapters/mobilebert_stability.py',
             'mlaas_data_generator/models/adapters/rocm_dpt_layout.py','mlaas_data_generator/contrastive_batch.py']
    (evidence_dir/f'{args.file.stem}_{args.family_id or "all"}_{time.time_ns()}.json').write_text(json.dumps({
        'manifest':str(args.file),'family_filter':args.family_id,'executable':sys.executable,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        'manifest_sha256':hashlib.sha256(args.file.read_bytes()).hexdigest(),
        'companion_sha256':hashlib.sha256(args.file.with_suffix('.families.json').read_bytes()).hexdigest(),
        'capture':'Worker start before preparation; actual recipe/metrics and patch behavior are recorded separately.'},indent=2),encoding='utf-8')
    if args.family_id:
        original_load=manifest.load_manifest
        def selected_load(path,*positional,**keywords):
            frame,defaults=original_load(path,*positional,**keywords)
            return frame[frame['service_id'].str.startswith(args.family_id+'__')].copy(),defaults
        manifest.load_manifest=selected_load
    def snapshots():
        while True:
            time.sleep(60)
            with (out/'pilot_stacks.log').open('a',encoding='utf-8') as log:
                log.write(f'\nSnapshot {time.time()}\n')
                for ident,frame in sys._current_frames().items():
                    log.write(f'Thread {ident}\n');traceback.print_stack(frame,file=log)
    threading.Thread(target=snapshots,daemon=True).start()
    for name, filename in [('MANIFEST_RESULTS_PATH',args.file.stem+'_results.csv'),('FAILED_MANIFEST_PATH',args.file.stem+'_failed.csv'),
            ('FAILURE_LOG_PATH',args.file.stem+'_failures.log'),('MULTIMODAL_AUDIT_PATH','multimodal_audit.csv'),
            ('MISSING_MULTIMODAL_MANIFEST_PATH','missing_multimodal.csv')]:
        setattr(manifest,name,out/filename)
    original=preparation.prepare
    def with_baseline(config,**kwargs):
        result=original(config,**kwargs)
        doc,ref,cfg,prepared,state,base=result
        target=out/'initial_baselines'/f"{doc['family_id']}.json"
        if target.is_file():
            previous=json.loads(target.read_text())
            if (previous.get('status')=='completed' and previous.get('runtime_matched')
                    and previous.get('document_hash')==ref['document_hash']
                    and previous.get('initial_state_hash')==doc['initial_state_hash']
                    and all(previous.get(k)==config.get(k) for k in ('batch_size','precision_type','mixed_precision'))):return result
            archive=target.with_name(target.stem+'_previous.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2),encoding='utf-8')
        train,test,meta,task,model,*_=prepared
        # Only tasks without the existing classification/correlation rubric need this comparison.
        if task not in {'segmentation','generation','regression','retrieval'}: return result
        import numpy as np
        import torch
        py_state,np_state,torch_state=random.getstate(),np.random.get_state(),torch.get_rng_state()
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        start=time.perf_counter()
        evidence={'family_id':doc['family_id'],'document_hash':ref['document_hash'],
                  'initial_state_hash':doc['initial_state_hash'],
                  'reservation_hash':doc['reservation']['reservation_hash'],
                  'catalogue_count':len(doc['reservation']['catalogue_metric_ids']),
                  'runtime_matched':True,'batch_size':config.get('batch_size'),
                  'precision_type':config.get('precision_type'),'mixed_precision':config.get('mixed_precision')}
        try:
            ids=np.asarray(doc['reservation']['catalogue_positions'],dtype=np.int64)
            x,y=_take_rows(test[0],ids),_take_rows(test[1],ids)
            _apply_runtime_knobs_to_model(model,config)
            baseline_config={**config,**cfg,'service_id':'review_initial_baseline'}
            loss,primary,secondary,qos=ServiceRunner(baseline_config)._evaluate_model(model,x,y,
                inference_only=False,task_family=task,enforce_time_limit=True)
            evidence.update(status='completed',loss=loss,primary=primary,secondary=secondary,metrics=qos)
        except Exception as exc:
            evidence.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.set_rng_state(torch_state)
            if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
        evidence['runtime_s']=time.perf_counter()-start
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(evidence_value(evidence),indent=2,allow_nan=False),encoding='utf-8')
        return result
    preparation.prepare=with_baseline
    manifest.run_manifest(str(args.file),db_path=str(out/'pilot.db'),workers=1,dry_run=args.dry_run)


if __name__=='__main__':main()
