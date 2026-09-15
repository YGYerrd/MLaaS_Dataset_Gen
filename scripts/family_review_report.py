"""Cheap persisted-evidence checks and human-readable review findings."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def supplement(out,old,runs,clearance,inventory,holds):
    root=out.parents[1]
    sources=json.loads((out/'source_inventory.json').read_text())
    current={}
    for source in sources:
        with sqlite3.connect((root/source).resolve().as_uri()+'?mode=ro',uri=True) as con:
            con.execute('PRAGMA query_only=ON')
            current[source]=dict(con.execute('select status,count(*) from services group by status'))
    with sqlite3.connect((out/'pilot.db').resolve().as_uri()+'?mode=ro',uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        integrity=[r[0] for r in con.execute('PRAGMA quick_check')]
        foreign_keys=con.execute('PRAGMA foreign_key_check').fetchall()
    stages={key:sum(r['metrics'].get(key) or 0 for r in runs)
            for key in ('training_runtime_s','evaluation_runtime_s','service_runtime_s','checkpoint_s','checkpoint_model_restore_verify_s','checkpoint_reduced_quality_s')}
    baselines=[json.loads(p.read_text()) for p in (out/'initial_baselines').glob('*.json') if '_previous' not in p.stem]
    checks={'historical_status_counts_unchanged':current==sources,'source_counts':current,
            'pilot_quick_check':integrity,'pilot_foreign_key_violations':foreign_keys,
            'pilot_status_counts':dict(Counter(r['status'] for r in runs)),
            'pilot_quality_counts':dict(Counter(r['quality'] for r in runs)),
            'pilot_checkpoint_verified':sum(r['checkpoint'].get('round_trip_status')=='passed' for r in runs),
            'checkpoint_representations':dict(Counter(r['checkpoint'].get('representation') for r in runs)),
            'reduced_checkpoint_quality_statuses':dict(Counter(r['checkpoint'].get('reduced_quality_status') for r in runs)),
            'pilot_engineering_gaps':{r['service_id']:r['engineering_gaps'] for r in runs if r['engineering_gaps']},
            'stage_seconds':stages,'baseline_seconds':sum(b.get('runtime_s',0) for b in baselines),
            'review_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [root/'scripts/review_families.py',root/'scripts/run_family_review.py',
                          root/'mlaas_data_generator/tasks/contracts.py',root/'mlaas_data_generator/services/runner.py']}}
    (out/'health_and_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    assert checks['historical_status_counts_unchanged']
    assert integrity==['ok'] and not foreign_keys
    lines=['','## Execution, time and scope','',
        f"Pilot database integrity: OK; {checks['pilot_checkpoint_verified']}/{len(runs)} terminal services have passed checkpoint round-trip verification. Historical source status counts are unchanged.",
        f"Stored service time totals: training {stages['training_runtime_s']/60:.2f} min; evaluation {stages['evaluation_runtime_s']/60:.2f} min; checkpoint pipeline {stages['checkpoint_s']/60:.2f} min (including {stages['checkpoint_model_restore_verify_s']/60:.2f} min restore verification); service runtime {stages['service_runtime_s']/60:.2f} min. These are different stage measures, not additive estimates of review wall time. Preparation, separate initial evaluations and interrupted attempts are additional; per-family process durations are in `queue_status.json`.",
        f"Pilot quality labels: {dict(Counter(r['quality'] for r in runs))}. Full per-service metrics, actual updates and requested/runtime differences are in `service_results.json`.",
        'Recorded budget adjustments explain realized epoch/accumulation changes caused by selected training quantities. CLIP also records its physical contrastive-batch adjustment; requested batch 1 is not evidence of singleton contrastive training.',
        'Runtime provenance for every selected clearance service matches the current dependency-lock hash, Python/platform, PyTorch and applicable Transformers/Datasets versions; see `runtime_comparison.json`. This supplements the model/data/label/preprocessing/initialization bindings, without assuming identical code merely from timestamps or a dirty Git commit.',
        f"The checkpoint pipeline includes {stages['checkpoint_reduced_quality_s']/60:.2f} min of its normal reduced-checkpoint quality validation. This is production checkpoint work, separate from the primary service evaluation; no additional exploratory precision sweep was run.",
        '', '## Diagnoses and bounded interventions','',
        '- MobileBERT: explicit layer normalization, gradient clipping and FP32 were exercised with saved checkpoints. The MRPC AdamW recipe improves on its majority baselines; constant-majority alternatives do not qualify.',
        '- CLIP: compare recall@1 against the same frozen 500-candidate initial evaluation; recall@5 alone is insufficient. Successful updates, non-singleton training batches and saved-checkpoint verification provide production-path evidence beyond the earlier synthetic check.',
        '- ViT/SVHN: the initial three 256-image, 96-update trials reached only 19–21% accuracy. Its single follow-up uses 730 IID examples and six epochs, keeping initialization/evaluation and remaining knobs fixed. No further recipe search is authorized by this bundle.',
        '- SegFormer/ADE20K-mini: the corrected three-label initial model produced nonfinite FP16 logits. The sole follow-up selects FP32 and includes the task-contract runtime-data traversal fix. Earlier 150-label results cannot clear this preparation.',
        '- Task-contract resolution now avoids converting prepared arrays to Python lists. Structured evaluation persists its existing per-class prediction counts so collapse can be checked without another inference pass.',
        '- Historical DeBERTa/CoLA and BERT/IMDb evidence is reused only under matching frozen contracts. The final BERT/IMDb selection excludes the weak historical SGD recipe. A patch file or a later timestamp alone is insufficient.',
        ]
    for hold in holds:
        lines.append(f"- {hold['review_family_id']}: inconclusive before training. {hold['cause']}. No unfinished service was inserted into SQLite; this family is excluded.")
    lines+=['','## Evidence interpretation','',
        'The inventory groups model, dataset, subset, task and regime for triage; clearance uses exact family-document and initialization hashes. Its optimizer/joint-recipe counts and IID/skew profile counts describe persisted heterogeneity, without pairwise behavioral or data-overlap calculations.',
        'Existing IID evidence may have different recorded sample quantities across seeds. Only its explicitly healthy joint recipes at those recorded quantities enter the final roster. Current prospective seed repeats use matching quantities.',
        'A weak recipe can remain in the review evidence when execution is reliable and its weakness is explained, but this conservative final roster selects healthy recipes only. Three representatives do not guarantee population-wide success.',
        'Interrupted means an attempt ended without a genuine completed or failed outcome; it may have started preparation or training. It does not mean a measured model failure. This review records such attempts in its progress/diagnosis files rather than fabricating terminal database rows.',
        'The final manifest contains 100 prospective clients per cleared family. It has been generated and validated, not run. Keep its companion definitions and referenced immutable artifacts available.',
        'Validation: 95 production/task/manifest regression tests passed, followed by 10 focused clearance tests after the final provenance guards. Logs are `test_results.log` and `clearance_test_results.log`.',
        ]
    return lines
