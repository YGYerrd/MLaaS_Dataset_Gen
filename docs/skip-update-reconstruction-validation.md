# Skipped-update reconstruction diagnostic — 16 September 2026

Three fresh-process GPU comparisons reproduced their native trained checkpoints
bit-for-bit. This supports relaxing the previous 5% eligibility cutoff. The
configured limit is now **20% of attempted optimizer updates**, inclusive. The
largest fraction measured here is **13.33%**; 20% is a policy allowance, not a
measured fidelity guarantee at that boundary.

| Model / dataset | Historical failed run | Diagnostic reference | Recipe replay | Model-state hash | Fidelity gate |
| --- | ---: | ---: | ---: | --- | --- |
| ALBERT / GLUE SST-2 | 4/66 (6.06%) | 1/66 (1.52%) | 1/66 (1.52%) | Identical | Passed |
| ELECTRA / WikiANN Spanish | 2/20 (10%) | 2/20 (10%) | 2/20 (10%) | Identical | Passed |
| RoBERTa / GLUE WNLI | 2/15 (13.33%) | 2/15 (13.33%) | 2/15 (13.33%) | Identical | Passed |

The lower case's historical skip rate did not recur. No skip decisions were
forced. The two higher cases retained their historical fractions in both runs.

All comparisons have zero parameter relative L2 error, update cosine similarity
1, identical training loss, and 100% prediction agreement. They evaluated 436,
2,048, and 36 examples respectively. Accuracy and F1 deltas are zero. ELECTRA's
evaluation loss differs by 0.00010014123623247606, within the unchanged fidelity
tolerance; the other evaluation-loss deltas are zero.

All reference and replay runs completed their configured epochs, with zero OOM
retries and zero loss-precision retries. Replay verified the selected source rows
and effective training settings. Guards prohibited client checkpoint reads during
reconstruction; the common initial checkpoint remained available.

## Method and scope

The experiment used the local AMD Radeon RX 9070 XT, existing FP16 mixed-precision
recipes, pinned models/data, original service IDs, sampling/training seeds, and
frozen family initial states. Each diagnostic reference used
`retain_validation_checkpoint=true` and `checkpoint_weight_format=native`.
Before comparison, the checkpoint's canonical state hash was checked against the
recipe's original-model hash, excluding INT8/FP16 storage approximation.

Separate diagnostic databases and artifact stores preserved the source database
and existing service artifacts. Each reference generation and reconstruction ran
in its own process. Only the comparison process temporarily allowed up to 20%
skips; production remained at 5% throughout the experiment. Completion, recovery,
integrity checks, and functional fidelity tolerances were not relaxed. After the
comparisons passed, the production cutoff was raised to 20%.

Two IMDb candidates (`svc_28bf9a145b2f` at 5.13% and
`svc_0863ed331b25` at 6%) failed the existing frozen-family verification before
training: `reloaded model, data, or task interface differs from frozen family`.
They were replaced with SST-2 and are not successful checkpoint comparisons.
Their logs are retained; their family mismatch was not bypassed or repaired.

This is one reference/replay pair per successful case, on one machine and the
current checkout. It establishes fidelity for those pairs, not universal
reproducibility, model quality, or recovery support. The 13.33% case provides
direct counterevidence to treating every run above 5% as unreconstructable.

## Evidence and regression checks

- Measured results and hashes
- ALBERT comparison
- ELECTRA comparison
- RoBERTa comparison
- Diagnostic harness

The case directories retain configurations, copied source failure records,
native references, authoritative recipes, and generation/comparison logs.

**78 targeted tests passed** in `test_weight_retention.py` and
`test_client_reconstruction.py`. Coverage includes acceptance at exactly 20%,
rejection above 20%, original and replay eligibility independently, recovery and
completion rejection, missing/inconsistent counters, and existing fidelity and
integrity checks. These boundary tests use simulated update counters; they do
not constitute GPU reconstruction measurements near 20%.
