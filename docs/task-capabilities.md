# Task capabilities and measured composition

The task extension adds separate capability decisions, frozen task contracts and
request schema v5. Single-label token classification, semantic segmentation and
seq2seq summarisation can use the independent-family and shared-parent workflows.
Backend declarations remain experimental where coverage is limited; a successful
run certifies its recorded model, dataset, runtime and protocol combination.

## Inspect capabilities

```powershell
python -m mlaas_data_generator.cli.main task-capabilities --task token_classification --backend hf_finetune
```

Omit the filters to report the whole matrix. Each capability has its
own implementation status, instance decision, stage, reason, remediation and
evidence references. `pending` means a required runtime check has not happened;
an implemented backend with a missing or corrupt checkpoint is still ineligible.
Manifest preflight reports declarations; publication records checkpoint checks;
v5 measured validation reports verified aggregation and audit decisions.

Independent raw-source reconstruction also reports `recipe_persistence`,
`materialization` and `reconstruction_validation`. These do not require retained
client weights. A passing reconstruction report establishes measured support only
for its recorded family/recipe/runtime; see [reconstruction coverage](client-reconstruction.md#inspect-reconstruction-coverage).

| Profile | Primary / secondary metric | Partition profiles | Sampling and aggregation unit |
| --- | --- | --- | --- |
| Single-label tokens | Supervised-token weighted F1 / accuracy | IID, shard, Dirichlet using the existing sequence entity-presence axis | Whole source sequences |
| Fill-mask | Masked-token accuracy / perplexity proxy | IID, Dirichlet and shard using raw text character-length buckets | Whole source sequences |
| Semantic segmentation | Corpus macro IoU / Dice over classes present in truth or predictions | IID with varying sample counts | Whole images |
| Seq2seq summarisation | Decoded ROUGE-1 / ROUGE-2 F1, example mean, 0–1 | IID with varying sample counts | Source-target examples |

Token F1 is not entity-span F1. Segmentation uses nearest-neighbour target resizing
to logits resolution, excludes void pixels, and excludes classes absent from both
truth and predictions. Summarisation uses the repository's versioned word and
punctuation tokenisation and ROUGE implementation. Teacher-forced loss is reported
separately; references never enter `generate`. Immediate EOS is scored as empty
text, while absent predictions or failed execution are errors.

## Frozen boundaries

`tasks/contracts.py` declares semantics, sampling, supervision and named evaluation
protocols. `tasks/runtime.py` executes structured inference with streaming metric
statistics; `tasks/diagnostics.py` streams per-example prediction artifacts and
pairwise comparisons. The central checkpoint, lineage, dataset-content and
reservation checks remain mandatory.

New independent families use `raw_source_family_v3`. New task reservations use
`reserved_evaluation_v2`: per-source target hashes replace embedded masks or token
arrays. Reconstruction verifies the immutable source content before inference.
`source_group_column` can declare document or crop provenance; its train/test
identities are checked and catalogue/audit reservations preserve complete groups.
Client overlap within the training source remains permitted and is reported.

Task semantics bind label/vocabulary meaning, alignment, masking and preprocessing.
Evaluation identities additionally bind metric reductions and deterministic
decoding settings. Different quantities, seeds and optimisers remain recorded
client heterogeneity. Aggregation still uses persisted **training-example counts**;
valid tokens/pixels are separate supervision counts. All-ignored batches and
non-finite losses fail before a successful update can be reported.

Saved request versions 1–4, legacy family/reservation readers and recipe hashes
retain their existing interpretation. Missing historical task settings are not
inferred into v5. Changed semantics require new artifacts.

## Generate and validate v5 requests

```powershell
python -m mlaas_composition generate-requests --db outputs/services.db --profile measured_tasks_v5 --compatibility-mode strict --output outputs/task_requests.json
python -m mlaas_composition evaluate --db outputs/services.db --requests-file outputs/task_requests.json --output-dir outputs/task_search
```

Use the existing `validate` command with the selected request and its frozen
reservation. V5 requires parameter aggregation and measured validation, binds both
task hashes, and freezes eligible membership. An incompatible selected member
fails validation; validation does not replace members or repeat selection.
Results include named task metrics, supervision counts, aggregate provenance,
timings and capability decisions. Execution failures include a reason code and
stage.

## Real backend verification

The opt-in harness uses cached registry models and datasets. Only dataset transport
is replaced with reads of identified cached Arrow files; preprocessing, training,
checkpoint storage, search and decoded/model inference use production code.
The exact Arrow hashes, first-N row selection and model revisions are recorded.
There are four frozen clients per mode, 48 source training examples and 16 source
evaluation examples, seed 42, FP32 and native checkpoint storage. The held-out
reservation divides evaluation into catalogue and audit partitions before training.

```powershell
python -m tests.smoke.run_task_extensions --profile token --device cuda --shared-parent --output-dir outputs/token_smoke
python -m tests.smoke.run_task_extensions --profile segmentation --device cpu --processor-size 128 --shared-parent --output-dir outputs/segmentation_smoke
python -m tests.smoke.run_task_extensions --profile summarization --device cuda --shared-parent --output-dir outputs/summarization_smoke
```

Use a new directory for a changed protocol or roster. Reusing a directory resumes
its frozen manifest. The harness checks all four clients, diagnostics, strict
search, selected aggregation, exact restored states, replay, contributor weights,
preserved tensors and resume without rewriting immutable artifacts. It measures
the parent and the individual selected using catalogue quality as audit baselines.
In-memory parent, schema and compatibility mismatches must fail before inference.
Fast tests additionally cover corrupted persisted checkpoints, changed source or
target content, reservation overlap and changed eligible membership.

Evidence is in `smoke_evidence.json`, `verification.json`, diagnostics, validation
reports, manifests and the run database. It includes code hashes, package versions,
hardware, revisions, contracts, source hashes, metrics and timings. Results on this
tiny subset establish execution mechanics; they impose no minimum accuracy and do
not establish unknown pretraining exposure as absent. An aggregate may perform
worse than either baseline.

The September 2026 local runs use TinyBERT/WikiANN and FLAN-T5-small/XSum on the
RX 9070 XT, and SegFormer-B0/ADE20K-mini on CPU with the frozen 128-pixel processor
protocol. See [recorded smoke results](task-smoke-results.md) for completion and
quality comparisons. The attempted segmentation ROCm path raised a MIOpen error
and has no passing runtime evidence. Reduced checkpoint formats do not inherit
native-storage evidence.

## Exclusions and incompatible combinations

Detection, causal LM, translation, captioning and other task variants keep their
individual adapter capabilities; this rollout does not give them a shared measured
composition protocol. Token multilabel and segmentation/generation label-skew
profiles are explicitly excluded until their target or sampling rules exist.

Task names alone do not make parameter aggregation impossible. Different parents,
incompatible tensor layouts, label meanings, tokenizers or audit protocols do make
particular selections incompatible with this operator. Trees and unaligned cluster
identities need a different composition operator. Functional catalogue matching
can remain useful independently of executable parameter composition.
