# Select client data before preparation (Stage A + B)

Independent clients now use:

```text
load_raw_dataset -> select raw candidates -> prepare selected rows -> train -> evaluate
```

`load_raw_dataset` and `prepare_dataset` are separate public loader functions.
`load_dataset` composes them inside the execution caller's `training_row_selection`
scope. Both row-local execution and grouped manifest execution use this scope,
including normal runs that retain checkpoints.

For an arXiv pool of 203,037 rows and `sample_size=1000`, the initial candidate set
has 1,000 rows. Only those training rows are tokenized. If preparation rejects
rows, the next candidates in the same seeded order are prepared until 1,000
valid examples are accepted or the raw source is exhausted. Evaluation has its
own split and configured selection; evaluation rows are additional to the client
training count. Causal generation caches prepared evaluation rows across retries.

## Sampling contract

New partitions record `independent_source_pool_v4`.

- **IID:** a seeded uniform permutation of raw row positions, without replacement.
  The requested client quantity is its prefix. No label quotas or token-derived
  metadata are resolved, even if an unused `skew_axis` is present.
- **Sample quantities:** every profile receives independently assigned `sample_size`
  values. They are rounded, shuffled midpoints of uniform strata around the mean,
  followed by any policy/capacity adjustment. The allocator does not scale an
  already requested count a second time; quantity is not a separate profile.
- **Dirichlet / shard:** require an available `skew_axis`; fill-mask defaults to
  `raw_text_length_bucket`. Scalar class labels, numeric score bins, raw text
  lengths and declared domain sidecars can be resolved from
  raw data. Unsupported axes fail instead of silently falling back to another
  axis. Dirichlet records its resolved concentration.
- **Fill-mask:** new recipes support IID, Dirichlet and shard. Skewed profiles
  use `raw_text_length_bucket`, computed from the configured `text_column` before
  tokenization and masking. Character counts use Python `len(text)` (Unicode code
  points, with no trimming or normalization). Fixed inclusive upper bounds are
  0, 64, 256, 1024 and 4096, followed by a 4097+ bucket. This produces length skew;
  the buckets do not represent topics or token identities. Missing/non-string
  text or fewer than two populated buckets fails explicitly.
  Grouped clients share one bounded scan of the raw text column; only selected
  candidates and necessary replacements are tokenized. Bucket evidence records
  the text column, bounds and `raw_text_char_length_v1` semantics. Tokenizer,
  truncation and masking settings do not affect candidate partitioning; masking
  can still affect which candidates survive preparation.
  Explicit `masked_token_id` requests remain rejected; change existing skewed
  rows to `raw_text_length_bucket` rather than silently changing their semantics.
  Recorded source IDs remain authoritative for replay, which prepares only those
  rows. Frozen IID-only task contracts keep their recorded capability lists.
- **Other prepared-feature skew:** explicitly requesting a token-length, aligned
  entity or other prepared axis retains the existing full-pool preparation path.
  Merely using generation or IID never requests this work.

Generated recipes declare their skew axis for skewed profiles. Tasks without a
raw scalar axis use IID unless a supported axis is explicitly configured.
Quantity ranges are still generated using the existing recipe policy; `sample_size`
is the resulting requested count for that client.

Candidates prefer shard support. If that support is exhausted, selection advances
through the rest of the source to honor the requested count, recording
`support_relaxation_count`. Realized bucket counts describe accepted rows.

## Row identity and filtering

Each client retains its candidate order and ordered accepted source IDs. Grouped
clients select independently; preparation processes the union once and reuses
shared rows, then restores each client's order. A client never samples the
prepared subset again. Source IDs, sidecars and group IDs remain aligned.

Generation and image/multimodal decoders report accepted row positions. Explicit
row-eligibility failures permit replacement; infrastructure and schema errors
still fail the run. A filtering custom preprocessor must report
`accepted_train_positions` relative to its input candidates. Row-preserving
custom preprocessors need no position map.
Text classification and sentence-pair label fitting currently require a single
HF preprocessor; additional custom stages fail explicitly rather than preparing
the full pool or changing row identity silently.

Shared text label vocabularies and sentence-pair normalization continue to use
cheap raw labels from the source pool. Image classification uses the source label
schema; VQA vocabulary fitting uses raw answers. This does not tokenize texts or
decode the full image source. Dynamic padding uses the selected data and evaluation,
so its width can differ from historical full-pool preparation.

`loader_meta.tokenization_selection` records raw pool size, candidate count,
accepted union count and rejected count. Split provenance records each client's
requested/realized counts, ordered source IDs, seed, attempted/rejected counts and
completion reason. An empty valid source fails explicitly; an exhausted source
with some valid rows records its shortfall.

## Boundary with later stages

Checkpoints are still retained. The existing family/checkpoint implementation now
hashes only the selected prepared union and evaluation data. Its preparation cache
includes the selection roster, and a saved roster is reused when reloading that
preparation. This is an interim Stage A/B arrangement: families can still depend
on the selected union. Stage C will replace this with a semantic raw-source family
identity. Authoritative training recipes, `recipe_only`, reconstruction and
composition/FedAvg consumer migration remain later stages.

Historical full-pool family documents must be regenerated using new manifests
and service IDs. Shared-parent/FedAvg execution retains its existing partition
contract for this stage.

## Verification

`tests/generator/test_tokenization_selection.py` uses real HF Dataset row selection
and offline recording tokenizers. It covers a 203,037-row source selecting 1,000,
replacement after filtering, full-source exhaustion, grouped overlap and client
order, shared label mappings, explicit skew, raw/prepared API separation, dynamic
padding, normal checkpoint retention and the runner's training handoff. These are
offline pipeline checks, not a full arXiv model training experiment.
