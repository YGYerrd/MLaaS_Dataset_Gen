# Dataset execution provenance

The final SQLite dataset owns execution evidence in
`service_execution_provenance`. It does not require the input manifest or a
manifest sidecar to read that evidence. Each successful execution appends a
separate, hashed JSON document containing:

- the resolved public configuration at runner entry (including defaults),
  the final configuration, and changes between them;
- the effective training configuration before restoration of the requested
  recipe, plus negotiated Transformers adapter fit arguments;
- all emitted metrics, including final timing and recorded training/memory
  adjustments;
- service metadata, split evidence, generation and checkpoint provenance,
  artifact references, and runtime versions;
- embedded generation documents, including selected partition row IDs,
  for normal checkpoint-producing runs.

These records use no diagnostic byte, string, collection, or node limits.
Private worker caches/models are excluded from configurations; unsupported
runtime objects are explicitly identified by type. Actual model weights remain
in checkpoint artifacts. Runtime adjustments are available in emitted metrics,
configuration changes, and the training-budget adjustment field; the record
does not infer events the training backend did not report.

Execution IDs distinguish reruns. SQLite triggers reject replacement, update,
and deletion of evidence records; service replacement does not cascade into
them. SHA-256 detects document changes when verified against the stored hash;
this is not an externally signed or tamper-proof audit log. Creation is additive
for older databases, and never invents evidence for historical services.

The manifest specifies intended work. `requested_config` here means the
resolved configuration received by the service runner, not proof of the original
CSV contents. This distinction remains explicit in `evidence_scope`. Missing
original manifests cannot be recovered from observations, and historical
request-to-result equality must not be asserted. This change does not backfill
historical services or archive original manifest files.

The JSON document includes the complete emitted metric set; the existing
`service_metrics`, `service_generation_provenance`, and split/checkpoint tables
remain available for querying. A flattened metrics CSV alone is not a complete
provenance export: distribute the SQLite dataset to retain this evidence.
