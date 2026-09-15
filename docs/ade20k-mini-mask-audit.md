# ADE20K-mini mask audit

Inspected the actual local `qubvel-hf/ade20k-mini` Arrow files at dataset revision
`0d17237f0ea54dd73dc1239d06f7c834dd27007f` on 2026-09-13. No models were loaded.
The sample contains 25 evenly spaced training masks and 25 evenly spaced validation
masks, including the first and last record of each split. Per-mask modes, shapes,
channel equality and unique values are saved in `outputs/checks/ade20k_mask_audit.json`.

All 50 annotations are RGB, H×W×3. R=G across every pixel in 16 masks; R=B in 3.
The channels are therefore not interchangeable. Across the sample, R contains only
0, 1 and 2; G contains IDs from 0 to 28; B is always zero.

Every sampled record declares `semantic_class_to_id = {background: 0, person: 1,
car: 2}`. All red-channel IDs are in this declared map. The dataset's own
[Hub viewer](https://huggingface.co/datasets/qubvel-hf/ade20k-mini) also exposes this
three-class mapping. These are reduced foreground classes, not the full ADE20K
150-class label space. The observed G values are consistent with instance IDs.

Keep the existing red-channel class conversion. The actual error was allowing
the pretrained model's 150-label count to override the dataset's explicit schema.
Fine-tuning now uses the declared three-label head and preserves class 0 as valid
background. It validates that the declaration is consistent across selected
records and that processed mask IDs fit the schema. Datasets without an explicit
mapping retain their existing handling. Inference-only execution retains its
checkpoint head and still needs semantic alignment to compare different label spaces.

This corrects the earlier inference that three observed target classes proved bad
mask decoding. Historical metrics are not rewritten. Regenerate frozen family
manifests before producing new saved providers with the corrected label contract.
