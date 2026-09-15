# UPerNet-ConvNeXt Windows ROCm layout diagnostic and fix

13 September 2026. Follow-up to the [DPT diagnostic](dpt-batch-size-diagnostic.md).

The user's UPerNet log shows approximately 3.85 seconds per physical batch of
two images after startup, projecting to about 8.8 minutes of training for 128
batches. This is separate from the logged 107 seconds loading dataset shards
and 31 seconds downloading the model.

## Result

The slow convolution-layout path also affects
`openmmlab/upernet-convnext-tiny`. The DPT-only guard correctly did not apply to
this model. The production guard now includes UPerNet with a ConvNeXt backbone
on Windows ROCm, keeping physical batch size and all model/training settings.
Other UPerNet backbones are excluded.

Two fixed FoodSeg103 validation images were prepared once using UPerNet's own
512 x 512 processor, and reused in both conditions. The model revision is
`876ffc56b819a829448f7e81e9f8606deef6fb65`. This reproduces the model's slow
compute path; it does not reconstruct the exact Apple dataset service partition.
Both cases use FP32 parameters with FP16 autocast, GPU-resident inputs, the same
initial checkpoint and seed, native ROCm BatchNorm, and the same gradient
checkpointing policy. Training uses SGD at 1e-5, weight decay 0.01 and a matched
effective batch of two. At physical batch two, accumulation is one in both
the baseline and corrected cases. The existing singleton BatchNorm handling
is also installed for physical-batch-one diagnostic runs.

| Physical batch 2, same two images | Baseline | Production layout fix |
|---|---:|---:|
| Warm evaluation | 1,785.28 ms | 35.42 ms |
| Successful optimizer update on two images (median of two) | 3,892.59 ms | 113.25 ms |
| Peak allocated memory in follow-up evaluation benchmark | 1.042 GiB | 1.041 GiB |
| Peak allocated memory in follow-up training benchmark | 2.395 GiB | 2.399 GiB |

This is approximately 50x faster evaluation and 34x faster training for the
tested batch, with ample VRAM free. Times exclude first-use kernel compilation
and data preparation. The original batch-size-one evaluation overlapped the
period when the user's manifest was being stopped, so it is not used for a
clean speedup claim. The user confirmed the manifest stopped; subsequent
batch-size-two verification reproduced the baseline at approximately 1.79 s.

## Cause and numerical checks

Layer events identify `decode_head.fpn_convs.0.conv` as the largest offender,
at approximately 1,300 ms on input shape `[2,512,128,128]`. The next FPN
convolution takes approximately 322 ms. MIOpen's saved solver records identify
NHWC inputs with `ConvDirectNaiveConvFwd`. Contiguous NCHW convolution inputs
remove these slow paths; the corrected model's largest measured convolution
is the FPN bottleneck at approximately 4.8 ms.

A direct physical-batch-two, same-weights comparison produced finite logits in
both cases, with **99.9866% pixel-label agreement**, mean absolute logit
difference **0.002401**, relative L2 difference **0.000244**, and losses
**16.090727** versus **16.091652**. These are FP16 numerical differences, not
bitwise equality or an end-to-end quality guarantee.

The diagnostic now records GradScaler scale and whether each optimizer update
was actually applied. Initial high-scale attempts can be skipped. The fixed
batch-size-two run recorded two successful updates at approximately 113 ms,
and loss decreased from 9.30964 to 9.13379. This avoids reporting skipped AMP
updates as proof that weight training succeeded. Baseline initial timing data
did not have that counter; the follow-up verification records it explicitly.
That baseline recorded two successful updates at 3,937 and 3,848 ms, with loss
decreasing from 9.30859 to 9.13573. The table uses these successful-update
measurements, rather than skipped AMP steps.

## Activation and validation

Restart the manifest process. No recipe changes or reduced batch size are
required. The expected startup message for this model is:

```text
[HFCore] Windows ROCm: using contiguous UPerNet-ConvNeXt convolution inputs for 37 layers
```

The shared helper retains the original DPT entry point for compatibility and
uses independent opt-outs for DPT and UPerNet. Regression tests cover supported
backbone selection, platform restrictions, independent opt-out, idempotence,
state-dict/parameter preservation, gradients, serialization and HFCore setup.
All 55 targeted tests pass. The full 256-image service has not been rerun.

Raw data and kernel records are in `outputs/checks/upernet_batch_diagnostic/`:

- `default-fp16-layers.json`: initial baseline and layer measurements.
- `default-fp16-layers-runtime.json`: production-path benchmark, six measured
  repetitions and explicit optimizer-update counts.
- `default-fp16-verify.json`: same-weights numerical comparison and follow-up
  batch-size-two baseline.

Adjacent `upernet_batch_diagnostic*.log` files contain console output. As in
the DPT diagnostic, these Windows runtime processes wrote completed result
files but returned shell exit code 1 without a terminating Python traceback.
This is recorded as a separate runtime limitation, not a passing exit-status
check. The full benchmark methodology and profiler limitations are described
in the DPT report.

Example production-path reproduction:

```powershell
.venv/Scripts/python.exe scripts/diagnose_dpt_batch.py --output outputs/checks/upernet_batch_diagnostic --model-path 'D:\huggingface\hub\models--openmmlab--upernet-convnext-tiny\snapshots\876ffc56b819a829448f7e81e9f8606deef6fb65' --weight-decay 0.01 --runtime-layout --layers --eval-order 1,2 --repeats 6
```
