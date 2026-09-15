"""Native GPU BatchNorm for Windows ROCm builds with broken MIOpen HIPRTC.

Keep the standard BatchNorm state/update rules, but dispatch directly to
torch.native_batch_norm. The cudnn_enabled argument of torch.batch_norm alone
does not bypass MIOpen in affected builds. No global backend flags or model
parameters are changed.
"""

from functools import partial
import os
import sys


def _native_batch_norm_forward(module, input):
    import torch

    module._check_input_dim(input)
    factor = 0.0 if module.momentum is None else module.momentum
    if module.training and module.track_running_stats:
        if module.num_batches_tracked is not None:
            module.num_batches_tracked.add_(1)
            if module.momentum is None:
                factor = 1.0 / float(module.num_batches_tracked)

    training = module.training or (module.running_mean is None and module.running_var is None)
    if training:
        # Match functional.batch_norm's singleton validation before dispatch.
        samples = input.size(0)
        for size in input.shape[2:]:
            samples *= size
        if samples == 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {input.size()}")
    return torch.native_batch_norm(
        input, module.weight, module.bias,
        module.running_mean if not module.training or module.track_running_stats else None,
        module.running_var if not module.training or module.track_running_stats else None,
        training, factor, module.eps,
    )[0]


def configure_native_batchnorm(model, torch, device):
    """Return the number of newly configured standard BatchNorm layers."""
    if (
        sys.platform != "win32"
        or not getattr(getattr(torch, "version", None), "hip", None)
        or str(device).split(":", 1)[0] != "cuda"
        or os.environ.get("MLAAS_ROCM_NATIVE_BATCHNORM", "1") == "0"
    ):
        return 0
    count = 0
    for module in model.modules():
        # Custom subclasses / SyncBatchNorm may have different forward semantics.
        if type(module) not in (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d):
            continue
        if "forward" in module.__dict__:
            continue
        # partial keeps deepcopy/pickling valid while retaining the same modules,
        # parameter objects, hooks, and state_dict keys.
        module.forward = partial(_native_batch_norm_forward, module)
        count += 1
    return count
