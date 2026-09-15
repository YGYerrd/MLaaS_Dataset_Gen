"""Avoid slow NHWC convolution paths in supported Windows ROCm models.

The batch-size diagnostic reproduced a >100x inference slowdown with ample
free VRAM. Contiguous NCHW inputs select fast MIOpen kernels at the same batch
size. Hooks preserve parameter objects, state_dict keys, and autograd.
"""

import os
import sys


def _contiguous_convolution_input(module, args, kwargs):
    if args:
        return (args[0].contiguous(), *args[1:]), kwargs
    if "input" in kwargs:
        return args, {**kwargs, "input": kwargs["input"].contiguous()}
    return args, kwargs


def convolution_layout_family(model):
    """Only include model/backbone combinations covered by GPU diagnostics."""
    config = getattr(model, "config", None)
    family = getattr(config, "model_type", None)
    if family == "dpt":
        return "DPT"
    backbone = getattr(config, "backbone_config", None)
    backbone_family = (backbone.get("model_type") if isinstance(backbone, dict)
                       else getattr(backbone, "model_type", None))
    if family == "upernet" and backbone_family == "convnext":
        return "UPerNet-ConvNeXt"
    return None


def configure_rocm_convolution_layout(model, torch, device):
    """Return the number of newly configured standard convolution layers."""
    family = convolution_layout_family(model)
    setting = ("MLAAS_ROCM_DPT_CONTIGUOUS_CONV" if family == "DPT"
               else "MLAAS_ROCM_UPERNET_CONTIGUOUS_CONV")
    if (
        sys.platform != "win32"
        or not getattr(getattr(torch, "version", None), "hip", None)
        or str(device).split(":", 1)[0] != "cuda"
        or family is None
        or os.environ.get(setting, "1") == "0"
    ):
        return 0
    count = 0
    for module in model.modules():
        if type(module) not in (torch.nn.Conv2d, torch.nn.ConvTranspose2d):
            continue
        # A top-level hook survives deepcopy/pickling. Checking the hook itself
        # also permits reinstalling it if a caller has explicitly removed it.
        if _contiguous_convolution_input in module._forward_pre_hooks.values():
            continue
        module.register_forward_pre_hook(_contiguous_convolution_input, with_kwargs=True)
        count += 1
    return count


def configure_dpt_convolution_layout(model, torch, device):
    """Keep the original DPT-only entry point for existing callers."""
    if convolution_layout_family(model) != "DPT":
        return 0
    return configure_rocm_convolution_layout(model, torch, device)
