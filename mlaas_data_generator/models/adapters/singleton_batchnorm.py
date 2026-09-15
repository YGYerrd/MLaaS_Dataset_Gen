"""Keep singleton training examples using BatchNorm's stored statistics."""

from functools import partial

from .rocm_batchnorm import _native_batch_norm_forward


def _singleton_batch_norm_forward(module, forward, input):
    module._check_input_dim(input)
    samples = input.size(0)
    for size in input.shape[2:]:
        samples *= size
    if (
        module.training
        and samples == 1
        and module.running_mean is not None
        and module.running_var is not None
    ):
        # Only this layer uses evaluation statistics. Autograd remains enabled,
        # and neither running statistics nor num_batches_tracked are updated.
        module.training = False
        try:
            return forward(input)
        finally:
            module.training = True
    return forward(input)


def configure_singleton_batchnorm(model, torch):
    """Wrap standard/native BatchNorm without changing parameters or state keys."""
    if not callable(getattr(model, "modules", None)):
        return 0
    count = 0
    for module in model.modules():
        if type(module) not in (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d):
            continue
        forward = module.forward
        # Preserve custom forwards and make repeated finetune calls idempotent.
        if "forward" in module.__dict__ and not (
            isinstance(forward, partial) and forward.func is _native_batch_norm_forward
        ):
            continue
        module.forward = partial(_singleton_batch_norm_forward, module, forward)
        count += 1
    return count
