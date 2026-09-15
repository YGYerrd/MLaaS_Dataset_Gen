"""Opt-in stabilization for MobileBERT's unnormalized classification features."""
import torch

POLICY = "layer_norm_v1"


class NormalizedLinear(torch.nn.Linear):
    """Parameter-free feature normalization; retain linear checkpoint key names."""

    def forward(self, inputs):
        inputs = torch.nn.functional.layer_norm(inputs, (self.in_features,), eps=1e-5)
        return torch.nn.functional.linear(inputs, self.weight, self.bias)


def apply_mobilebert_head_normalization(model, policy):
    if policy is None or policy == "none":
        return
    if policy != POLICY:
        raise ValueError(f"Unsupported MobileBERT head normalization: {policy}")
    if getattr(model.config, "model_type", None) != "mobilebert":
        raise ValueError("MobileBERT head normalization requires a MobileBERT model")
    head = getattr(model, "classifier", None)
    if isinstance(head, NormalizedLinear):
        return
    if not isinstance(head, torch.nn.Linear):
        raise ValueError("MobileBERT head normalization requires a linear classifier")
    # Preserve parameters, device, dtype and RNG state exactly.
    replacement = NormalizedLinear.__new__(NormalizedLinear)
    torch.nn.Module.__init__(replacement)
    replacement.in_features, replacement.out_features = head.in_features, head.out_features
    replacement.weight, replacement.bias = head.weight, head.bias
    replacement.train(head.training)
    model.classifier = replacement
    model.config.mobilebert_head_normalization = policy
