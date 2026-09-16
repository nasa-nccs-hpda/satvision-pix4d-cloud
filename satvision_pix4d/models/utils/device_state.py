"""Place non-parameter state without touching ZeRO-sharded parameters."""
from torchmetrics import Metric


def move_non_parameter_state(module, device):
    # zero.Init places parameters, but DeepSpeed then skips module.to(device).
    # Buffers created outside zero.Init and TorchMetrics states need a move too.
    for child in module.modules():
        for name, buffer in child.named_buffers(recurse=False):
            setattr(child, name, buffer.to(device=device))
        if isinstance(child, Metric):
            child.to(device=device)
