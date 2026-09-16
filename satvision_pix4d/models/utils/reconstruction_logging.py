"""Small TensorBoard panels with consistent target-derived display limits."""
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


@torch.no_grad()
def reconstruction_grid(target, prediction, pixel_mask, bands=(1, 12), max_times=3):
    # Rows: timestep/band. Columns: target, masked input, reconstruction.
    tiles = []
    selected = [b for b in bands if 0 <= b < target.shape[2]]
    if not selected:
        selected = [0]
    for t in range(min(max_times, target.shape[1])):
        for band in selected:
            truth = target[0, t, band].float()
            lo, hi = truth.min(), truth.max()
            scale = (hi - lo).clamp_min(1e-6)
            truth = ((truth - lo) / scale).clamp(0, 1)
            pred = ((prediction[0, t, band].float() - lo) / scale).clamp(0, 1)
            observed = truth * (1 - pixel_mask[0, t, 0])
            tiles.extend([truth[None], observed[None], pred[None]])
    tiles = F.interpolate(torch.stack(tiles), size=(128, 128), mode="bilinear", align_corners=False)
    return make_grid(tiles.cpu(), nrow=3, padding=2)
