"""Masked pixel fidelity, Sobel gradients, and VIS/NIR-only MS-SSIM.

Inputs are channel-standardized images. SSIM bounds must be expressed in that
same space (the model builder converts the configured raw channel bounds).
"""
import torch
from torch import nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type, in_chans, visnir_channels=(), ssim_min=(), ssim_max=()):
        super().__init__()
        if loss_type not in {"mse", "pixel_structure"}:
            raise ValueError("LOSS_TYPE must be mse or pixel_structure")
        self.loss_type = loss_type
        self.last_components = {}
        self.visnir_channels = tuple(visnir_channels)
        if loss_type == "pixel_structure":
            if (not self.visnir_channels or len(set(self.visnir_channels)) != len(self.visnir_channels)
                    or any(c < 0 or c >= in_chans for c in self.visnir_channels)):
                raise ValueError("Specify unique, zero-based VIS/NIR channel indices within IN_CHANS")
            if len(ssim_min) != in_chans or len(ssim_max) != in_chans:
                raise ValueError("SSIM bounds must have one entry per input channel")
            if any(hi <= lo for lo, hi in zip(ssim_min, ssim_max)):
                raise ValueError("SSIM maximum must exceed minimum for every channel")
        self.register_buffer("lower", torch.tensor(ssim_min).float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer("upper", torch.tensor(ssim_max).float().view(1, -1, 1, 1), persistent=False)

    @staticmethod
    def _masked_mean(value, mask):
        return (value * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1)

    def _ms_ssim(self, pred, target, mask):
        # Five scales, Gaussian 7x7 windows: supports chips >=112 pixels.
        # Mask-weighted local statistics prevent visible-only windows from
        # diluting the objective. Visible pixels provide boundary context only.
        if min(pred.shape[-2:]) < 112:
            raise ValueError("Five-scale MS-SSIM requires height and width >=112")
        channels = pred.shape[1]
        coords = torch.arange(7, device=pred.device, dtype=pred.dtype) - 3
        g = torch.exp(-coords.square() / (2 * 1.5 ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).expand(channels, 1, 7, 7)
        weights = pred.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        terms = []
        for level in range(5):
            def smooth(x):
                return F.conv2d(x, kernel, groups=channels)
            mu_p, mu_t = smooth(pred), smooth(target)
            var_p = (smooth(pred.square()) - mu_p.square()).clamp_min(0)
            var_t = (smooth(target.square()) - mu_t.square()).clamp_min(0)
            cov = smooth(pred * target) - mu_p * mu_t
            cs = (2 * cov + 0.03 ** 2) / (var_p + var_t + 0.03 ** 2)
            score = cs
            if level == 4:
                score = score * (2 * mu_p * mu_t + 0.01 ** 2) / (mu_p.square() + mu_t.square() + 0.01 ** 2)
            local_mask = F.avg_pool2d(mask, 7, stride=1)
            per_channel = (score * local_mask).sum((-2, -1)) / local_mask.sum((-2, -1)).clamp_min(1e-8)
            # Nonnegative factors are required for fractional scale powers.
            terms.append(per_channel.clamp(1e-6, 1))
            if level < 4:
                pred, target, mask = [F.avg_pool2d(v, 2, ceil_mode=True) for v in (pred, target, mask)]
        score = torch.stack(terms).pow(weights[:, None, None]).prod(0).mean(1)
        active = (mask.sum((1, 2, 3)) > 0).to(score.dtype)
        return ((1 - score) * active).sum() / active.sum().clamp_min(1)

    def _region_loss(self, pred, target, mask):
        char = self._masked_mean(((pred - target).square() + 1e-6).sqrt() - 1e-3, mask)
        merged = torch.where(mask.bool(), pred, target)
        channels = target.shape[1]
        sobel = pred.new_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
        kernels = torch.stack([sobel, sobel.T])[:, None].repeat(channels, 1, 1, 1)
        error = F.conv2d(F.pad(merged - target, (1, 1, 1, 1), mode="replicate"), kernels, groups=channels).abs()
        affected = F.max_pool2d(mask, 3, stride=1, padding=1)
        grad = self._masked_mean(error, affected)
        idx = self.visnir_channels
        lo, span = self.lower[:, idx], (self.upper - self.lower)[:, idx]
        # Fixed channel bounds, no prediction clipping (preserve gradients).
        pred_vis = (merged[:, idx] - lo) / span
        target_vis = (target[:, idx] - lo) / span
        structural = self._ms_ssim(pred_vis, target_vis, mask)
        components = {"charbonnier": char, "sobel": grad, "ms_ssim_loss": structural}
        return components

    def forward(self, pred, target, mask, visible_loss_weight=0.0):
        # Explicitly disable autocast: squared moments/SSIM are unstable in fp16.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target, mask = [v.float().flatten(0, 1) for v in (pred, target, mask)]
            components = self._region_loss(pred, target, mask)
            if visible_loss_weight > 0:
                visible = self._region_loss(pred, target, 1 - mask)
                components = {k: v + visible_loss_weight * visible[k] for k, v in components.items()}
            self.last_components = {k: v.detach() for k, v in components.items()}
            return (0.7 * components["charbonnier"] + 0.2 * components["sobel"]
                    + 0.1 * components["ms_ssim_loss"])
