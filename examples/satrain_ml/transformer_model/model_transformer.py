"""Experiment #2: Hybrid CNN + factorized space-time Transformer U-Net.

Loss, targets, thresholds, optimizer family, augmentation philosophy, and precision are kept
identical to the baseline (see ../baseline_model/model.py). The only intentional differences are
the architecture and the temporal ABI input (geo_t: 7 timesteps x 16 channels, time-major).

# ---------------------------------------------------------------------------
# AUDIT REPORT (fair-comparison correctness audit against the baseline contract)
# ---------------------------------------------------------------------------
# 1. _compute_losses: active regression term is raw mm/h MSE
#    ((surface_precip_pred - surface_precip) ** 2).mean() -- no log1p. The log1p/expm1
#    variant exists only as a commented, disabled "OLD COMPUTE LOSSES" block. CONFIRMED.
# 2. FastSatRainTemporal.__getitem__ returns exactly {"surface_precip","precip_mask",
#    "heavy_precip_mask"}; static = cat([gmi, anc]) = (19,64,64), asserted at construction.
#    CONFIRMED.
# 3. Augmentation samples ONE affine (angle,scale,shear) per sample and applies it
#    identically to geo_t (all 7x16 "channels", via a flatten/reshape so TF.affine's
#    single transform covers every timestep), to static (fill=input_nan_fill), and to
#    sp/pm/hp (fill=nan) -- timesteps stay spatially aligned. CONFIRMED.
# 4. geo_t is time-major everywhere: on-disk geo_t.npy is already (N,7,16,64,64); the
#    eval-time reshape in retrieval_transformer.py maps flat index t*16+c -> (T,C) (T
#    outer, C inner), verified to round-trip in verify_transformer.py. CONFIRMED.
# 5. Skip connections are captured only for t == REF_TIMESTEP (3), the overpass step --
#    not the last timestep -- and are exactly what the decoder consumes. CONFIRMED.
# 6. retrieval_fn (retrieval_transformer.py) reshapes obs_geo -> (B,7,16,H,W) (never
#    (B,16,7,...)), builds static = cat([obs_gmi, ancillary]), returns raw surface_precip
#    plus sigmoid(logits) + 0.5-threshold flags for both probability heads, using the
#    input's own dims tuple. CONFIRMED.
# 7. Heads (TransformerUNet.forward) return raw ResNetBlock(k=1)+Conv2d outputs -- no
#    sigmoid anywhere inside the model; sigmoid only applied in retrieval_fn at eval time.
#    CONFIRMED.
# 8. Fail-loud shape asserts are present at every boundary (TransformerUNet.forward,
#    FastSatRainTemporal.__init__, retrieval_fn, verify_transformer.py) -- no silent
#    shape hardcoding found. CONFIRMED.
# ---------------------------------------------------------------------------
"""
import math
from typing import Callable, List, Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF
import lightning as L

# ---------------------------------------------------------------------------
# Data contract constants (see DESIGN_NOTES.md) — asserted at runtime, never guessed.
# ---------------------------------------------------------------------------
N_TIMESTEPS = 7
N_ABI_CH = 16
N_GMI_CH = 13
N_ANC_CH = 6
STATIC_CH = N_GMI_CH + N_ANC_CH  # 19
REF_TIMESTEP = 3  # index of the overpass (t0) among the 7 ABI steps


class ResNetBlock(nn.Module):
    """Verbatim copy of the baseline's ResNetBlock (model.py) for architectural parity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        normalization_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        shortcut = x if self.projection is None else self.projection(x)
        return y + shortcut


def _build_encoder(input_features: int, internal_features: List[int], **block_kwargs):
    """Shared stem+encoder builder, mirrors baseline UNet's construction pattern."""
    stem = ResNetBlock(input_features, internal_features[0])
    chans_in = internal_features[0]
    stages = []
    for n_features in internal_features:
        stages.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlock(chans_in, n_features, **block_kwargs),
        ))
        chans_in = n_features
    return stem, nn.ModuleList(stages)


class SiameseABIEncoder(nn.Module):
    """Shared-weight per-timestep CNN encoder applied to each of the 7 ABI steps.

    Input:  geo_t (B, T, C, H, W)
    Output: bottleneck (B, T, D, h, w), shortcuts from the reference timestep only
            (list of (B, d_i, h_i, w_i), highest resolution first).
    """

    def __init__(self, abi_ch: int, internal_features: List[int], ref_timestep: int, **block_kwargs):
        super().__init__()
        self.ref_timestep = ref_timestep
        self.stem, self.encoder = _build_encoder(abi_ch, internal_features, **block_kwargs)

    def forward(self, geo_t: torch.Tensor):
        B, T, C, H, W = geo_t.shape
        assert 0 <= self.ref_timestep < T, f"ref_timestep={self.ref_timestep} out of range for T={T}"

        # Encode all T timesteps in ONE CNN pass over (B*T,C,H,W) instead of a Python loop
        # of T separate (B,C,H,W) passes -- same total FLOPs, far better GPU utilization
        # (the single biggest throughput win on a single H100; see Part C of the audit).
        #
        # NOTE (documented, intentional): BatchNorm2d computes training-mode statistics
        # over whatever batch it's given. The old loop ran 7 separate BN forward calls,
        # each normalizing over B samples (7 running-stat EMA updates per training step).
        # This batched version runs BN once over B*T samples pooled together (1 EMA
        # update per step). At eval time this is a no-op difference: BatchNorm uses
        # running stats, not batch stats, in .eval() mode, so the frozen-checkpoint
        # retrieval path (retrieval_fn) is unaffected either way.
        y = self.stem(geo_t.reshape(B * T, C, H, W))
        shortcuts_flat = []
        for layer in self.encoder:
            shortcuts_flat.append(y)
            y = layer(y)

        bottleneck_abi = y.reshape(B, T, *y.shape[1:])  # (B, T, D, h, w)
        ref_shortcuts = [s.reshape(B, T, *s.shape[1:])[:, self.ref_timestep] for s in shortcuts_flat]
        return bottleneck_abi, ref_shortcuts


class StaticEncoder(nn.Module):
    """Separate-weight CNN encoder for concat([gmi, anc]) -> bottleneck-resolution feature map."""

    def __init__(self, static_ch: int, internal_features: List[int], **block_kwargs):
        super().__init__()
        self.stem, self.encoder = _build_encoder(static_ch, internal_features, **block_kwargs)

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        y = self.stem(static)
        for layer in self.encoder:
            y = layer(y)
        return y  # (B, D, h, w)


class DropPath(nn.Module):
    """Stochastic-depth per-sample residual dropping (Part D: [low-risk, implement]).
    Identity whenever drop_prob=0.0 (the default) or in eval mode -- zero behavioral
    change unless a future run explicitly turns it on."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


def _mhsa(
    x: torch.Tensor, qkv: nn.Linear, proj: nn.Linear, n_heads: int, fp32: bool = False,
) -> torch.Tensor:
    """Multi-head self-attention over the token axis of x: (N, L, D).

    fp32=True upcasts q/k/v to float32 before scaled_dot_product_attention (under
    bf16-mixed autocast, SDPA's softmax otherwise runs in bf16, which has ~3 decimal digits
    of precision -- a numerically dangerous spot for divergence). Output is cast back to
    x's original dtype so the residual add stays dtype-consistent."""
    N, L, D = x.shape
    d_head = D // n_heads
    qkv_out = qkv(x).reshape(N, L, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)  # (3,N,heads,L,d)
    q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]
    if fp32:
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(x.dtype)
    else:
        out = F.scaled_dot_product_attention(q, k, v)  # (N, heads, L, d)
    out = out.transpose(1, 2).reshape(N, L, D)
    return proj(out)


class SpaceTimeBlock(nn.Module):
    """One factorized space-time transformer block (ViViT/TimeSformer-style):
    prenorm temporal self-attention -> prenorm spatial self-attention -> prenorm GELU MLP,
    each with a residual connection. Operates on channel-last (B, T, H, W, D) tensors."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        attn_fp32: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.attn_fp32 = attn_fp32
        self.norm_t = nn.LayerNorm(dim)
        self.qkv_t = nn.Linear(dim, dim * 3)
        self.proj_t = nn.Linear(dim, dim)
        self.norm_s = nn.LayerNorm(dim)
        self.qkv_s = nn.Linear(dim, dim * 3)
        self.proj_s = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # Part D: [low-risk, implement] regularization knobs, default off (no behavioral
        # change unless a future run explicitly sets drop_path/dropout > 0).
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, D = x.shape

        # Temporal attention: tokens = the T steps at each fixed spatial location.
        xt = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        xt = xt + self.drop_path(
            _mhsa(self.norm_t(xt), self.qkv_t, self.proj_t, self.n_heads, fp32=self.attn_fp32)
        )
        x = xt.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)  # (B, T, H, W, D)

        # Spatial attention: tokens = the H*W locations within each timestep.
        xs = x.reshape(B * T, H * W, D)
        xs = xs + self.drop_path(
            _mhsa(self.norm_s(xs), self.qkv_s, self.proj_s, self.n_heads, fp32=self.attn_fp32)
        )
        x = xs.reshape(B, T, H, W, D)

        x_flat = x.reshape(B * T * H * W, D)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm_mlp(x_flat)))
        return x_flat.reshape(B, T, H, W, D)


class TemporalAggregation(nn.Module):
    """Collapses the T axis via a learned query token that cross-attends over time,
    independently at each spatial location. (B,T,H,W,D) -> (B,D,H,W)."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0, attn_fp32: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.attn_fp32 = attn_fp32
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        # Part D: [low-risk, implement] regularization knob, default off.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, D = x.shape
        d_head = D // self.n_heads
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        N = x.shape[0]
        x_n = self.norm(x)

        kv = self.kv(x_n).reshape(N, T, 2, self.n_heads, d_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_proj(self.query.expand(N, -1, -1)).reshape(N, 1, self.n_heads, d_head)
        q = q.permute(0, 2, 1, 3)

        if self.attn_fp32:
            out = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v)  # (N, heads, 1, d)
        out = out.transpose(1, 2).reshape(N, 1, D)
        out = self.dropout(self.proj(out).squeeze(1))  # (N, D)
        return out.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)


class TransformerUNet(nn.Module):
    """Hybrid CNN + factorized space-time Transformer U-Net.

    Tensor-shape flow (default internal_features=[64,128,256], bottleneck 8x8, D=256 --
    widened to be param-comparable to the baseline UNet, see IPWGTransformer docstring):
        geo_t (B,7,16,64,64), static (B,19,64,64)
          -> SiameseABIEncoder -> bottleneck_abi (B,7,256,8,8), ref-timestep shortcuts
          -> StaticEncoder     -> static_feat (B,256,8,8)
          -> + learned temporal/spatial pos enc, channel-last (B,7,8,8,256)
          -> SpaceTimeBlock x n_blocks (same shape throughout)
          -> TemporalAggregation -> temporal_agg (B,256,8,8)
          -> fuse(cat(temporal_agg, static_feat)) -> (B,256,8,8) [new bottleneck]
          -> U-Net decoder (upsample+concat(shortcut)+ResNetBlock) x len(internal_features_abi)
             -> (B,64,64,64)
          -> 3 heads (ResNetBlock(k=1)+Conv2d) -> each (B,1,64,64), raw logits/mm-h
    """

    def __init__(
        self,
        abi_ch: int = N_ABI_CH,
        static_ch: int = STATIC_CH,
        internal_features_abi: Optional[List[int]] = None,
        internal_features_static: Optional[List[int]] = None,
        n_blocks: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        n_timesteps: int = N_TIMESTEPS,
        ref_timestep: int = REF_TIMESTEP,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        attn_fp32: bool = False,
        **block_kwargs,
    ):
        super().__init__()
        # Widened from [64,64,128] to [64,128,256] (Part: capacity parity) -- lands the
        # transformer's param count within 1.003x of the baseline UNet's 8,354,883 (measured
        # empirically: 8,380,739 params at n_blocks=4), vs. 0.301x at the old default. Spatial
        # downsampling is unchanged (still 3 stages, h=w=8) -- only channel widths grew.
        internal_features_abi = internal_features_abi or [64, 128, 256]
        internal_features_static = internal_features_static or [64, 128, 256]
        assert internal_features_abi[-1] == internal_features_static[-1], (
            "ABI and static encoders must produce the same bottleneck channel width to fuse"
        )
        assert len(internal_features_abi) == len(internal_features_static), (
            "ABI and static encoders must downsample by the same factor to fuse spatially"
        )
        assert 0 <= ref_timestep < n_timesteps

        self.n_timesteps = n_timesteps
        self.ref_timestep = ref_timestep
        dim = internal_features_abi[-1]
        h = w = 64 // (2 ** len(internal_features_abi))

        self.abi_encoder = SiameseABIEncoder(abi_ch, internal_features_abi, ref_timestep, **block_kwargs)
        self.static_encoder = StaticEncoder(static_ch, internal_features_static, **block_kwargs)

        self.temporal_pos = nn.Parameter(torch.zeros(1, n_timesteps, 1, 1, dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, h, w, dim))
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.spatial_pos, std=0.02)

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim, n_heads=n_heads, mlp_ratio=mlp_ratio, drop_path=drop_path, dropout=dropout,
                attn_fp32=attn_fp32,
            )
            for _ in range(n_blocks)
        ])
        self.temporal_agg = TemporalAggregation(dim, n_heads=n_heads, dropout=dropout, attn_fp32=attn_fp32)
        self.fuse = ResNetBlock(dim * 2, dim, kernel_size=1, **block_kwargs)
        # Part D: [low-risk, implement] reference-timestep residual bypass -- the
        # t=ref_timestep CNN bottleneck feature (already computed for free by the
        # batched SiameseABIEncoder) is projected and added back after fuse(), so the
        # transformer path only has to REFINE a single-frame CNN baseline rather than
        # fully replace it: temporal context can help but can't make the model worse
        # than dropping the transformer path to zero.
        self.ref_bypass = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.ref_bypass.weight)
        nn.init.zeros_(self.ref_bypass.bias)

        decoder_stages = []
        chans_in = dim
        for nf in internal_features_abi[-2::-1]:
            decoder_stages.append(nn.Sequential(ResNetBlock(chans_in + nf, nf, **block_kwargs)))
            chans_in = nf
        decoder_stages.append(ResNetBlock(nf + nf, nf, **block_kwargs))
        self.decoder = nn.ModuleList(decoder_stages)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        heads = {}
        for output in ["surface_precip", "probability_of_precip", "probability_of_heavy_precip"]:
            heads[output] = nn.Sequential(
                ResNetBlock(nf, nf, kernel_size=1, **block_kwargs),
                nn.Conv2d(nf, 1, kernel_size=1),
            )
        self.heads = nn.ModuleDict(heads)

    def forward(self, geo_t: torch.Tensor, static: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert geo_t.shape[1:] == (self.n_timesteps, geo_t.shape[2], 64, 64), (
            f"geo_t must be (B,{self.n_timesteps},C,64,64), got {tuple(geo_t.shape)}"
        )

        bottleneck_abi, shortcuts = self.abi_encoder(geo_t)  # (B,T,D,h,w)
        static_feat = self.static_encoder(static)  # (B,D,h,w)

        x = bottleneck_abi.permute(0, 1, 3, 4, 2)  # channel-last: (B,T,h,w,D)
        x = x + self.temporal_pos + self.spatial_pos
        for block in self.blocks:
            x = block(x)
        temporal_agg = self.temporal_agg(x)  # (B,D,h,w)

        y = self.fuse(torch.cat([temporal_agg, static_feat], dim=1))
        y = y + self.ref_bypass(bottleneck_abi[:, self.ref_timestep])  # single-frame residual bypass

        for shortcut, layer in zip(reversed(shortcuts), self.decoder):
            y = self.upsample(y)
            y = torch.cat([y, shortcut], dim=1)
            y = layer(y)

        return {name: head(y) for name, head in self.heads.items()}


class IPWGTransformer(L.LightningModule):
    """Mirrors baseline IPWGUNet's Lightning conventions exactly (loss/metrics/optimizer);
    only the forward signature and underlying model differ (temporal ABI + static context)."""

    def __init__(
        self,
        internal_features_abi: Optional[List[int]] = None,
        internal_features_static: Optional[List[int]] = None,
        n_blocks: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        n_epochs: int = 200,
        base_lr: float = 1e-3,
        input_nan_fill: float = -1.5,
        drop_path: float = 0.0,
        dropout: float = 0.0,
        attn_fp32: bool = False,
        fp32_loss: bool = False,
        **block_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_epochs = n_epochs
        self.base_lr = base_lr
        self.input_nan_fill = input_nan_fill
        self.fp32_loss = fp32_loss
        self.model = TransformerUNet(
            abi_ch=N_ABI_CH,
            static_ch=STATIC_CH,
            internal_features_abi=internal_features_abi,
            internal_features_static=internal_features_static,
            n_blocks=n_blocks,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            n_timesteps=N_TIMESTEPS,
            ref_timestep=REF_TIMESTEP,
            drop_path=drop_path,
            dropout=dropout,
            attn_fp32=attn_fp32,
            **block_kwargs,
        )

    def forward(self, geo_t: torch.Tensor, static: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(geo_t, static)

    def _compute_losses(self, inpt, target):
        geo_t, static = inpt
        pred = self(geo_t, static)
        surface_precip = target["surface_precip"]
        precip_mask = target["precip_mask"]
        heavy_precip_mask = target["heavy_precip_mask"]

        valid = torch.isfinite(surface_precip)
        surface_precip = surface_precip[valid]
        precip_mask = precip_mask[valid]
        heavy_precip_mask = heavy_precip_mask[valid]
        surface_precip_pred = pred["surface_precip"][:, 0][valid]
        pop = pred["probability_of_precip"][:, 0][valid]
        pohp = pred["probability_of_heavy_precip"][:, 0][valid]

        if valid.sum() == 0:
            # Empty-valid-batch guard: an all-NaN target patch (e.g. an augmentation-filled
            # edge tile) makes .mean() over an empty tensor return NaN, silently poisoning the
            # loss and every downstream gradient for the rest of training. Fall back to a
            # zero loss that still carries a gradient path through pred (all-zero grad), so
            # backward()/optimizer.step() stay well-defined for this one batch.
            zero = pred["surface_precip"].sum() * 0.0
            return zero, zero, zero, surface_precip_pred, surface_precip

        sp_true_log = torch.log1p(surface_precip.clamp(min=0))

        if self.fp32_loss:
            # Numerical-safety upcast, NOT a fairness change -- same loss definition, just
            # computed in fp32 instead of whatever bf16-mixed autocast handed back. No longer
            # required for stability (log1p compresses the 0-141 mm/h target range to ~0-5,
            # removing the exploding-MSE-gradient cause), kept as an optional belt-and-suspenders.
            surface_precip_pred = surface_precip_pred.float()
            sp_true_log = sp_true_log.float()
            pop = pop.float()
            pohp = pohp.float()
            precip_mask = precip_mask.float()
            heavy_precip_mask = heavy_precip_mask.float()

        # log1p MSE (matches baseline's final v4log1p variant — fair comparison). Model predicts
        # in log-space; compresses the heavy-precip tail so raw-mm/h outliers no longer dominate
        # the gradient the way they did under the old raw-MSE loss.
        loss_estim = ((surface_precip_pred - sp_true_log) ** 2).mean()
        loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
        loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
        sp_pred_mmh = torch.expm1(surface_precip_pred.clamp(min=0))
        return loss_estim, loss_detect, loss_detect_heavy, sp_pred_mmh, surface_precip

    ### OLD COMPUTE LOSSES (raw mm/h MSE variant) — disabled hook, kept for reference. DO NOT
    ### enable without also updating retrieval_fn's expm1 hook in retrieval_transformer.py
    ### (the raw-MSE variant predicts mm/h directly; no expm1 needed there).
    # def _compute_losses(self, inpt, target):
    #     geo_t, static = inpt
    #     pred = self(geo_t, static)
    #     surface_precip = target["surface_precip"]
    #     precip_mask = target["precip_mask"]
    #     heavy_precip_mask = target["heavy_precip_mask"]
    #
    #     valid = torch.isfinite(surface_precip)
    #     surface_precip = surface_precip[valid]
    #     precip_mask = precip_mask[valid]
    #     heavy_precip_mask = heavy_precip_mask[valid]
    #     surface_precip_pred = pred["surface_precip"][:, 0][valid]
    #     pop = pred["probability_of_precip"][:, 0][valid]
    #     pohp = pred["probability_of_heavy_precip"][:, 0][valid]
    #
    #     loss_estim = ((surface_precip_pred - surface_precip) ** 2).mean()
    #     loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
    #     loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
    #     return loss_estim, loss_detect, loss_detect_heavy, surface_precip_pred, surface_precip

    def _log_precip_metrics(self, sp_pred, sp_true):
        """Read-only mm/h-space metrics on valid pixels. No effect on training.
        sp_pred, sp_true are already masked to finite pixels, in raw mm/h."""
        with torch.no_grad():
            err = sp_pred - sp_true
            mae = err.abs().mean()
            bias = err.mean()
            rain = sp_true >= 0.1
            if rain.any():
                mae_rain = (sp_pred[rain] - sp_true[rain]).abs().mean()
                bias_rain = (sp_pred[rain] - sp_true[rain]).mean()
            else:
                mae_rain = torch.tensor(0.0, device=sp_true.device)
                bias_rain = torch.tensor(0.0, device=sp_true.device)
            if sp_true.numel() > 1:
                vp = sp_pred - sp_pred.mean()
                vt = sp_true - sp_true.mean()
                denom = (vp.norm() * vt.norm()).clamp_min(1e-8)
                corr = (vp * vt).sum() / denom
            else:
                corr = torch.tensor(0.0, device=sp_true.device)
            self.log_dict({
                "val_mae_mmh": mae, "val_bias_mmh": bias,
                "val_mae_rain": mae_rain, "val_bias_rain": bias_rain,
                "val_corr": corr,
            }, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inpt, target = batch
        loss_estim, loss_detect, loss_detect_heavy, _, _ = self._compute_losses(inpt, target)
        tot_loss = loss_estim + loss_detect + loss_detect_heavy

        self.log_dict(
            {
                "train_loss": tot_loss,
                "train_loss_estim": loss_estim,
                "train_loss_detect": loss_detect,
                "train_loss_detect_heavy": loss_detect_heavy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return tot_loss

    def validation_step(self, batch, batch_idx) -> None:
        inpt, target = batch
        loss_estim, loss_detect, loss_detect_heavy, sp_pred, sp_true = self._compute_losses(inpt, target)
        tot_loss = loss_estim + loss_detect + loss_detect_heavy
        learning_rate = self.optimizers().param_groups[0]["lr"]
        self._log_precip_metrics(sp_pred, sp_true)
        self.log_dict(
            {
                "val_loss": tot_loss,
                "val_loss_estim": loss_estim,
                "val_loss_detect": loss_detect,
                "val_loss_detect_heavy": loss_detect_heavy,
                "learning_rate": learning_rate,
            },
            on_epoch=True,
            prog_bar=True,
        )

    def on_before_optimizer_step(self, optimizer):
        # Pre-clip grad norm (Lightning calls this hook before configure_gradient_clipping),
        # so it's the true divergence signal -- gradient_clip_val=1.0 below would otherwise
        # mask an exploding gradient by silently capping it every step.
        total_norm_sq = torch.zeros((), device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().pow(2).sum()
        self.log("grad_norm", total_norm_sq.sqrt(), on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.base_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """~500-step linear LR warmup, applied as a manual per-step override on top of the
        unmodified Adam+CosineAnnealingLR(T_max=n_epochs) family above (fair-comparison HARD
        CONSTRAINT). steps_per_epoch is ~550 for this dataset/batch_size=128 (measured from
        the real run's metrics.csv: epoch 6 ended at global_step 3296), so warmup fully
        completes within epoch 0 -- entirely before CosineAnnealingLR's first .step() call
        (Lightning steps epoch-interval schedulers once per epoch, at epoch end). This avoids
        ever mixing a step-interval and an epoch-interval scheduler on the same optimizer,
        which would otherwise clobber cosine's decayed LR back to base every step post-warmup.
        """
        warmup_steps = 500
        if self.trainer.global_step < warmup_steps:
            warmup_factor = (self.trainer.global_step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = self.base_lr * warmup_factor
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)


class FastSatRainTemporal(Dataset):
    """Loads geo_t/gmi/anc/sp/pm/hp memmaps. Returns ((geo_t, static), targets) where
    static = concat([gmi, anc]) = (19,64,64). Augmentation is per-sample: one random affine
    applied identically across all 7 ABI timesteps + static + targets (never per-batch, never
    with different params per timestep — see DESIGN_NOTES.md pitfall #2)."""

    def __init__(self, dir_: str, augment: bool = False, input_nan_fill: float = -1.5):
        self.geo_t = np.load(f"{dir_}/geo_t.npy", mmap_mode="r")
        self.gmi = np.load(f"{dir_}/gmi.npy", mmap_mode="r")
        self.anc = np.load(f"{dir_}/anc.npy", mmap_mode="r")
        self.sp = np.load(f"{dir_}/sp.npy", mmap_mode="r")
        self.pm = np.load(f"{dir_}/pm.npy", mmap_mode="r")
        self.hp = np.load(f"{dir_}/hp.npy", mmap_mode="r")
        assert self.geo_t.shape[1:] == (N_TIMESTEPS, N_ABI_CH, 64, 64), (
            f"geo_t.npy has unexpected shape {self.geo_t.shape}"
        )
        assert self.gmi.shape[1:] == (N_GMI_CH, 64, 64), f"gmi.npy has unexpected shape {self.gmi.shape}"
        assert self.anc.shape[1:] == (N_ANC_CH, 64, 64), f"anc.npy has unexpected shape {self.anc.shape}"
        n = self.geo_t.shape[0]
        assert n == self.gmi.shape[0] == self.anc.shape[0] == self.sp.shape[0]

        self.augment = augment
        self.fill = input_nan_fill
        self.rng = np.random.default_rng()

        inpt0, _ = self[0]
        geo_t0, static0 = inpt0
        assert geo_t0.shape == (N_TIMESTEPS, N_ABI_CH, 64, 64), f"sample 0 geo_t shape {geo_t0.shape}"
        assert static0.shape == (STATIC_CH, 64, 64), f"sample 0 static shape {static0.shape}"

    def __len__(self):
        return self.geo_t.shape[0]

    def __getitem__(self, i):
        geo_t = torch.from_numpy(np.ascontiguousarray(self.geo_t[i]).copy())  # (7,16,64,64)
        gmi = torch.from_numpy(np.ascontiguousarray(self.gmi[i]).copy())      # (13,64,64)
        anc = torch.from_numpy(np.ascontiguousarray(self.anc[i]).copy())      # (6,64,64)
        static = torch.cat([gmi, anc], dim=0)                                 # (19,64,64)
        sp = torch.from_numpy(np.ascontiguousarray(self.sp[i]).copy())
        pm = torch.from_numpy(np.ascontiguousarray(self.pm[i]).copy())
        hp = torch.from_numpy(np.ascontiguousarray(self.hp[i]).copy())

        if self.augment:
            angle = float(self.rng.uniform(-180, 180))
            scale = float(self.rng.uniform(0.8, 1.2))
            shear = float(self.rng.uniform(-30.0, 30.0))
            kw = dict(angle=angle, translate=[0.0, 0.0], scale=scale, shear=[shear])

            # TF.affine applies the SAME spatial transform to every channel of a (C,H,W) tensor,
            # so flattening (T,C,H,W) -> (T*C,H,W) keeps all 7 timesteps aligned with each other.
            T, C, H, W = geo_t.shape
            geo_t = TF.affine(geo_t.reshape(T * C, H, W), **kw, fill=self.fill).reshape(T, C, H, W)
            static = TF.affine(static, **kw, fill=self.fill)
            sp = TF.affine(sp.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)
            pm = TF.affine(pm.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)
            hp = TF.affine(hp.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)

        return (geo_t, static), {"surface_precip": sp, "precip_mask": pm, "heavy_precip_mask": hp}
