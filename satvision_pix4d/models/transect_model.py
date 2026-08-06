"""
Decoder and finetuning model for 1D CloudSat/ABI transect reconstruction.

The adapted SatMAE ViT encoder (with asymmetric ``(16, 1)`` patches)
produces a 1D sequence of tokens along the CloudSat transect.  This
module defines a lightweight decoder that maps those encoder tokens to
a ``(512, 40)`` cloud-property curtain (cloud type per footprint per
vertical bin).

Architecture overview
---------------------

::

    SatMAE encoder
         │
    (B, T*L_s + 1, D)    <- token sequence + CLS
         │  drop CLS, temporal-pool
         ▼
    (B, L_s, D)           <- 1 spatial token per patch (L_s = 32)
         │
    ┌────▼────┐
    │ 1D UNet │           <- Conv1d encoder-decoder w/ skip connections
    │ decoder │
    └────┬────┘
         │
    (B, 512, 40)          <- cloud curtain: 512 footprints × 40 bins

The 1D UNet uses transposed-conv upsampling to go from 32 → 64 → 128
→ 256 → 512, while a parallel classification head maps each spatial
position to 40 vertical bins (or ``num_classes × 40`` for multi-class).
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # type: ignore[no-redef]

LOG = logging.getLogger(__name__)


# ─── 1D building blocks ───────────────────────────────────────────────────────

class DoubleConv1d(nn.Module):
    """Two Conv1d → BN → GELU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock1d(nn.Module):
    """Transposed convolution ×2 upsampling + DoubleConv1d."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv1d(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x


# ─── 1D Transect Decoder ──────────────────────────────────────────────────────

class TransectDecoder(nn.Module):
    """1D UNet-style decoder for transect cloud-property prediction.

    Takes the encoder's token embedding (one token per spatial patch) and
    progressively upsamples from ``L_s`` (e.g. 32) to ``target_len``
    (e.g. 512) while mapping each spatial position to ``num_bins``
    (e.g. 40) vertical bins.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of each encoder token (default 1024).
    num_classes : int
        Number of output classes per bin.  ``1`` for binary
        cloud/no-cloud, ``9`` for cloud-type segmentation.
    num_bins : int
        Number of vertical altitude bins per footprint (default 40).
    target_len : int
        Number of CloudSat footprints in the output (default 512).
    base_channels : int
        Channel width at the first decoder stage.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 9,
        num_bins: int = 40,
        target_len: int = 512,
        base_channels: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.target_len = target_len
        b = base_channels

        # Compress encoder tokens: D → base_channels
        self.stem = DoubleConv1d(embed_dim, b)         # at L_s

        # Upsampling path: L_s → 2L_s → 4L_s → 8L_s → 16L_s
        # e.g. 32 → 64 → 128 → 256 → 512
        self.up1 = UpBlock1d(b,     b // 2)            # ×2
        self.up2 = UpBlock1d(b // 2, b // 4)           # ×2
        self.up3 = UpBlock1d(b // 4, b // 8)           # ×2
        self.up4 = UpBlock1d(b // 8, b // 16)          # ×2

        # Classification head: each spatial position → num_classes × num_bins
        out_features = num_classes * num_bins
        self.head = nn.Sequential(
            nn.Conv1d(b // 16, b // 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(b // 16, out_features, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, L_s, D)
            Spatial encoder tokens (CLS and temporal already removed/pooled).

        Returns
        -------
        logits : (B, num_classes, target_len, num_bins)
            Per-footprint, per-bin class logits.
            For binary segmentation (num_classes=1), squeeze dim 1.
        """
        B, L_s, D = tokens.shape

        # (B, L_s, D) → (B, D, L_s) for Conv1d
        x = tokens.permute(0, 2, 1).contiguous()

        x = self.stem(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x is now (B, base//16, L_s * 16)

        # If upsampled length doesn't match target exactly, interpolate
        if x.shape[2] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear",
                              align_corners=False)

        # Predict class logits per spatial position
        x = self.head(x)  # (B, num_classes * num_bins, target_len)

        # Reshape to (B, num_classes, target_len, num_bins)
        x = x.view(B, self.num_classes, self.num_bins, self.target_len)
        x = x.permute(0, 1, 3, 2)  # (B, num_classes, target_len, num_bins)

        return x


# ─── End-to-end encoder + decoder ─────────────────────────────────────────────

class SatMAETransectModel(nn.Module):
    """SatMAE encoder + 1D decoder for transect cloud-property prediction.

    Wraps a pretrained ``MaskedAutoencoderViT`` encoder and a
    ``TransectDecoder``.  During the forward pass the encoder is run
    with ``mask_ratio=0`` (no masking) and the resulting tokens are
    pooled across the temporal dimension before being fed to the decoder.

    Parameters
    ----------
    mae_model : nn.Module
        A ``MaskedAutoencoderViT`` (the ``.model`` attribute of the
        pretrained pipeline).
    num_classes : int
        ``1`` for binary, ``9`` for cloud-type segmentation.
    num_bins : int
        Vertical altitude bins (default 40).
    target_len : int
        Number of CloudSat footprints (default 512).
    patch_size : int | tuple[int, int]
        Patch size used by the encoder (default 16).
    freeze_encoder : bool
        Whether to freeze encoder weights at init.
    temporal_pool : str
        How to pool across timesteps before the decoder.
        ``"mean"`` averages, ``"last"`` takes the center timestep.
    """

    def __init__(
        self,
        mae_model: nn.Module,
        num_classes: int = 9,
        num_bins: int = 40,
        target_len: int = 512,
        patch_size: int | tuple[int, int] = 16,
        freeze_encoder: bool = True,
        temporal_pool: str = "mean",
        decoder_base_channels: int = 256,
    ):
        super().__init__()
        self.mae = mae_model
        self.embed_dim = mae_model.cls_token.shape[-1]
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.target_len = target_len
        self.temporal_pool = temporal_pool

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.decoder = TransectDecoder(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            num_bins=num_bins,
            target_len=target_len,
            base_channels=decoder_base_channels,
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.mae.parameters():
            p.requires_grad = False
        LOG.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for fine-tuning."""
        for p in self.mae.parameters():
            p.requires_grad = True
        LOG.info("Encoder unfrozen")

    def _extract_tokens(
        self, imgs: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Run encoder with mask_ratio=0 and return patch tokens (no CLS).

        Parameters
        ----------
        imgs : (B, T, C, H, W)
        timestamps : (B, T, n_components)

        Returns
        -------
        tokens : (B, T * L_s, D)
            All encoder patch tokens, CLS removed.
        """
        out = self.mae.forward_encoder(imgs, timestamps, mask_ratio=0.0)
        latent = out[0] if isinstance(out, (tuple, list)) else out
        # Drop CLS token (first position)
        tokens = latent[:, 1:, :]
        return tokens

    def _temporal_pool(
        self, tokens: torch.Tensor, T: int
    ) -> torch.Tensor:
        """Pool tokens across the temporal dimension.

        Parameters
        ----------
        tokens : (B, T * L_s, D)
        T : int
            Number of timesteps.

        Returns
        -------
        pooled : (B, L_s, D)
        """
        B, TL, D = tokens.shape
        L_s = TL // T
        # (B, T * L_s, D) → (B, T, L_s, D)
        tokens = tokens.view(B, T, L_s, D)

        if self.temporal_pool == "mean":
            return tokens.mean(dim=1)
        elif self.temporal_pool == "last":
            # Take the center timestep
            return tokens[:, T // 2, :, :]
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")

    def forward(
        self, chips: torch.Tensor, timestamps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        chips : (B, C, T, H, W)
            ABI transect chip in channel-first format.
        timestamps : (B, T, n_components), optional
            Temporal metadata.  If ``None``, zeros are used.

        Returns
        -------
        logits : (B, num_classes, 512, 40)
        """
        B, C, T, H, W = chips.shape

        # SatMAE encoder expects (B, T, C, H, W)
        imgs = chips.permute(0, 2, 1, 3, 4).contiguous()

        if timestamps is None:
            n_comp = getattr(self.mae, "n_time_components", 3)
            timestamps = torch.zeros(
                (B, T, n_comp), device=chips.device, dtype=chips.dtype
            )

        # Encode with no masking
        tokens = self._extract_tokens(imgs, timestamps)  # (B, T*L_s, D)

        # Pool across time → (B, L_s, D)
        spatial_tokens = self._temporal_pool(tokens, T)

        # Decode → (B, num_classes, 512, 40)
        logits = self.decoder(spatial_tokens)

        return logits


# ─── Lightning training module ────────────────────────────────────────────────

class TransectLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for transect cloud-property prediction.

    Supports both binary segmentation (``num_classes=1``, uses
    ``BCEWithLogitsLoss``) and multi-class segmentation
    (``num_classes > 1``, uses ``CrossEntropyLoss``).

    Parameters
    ----------
    backbone : SatMAETransectModel
        The combined encoder + decoder model.
    encoder_lr : float
        Learning rate for the (optionally unfrozen) encoder.
    decoder_lr : float
        Learning rate for the decoder.
    dice_weight : float
        Weight of the Dice loss term relative to CE/BCE loss.
    unfreeze_encoder_at_epoch : int
        Epoch at which to unfreeze encoder weights.  ``-1`` to never
        unfreeze.
    """

    def __init__(
        self,
        backbone: SatMAETransectModel,
        encoder_lr: float = 1e-5,
        decoder_lr: float = 1e-4,
        dice_weight: float = 0.5,
        unfreeze_encoder_at_epoch: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        self.model = backbone
        self.num_classes = backbone.num_classes
        self.dice_weight = dice_weight
        self.unfreeze_encoder_at_epoch = unfreeze_encoder_at_epoch
        self._encoder_unfrozen = False

        if self.num_classes == 1:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)

    # ── Loss helpers ──────────────────────────────────────────────────

    @staticmethod
    def _dice_loss(logits: torch.Tensor, targets: torch.Tensor,
                   smooth: float = 1.0) -> torch.Tensor:
        """Soft Dice loss (works for both binary and multi-class)."""
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        intersection = (probs * targets).sum()
        cardinality = probs.sum() + targets.sum()
        return 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, chips: torch.Tensor,
                timestamps: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(chips, timestamps)

    # ── Training / validation / test steps ────────────────────────────

    def _common_step(self, batch: dict, stage: str) -> dict:
        if batch is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        chips = batch["chip"]     # (B, C, T, H, W)
        masks = batch["mask"]     # (B, 512, 40)

        logits = self.forward(chips)  # (B, num_classes, 512, 40)

        if self.num_classes == 1:
            # Binary: logits (B, 1, 512, 40) → (B, 512, 40)
            logits_squeezed = logits.squeeze(1)
            ce = self.ce_loss(logits_squeezed, masks.float())
            dice = self._dice_loss(logits_squeezed, masks)
        else:
            # Multi-class: CE expects (B, C, *) with targets (B, *)
            # logits: (B, num_classes, 512, 40), masks: (B, 512, 40)
            ce = self.ce_loss(logits, masks.long())
            # Dice on argmax prediction vs target (approximate)
            pred = logits.argmax(dim=1)
            dice = self._dice_loss(
                (pred > 0).float(), (masks > 0).float()
            )

        loss = self.dice_weight * dice + (1.0 - self.dice_weight) * ce

        self.log(f"{stage}_loss", loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log(f"{stage}_ce", ce, on_epoch=True, on_step=False)
        self.log(f"{stage}_dice", dice, on_epoch=True, on_step=False)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    # ── Encoder unfreezing ────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        if (
            not self._encoder_unfrozen
            and self.unfreeze_encoder_at_epoch >= 0
            and self.current_epoch >= self.unfreeze_encoder_at_epoch
        ):
            self.model.unfreeze_encoder()
            self._encoder_unfrozen = True

    # ── Optimizer with differential learning rates ────────────────────

    def configure_optimizers(self):
        encoder_params = list(self.model.mae.parameters())
        decoder_params = list(self.model.decoder.parameters())

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.hparams.encoder_lr},
            {"params": decoder_params, "lr": self.hparams.decoder_lr},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
