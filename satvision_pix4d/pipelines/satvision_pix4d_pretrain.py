import os
import logging
import torch
import torchmetrics
import lightning.pytorch as pl

from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.optimizers.build import build_optimizer


class SatVisionPix4DSatMAEPretrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.config = config

        # ------------------------------
        # Build model
        # ------------------------------
        self.model = build_satmae_model(self.config)

        # ------------------------------
        # Training/data params
        # ------------------------------
        self.batch_size  = config.DATA.BATCH_SIZE
        self.num_workers = config.DATA.NUM_WORKERS
        self.img_size    = config.DATA.IMG_SIZE
        self.pin_memory  = config.DATA.PIN_MEMORY
        self.mask_ratio  = config.DATA.MASK_RATIO

        # ------------------------------
        # Per-channel stats (buffers for broadcast over (B,T,C,H,W))
        # ------------------------------
        mean = torch.as_tensor(self.config.DATA.MEAN, dtype=torch.float32)  # (C,)
        std  = torch.as_tensor(self.config.DATA.STD,  dtype=torch.float32)  # (C,)
        self.register_buffer("ch_mean", mean.view(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("ch_std",  std.view(1, 1, -1, 1, 1),  persistent=False)

        # Optional global min/max → scalar data_range for TorchMetrics
        data_min = getattr(self.config.DATA, "MIN", None)
        data_max = getattr(self.config.DATA, "MAX", None)
        if data_min is not None and data_max is not None:
            mn = torch.as_tensor(data_min, dtype=torch.float32).min().item()
            mx = torch.as_tensor(data_max, dtype=torch.float32).max().item()
            metric_data_range = max(mx - mn, 1e-6)
        else:
            approx_min = (mean - 4 * std).min().item()
            approx_max = (mean + 4 * std).max().item()
            metric_data_range = max(approx_max - approx_min, 1e-6)
        self.metric_data_range = float(metric_data_range)

        # ------------------------------
        # Metrics (computed in RAW scale)
        # ------------------------------
        self.train_loss_avg = torchmetrics.MeanMetric()
        self.val_loss_avg   = torchmetrics.MeanMetric()

        self.train_psnr = torchmetrics.image.PeakSignalNoiseRatio(
            data_range=self.metric_data_range
        )
        self.train_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=self.metric_data_range
        )
        self.val_psnr = torchmetrics.image.PeakSignalNoiseRatio(
            data_range=self.metric_data_range
        )
        self.val_ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=self.metric_data_range
        )

        # log masked-only PSNR (computed manually)
        self.log_masked_metrics = True

        # small debug flag
        self._z_dbg_steps = 0

    # ------------------------------
    # Helpers
    # ------------------------------
    @torch.no_grad()
    def _inverse_standardize(self, x):
        """x: (B,T,C,H,W) in z-score → returns raw physical scale."""
        return x * self.ch_std + self.ch_mean

    def _standardize(self, x):
        """x: (B,T,C,H,W) RAW → z-score in fp32, then cast to model dtype."""
        z = (x.float() - self.ch_mean) / self.ch_std.clamp_min(1e-6)
        model_dtype = next(self.model.parameters()).dtype
        return z.to(model_dtype)

    def _tokens_to_pixel_mask(self, mask_tokens, T, H, W):
        """
        Convert patch-token mask (B, L=T*h*w) with 1=masked to per-time pixel mask (B,T,1,H,W),
        without unpatchify: reshape to (B,T,h,w) and repeat each patch to (p,p).
        """
        B, L = mask_tokens.shape
        p = int(self.model.patch_embed.patch_size[0])
        h, w = H // p, W // p
        m = mask_tokens.view(B, T, h, w)            # (B,T,h,w)
        m = m.unsqueeze(2)                          # (B,T,1,h,w)
        m = m.repeat_interleave(p, dim=3).repeat_interleave(p, dim=4)  # (B,T,1,H,W)
        return m.float()

    def _merge_pred_with_visible(self, pred_tokens, gt_imgs_z):
        """
        MAE-style full reconstruction in z-space:
        - use GT tokens at visible positions
        - use predicted tokens at masked positions
        pred_tokens: (B, L, D) for ALL tokens (original order)
        gt_imgs_z:   (B,T,C,H,W) in z-score space
        returns:     recon_z: (B,T,C,H,W) in z-score space
        """
        B, T, C, H, W = gt_imgs_z.shape
        gt_tokens = self.model.patchify(gt_imgs_z)              # (B,L,D)
        mask_tokens = self._last_mask_tokens                    # (B,L) 1=masked

        m = mask_tokens.bool().unsqueeze(-1)
        full_tokens = torch.where(m, pred_tokens, gt_tokens)     # replace masked with preds
        recon_z = self.model.unpatchify(full_tokens, T, H, W)
        return recon_z

    def _avg_over_time(self, metric_fn, pred_raw, tgt_raw, pixel_mask_per_t=None):
        """
        pred_raw, tgt_raw: (B,T,C,H,W) in RAW scale.
        If pixel_mask_per_t is provided: (B,T,1,H,W) with 1=masked (per time).
        """
        B, T, C, H, W = pred_raw.shape
        is_psnr = (metric_fn is self.train_psnr) or (metric_fn is self.val_psnr)
        vals = []
        for t in range(T):
            if pixel_mask_per_t is not None and is_psnr:
                m = pixel_mask_per_t[:, t]                      # (B,1,H,W)
                num = m.sum().clamp_min(1)
                mse = (((pred_raw[:, t] - tgt_raw[:, t]) ** 2) * m).sum() / num
                psnr_t = 10.0 * torch.log10((self.metric_data_range ** 2) / mse.clamp_min(1e-12))
                vals.append(psnr_t)
            else:
                vals.append(metric_fn(pred_raw[:, t], tgt_raw[:, t]))
        return torch.stack(vals).mean()

    @classmethod
    def load_checkpoint(cls, ckpt_path, config):
        if ckpt_path.endswith(".pt") or "mp_rank" in ckpt_path:
            logging.info(f"Loading DeepSpeed checkpoint: {ckpt_path}")
            model = cls(config)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("module", checkpoint)
            cleaned_state_dict = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state_dict.items()
            }
            missing_keys, unexpected_keys = model.model.load_state_dict(cleaned_state_dict, strict=False)
            logging.info(
                f"Loaded DeepSpeed weights with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys."
            )
            return model
        else:
            logging.info(f"Loading Lightning checkpoint: {ckpt_path}")
            return cls.load_from_checkpoint(ckpt_path, config=config)

    # ------------------------------
    # Forward → delegate to MAE model
    # ------------------------------
    def forward(self, samples_z, timestamps):
        return self.model(samples_z, timestamps, self.mask_ratio)

    # ------------------------------
    # Training / Validation
    # ------------------------------
    def training_step(self, batch, batch_idx):
        samples_raw, timestamps = batch
        samples_raw = samples_raw.to(self.device, non_blocking=True)  # RAW
        timestamps  = timestamps.to(self.device,  non_blocking=True)

        # standardize once here
        z = self._standardize(samples_raw)

        # small debug for first few steps
        if self._z_dbg_steps < 3:
            with torch.no_grad():
                zm = z.float().mean().item()
                zs = z.float().std().item()
            self.print(f"[dbg] train z mean={zm:.3f} std={zs:.3f}")
            self._z_dbg_steps += 1

        loss, pred_tokens, mask_tokens = self.forward(z, timestamps)
        self.train_loss_avg.update(loss.detach())

        B, T, C, H, W = z.shape
        # full-image reconstruction for metrics (pred at masked, GT at visible)
        self._last_mask_tokens = mask_tokens  # needed by _merge_pred_with_visible
        recon_z = self._merge_pred_with_visible(pred_tokens, z)  # z-space
        pred_raw = self._inverse_standardize(recon_z)            # RAW
        tgt_raw  = samples_raw                                   # RAW

        # per-time pixel mask (1=masked) for masked-only PSNR
        pixel_mask_t = self._tokens_to_pixel_mask(mask_tokens, T, H, W)

        # metrics
        train_psnr_full   = self._avg_over_time(self.train_psnr, pred_raw, tgt_raw, pixel_mask_per_t=None)
        train_ssim_full   = self._avg_over_time(self.train_ssim, pred_raw, tgt_raw, pixel_mask_per_t=None)
        train_psnr_masked = self._avg_over_time(self.train_psnr, pred_raw, tgt_raw, pixel_mask_per_t=pixel_mask_t)

        self.log("train_loss", self.train_loss_avg.compute(), prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("train_psnr", train_psnr_full,               prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("train_ssim", train_ssim_full,               prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("train_psnr_masked", train_psnr_masked,      prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        samples_raw, timestamps = batch
        samples_raw = samples_raw.to(self.device, non_blocking=True)  # RAW
        timestamps  = timestamps.to(self.device,  non_blocking=True)

        # standardize here too (this was missing before → caused mismatch)
        z = self._standardize(samples_raw)

        loss, pred_tokens, mask_tokens = self.forward(z, timestamps)
        self.val_loss_avg.update(loss.detach())
        self._last_mask_tokens = mask_tokens

        B, T, C, H, W = z.shape
        # MAE-style merge in z-space (pred for masked, GT for visible)
        recon_z = self._merge_pred_with_visible(pred_tokens, z)
        pred_raw = self._inverse_standardize(recon_z)  # RAW
        tgt_raw  = samples_raw                          # RAW

        pixel_mask_t = self._tokens_to_pixel_mask(mask_tokens, T, H, W)

        val_psnr_full   = self._avg_over_time(self.val_psnr, pred_raw, tgt_raw, pixel_mask_per_t=None)
        val_ssim_full   = self._avg_over_time(self.val_ssim, pred_raw, tgt_raw, pixel_mask_per_t=None)
        val_psnr_masked = self._avg_over_time(self.val_psnr, pred_raw, tgt_raw, pixel_mask_per_t=pixel_mask_t)

        self.log("val_loss", self.val_loss_avg.compute(), prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("val_psnr", val_psnr_full,               prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("val_ssim", val_ssim_full,               prog_bar=True,  sync_dist=True, batch_size=self.batch_size)
        self.log("val_psnr_masked", val_psnr_masked,      prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    # ------------------------------
    # Optimizer / Schedulers
    # ------------------------------
    def configure_optimizers(self):
        optimizer = build_optimizer(self.config, self.model, is_pretrain=True)

        # Robust scheduler when dataset is tiny
        total_steps  = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(self.config.TRAIN.WARMUP_EPOCHS * (total_steps / self.config.TRAIN.EPOCHS))
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=self.config.TRAIN.MIN_LR
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self.train_psnr.reset()
        self.train_ssim.reset()

    def on_validation_epoch_start(self):
        self.val_loss_avg.reset()
        self.val_psnr.reset()
        self.val_ssim.reset()
