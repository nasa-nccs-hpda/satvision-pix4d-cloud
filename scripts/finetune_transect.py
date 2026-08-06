#!/usr/bin/env python3
"""
Finetune SatMAE on 1D CloudSat/ABI transect data.

This script wires together:
  - Phase 2: Pre-trained SatMAE ViT encoder (weights from HuggingFace)
  - Task 1:  TransectDataModule (1D `.npz` loader)
  - Task 2:  SatMAETransectModel + TransectLightningModule

Usage
-----
    # Single-GPU, local:
    python scripts/finetune_transect.py

    # Multi-GPU via Slurm (set env vars in your .sh script):
    TRAIN_BATCH_SIZE=8 TRAIN_NUM_DEVICES=4 python scripts/finetune_transect.py

    # Resume from checkpoint:
    RESUME_CHECKPOINT=./checkpoints/transect/last.ckpt python scripts/finetune_transect.py

Environment variables
---------------------
    DATA_DIR              Path(s) to transect .npz files (comma-separated)
    PRETRAINED_WEIGHTS    Path to mp_rank_00_model_states.pt (or .ckpt)
    PRETRAINED_CONFIG     Path to the SatMAE YAML config
    TRAIN_BATCH_SIZE      Batch size per GPU (default: 4)
    TRAIN_NUM_DEVICES     Number of GPUs (default: 1)
    TRAIN_STRATEGY        Lightning strategy (default: auto)
    RESUME_CHECKPOINT     Resume training from this .ckpt
    CHECKPOINT_DIR        Where to save checkpoints (default: ./checkpoints/transect)
"""

from __future__ import annotations

import logging
import os
import sys
import time

import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import TensorBoardLogger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from satvision_pix4d.datasets.transect_dataset import TransectDataModule
from satvision_pix4d.models.transect_model import (
    SatMAETransectModel,
    TransectLightningModule,
)
from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.configs.config import _C, _update_config_from_file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOG = logging.getLogger(__name__)


# ─── Slurm-friendly progress callback ────────────────────────────────────────

class SlurmProgressCallback(L.Callback):
    """Prints plain-text progress lines that show up in Slurm .log files."""

    def __init__(self, log_every_n_batches: int = 50):
        super().__init__()
        self.log_every = log_every_n_batches
        self.epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()
        try:
            total = len(trainer.train_dataloader)
        except (TypeError, AttributeError):
            total = "?"
        print(
            f"\n{'='*60}\n"
            f"  Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"| {total} batches\n"
            f"{'='*60}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.log_every == 0:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            loss_str = f"{loss.item():.4f}" if isinstance(loss, torch.Tensor) else "N/A"
            elapsed = time.time() - self.epoch_start
            try:
                total = len(trainer.train_dataloader)
            except (TypeError, AttributeError):
                total = "?"
            print(
                f"  [Batch {batch_idx + 1}/{total}] "
                f"train_loss={loss_str} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start
        loss = trainer.callback_metrics.get("train_loss")
        loss_str = f"{loss.item():.4f}" if isinstance(loss, torch.Tensor) else "N/A"
        print(
            f"  >>> Epoch {trainer.current_epoch + 1} done in {elapsed:.1f}s "
            f"| train_loss={loss_str}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        val_loss = m.get("val_loss")
        val_loss_str = f"{val_loss.item():.4f}" if isinstance(val_loss, torch.Tensor) else "N/A"
        print(f"  >>> Validation: val_loss={val_loss_str}", flush=True)


# ─── Pretrained weight loading ───────────────────────────────────────────────

def load_pretrained_mae(config, weights_path: str) -> torch.nn.Module:
    """Build a SatMAE model and load pretrained weights.

    Handles both DeepSpeed (`mp_rank_*.pt`) and Lightning (`.ckpt`)
    checkpoint formats.  When the config specifies an asymmetric
    ``PATCH_SIZE`` (e.g. ``(16, 1)`` for 1D transects), the pretrained
    patch-embedding kernel is reshaped by averaging over the collapsed
    spatial dimension.

    Returns
    -------
    mae : MaskedAutoencoderViT
        The encoder model with pretrained weights loaded.
    """
    mae = build_satmae_model(config)

    if weights_path.endswith(".pt") or "mp_rank" in weights_path:
        LOG.info("Loading DeepSpeed checkpoint: %s", weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("module", checkpoint)
        # Strip 'model.' prefix from DeepSpeed-wrapped keys
        cleaned = {
            (k.replace("model.", "", 1) if k.startswith("model.") else k): v
            for k, v in state_dict.items()
        }

        # ── Phase 2: Reshape weights for asymmetric patch sizes ───────
        model_state = mae.state_dict()
        for key in list(cleaned.keys()):
            if key not in model_state:
                continue
            if cleaned[key].shape != model_state[key].shape:
                if key == "patch_embed.proj.weight":
                    # Pretrained: (E, C, Ph, Pw_old) → New: (E, C, Ph, Pw_new)
                    # Average over the width dimension to collapse 16→1
                    old_w = cleaned[key]
                    new_pw = model_state[key].shape[-1]
                    if old_w.shape[-1] != new_pw:
                        LOG.info(
                            "Reshaping %s: %s → %s (averaging width dim)",
                            key, list(old_w.shape), list(model_state[key].shape),
                        )
                        cleaned[key] = old_w.mean(dim=-1, keepdim=True)
                        if new_pw != 1:
                            # For patch widths other than 1, use adaptive avg pool
                            cleaned[key] = cleaned[key].expand_as(model_state[key])
                    new_ph = model_state[key].shape[-2]
                    if old_w.shape[-2] != new_ph:
                        LOG.info(
                            "Reshaping %s height: %d → %d (averaging height dim)",
                            key, old_w.shape[-2], new_ph,
                        )
                        cleaned[key] = cleaned[key].mean(dim=-2, keepdim=True)
                else:
                    # Drop mismatched keys (e.g. decoder_pred — not needed for finetuning)
                    LOG.warning(
                        "Dropping %s: shape %s != %s",
                        key, list(cleaned[key].shape), list(model_state[key].shape),
                    )
                    del cleaned[key]

        missing, unexpected = mae.load_state_dict(cleaned, strict=False)
        LOG.info(
            "Loaded weights: %d missing, %d unexpected keys",
            len(missing), len(unexpected),
        )
        if missing:
            LOG.info("Missing keys (first 10): %s", missing[:10])
        if unexpected:
            LOG.info("Unexpected keys (first 10): %s", unexpected[:10])
    else:
        LOG.info("Loading Lightning checkpoint: %s", weights_path)
        from satvision_pix4d.pipelines import PIPELINES
        pipeline_cls = PIPELINES[config.PIPELINE]
        ptl = pipeline_cls.load_from_checkpoint(weights_path, config=config)
        mae = ptl.model

    return mae


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration from environment ────────────────────────────────
    data_dirs = os.environ.get(
        "DATA_DIR",
        "/explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/chips/allChips",
    ).split(",")

    pretrained_weights = os.environ.get(
        "PRETRAINED_WEIGHTS",
        "/explore/nobackup/projects/ilab/projects/SatVisionPix4D/pretraining/mp_rank_00_model_states.pt",
    )
    pretrained_config = os.environ.get(
        "PRETRAINED_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "tests", "configs", "test_satmae_dev_dgx.yaml"),
    )

    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", 4))
    num_devices = int(os.environ.get("TRAIN_NUM_DEVICES", 1))
    strategy = os.environ.get("TRAIN_STRATEGY", "auto")
    resume_ckpt = os.environ.get("RESUME_CHECKPOINT", None)
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "./checkpoints/transect")
    num_workers = int(os.environ.get("NUM_WORKERS", 4))

    # ── Hyperparameters ───────────────────────────────────────────────
    MAX_EPOCHS = 100
    DECODER_LR = 1e-4
    ENCODER_LR = 1e-5
    DICE_WEIGHT = 0.5
    FREEZE_ENCODER = True
    UNFREEZE_AT_EPOCH = 5
    NUM_CLASSES = 9          # 0=clear + 8 cloud types
    NUM_BINS = 40
    TARGET_LEN = 512
    LABEL_KEY = "cloud_class"
    NORMALIZATION = "global"
    LOG_EVERY_N_BATCHES = 50
    SAVE_EVERY_N_EPOCHS = 5

    # ── 1D Transect geometry ──────────────────────────────────────────
    TRANSECT_IMG_SIZE = (512, 1)     # 512 footprints × 1 pixel wide
    TRANSECT_PATCH_SIZE = (16, 1)    # 16-footprint patches × 1 pixel

    L.seed_everything(42)

    # ── Load SatMAE config ────────────────────────────────────────────
    config = _C.clone()
    if os.path.isfile(pretrained_config):
        _update_config_from_file(config, pretrained_config)
        LOG.info("Loaded config from %s", pretrained_config)
    else:
        LOG.warning("Config file %s not found, using defaults", pretrained_config)

    # Override for 1D transect geometry
    config.defrost()
    config.MODEL.PRETRAINED = pretrained_weights
    config.DATA.IMG_SIZE = TRANSECT_IMG_SIZE
    config.MODEL.MAE_VIT.PATCH_SIZE = TRANSECT_PATCH_SIZE
    config.freeze()

    # ── Print configuration ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Transect Finetuning Configuration")
    print("=" * 60)
    print(f"  Data dirs:        {data_dirs}")
    print(f"  Pretrained:       {pretrained_weights}")
    print(f"  Config:           {pretrained_config}")
    print(f"  IMG_SIZE:         {config.DATA.IMG_SIZE}")
    print(f"  PATCH_SIZE:       {config.MODEL.MAE_VIT.PATCH_SIZE}")
    print(f"  Batch size:       {batch_size}")
    print(f"  Devices:          {num_devices} ({strategy})")
    print(f"  Epochs:           {MAX_EPOCHS}")
    print(f"  LR (encoder):     {ENCODER_LR}")
    print(f"  LR (decoder):     {DECODER_LR}")
    print(f"  Freeze encoder:   {FREEZE_ENCODER} (unfreeze at epoch {UNFREEZE_AT_EPOCH})")
    print(f"  Num classes:      {NUM_CLASSES}")
    print(f"  Label key:        {LABEL_KEY}")
    print(f"  Normalization:    {NORMALIZATION}")
    print(f"  Checkpoint dir:   {checkpoint_dir}")
    print("=" * 60 + "\n")

    # ── Data module ───────────────────────────────────────────────────
    datamodule = TransectDataModule(
        data_dir=data_dirs,
        label_key=LABEL_KEY,
        normalization=NORMALIZATION,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Build model ───────────────────────────────────────────────────
    if os.path.isfile(pretrained_weights):
        LOG.info("Loading pretrained SatMAE encoder...")
        mae = load_pretrained_mae(config, pretrained_weights)
    else:
        LOG.warning(
            "Pretrained weights not found at %s — training from scratch!",
            pretrained_weights,
        )
        mae = build_satmae_model(config)

    # Disable strict_img_size for dynamic input shapes
    pe = mae.patch_embed
    if hasattr(pe, "strict_img_size"):
        pe.strict_img_size = False
    if hasattr(pe, "img_size"):
        pe.img_size = None
    if hasattr(pe, "dynamic_img_pad"):
        pe.dynamic_img_pad = True

    # Wrap encoder + decoder
    backbone = SatMAETransectModel(
        mae_model=mae,
        num_classes=NUM_CLASSES,
        num_bins=NUM_BINS,
        target_len=TARGET_LEN,
        patch_size=config.MODEL.MAE_VIT.PATCH_SIZE,
        freeze_encoder=FREEZE_ENCODER,
        temporal_pool="mean",
    )

    # Lightning training module
    model = TransectLightningModule(
        backbone=backbone,
        encoder_lr=ENCODER_LR,
        decoder_lr=DECODER_LR,
        dice_weight=DICE_WEIGHT,
        unfreeze_encoder_at_epoch=UNFREEZE_AT_EPOCH,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable\n")

    # ── Callbacks ─────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=-1,
        every_n_epochs=SAVE_EVERY_N_EPOCHS,
        filename="epoch{epoch:03d}-loss{val_loss:.4f}",
    )
    best_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-{epoch:03d}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_cb = SlurmProgressCallback(log_every_n_batches=LOG_EVERY_N_BATCHES)

    # ── Loggers ───────────────────────────────────────────────────────
    csv_logger = CSVLogger(checkpoint_dir, name="logs")
    loggers = [csv_logger]
    if TENSORBOARD_AVAILABLE:
        tb_logger = TensorBoardLogger(checkpoint_dir, name="tb_logs")
        loggers.append(tb_logger)

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        strategy=strategy,
        callbacks=[checkpoint_cb, best_cb, lr_monitor, progress_cb],
        logger=loggers,
        default_root_dir=checkpoint_dir,
        enable_progress_bar=False,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # ── Train ─────────────────────────────────────────────────────────
    LOG.info("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=resume_ckpt,
    )

    # ── Test ──────────────────────────────────────────────────────────
    if trainer.is_global_zero:
        LOG.info("Running test evaluation on best checkpoint...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
