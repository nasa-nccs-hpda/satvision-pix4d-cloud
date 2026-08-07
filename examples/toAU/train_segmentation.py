#!/usr/bin/env python
# coding: utf-8

# # Cloud Height Binary Segmentation with SatVision-Pix4D
# 
# This notebook implements binary cloud segmentation using:
# - **Encoder**: Pre-trained SatMAE from SatVision-Pix4D
# - **Decoder**: Custom U-Net decoder
# - **Task**: Binary classification (cloud vs. no-cloud)
# - **Metrics**: Binary Accuracy, F1-Score, IoU

# In[1]:


# !pip install pytorch-lightning lightning segmentation_models_pytorch yacs deepspeed xarray torchmetrics


# ## 1. Setup and Imports

# In[1]:


import os
import sys

sys.path.append('/home/al8425b-hpc/freshStart/satvision-pix4d')

import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import pandas as pd

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar
)
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.pipelines import PIPELINES, get_available_pipelines

print("✅ All imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pl.__version__}")


# ## 2. Configuration

# ## 3. Load Pre-trained SatMAE Model

# In[2]:


# Data and paths
# Update these paths to where the ABI dataset and configs are stored on Lovelace
DATA_DIRECTORY = "/home/al8425b-hpc/testLoad/checkModel/testData"
WORK_DIR = "/home/al8425b-hpc/cloud_height_satmae_workspace"
SATVISION_PIX4D_CONFIG = "/home/al8425b-hpc/freshStart/satvision-pix4d/tests/configs/test_satmae_dev.yaml"

# Point directly to the verified Hugging Face cache snapshot
SATVISION_PIX4D_MODEL = "/home/al8425b-hpc/.cache/huggingface/hub/models--nasa-cisto-data-science-group--satvision-pix4d-base/snapshots/b229dfef33f1b48d97fa754e166424e0a118ab41/mp_rank_00_model_states.pt"

# Training hyperparameters
BATCH_SIZE = 256
NUM_WORKERS = 10
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
ENCODER_LR = 1e-5  # Smaller LR for pre-trained encoder

# Model architecture (Updated for 512x512 spatial input and 512x40 output mask)
TARGET_HEIGHT = 512 
TARGET_WIDTH = 40
IN_CHANNELS = 16
NUM_CLASSES = 1  
PATCH_SIZE = 16

# Training strategy
FREEZE_ENCODER = True
UNFREEZE_ENCODER_AT_EPOCH = 2

# Visualization
NUM_SAMPLES_TO_TEST = 5

# Set random seed
pl.seed_everything(42)

print("\n📋 Configuration:")
print(f"  Data directory: {DATA_DIRECTORY}")
print(f"  Work directory: {WORK_DIR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rates: Encoder={ENCODER_LR}, Decoder={LEARNING_RATE}")
print(f"  Target size: ({TARGET_HEIGHT}, {TARGET_WIDTH})")
print(f"  Task: Binary Cloud Segmentation with SatMAE Encoder")


# In[3]:


# Load SatVision-Pix4D configuration
config = _C.clone()
_update_config_from_file(config, SATVISION_PIX4D_CONFIG)
print("✅ Loaded configuration file")

# Update config with checkpoint path
config.defrost()
config.MODEL.PRETRAINED = SATVISION_PIX4D_MODEL
config.OUTPUT = '.'
config.freeze()
print("✅ Updated configuration file")

# Get pipeline
available_pipelines = get_available_pipelines()
print(f"\nAvailable pipelines: {available_pipelines}")

pipeline = PIPELINES[config.PIPELINE]
print(f"Using pipeline: {pipeline}")

# Load pre-trained model
ptlPipeline = pipeline(config)
print(f"\n🔄 Loading checkpoint from: {config.MODEL.PRETRAINED}")
sat_pretrain = ptlPipeline.load_checkpoint(config.MODEL.PRETRAINED, config)

# Configure patch embedding for dynamic image sizes
pe = sat_pretrain.model.patch_embed
if hasattr(pe, "strict_img_size"):
    pe.strict_img_size = False
if hasattr(pe, "img_size"):
    pe.img_size = None
if hasattr(pe, "dynamic_img_pad"):
    pe.dynamic_img_pad = True

print("\n✅ Successfully loaded SatVision pre-trained model")
print(f"   Pipeline type: {type(sat_pretrain)}")
print(f"   Underlying MAE: {type(sat_pretrain.model)}")


# ## 4. Dataset Definition

# In[4]:


class CloudSatBinaryDataset(Dataset):
    """Dataset for CloudSat binary cloud mask segmentation."""
    
    def __init__(self, file_paths, target_size=(91, 40)):
        self.file_paths = file_paths
        self.target_height, self.target_width = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            with np.load(self.file_paths[idx], allow_pickle=True) as data:
                # Load ABI chip and extract the middle timestep (index 3)
                chip_g16_raw = data['chip_g16'].astype(np.float32)
                chip_g16 = chip_g16_raw[:, :, :, 3]

                # Normalize each channel independently
                for i in range(chip_g16.shape[2]):
                    channel = chip_g16[:, :, i]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val > min_val:
                        chip_g16[:, :, i] = (channel - min_val) / (max_val - min_val)
                    else:
                        chip_g16[:, :, i] = 0.0
                
                # Transpose to channel-first: (H, W, C) -> (C, H, W)
                chip_g16 = np.transpose(chip_g16, (2, 0, 1))

                # Load companion ABI chip and extract the middle timestep (index 3)
                chip_g17_raw = data['chip_g17'].astype(np.float32)
                chip_g17 = chip_g17_raw[:, :, :, 3]

                for i in range(chip_g17.shape[2]):
                    channel = chip_g17[:, :, i]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val > min_val:
                        chip_g17[:, :, i] = (channel - min_val) / (max_val - min_val)
                    else:
                        chip_g17[:, :, i] = 0.0 
                        
                # Transpose to channel-first: (H, W, C) -> (C, H, W)    
                chip_g17 = np.transpose(chip_g17, (2, 0, 1))
                
                # Load BINARY cloud mask
                cloud_mask_raw = data['data'].item()['Cloud_mask_binary'].astype(np.float32)
                
                # Pad mask to target size (512, 40)
                pad_height_total = self.target_height - cloud_mask_raw.shape[0]
                pad_top = pad_height_total // 2
                pad_bottom = pad_height_total - pad_top
                
                pad_width_total = self.target_width - cloud_mask_raw.shape[1]
                pad_left = pad_width_total // 2
                pad_right = pad_width_total - pad_left
                
                cloud_mask_padded = np.pad(
                    cloud_mask_raw,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    'constant',
                    constant_values=0
                )
               
                return {
                    "chip_g16": torch.from_numpy(chip_g16),
                    "chip_g17": torch.from_numpy(chip_g17),
                    "mask": torch.from_numpy(cloud_mask_padded),
                    "path": self.file_paths[idx]
                }
        except Exception as e:
            print(f"⚠️  Error loading {self.file_paths[idx]}: {e}")
            return None


# ## 5. Data Module

# In[5]:


class CloudSatDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for CloudSat binary segmentation."""
    
    def __init__(self, data_dir, batch_size=16, num_workers=0, train_val_test_split=(0.8, 0.1, 0.1)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.npz')))
        
        print(f"\n📂 Found {len(self.file_paths)} data files in {data_dir}")

    def setup(self, stage=None):
        n_total = len(self.file_paths)
        n_train = int(n_total * self.train_val_test_split[0])
        n_val = int(n_total * self.train_val_test_split[1])
       
        self.train_files = self.file_paths[:n_train]
        self.val_files = self.file_paths[n_train : n_train + n_val]
        self.test_files = self.file_paths[n_train + n_val :]

        self.train_dataset = CloudSatBinaryDataset(self.train_files)
        self.val_dataset = CloudSatBinaryDataset(self.val_files)
        self.test_dataset = CloudSatBinaryDataset(self.test_files)
        
        print(f"\n📊 Data split:")
        print(f"  Train: {len(self.train_files)} files")
        print(f"  Val:   {len(self.val_files)} files")
        print(f"  Test:  {len(self.test_files)} files")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Filter out None samples and collate."""
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)


# ## 6. Model Architecture

# In[6]:


class DoubleConv(nn.Module):
    """Double convolution block for U-Net decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch, out_ch)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class UNetDecoderBinary(nn.Module):
    """U-Net decoder for binary segmentation.
    
    Input: (B, 1024, 8, 8) from SatMAE encoder
    Output: (B, 1, 128, 128) for binary segmentation
    """
    def __init__(self, in_dim=1024, num_classes=1, base=256):
        super().__init__()
        # Compress encoder features
        self.stem = DoubleConv(in_dim, base)      # 1024 -> 256 at 8x8
        
        # Upsampling path
        self.up1 = UpBlock(base, base//2)         # 8 -> 16
        self.up2 = UpBlock(base//2, base//4)      # 16 -> 32
        self.up3 = UpBlock(base//4, base//8)      # 32 -> 64
        self.up4 = UpBlock(base//8, base//16)     # 64 -> 128
        
        # Final classification head
        self.head = nn.Conv2d(base//16, num_classes, 1)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)


class SatMAEBinarySegmentation(nn.Module):
    """SatMAE encoder + U-Net decoder for binary cloud segmentation."""
    
    def __init__(
        self,
        ptl_pretrain_model,
        num_classes=1,
        patch_size=16,
        output_shape=(91, 40),
        freeze_encoder=True,
    ):
        super().__init__()
        self.ptl = ptl_pretrain_model
        self.mae = ptl_pretrain_model.model  # MaskedAutoencoderViT
        self.patch_size = patch_size
        self.embed_dim = 1024
        self.output_shape = output_shape
        
        # U-Net decoder (8x8 -> 128x128)
        self.decoder = UNetDecoderBinary(
            in_dim=self.embed_dim,
            num_classes=num_classes,
            base=256
        )
        
        # Final resize to target shape
        self.final_upsample = nn.Upsample(
            size=output_shape,
            mode="bilinear",
            align_corners=False
        )
        
        if freeze_encoder:
            self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder weights."""
        for p in self.mae.parameters():
            p.requires_grad = False
        print("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for p in self.mae.parameters():
            p.requires_grad = True
        print("🔓 Encoder unfrozen")
    
    def _encode_tokens(self, imgs_btchw, ts_bt3):
        """Extract encoder tokens."""
        if not hasattr(self.mae, "forward_encoder"):
            raise RuntimeError(
                f"Expected SatMAE to implement forward_encoder. Got type={type(self.mae)}"
            )
        out = self.mae.forward_encoder(imgs_btchw, ts_bt3, mask_ratio=0.0)
        latent = out[0] if isinstance(out, (tuple, list)) else out
        return latent
    
    def forward(self, chips_bchw):
        """
        Args:
            chips_bchw: (B, 16, 128, 128)
        
        Returns:
            logits: (B, 1, 91, 40) for binary segmentation
        """
        B, C, H, W = chips_bchw.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, f"Input H,W must be divisible by patch_size={P}"
        
        # Add time dimension for SatMAE: (B, 1, C, H, W)
        imgs = chips_bchw.unsqueeze(1)
        
        # Dummy timestamps (B, 1, 3)
        ts = torch.zeros((B, 1, 3), device=chips_bchw.device, dtype=torch.float32)
        
        # Get encoder tokens: (B, N, D)
        tokens = self._encode_tokens(imgs, ts)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected tokens (B,N,D), got {tokens.shape}")
        
        h = H // P
        w = W // P
        N_expected = 1 * h * w
        
        # Drop CLS token if present
        if tokens.shape[1] == N_expected + 1:
            tokens = tokens[:, 1:, :]
        
        if tokens.shape[1] != N_expected:
            raise RuntimeError(
                f"Token count mismatch: got {tokens.shape[1]}, expected {N_expected}"
            )
        
        # Reshape to spatial feature map: (B, N, D) -> (B, D, h, w)
        feat_map = tokens.view(B, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        
        # Decode to full resolution
        logits = self.decoder(feat_map)  # (B, 1, 128, 128)
        
        # Resize to target output shape
        logits = self.final_upsample(logits)  # (B, 1, 91, 40)
        
        return logits


# ## 7. Loss Functions

# In[7]:


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = torch.sum(probs * targets)
        cardinality = torch.sum(probs + targets)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score


# ## 8. Lightning Module

# In[8]:


class BinarySegmentationLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for binary cloud segmentation."""
    
    def __init__(
        self,
        backbone: nn.Module,
        encoder_lr=1e-5,
        decoder_lr=1e-4,
        dice_weight=0.5,
        unfreeze_encoder_at_epoch=2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        
        self.model = backbone
        
        # Binary losses
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        
        # Binary metrics
        threshold = 0.5
        self.train_acc = BinaryAccuracy(threshold=threshold)
        self.val_acc = BinaryAccuracy(threshold=threshold)
        self.test_acc = BinaryAccuracy(threshold=threshold)
        
        self.train_f1 = BinaryF1Score(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)
        self.test_f1 = BinaryF1Score(threshold=threshold)
        
        self.train_iou = BinaryJaccardIndex(threshold=threshold)
        self.val_iou = BinaryJaccardIndex(threshold=threshold)
        self.test_iou = BinaryJaccardIndex(threshold=threshold)
        
        # Training strategy
        self.unfreeze_encoder_at_epoch = unfreeze_encoder_at_epoch
        self._encoder_unfrozen = False
    
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx, stage: str):
        if batch is None:
            return None
        
        chip_g16 = batch["chip_g16"]  # [B, 16, H, W]
        chip_g17 = batch["chip_g17"]  # [B, 16, H, W]
        masks = batch["mask"]  # [B, H, W] binary

        logits_g16 = self.forward(chip_g16).squeeze(1)  # [B, H, W]
        logits_g17 = self.forward(chip_g17).squeeze(1)  # [B, H, W]
        
        # Compute losses
        ce_g16 = self.ce_loss(logits_g16, masks.float())
        ce_g17 = self.ce_loss(logits_g17, masks.float())
        dice_g16 = self.dice_loss(logits_g16, masks)
        dice_g17 = self.dice_loss(logits_g17, masks)

        loss_g16 = self.dice_weight * dice_g16 + (1 - self.dice_weight) * ce_g16
        loss_g17 = self.dice_weight * dice_g17 + (1 - self.dice_weight) * ce_g17
        
        # Consistency loss: same location/time should produce similar prediction
        prob_g16 = torch.sigmoid(logits_g16)
        prob_g17 = torch.sigmoid(logits_g17)

        consistency_loss = F.mse_loss(prob_g16, prob_g17)

        lambda_consistency = 0.1
        loss = 0.5 * (loss_g16 + loss_g17) + lambda_consistency * consistency_loss

        # Use averaged prediction for metrics
        preds = ((prob_g16 + prob_g17) / 2.0 > 0.5).float()

        bs = chip_g16.size(0)
        
        # Update metrics
        if stage == "train":
            self.train_acc.update(preds, masks)
            self.train_f1.update(preds, masks)
            self.train_iou.update(preds, masks)
        elif stage == "val":
            self.val_acc.update(preds, masks)
            self.val_f1.update(preds, masks)
            self.val_iou.update(preds, masks)
        elif stage == "test":
            self.test_acc.update(preds, masks)
            self.test_f1.update(preds, masks)
            self.test_iou.update(preds, masks)
        
        # Log losses
        self.log(f"{stage}_loss", loss, prog_bar=True,
                on_step=True if stage == "train" else False,
                on_epoch=True, batch_size=bs,
                sync_dist=(stage != "train"))
        self.log(f"{stage}_ce_loss", 0.5 * (ce_g16 + ce_g17), prog_bar=False,
                on_step=False, on_epoch=True, batch_size=bs,
                sync_dist=(stage != "train"))

        self.log(f"{stage}_dice_loss", 0.5 * (dice_g16 + dice_g17), prog_bar=False,
                on_step=False, on_epoch=True, batch_size=bs,
                sync_dist=(stage != "train"))

        self.log(f"{stage}_consistency_loss", consistency_loss, prog_bar=False,
                on_step=False, on_epoch=True, batch_size=bs,
                sync_dist=(stage != "train"))
        
        # Log metrics
        if stage == "train":
            self.log("train_acc", self.train_acc, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs)
            self.log("train_f1", self.train_f1, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs)
            self.log("train_iou", self.train_iou, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs)
        elif stage == "val":
            self.log("val_acc", self.val_acc, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("val_f1", self.val_f1, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("val_iou", self.val_iou, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        elif stage == "test":
            self.log("test_acc", self.test_acc, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("test_f1", self.test_f1, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
            self.log("test_iou", self.test_iou, prog_bar=True,
                    on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        
        return loss
    
    def on_train_epoch_start(self):
        """Unfreeze encoder after specified epoch."""
        if (
            (not self._encoder_unfrozen)
            and (self.current_epoch >= self.unfreeze_encoder_at_epoch)
            and hasattr(self.model, "unfreeze_encoder")
        ):
            self.model.unfreeze_encoder()
            self._encoder_unfrozen = True
            print(f"\n🔓 Unfroze encoder at epoch {self.current_epoch}")
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        """Configure separate learning rates for encoder and decoder."""
        if hasattr(self.model, "mae") and hasattr(self.model, "decoder"):
            enc_params = list(self.model.mae.parameters())
            dec_params = (
                list(self.model.decoder.parameters())
                + list(self.model.final_upsample.parameters())
            )
            
            optimizer = torch.optim.AdamW(
                [
                    {"params": enc_params, "lr": self.hparams.encoder_lr},
                    {"params": dec_params, "lr": self.hparams.decoder_lr},
                ],
                weight_decay=1e-4,
            )
            return optimizer
        
        # Fallback
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.decoder_lr,
            weight_decay=1e-4
        )


# ## 9. Visualization Functions

# In[15]:


def visualize_prediction(chip_tensor, true_mask_padded, pred_mask_padded, file_path, save_dir):
    """Visualize binary cloud segmentation results."""
    chip = chip_tensor[0].cpu().numpy()
    true_mask = true_mask_padded.cpu().numpy()
    pred_mask = pred_mask_padded.cpu().numpy()
    
    # Unpad to original size
    original_height, original_width = 91, 40
    ##91, 34
    pad_height_total = true_mask.shape[0] - original_height
    pad_top = pad_height_total // 2
    pad_width_total = true_mask.shape[1] - original_width
    pad_left = pad_width_total // 2
    
    true_mask_unpadded = true_mask[
        pad_top : pad_top + original_height,
        pad_left : pad_left + original_width
    ]
    pred_mask_unpadded = pred_mask[
        pad_top : pad_top + original_height,
        pad_left : pad_left + original_width
    ]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f"File: {os.path.basename(file_path)}", fontsize=16)
    
    # Input chip
    ax1.imshow(chip, cmap='gray')
    ax1.set_title("Input ABI Chip (Channel 1)", fontsize=14)
    ax1.axis('off')
    
    # Ground truth
    plot_binary_curtain(ax2, true_mask_unpadded, "Ground Truth (Binary)", fig)
    
    # Prediction
    plot_binary_curtain(ax3, pred_mask_unpadded, "Prediction (Binary)", fig)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_filename = os.path.basename(file_path).replace('.npz', '.png')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"💾 Saved: {save_path}")


def plot_binary_curtain(ax, mask, title, fig):
    """Plot binary cloud mask as curtain plot."""
    zz = mask.T
    
    x_axis_points = np.arange(zz.shape[1])
    y_axis_km = np.arange(0, zz.shape[0] * 0.5, 0.5)
    
    # Binary colormap
    mesh = ax.pcolormesh(
        x_axis_points, y_axis_km, zz,
        cmap='Blues', shading='nearest',
        vmin=0, vmax=1
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Point Along Track", fontsize=12)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_ylim(0, zz.shape[0] * 0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['No Cloud', 'Cloud'])


def calculate_binary_iou(pred_mask, true_mask):
    """Calculate IoU for binary segmentation."""
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    
    # Class 0 (No Cloud)
    intersection_0 = np.sum((pred_mask == 0) & (true_mask == 0))
    union_0 = np.sum((pred_mask == 0) | (true_mask == 0))
    iou_0 = intersection_0 / union_0 if union_0 > 0 else 0.0
    
    # Class 1 (Cloud)
    intersection_1 = np.sum((pred_mask == 1) & (true_mask == 1))
    union_1 = np.sum((pred_mask == 1) | (true_mask == 1))
    iou_1 = intersection_1 / union_1 if union_1 > 0 else 0.0
    
    # Mean IoU
    mIoU = (iou_0 + iou_1) / 2
    
    return mIoU, np.array([iou_0, iou_1])


# ## 10. Initialize Data and Model

# In[9]:


# Initialize data module
datamodule = CloudSatDataModule(
    data_dir=DATA_DIRECTORY,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# Create SatMAE + U-Net model
sat_binary_seg = SatMAEBinarySegmentation(
    ptl_pretrain_model=sat_pretrain,
    num_classes=NUM_CLASSES,
    patch_size=PATCH_SIZE,
    output_shape=(TARGET_HEIGHT, TARGET_WIDTH),
    freeze_encoder=FREEZE_ENCODER,
)

# Wrap in Lightning module
model = BinarySegmentationLightning(
    backbone=sat_binary_seg,
    encoder_lr=ENCODER_LR,
    decoder_lr=LEARNING_RATE,
    dice_weight=0.5,
    unfreeze_encoder_at_epoch=UNFREEZE_ENCODER_AT_EPOCH,
)

print("\n✅ Model initialized:")
print(f"   Encoder: Pre-trained SatMAE (frozen={FREEZE_ENCODER})")
print(f"   Decoder: Custom U-Net")
print(f"   Task: Binary cloud segmentation")


# ## 11. Training Setup

# In[10]:


# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=os.path.join(WORK_DIR, 'checkpoints'),
    filename='cloudsat-satmae-binary-p128-{epoch:02d}-{val_loss:.4f}',
    save_top_k=3,
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
)

# Trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback, TQDMProgressBar(refresh_rate=20)],
    max_epochs=MAX_EPOCHS,
    accelerator='auto',
    log_every_n_steps=10,
    default_root_dir=WORK_DIR,
    gradient_clip_val=1.0,
)

print("\n✅ Training setup complete")


# ## 12. Train Model

# In[11]:


print("\n🚀 Starting training...\n")
trainer.fit(model, datamodule)


# ## 13. Model Evaluation

# In[13]:


# Load best model
best_model_path = checkpoint_callback.best_model_path
print(f"\n📂 Loading best model: {os.path.basename(best_model_path)}")

trained_model = BinarySegmentationLightning.load_from_checkpoint(
    best_model_path,
    backbone=sat_binary_seg,
)
trained_model.eval()

# Get device
device = next(trained_model.parameters()).device
print(f"Model device: {device}")

# Setup test data
datamodule.setup('test')
test_loader = datamodule.test_dataloader()

all_preds_g16 = []
all_preds_g17 = []
all_true_masks = []

with torch.no_grad():
    for batch in tqdm.tqdm(test_loader, desc="Testing"):
        if batch is None:
            continue

        chip_g16 = batch["chip_g16"].float().to(device)
        chip_g17 = batch["chip_g17"].float().to(device)
        true_masks = batch["mask"].float().to(device)

        logits_g16 = trained_model(chip_g16)
        logits_g17 = trained_model(chip_g17)

        probs_g16 = torch.sigmoid(logits_g16.squeeze(1))
        probs_g17 = torch.sigmoid(logits_g17.squeeze(1))

        pred_g16 = (probs_g16 > 0.5).float()
        pred_g17 = (probs_g17 > 0.5).float()

        all_preds_g16.append(pred_g16.cpu().numpy())
        all_preds_g17.append(pred_g17.cpu().numpy())
        all_true_masks.append(true_masks.cpu().numpy())


# ## 14. Calculate Metrics

# In[17]:


if len(all_preds_g16) > 0 and len(all_preds_g17) > 0:
    all_preds_g16 = np.concatenate(all_preds_g16, axis=0)
    all_preds_g17 = np.concatenate(all_preds_g17, axis=0)
    all_true_masks = np.concatenate(all_true_masks, axis=0)

    mean_iou_g16, class_iou_g16 = calculate_binary_iou(all_preds_g16, all_true_masks)
    mean_iou_g17, class_iou_g17 = calculate_binary_iou(all_preds_g17, all_true_masks)

    # Optional: consistency between G16 and G17 predictions
    consistency_iou, consistency_class_iou = calculate_binary_iou(all_preds_g16, all_preds_g17)

    CLASS_NAMES = ["No Cloud", "Cloud"]

    iou_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "G16 IoU": class_iou_g16,
        "G17 IoU": class_iou_g17,
        "G16-G17 Consistency IoU": consistency_class_iou,
    })

    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE ON TEST SET")
    print("=" * 60)
    print(f"\n🎯 G16 Mean IoU: {mean_iou_g16:.4f}")
    print(f"🎯 G17 Mean IoU: {mean_iou_g17:.4f}")
    print(f"🔁 G16-G17 Consistency IoU: {consistency_iou:.4f}\n")

    print("Per-Class IoU:")
    print(iou_df.to_string(index=False, float_format="%.4f"))
    print("\n" + "=" * 60 + "\n")
else:
    print("⚠️  No predictions generated")


# ## 15. Visualize Sample Predictions

# In[18]:


# Create visualization loader
vis_loader = DataLoader(
    datamodule.test_dataset,
    batch_size=NUM_SAMPLES_TO_TEST,
    shuffle=True,
    collate_fn=datamodule._collate_fn
)

print(f"\n🎨 Visualizing {NUM_SAMPLES_TO_TEST} random test samples...\n")

vis_batch = next(iter(vis_loader))

if vis_batch:
    vis_chips_g16 = vis_batch["chip_g16"].float().to(device)
    vis_chips_g17 = vis_batch["chip_g17"].float().to(device)
    vis_true_masks = vis_batch["mask"].float().to(device)
    vis_file_paths = vis_batch["path"]

    with torch.no_grad():
        vis_logits_g16 = trained_model(vis_chips_g16)
        vis_logits_g17 = trained_model(vis_chips_g17)

        vis_probs_g16 = torch.sigmoid(vis_logits_g16.squeeze(1))
        vis_probs_g17 = torch.sigmoid(vis_logits_g17.squeeze(1))

        vis_pred_masks_g16 = (vis_probs_g16 > 0.5).float()
        vis_pred_masks_g17 = (vis_probs_g17 > 0.5).float()

    save_dir = os.path.join(WORK_DIR, "predictions_paired")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(vis_chips_g16)):
        base_path = vis_file_paths[i]

        visualize_prediction(
            vis_chips_g16[i],
            vis_true_masks[i],
            vis_pred_masks_g16[i],
            base_path.replace(".npz", "_G16.npz"),
            save_dir
        )

        visualize_prediction(
            vis_chips_g17[i],
            vis_true_masks[i],
            vis_pred_masks_g17[i],
            base_path.replace(".npz", "_G17.npz"),
            save_dir
        )

    print(f"\n✅ All visualizations saved to: {save_dir}")
else:
    print("⚠️  No visualization batch available")


# ## 16. Summary
# 
# ### 📈 Model Architecture:
# ```
# Input: (B, 16, 128, 128) ABI chips
#   ↓
# SatMAE Encoder: (B, 16, 128, 128) → (B, 1024, 8, 8)
#   ↓
# U-Net Decoder: (B, 1024, 8, 8) → (B, 1, 128, 128)
#   ↓
# Resize: (B, 1, 128, 128) → (B, 1, 96, 40)
#   ↓
# Output: Binary logits
# ```

# In[ ]:




