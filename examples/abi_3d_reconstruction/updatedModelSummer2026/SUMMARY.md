# 2026 Cloud Transect Reconstruction Pipeline Summary

This document summarizes the complete overhaul of the `cloud_transect_reconstruction` pipeline to support the new 2026 satellite chip format.

## Overview of Changes

The original pipeline processed 14-channel, 2D chips (128x128). The pipeline has been completely rewritten to support the new 2026 chips which are **4D Tensors: `(7 timesteps, 512 Height, 512 Width, 16 Channels)`**.

### 1. Simplified Normalization (`transforms.py`)
*   **Action**: Stripped out the complex, physics-based radiometric calibration (ESUN, Planck constants) which was hardcoded for only 14 channels.
*   **Result**: Replaced with a fast, deep-learning standard `SimpleMinMaxScale`.
*   **Note**: We built a high-performance multiprocessing Jupyter notebook (`calculate_min_max.ipynb`) to scan all 21,638 chips to find the true global minimums and maximums across the dataset to plug into this transform.

### 2. Dataloader & On-The-Fly Downsampling (`abidatamodule.py`)
*   **Data Aggregation**: Automatically globs all 21k `.npz` chips recursively from 12 distinct month directories and handles the 80/10/10 Train/Val/Test splitting dynamically.
*   **Key Updates**: Extracts the new 16-channel `ABI/chip` matrix and the 9-class `CloudSat/cloud_class` mask. 
*   **Dynamic Transect Downsampling Algorithm**: 
    *   Instead of rewriting terabytes of `.npz` files, the dataloader performs in-memory downsampling.
    *   It identifies overlapping CloudSat footprints (caused by ABI pixel distortion) using the `CloudSat/abi_row` and `CloudSat/abi_column` arrays.
    *   It logically drops these exact (row, col) duplicates, and if necessary, drops an evenly spaced subset of remaining footprints to bring the transect length to exactly **`478`**.

### 3. Model Architecture (`models.py`)
*   **Action**: Upgraded the standard 2D U-Net to a baseline **3D U-Net (`UNET3D`)**.
*   **Architecture**: Uses `Conv3d` and `MaxPool3d` to ingest the temporal (`T=7`) dimension. A final 3D convolution collapses the time dimension before passing the features to a 2D prediction head, outputting a spatial mask of `(478, 40)`.
*   **Multi-Class Segmentation**: 
    *   Switched from binary predictions to 9-class segmentation (`num_classes=9`).
    *   Utilizes `nn.CrossEntropyLoss(ignore_index=-1)` and `JaccardIndex(task="multiclass", ignore_index=-1)` to accurately penalize the model while ignoring any "-1" (missing retrieval) CloudSat labels.

### 4. Training & Execution (`3dcloudpipeline.py`)
*   A clean PyTorch Lightning training script that stitches the updated datamodule and `UNET3D` model together.
*   **Configurable via Environment Variables**: Batch size, device count, and DDP strategy are read from environment variables (`TRAIN_BATCH_SIZE`, `TRAIN_NUM_DEVICES`, `TRAIN_STRATEGY`) set by the submit scripts. This allows the same Python file to work for both V100 multi-GPU and H100 single-GPU configurations.
*   **OOM Prevention**: On V100s (32GB VRAM), `BATCH_SIZE=1` with 4-GPU DDP gives an effective batch size of 4. On H100s (96GB VRAM), `BATCH_SIZE=8` runs directly on a single GPU.
*   **Logging**: Uses both `CSVLogger` and `TensorBoardLogger` to log epoch metrics into `./checkpoints/unet3d_baseline/`. A custom `SlurmProgressCallback` prints plain-text batch progress every 100 batches for Slurm log file readability.
*   **Metric Sync**: All `self.log()` calls use `sync_dist=True` to properly average metrics across GPUs during DDP training.
*   **Imports**: Upgraded all `import lightning` syntax to `import pytorch_lightning` to support the specific older Lightning package installed in the ADAPT environment.

### 5. Slurm Submission (Two Configurations)
Two submit scripts are provided for the NASA ADAPT cluster, both using the project's official Singularity container:

*   **`submit_training_v100.sh`** — Multi-GPU DDP on V100s:
    *   Runs on the `compute` partition (default) with 4× V100 32GB GPUs.
    *   Uses `docker://nasanccs/satvision-pix4d:v100` container (x86/amd64 only).
    *   Launches via `srun --cpu-bind=none` for DDP multi-task execution.
    *   `BATCH_SIZE=1` per GPU, 4 GPUs = effective batch size of 4.
    *   Estimated epoch time: ~2.75 hours.

*   **`submit_training_h100.sh`** — Single H100 on Grace Hopper:
    *   Runs on the `grace` partition with 1× H100 96GB GPU.
    *   Uses `docker://nasanccs/satvision-pix4d:latest` container (multi-arch: amd64 + arm64).
    *   The `grace` partition nodes (`gh[001-062]`) are NVIDIA Grace Hopper (ARM CPU + H100 GPU), so the ARM-compatible `:latest` container is required.
    *   `BATCH_SIZE=8`, single GPU, no DDP needed.
    *   Estimated epoch time: ~47 minutes (~3.5× faster than 4× V100).
    *   Patches `huggingface-hub` at runtime alongside numpy to fix a dependency mismatch in the `:latest` container.
