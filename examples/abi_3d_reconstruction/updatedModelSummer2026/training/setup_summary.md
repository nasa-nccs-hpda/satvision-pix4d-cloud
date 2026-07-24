# 3D U-Net Cloud Reconstruction: Training Setup & Fixes Summary

This document summarizes the recent updates, bug fixes, and the final state of the training pipeline for the updated 2026 3D U-Net model.

## 1. Container & Environment Setup
The training environment was transitioned from a local conda environment (`ilab-pytorch`) to the project's official **Singularity container**, which resolves various dependency and architecture mismatch issues.

*   **Container Image Used**: `docker://nasanccs/satvision-pix4d:v100`
    *   *Why `:v100` instead of `:latest`?* The cluster nodes assigned for this job have Tesla V100 GPUs (Compute Capability 7.0). The `:latest` container uses a PyTorch version that dropped support for V100s. The `:v100` tag maintains this legacy support.
*   **Automatic Build Process**: The Slurm submission script (`submit_training.sh`) is now designed to automatically build the container sandbox in `/lscratch/$USER/container/satvision-pix4d-v100-clean` if it doesn't already exist on the compute node.
*   **Numpy Binary Fix**: The container is automatically patched via `singularity exec --writable pip install "numpy<2"` at runtime to resolve a binary incompatibility between `scikit-learn` and `numpy==2.x`. (The resulting pip conflict warnings during installation are harmless and can be ignored).
*   **Home Directory Isolation**: The `--env PYTHONNOUSERSITE=1` flag was added to the `singularity exec` call. This prevents Python from accidentally loading conflicting packages from the user's local `~/.local/` directory instead of the container's isolated environment.

## 2. Pipeline Logging & TensorBoard
Previously, training progress was completely silent in the Slurm `.log` files due to how PyTorch Lightning's default `tqdm` progress bar handles carriage returns.

*   **Slurm-Friendly Progress**: A custom `SlurmProgressCallback` was added to `3dcloudpipeline.py`. It prints clean, newline-separated updates every 100 batches, showing the current batch, elapsed time, and `train_loss`.
*   **TensorBoard Integration**: `TensorBoardLogger` was added alongside the existing `CSVLogger`. Metrics are now logged to `./checkpoints/unet3d_baseline/` and can be visualized using `tensorboard --logdir ./checkpoints/unet3d_baseline`.

## 3. Data Transformation Fixes (`NaN` Loss)
During the initial successful execution, the model immediately reported `train_loss=nan`.

*   **The Cause**: The raw ABI satellite chips occasionally contain missing retrieval data represented as `NaN`. When passed into a 3D convolution, a single `NaN` value propagates and causes the gradients (and the loss) to instantly explode to `NaN`.
*   **The Fix**: A step was added to `transforms.py` to convert all `NaN` values to `0.0` *after* the min-max scaling step: `img = np.nan_to_num(img, nan=0.0)`.

## 4. Multi-GPU & H100 Migration

### 4a. Container Architecture Investigation
We investigated the Docker Hub manifests for the `nasanccs/satvision-pix4d` container to determine GPU/architecture compatibility:

| Tag | x86 (`amd64`) | ARM (`arm64`) | V100 Support | H100 Support |
|-----|:---:|:---:|:---:|:---:|
| `:v100` | ✅ | ❌ | ✅ | ❌ |
| `:latest` | ✅ | ✅ | ❌ | ✅ |

### 4b. ADAPT Cluster GPU Inventory

| Nodes | GPUs | VRAM | Partition | CPU Architecture |
|-------|------|------|-----------|-----------------|
| `gpu[001-022]` | 4× NVIDIA V100 | 32GB | `compute` (default) | x86 |
| `gh[001-062]` | 1× NVIDIA H100 | 96GB | `grace` | ARM (Grace Hopper) |
| `gpu100` | 8× NVIDIA A100 | 40GB | `dgx` | x86 |

**Key finding:** The `grace` partition uses NVIDIA Grace Hopper nodes (ARM CPUs + H100 GPUs). The `:v100` container is x86-only and **will not run** on these nodes. The `:latest` container has an `arm64` build and is required for the `grace` partition.

### 4c. Two Submit Script Configurations
The training pipeline now has two submit scripts. `3dcloudpipeline.py` reads configuration from environment variables (`TRAIN_BATCH_SIZE`, `TRAIN_NUM_DEVICES`, `TRAIN_STRATEGY`) set by the submit scripts, so the same Python code works for both.

**`submit_training_v100.sh`** — Multi-GPU DDP on V100s:
*   Partition: `compute` (default), 4× V100 32GB
*   `BATCH_SIZE=1`, `DEVICES=4`, `STRATEGY=ddp`
*   Uses `srun --cpu-bind=none` for multi-task DDP launch
*   Container: `docker://nasanccs/satvision-pix4d:v100`
*   Effective batch size: 4 (1 per GPU × 4 GPUs)

**`submit_training_h100.sh`** — Single H100 on Grace Hopper:
*   Partition: `grace`, 1× H100 96GB
*   `BATCH_SIZE=4`, `DEVICES=1`, `STRATEGY=auto`
*   Direct `singularity exec` (no srun needed for single GPU)
*   Container: `docker://nasanccs/satvision-pix4d:latest` (arm64)
*   Effective batch size: 4

### 4d. DDP Lessons Learned (V100 Multi-GPU)
Several issues were encountered and resolved when enabling multi-GPU DDP on V100s:
1.  **`--ntasks-per-node` must match `devices`**: Lightning DDP on Slurm expects one Slurm task per GPU. Mismatched values cause a `ValueError`.
2.  **`srun` is required for DDP**: Without `srun`, only rank 0 starts and hangs waiting for other ranks to connect.
3.  **`--cpu-bind=none` on ADAPT**: The cluster's default CPU affinity settings conflict with multi-task layouts. Adding `--cpu-bind=none` to `srun` resolves the error.
4.  **`sync_dist=True` in `self.log()`**: Required in `models.py` so that loss/IoU metrics are properly averaged across all GPUs during DDP training.

## 5. Current Execution State
To run the training pipeline, submit the appropriate script:

**For V100 (4-GPU DDP):**
```bash
cd /home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026/training
sbatch submit_training_v100.sh
```

**For H100 (single GPU):**
```bash
cd /home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026/training
sbatch submit_training_h100.sh
```

**What to expect:**
1.  If the node doesn't have the container built yet, it will spend a few minutes building it in `/lscratch/`.
2.  It will patch `numpy`, throwing a few harmless pip dependency warnings.
3.  Training will commence, and you will see line-by-line batch progress in your `training_<id>.log` file.
