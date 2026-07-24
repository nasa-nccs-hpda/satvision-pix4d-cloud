#!/bin/bash
#SBATCH --job-name=unet3d-summer-h100
#SBATCH --time=72:00:00
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10240
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --export=ALL

echo "Starting training job on $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Config: H100 x1 (Grace Hopper)"

# --- Training Parameters (read by 3dcloudpipeline.py) ---
# H100 has 96GB VRAM — can fit batch_size=8 with the 3D U-Net.
# If OOM, fall back to batch_size=4.
# Single GPU, no DDP needed.
export TRAIN_BATCH_SIZE=8
export TRAIN_NUM_DEVICES=1
export TRAIN_STRATEGY=auto

# --- Container Setup ---
# IMPORTANT: Must use :latest (NOT :v100) for the grace partition.
# The gh[001-062] nodes are NVIDIA Grace Hopper (ARM CPU + H100 GPU).
# The :latest container is multi-arch (amd64 + arm64), while :v100 is x86-only.
# Verified via Docker Hub manifest: nasanccs/satvision-pix4d:latest has arm64 support.
CONTAINER="/lscratch/$USER/container/satvision-pix4d-latest"
WORK_DIR="/home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026"
REPO_ROOT="/home/aliewehr/satvision-pix4d"

module load singularity

# Build the container automatically if it doesn't exist yet
if [ -d "$CONTAINER" ]; then
    echo "Container found at $CONTAINER"
else
    echo "Container not found at $CONTAINER — building it now..."
    mkdir -p "$(dirname "$CONTAINER")"
    singularity build --sandbox "$CONTAINER" docker://nasanccs/satvision-pix4d:latest
    echo "Container build finished at $(date)"
fi

# Fix dependency issues in the :latest container:
#   - numpy<2: binary incompatibility between scikit-learn and numpy>=2
#   - huggingface-hub>=1.5.0: required by transformers 5.x (container ships 1.2.3)
singularity exec --writable "$CONTAINER" pip install "numpy<2" "huggingface-hub>=1.5.0,<2"

# Run training inside the Singularity container
#   --nv          : Enable NVIDIA GPU support (passes through CUDA drivers)
#   --env         : Set PYTHONPATH so imports from the repo work
#                   Also passes training config env vars into the container
#   -B            : Bind-mount the filesystem paths needed for data and code
#   --pwd         : Set working directory inside the container
singularity exec \
  --nv \
  --env PYTHONPATH="$REPO_ROOT",PYTHONNOUSERSITE=1,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,TRAIN_NUM_DEVICES=$TRAIN_NUM_DEVICES,TRAIN_STRATEGY=$TRAIN_STRATEGY \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css \
  --pwd "$WORK_DIR" \
  "$CONTAINER" \
  python3 3dcloudpipeline.py

echo "Training completed at $(date)"
