#!/bin/bash
#SBATCH --job-name=unet3d-summer-v100
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=10240
#SBATCH --output=training_%j.log
#SBATCH --error=training_%j.err
#SBATCH --export=ALL

echo "Starting training job on $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Config: V100 x4 DDP"

# --- Training Parameters (read by 3dcloudpipeline.py) ---
# V100 has 32GB VRAM — only fits batch_size=1 with the 3D U-Net.
# 4 GPUs with DDP gives an effective batch size of 4.
export TRAIN_BATCH_SIZE=1
export TRAIN_NUM_DEVICES=4
export TRAIN_STRATEGY=ddp

# --- Container Setup ---
# The :v100 container is x86-only (amd64) and supports V100 GPUs (Compute Capability 7.0).
# The :latest container dropped V100 support.
CONTAINER="/lscratch/$USER/container/satvision-pix4d-v100-clean"
WORK_DIR="/home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026"
REPO_ROOT="/home/aliewehr/satvision-pix4d"

module load singularity

# Build the container automatically if it doesn't exist yet
if [ -d "$CONTAINER" ]; then
    echo "Container found at $CONTAINER"
else
    echo "Container not found at $CONTAINER — building it now..."
    mkdir -p "$(dirname "$CONTAINER")"
    singularity build --sandbox "$CONTAINER" docker://nasanccs/satvision-pix4d:v100
    echo "Container build finished at $(date)"
fi

# Fix numpy 2.0 binary incompatibility with scikit-learn in the base container
singularity exec --writable "$CONTAINER" pip install "numpy<2"

# Run training inside the Singularity container
# srun is required for multi-GPU DDP to launch one process per task.
# --cpu-bind=none avoids CPU affinity conflicts on this cluster.
srun --cpu-bind=none singularity exec \
  --nv \
  --env PYTHONPATH="$REPO_ROOT",PYTHONNOUSERSITE=1,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,TRAIN_NUM_DEVICES=$TRAIN_NUM_DEVICES,TRAIN_STRATEGY=$TRAIN_STRATEGY \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css \
  --pwd "$WORK_DIR" \
  "$CONTAINER" \
  python3 3dcloudpipeline.py

echo "Training completed at $(date)"
