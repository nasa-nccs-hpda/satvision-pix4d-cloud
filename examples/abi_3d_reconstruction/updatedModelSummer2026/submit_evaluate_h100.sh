#!/bin/bash
#SBATCH --job-name=eval-unet3d
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10240
#SBATCH --output=evaluate_%j.log
#SBATCH --error=evaluate_%j.err
#SBATCH --export=ALL

echo "Starting evaluation job on $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

export TRAIN_BATCH_SIZE=8
export TRAIN_NUM_DEVICES=1
export TRAIN_STRATEGY=auto

CONTAINER="/lscratch/$USER/container/satvision-pix4d-latest"
WORK_DIR="/home/aliewehr/satvision-pix4d/examples/abi_3d_reconstruction/updatedModelSummer2026"
REPO_ROOT="/home/aliewehr/satvision-pix4d"

module load singularity

# Build the container automatically if it doesn't exist yet on this node
if [ -d "$CONTAINER" ]; then
    echo "Container found at $CONTAINER"
else
    echo "Container not found at $CONTAINER — building it now..."
    mkdir -p "$(dirname "$CONTAINER")"
    singularity build --sandbox "$CONTAINER" docker://nasanccs/satvision-pix4d:latest
    echo "Container build finished at $(date)"
fi

# Fix numpy and huggingface-hub in the container
singularity exec --writable "$CONTAINER" pip install "numpy<2" "huggingface-hub>=1.5.0,<2"

singularity exec \
  --nv \
  --env PYTHONPATH="$REPO_ROOT",PYTHONNOUSERSITE=1,TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE,TRAIN_NUM_DEVICES=$TRAIN_NUM_DEVICES,TRAIN_STRATEGY=$TRAIN_STRATEGY \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css \
  --pwd "$WORK_DIR" \
  "$CONTAINER" \
  python3 evaluate.py

echo "Evaluation completed at $(date)"
