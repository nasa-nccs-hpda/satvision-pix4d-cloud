#!/bin/bash
#SBATCH --job-name=transect-finetune
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/transect_finetune_%j.log
#SBATCH --error=logs/transect_finetune_%j.err
# ─────────────────────────────────────────────────────────────
# Transect Finetuning — SatMAE encoder + 1D cloud curtain decoder
#
# Usage:
#   mkdir -p logs
#   sbatch scripts/submit_finetune_transect.sh
#
# Multi-GPU (4x V100):
#   Uncomment the 4-GPU lines below and comment the single-GPU ones.
# ─────────────────────────────────────────────────────────────

# ── Activate environment ──────────────────────────────────────
# Adjust to your conda/module setup:
# module load anaconda
# source activate satvision-pix4d
# -- or --
# module load python/3.12
# source /path/to/venv/bin/activate

# ── Configuration ─────────────────────────────────────────────
# Data: path to directory containing .npz transect files
export DATA_DIR="/explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/chips/allChips"

# Pretrained SatMAE weights
export PRETRAINED_WEIGHTS="/explore/nobackup/projects/ilab/projects/SatVisionPix4D/pretraining/mp_rank_00_model_states.pt"
export PRETRAINED_CONFIG="/explore/nobackup/projects/ilab/projects/SatVisionPix4D/pretraining/test_satmae_dev_dgx.yaml"

# Training
export TRAIN_BATCH_SIZE=4
export TRAIN_NUM_DEVICES=1
export TRAIN_STRATEGY="auto"
export NUM_WORKERS=8

# Checkpointing
export CHECKPOINT_DIR="./checkpoints/transect"

# To resume from a checkpoint, uncomment:
# export RESUME_CHECKPOINT="./checkpoints/transect/last.ckpt"

# ── Multi-GPU (uncomment for 4x GPU) ─────────────────────────
# #SBATCH --gres=gpu:4
# export TRAIN_NUM_DEVICES=4
# export TRAIN_STRATEGY="ddp"
# export TRAIN_BATCH_SIZE=8

# ── Run ───────────────────────────────────────────────────────
mkdir -p logs "${CHECKPOINT_DIR}"

echo "=================================================="
echo "  Job:     ${SLURM_JOB_ID}"
echo "  Node:    $(hostname)"
echo "  GPUs:    ${TRAIN_NUM_DEVICES}"
echo "  Data:    ${DATA_DIR}"
echo "  Weights: ${PRETRAINED_WEIGHTS}"
echo "=================================================="

cd /home/aliewehr/satvision-pix4d

srun python scripts/finetune_transect.py
