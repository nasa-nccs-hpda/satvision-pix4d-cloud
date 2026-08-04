#!/bin/bash
#SBATCH -J abi_only_cnn                      
#SBATCH -t 48:00:00 
#SBATCH -N 1                                 
#SBATCH -n 1                                 
#SBATCH -c 72
#SBATCH --mem=256G
#SBATCH -G 1                                 
#SBATCH -p grace
#SBATCH -o %x_%j.out                         
#SBATCH -e %x_%j.err                         
#SBATCH --export=ALL                         

# Load ONLY nvidia, NOT miniforge
module load nvidia
# Source aarch64 conda
source /panfs/ccds02/app/modules/miniforge/platform/aarch64/rocky/9.4/24.9.2/etc/profile.d/conda.sh

# Activate your environment
conda activate satrain_torch_arm
conda list
# Navigate and run
cd /explore/nobackup/projects/pix4dcloud/sanumolu/satrain_ml/abi_only
python train_abi_only.py