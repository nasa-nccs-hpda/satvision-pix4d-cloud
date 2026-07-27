#!/bin/bash
#SBATCH -t 60                                     # Request 60 minutes max runtime
#SBATCH -n 1                                      # Number of tasks
#SBATCH -N 1                                      # Run on a single node
#SBATCH -c 4                                      # Request 4 CPU cores
#SBATCH --mem=16G                                 # Request 16 GB of memory
#SBATCH -J abi_cloudsat_collocation               # Job name
#SBATCH --export=ALL                              # Export environment variables

# Initialize conda for non-interactive shell sessions
if [ -f /home/aliewehr/miniconda3/etc/profile.d/conda.sh ]; then
    source /home/aliewehr/miniconda3/etc/profile.d/conda.sh
    conda activate /panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/envs/ilab-pytorch
else
    # Fallback path modification if the home conda shell initialization script is not found
    export PATH="/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/envs/ilab-pytorch/bin:$PATH"
fi

# Run the pipeline with your own PYTHONPATH and CLI script paths
PYTHONPATH=/explore/nobackup/projects/pix4dcloud/aliewehr/satvision-pix4d \
python /explore/nobackup/projects/pix4dcloud/aliewehr/satvision-pix4d/satvision_pix4d/view/cloudsat_abi_cropping_cli.py \
  --abi-root /css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD \
  --cloudsat-root /explore/nobackup/projects/pix4dcloud/szhang16/cloudsat \
  --output-dir /explore/nobackup/projects/pix4dcloud/aliewehr/abi-cloudsat-crop-test-inner-disk-2500 \
  --year 2019 \
  --satellite goes16 \
  --metadata cloudsat \
  --require-cloud \
  --day-start 100 \
  --profile-selection chip \
  --chip-size 512 \
  --inner-disk-margin 2500 
