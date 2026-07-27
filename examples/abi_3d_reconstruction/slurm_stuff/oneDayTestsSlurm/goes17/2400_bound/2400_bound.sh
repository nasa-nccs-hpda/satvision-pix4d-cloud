#!/bin/bash
#SBATCH -t 8:00:00 #request 8 hours max runtime
#SBATCH -n 1                                      # Number of tasks
#SBATCH -c 10                                      # Request 10 CPU cores
#SBATCH --mem=64G                                 # Request 64 GB of memory
#SBATCH -J abi_cloudsat_collocation_2400_1day_goes17               # Job name
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
  --abi-root /css/geostationary/NonOptimized/L1/GOES-17-ABI-L1B-FULLD \
  --cloudsat-root /explore/nobackup/projects/pix4dcloud/szhang16/cloudsat \
  --output-dir /explore/nobackup/projects/pix4dcloud/aliewehr/oneDayTests/goes17/2400_bound \
  --offsets -60 -40 -20 0 20 40 60 \
  --year 2019 \
  --satellite goes17 \
  --metadata cloudsat \
  --require-cloud \
  --day-start 100 \
  --day-end 101 \
  --profile-selection chip \
  --chip-size 512 \
  --inner-disk-margin 2400
