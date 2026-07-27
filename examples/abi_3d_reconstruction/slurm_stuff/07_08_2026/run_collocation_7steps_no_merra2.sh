#!/bin/bash
#SBATCH -t 24:00:00                               # Request 24 hours max runtime
#SBATCH -n 1                                      # Number of tasks
#SBATCH -c 40                                     # Request 40 CPU cores
#SBATCH --mem=128G                                # Request 128 GB of memory
#SBATCH -J abi_cloudsat_collocation_7steps_no_merra2 # Job name
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
# - Set 7 time steps using --offsets (here, -60, -40, -20, 0, 20, 40, 60 minutes)
# - Exclude MERRA-2 by specifying --metadata cloudsat cloudsat_aux (omits merra2)
PYTHONPATH=/explore/nobackup/projects/pix4dcloud/aliewehr/satvision-pix4d \
python /explore/nobackup/projects/pix4dcloud/aliewehr/satvision-pix4d/satvision_pix4d/view/cloudsat_abi_cropping_cli.py \
  --abi-root /css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD \
  --cloudsat-root /explore/nobackup/projects/pix4dcloud/szhang16/cloudsat \
  --output-dir /explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/7steps_no_merra2_2400margin_180stride \
  --offsets -60 -40 -20 0 20 40 60 \
  --year 2019 \
  --satellite goes16 \
  --metadata cloudsat cloudsat_aux \
  --require-cloud \
  --day-start 100 \
  --day-end 101 \
  --profile-selection chip \
  --profile-stride 180 \
  --chip-size 512 \
  --inner-disk-margin 2400 \
  --workers 4
