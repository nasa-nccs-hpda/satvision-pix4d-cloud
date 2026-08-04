# Conda environments used by this project

Conda environment.yaml files.

Recreate with: `conda env create -f environment_<name>.yaml`

- **environment_satrain_env.yaml** — used by `download_data/download.sbatch`
  (data download/crawl jobs, x86_64)
- **environment_satrain_torch.yaml** — used in preprocessing, debugging, and evaluation.
  (`satrain_data/preprocess_train.sh`, `preprocess_val.sh`, x86_64)
- **environment_satrain_torch_arm.yaml** — used by all model training jobs
  on the grace/ARM GPU partition (`baseline_model/submit_baseline.sh`,
  `abi_only/submit_abi_only.sh`, `transformer_model/*.slurm`, aarch64).

