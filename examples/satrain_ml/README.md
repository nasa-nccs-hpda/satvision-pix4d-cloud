SatRain precipitation retrieval project. Three model experiments trained on the SatRain
dataset.
1. `baseline_model/` - THis folder has the baselien U-Net CNN (GMI + single-timestep ABI).
2. `abi_only/` - This is a CNN using only ABI data (no GMI).
3. `transformer_model/` - This is the hybrid temporal transformer.

## Layout

- `download_data/` - scripts that crawled and pulled the raw SatRain dataset from the source archive.
- `satrain_data/` - preprocessing scripts (raw NetCDF/HDF -> memmapped `.npy` tensors) and the
  preprocessed output itself, under `satrain_data/preprocessed/`.
> The raw satrain data is in `../satrain` or `/explore/nobackup/projects/pix4dcloud/sanumolu/satrain`
- `baseline_model/`, `abi_only/`, `transformer_model/` - Each has the
  same internal shape: model code, a training/submit script, `checkpoints/`, `logs/`
  (TensorBoard/Lightning logs), `figures/`, and analysis notebooks. See each folder's own README.
- `poster_figures/` - final poster figures.
- `diagnostics/` - diagnostic figs.
  See `diagnostics/README.md`.
- `PlotNeuralNet/` - Tool used to draw the architecture diagrams
  in `poster_figures/`.
- `envs/` - conda environment exports (see below).
- `Testing.ipynb` - scratch notebook.

## Data paths

Raw data lives one level up, at `../satrain/` (`gmi/` and `atms/`, each split into
training/validation/testing). Preprocessed training tensors live at
`satrain_data/preprocessed/satrain/gmi/{training,validation}/{xs,l,xl}/on_swath/`. Scripts read
these via the `SATRAIN_DATA_PATH` env var (raw data) and a `DATA_ROOT`/`PP_ROOT` constant near the
top of each training script (preprocessed data)

## Environments

Three conda environments are used. environment.yamls and a description of what each
one is for are in `envs/`. Recreate with `conda env create -f envs/environment_<name>.yaml`.