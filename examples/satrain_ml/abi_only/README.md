CNN using only ABI geostationary imagery

## Files

- `model.py` - network and LightningModule (`FastSatRain`, `IPWGUNet`), ABI-only input.
- `train_abi_only.py` - training entry point. Submit via `submit_abi_only.sh` (SLURM, grace
  partition, `satrain_torch_arm` env).
- `abi_only.ipynb` - exploratory notebook used while building the model.
- `abi_only_results.ipynb` - post-hoc model evaluation & plotting notebook.

## Runs

 - Model runs are in `checkpoints/` and `logs/`. 
 - `abi_only_cnn`, `_v2`, `_v3` use a raw-MSE
loss (val_loss ~0.87-1.15). 
 - `_v4_log1p` and `_v5log1p` use a log1p-transformed loss instead
