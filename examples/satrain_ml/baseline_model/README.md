U-Net CNN baseline: GMI passive-microwave + single-timestep ABI 
## Files

- `model.py` - `IPWGUNet` (the network) and `FastSatRain` (the LightningModule/data wrapper).
- `train_baseline.py` - training entry point. Submit via `submit_baseline.sh` (SLURM, grace
  partition, `satrain_torch_arm` env).
- `baseline_cnn_results.ipynb` - post-hoc evaluation & plotting notebook

## Runs

 - Five training runs, in `checkpoints/` and `logs/`
 - `baseline_cnn`, `_v2`, `_v3`, `_v5` use a raw-MSE
 - `_v4log1p` uses a log1p-transformed loss instead 