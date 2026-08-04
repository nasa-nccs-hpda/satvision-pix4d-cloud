Hybrid temporal transformer

## Core files

- `model_transformer.py` — the actual model used for training: `IPWGTransformer` +
  `FastSatRainTemporal` (LightningModule/data wrapper). This is the one everything below imports.
- `retrieval_transformer.py` — temporal retrieval helper functions, built on
  `model_transformer.py`.
- `train_transformer.py` — training entry point. Submit via `run_transformer.slurm` (SLURM,
  grace partition, `satrain_torch_arm` env).
- `transformer_results.ipynb` — post-hoc evaluation & plotting notebook.

## Runs

`checkpoints/` and `logs/` contain several experiment lines: 
- `transformer_v1`, `transformer_v2` - early checkpoints, these diverged to NaN early during training
- `transformer_v2_fixed` - fixed model, uses log1p transform
