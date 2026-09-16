# uv environment for H100 training

This environment covers the synthetic benchmarks and NumPy/Zarr pretraining
workflows. It is separate from the full satellite-ingestion/geospatial container
(GDAL, Satpy, tile generation and downstream examples are not included).

The checked-in `uv.lock` pins the complete dependency resolution. Python is 3.12,
PyTorch is 2.7.1 with torchvision 0.22.1, Lightning is 2.6.6, timm is 1.0.29, and
DeepSpeed is 0.17.6 for the Linux CUDA extras. TensorBoard and the targeted test
suite dependencies are included. The PyTorch version is a deliberate compatibility
pin, not a claim that it is the latest release. CUDA wheels come from PyTorch's
explicit package indexes; generic dependencies come from PyPI.

## H100 setup

From the repository root on Linux x86_64:

```bash
# If uv is not already installed:
python3 -m pip install --user 'uv>=0.9'

# uv downloads Python 3.12 if needed and creates .venv.
uv sync --locked --extra cu126
source .venv/bin/activate

# Run this on your allocated GPU node, not a CPU-only login node.
nvidia-smi
python -m satvision_pix4d.check_environment --require-deepspeed
```

If pip's user scripts are not in PATH, add the user-base `bin` directory to PATH
or use your site's uv module. Python installation and package downloads need
network access during setup. For restricted compute nodes, install on a login
node using a filesystem accessible from the compute allocation.

H100 supports bf16; keep the existing `PRECISION: bf16-mixed`. The driver must
support the selected CUDA runtime. `nvidia-smi` reports the driver's CUDA support,
not an installed toolkit. The preflight checks actual imports, visible devices,
and a bf16 FlashAttention forward/backward pass on each visible GPU. It does not
validate multi-node networking or a full ZeRO-3 training run.

CUDA 12.8 is an alternative when supported by the driver:

```bash
uv sync --locked --extra cu128
```

Choose one of `cpu`, `cu126`, and `cu128`; they are mutually exclusive. Do not use
`--all-extras`. CUDA environments are intended for Linux x86_64. Apple Silicon
uses native PyTorch from PyPI for local CPU tests; CUDA/DeepSpeed are not available
there. This lockfile does not target Windows or ARM Linux.

DeepSpeed builds have DS_BUILD_OPS=0 to avoid compiling all optional operators at
installation. If your chosen runtime optimizer/offload configuration JIT-compiles
an operator, a compatible CUDA toolkit/compiler must also be available on that
node. The default workflows use torch AdamW without CPU/NVMe offload. You do not
need the separate `flash-attn` package: the model uses PyTorch SDPA.

## Run the models

After activation, use the commands already in the
[training workflow guide](training-workflows.md). For example:

```bash
python -m satvision_pix4d.benchmark \
  --model 330m --devices 4 --steps 100 --warmup-steps 10 \
  --output benchmark_runs/330m-h100

python -m satvision_pix4d.satvision_pix4d_cli \
  -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation

tensorboard --logdir outputs
```

Repeat with `700m` and `3b`, adjusting the GPU allocation and paths. To use uv
without activating, include the selected extra on every command:

```bash
uv run --locked --extra cu126 python -m satvision_pix4d.benchmark \
  --model 3b --dry-run --output benchmark_runs/3b-check
```

Plain `uv run` may synchronize without the optional CUDA dependencies; avoid mixing
it with an environment installed using `--extra cu126`. Alternatively, after a
successful sync, `uv run --no-sync ...` uses that environment unchanged.

Optional MLflow support:

```bash
uv sync --locked --extra cu126 --extra mlflow
```

Then set MLFLOW.ENABLED in your configuration. TensorBoard requires no extra.

## Local CPU verification

```bash
uv sync --locked --extra cpu
source .venv/bin/activate
python -m satvision_pix4d.check_environment --device cpu
python -m pytest tests/test_mae_architecture.py tests/test_abi_temporal_chips.py \
  tests/test_training_workflows.py -q
```

CPU execution of large presets is not a practical training configuration. Use
`--tiny --accelerator cpu --strategy auto --precision 32-true` for functional
benchmark checks. H100/CUDA execution must be verified on the target system.

References: [uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/),
[PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/),
[DeepSpeed installation](https://www.deepspeed.ai/tutorials/advanced-install/).
