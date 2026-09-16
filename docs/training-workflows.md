# Synthetic benchmarking and NumPy foundation-model training

The supported capacity targets for these workflows are **330M** (336.49M actual
parameters), **700M** (705.14M), and **3B** (3.037B). The benchmark also accepts
`--model 300M` as an alias for 330M. Inputs default to **[B,7,16,512,512]**;
B is the per-device microbatch size. All commands below run from the repository
root in the training environment. The container definitions include TensorBoard.

## 1. Short synthetic benchmark

Check a preset without allocating weights or needing a GPU:

```bash
python -m satvision_pix4d.benchmark --model 3B --dry-run --output benchmark_runs/check
```

Measure training throughput and peak GPU memory, excluding the first ten optimizer
updates. This example uses four GPUs on one node; set `--devices` for your machine.
ZeRO-3 is the default strategy and needs DeepSpeed in the environment.

```bash
python -m satvision_pix4d.benchmark \
  --model 330M --mode throughput --devices 4 \
  --steps 100 --warmup-steps 10 --batch-size 1 --accumulation-steps 1 \
  --output benchmark_runs/330m-throughput
```

Repeat with `--model 700M` and `--model 3B`, using a different output directory for
each. For an unsharded single-GPU run, select `--strategy auto --devices 1`.
For replicated multi-GPU training use `--strategy ddp`; each GPU must hold a full
model and optimizer. ZeRO-3 initializes weights in Lightning's sharded context.
Large parameter initialization explicitly gathers each partitioned child layer
before applying custom Xavier initialization.

### Throughput tuning on H200 (330M first)

The presets prioritize memory savings: ZeRO-3, activation checkpointing, and
batch size one. For 330M on H200, measure DDP with checkpointing disabled before
assuming parameter sharding is needed. This trades more memory for less parameter
communication and recomputation. More GPU memory used is not itself a speedup;
compare `summary.json` sequences/sec and peak allocated/reserved GiB.

Use exclusively allocated GPUs for repeatable comparisons. Other workloads can
delay a rank and stall synchronous training even if they use little memory.
A starting comparison on four available GPUs is:

```bash
# Use these IDs only if they are available in your allocation.
export CUDA_VISIBLE_DEVICES=0,1,3,5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m satvision_pix4d.benchmark \
  --model 330M --mode throughput --strategy ddp --devices 4 \
  --batch-size 2 --workers 2 --cpu-threads 4 \
  --no-activation-checkpointing \
  --steps 100 --warmup-steps 10 \
  --output "benchmark_runs/330M-ddp-b2-$(date +%Y%m%d-%H%M%S)"
```

Four GPUs with batch two preserves the global batch of eight from eight GPUs
with batch one (accumulation one). To isolate each tuning effect, hold device
allocation fixed and vary one setting at a time: strategy, checkpointing, workers,
then microbatch. Try batch four next if memory permits; that changes the global
batch and samples processed per optimizer step. Larger batches do not guarantee
faster convergence, and 500 steps with a larger batch is more work.

The loader overlaps CPU generation using two workers per rank, pinned memory,
persistent workers and one prefetched batch per worker. `--cpu-threads` caps main
process PyTorch CPU threads per rank; loader workers use one each. Do not blindly
use the warning's suggested worker count on every GPU rank. These options and
the effective batch size are recorded in `metadata.json` and `config.yaml`.

After selecting the fastest stable setup, use `--mode overfit --steps 500
--fixed-samples 4 --probe-samples 4` to recheck convergence. These are short
benchmark controls, not changes to the 100-epoch pretraining presets. For real
training, set `TRAIN.STRATEGY: ddp`, `TRAIN.USE_CHECKPOINT: False`,
`DATA.BATCH_SIZE` and `DATA.NUM_WORKERS` in a local derived YAML once validated.
700M and 3B require their own memory/throughput measurements.

Test whether the training implementation can learn a fixed small synthetic set:

```bash
python -m satvision_pix4d.benchmark \
  --model 330M --mode overfit --devices 4 \
  --steps 500 --warmup-steps 10 --fixed-samples 4 \
  --probe-samples 4 --min-relative-improvement 0.10 \
  --output benchmark_runs/330m-overfit
```

The generator produces reproducible moving spatial patterns and correlated bands,
then converts them to raw values using DATA.MEAN/STD. This is learnable structure,
not independent white noise and not a physically realistic atmospheric simulation.
Training masks remain random; before/after probes use the same samples and fixed
masks. A separate synthetic validation split tests different patterns. The probe
also reports a channel-mean predictor baseline. Passing the relative-loss test
only demonstrates optimization on synthetic examples; it does not demonstrate
real-world generalization or require beating that baseline.

- `steps.jsonl`: one record per optimizer update, including raw loss, elapsed
  seconds, global sequences/sec and frames/sec. Accumulation is accounted for.
- `summary.json`: initial/final fixed-probe losses, relative improvement,
  held-out loss, baseline loss, post-warmup throughput, and maximum per-rank CUDA
  allocated/reserved memory. CPU memory fields are null.
- `metadata.json` and `config.yaml`: parameter count, input shape, effective batch
  size, device, precision, versions, seed, and resolved configuration.
- `tensorboard/`: scalar curves and initial/final reconstruction panels.
- `failure.json`: runtime exception details when the callback can catch a failure.

Throughput mode does not claim convergence. Overfit mode returns **exit code 2**
if it does not meet the chosen relative-loss-reduction threshold. Other runtime
failures are nonzero. Use a fresh output directory for every benchmark. Timing
includes synthetic generation, transfer, forward/backward, optimizer updates, and
training synchronization; probe evaluation and report-writing time are excluded.
It is not a pure GPU-kernel benchmark. Peak memory is measured for training after
timing warmup and excludes final probes. Fixed learning rate is used in this
short benchmark. It does not write large model checkpoints.

For a local functional check without large weight allocation:

```bash
python -m satvision_pix4d.benchmark \
  --tiny --image-size 128 --accelerator cpu --strategy auto --precision 32-true \
  --mode overfit --steps 100 --warmup-steps 5 --fixed-samples 1 --lr 0.001 \
  --output benchmark_runs/tiny-overfit
```

`--tiny` is strictly a software check, not a performance proxy for the named
capacity. Omitting `--image-size 128` checks the complete 7×16×512×512 input shape.

## 2. Real NumPy training for 100 epochs

Use the configurations in `configs/train/{330m,700m,3b}.yaml`. They select
512×512 chips, seven timesteps, 16 channels, five timestamp components, the
70/20/10 reconstruction loss, bf16, activation checkpointing, ZeRO-3, TensorBoard,
100 epochs, and one epoch checkpoint per epoch. Learning rate, gradient
accumulation, GPU strategy, and dataset statistics remain configurable through a
local YAML with `BASE: [path/to/configs/train/330m.yaml]`.

### Array contract

Supported float arrays:

- Single chip: `[7,16,512,512]` or `[7,512,512,16]`.
- Batched file: `[N,7,16,512,512]` or `[N,7,512,512,16]`.

Provide **raw values in the units of DATA.MEAN/STD**, with ABI C01–C16 in order.
The training module standardizes exactly once. If arrays are already standardized,
configure MEAN=0 and STD=1 for each channel and express MIN/MAX in those same
standardized units. Validate the inherited statistics against your actual corpus.
Non-finite pixels and wrong channel/spatial/temporal sizes fail explicitly. Missing
pixels need a preprocessing or validity-mask strategy before this workflow.

Preferred NPZ format:

```python
import numpy as np

# chip is float32 [7,16,512,512]. All observations refer to the same geographic tile.
timestamps = np.array([
    "2020-01-01T00:00:00", "2020-01-01T00:10:00", "2020-01-01T00:20:00",
    "2020-01-01T00:30:00", "2020-01-01T00:40:00", "2020-01-01T00:50:00",
    "2020-01-01T01:00:00",
], dtype="datetime64[s]")
np.savez("chip.npz", chip=chip, timestamps=timestamps)
```

NPZ chip keys accepted: `chip`, `chips`, `rad`, `data`, `image`, `images`, `arr_0`.
Timestamp keys: `timestamps`, `timestamp`, `times`, `time`, `t`.
For batched chips, timestamps may be `[N,7]` date strings/datetimes or `[N,7,5]`
components. Shared `[7]` dates or numeric `[7,5]` components are also supported.
Components are **[year−2000, month−1, day−1, hour, minute]**, in UTC. No pickle/object
arrays are loaded. One-dimensional numeric timestamps retain the legacy meaning
of hour coordinates; use datetime arrays to avoid ambiguous epoch units.

For NPY, add a sidecar with the same stem:

```python
np.save("chip.npy", chip)
np.save("chip.timestamps.npy", timestamps)
```

Timestamp sidecars are excluded from chip discovery. The production presets
require timestamps. Older configurations can still allow the synthetic fallback,
which now emits a warning. NPY data is memory-mapped and copied one selected chip
at a time. NPZ members must be decompressed when accessed: prefer individual/small
NPZ shards or NPY for large batched arrays. Discovery scans the supplied directory
itself, not nested directories; list the required day directories explicitly.

Use separate training/validation events and time periods. Identical input files
in both splits are rejected, but geographically or temporally overlapping chips
in different files still require your own split design.

### Validate, train, inspect, resume

Check sample layouts and timestamps **without allocating the model**:

```bash
python -m satvision_pix4d.satvision_pix4d_cli \
  -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation \
  --check-data-only --check-samples 8
```

This checks the first eight samples in each split, not the entire corpus. Samples
are also validated as they are loaded during training.

Train for the configured **100 epochs**:

```bash
python -m satvision_pix4d.satvision_pix4d_cli \
  -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation
```

Switch the YAML to `700m.yaml` or `3b.yaml` for the other capacities. Restrict
visible GPUs with CUDA_VISIBLE_DEVICES if needed. Use your normal Lightning/Slurm
launcher for multi-node production jobs; distributed GPU execution has not been
validated in this local environment.

```bash
tensorboard --logdir outputs
```

TensorBoard records:

- Per-step training loss and epoch-averaged train/validation loss.
- Charbonnier, Sobel, and MS-SSIM loss components.
- Existing full-image PSNR/SSIM and masked-only PSNR.
- Validation masked MAE separately for each band, in that band's raw units.
- Learning rate, epoch wall time (including validation), samples per wall second,
  and peak allocated GPU memory.
- Reconstruction panels each epoch: **target | masked input | reconstruction**.
  Rows show selected timesteps/bands (default zero-based bands 1 and 12, first
  three timesteps). Display limits are shared between prediction and target for
  each row, images are downsampled to 128×128, and only the first validation batch
  on rank zero is visualized. Visible pixels in the reconstruction come from the
  input. Validation masks are repeatable between epochs for stable comparisons.

Aggregate PSNR/SSIM use a common raw range across bands; per-band MAE is more
interpretable when channel units differ. A lower reconstruction objective alone
is not evidence of better downstream cloud retrieval or forecasting.

Checkpoints live in `outputs/<model-name>/numpy-pretrain-100epochs/`:
`epoch-000`, `epoch-001`, …, `epoch-099`, a `last` checkpoint, and the three best
validation checkpoints. Single-device/DDP checkpoints have `.ckpt` filenames;
DeepSpeed may use checkpoint directories with sharded files. Keeping every epoch
requires storage for full training state, including optimizer state.

```bash
python -m satvision_pix4d.satvision_pix4d_cli \
  -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation \
  --resume /path/to/last.ckpt
```

Use the same architecture/configuration and pass the DeepSpeed checkpoint directory
when applicable. Resume restores model, optimizer, scheduler, and progress toward
the total 100 epochs. New checkpoints use primitive configuration metadata so
safe PyTorch loading can restore them. Very old checkpoints containing pickled
YACS config objects may require a separate trusted-checkpoint migration.
TensorBoard starts a new version directory for a resumed invocation while keeping
the restored global steps. MLflow is optional (`MLFLOW.ENABLED: True`).

For a **100-epoch synthetic training run with the same tracking/checkpoints**:

```bash
python -m satvision_pix4d.satvision_pix4d_cli -c configs/benchmark/330m.yaml
```

Use `700m.yaml`/`3b.yaml` similarly. This regular training path uses the pipeline's
cosine schedule; the short benchmark above uses constant LR. Each synthetic epoch
has DATA.LENGTH samples (default 256). A local override with
`BENCHMARK.MODE: overfit` repeatedly uses BENCHMARK.FIXED_SAMPLES patterns instead.

## Local verification

```bash
python -m pytest tests/test_mae_architecture.py tests/test_abi_temporal_chips.py \
  tests/test_training_workflows.py -q
```

The tiny CPU test model reduced fixed-probe loss from 0.31454 to 0.06904 (~78%) in
100 updates on a single 7×16×128×128 synthetic sequence. Held-out loss did not
improve in that deliberate overfit run. A separate tiny-model smoke run performed
two updates with the exact 7×16×512×512 input. These demonstrate functional data,
loss, optimizer and logging paths, not full-size GPU throughput or convergence.
All 33 targeted tests passed. Tests read TensorBoard events/images, verify epoch
checkpoints, and resume training from the last checkpoint. A separate two-process
CPU DDP smoke run verified distributed loss/throughput aggregation. Full 330M/700M/3B GPU runs, CUDA memory values,
and ZeRO-3 checkpoint/resume still need validation on the target cluster.
