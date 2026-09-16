# Foundation-model training handoff — 2026-09-16

This is a working summary of the desktop conversation for continuation with
Codex CLI on the training node. It preserves decisions, implementation state,
observations, and unresolved work; it is not a verbatim chat transcript. Read the
current code and local run artifacts before assuming these observations remain
current. Update this document with new measured results and decisions.

### Additional request: stratified training dataset

The user now also wants datasets stratified using convection, cloud-top height,
MODIS land cover, and background/random chips, with an eventual ~200K total
sequences. **They confirmed seven frames at 20-minute spacing** (two-hour extent).
Source archive/catalog paths are still pending; old repo paths are clues, not
confirmed accessible sources. A provisional 1K total pilot and metadata-only
selector were added: `docs/stratified-training-data.md`,
`configs/data/stratified_pilot.yaml`, and
`python -m satvision_pix4d.preprocessing.stratified_manifest`.
This consumes preannotated CSV candidates, respects preassigned splits or hashes
audited leakage groups, balances sources/classes for enriched training pools,
deduplicates selections and reports unmet quotas. Five targeted tests pass.
No real imagery was downloaded/extracted and CTH/MODIS annotation adapters are
not implemented yet. Read the design document before building those adapters.
Do not claim the pilot CSVs can be fed directly to the current training loader.
200K float32 arrays alone require ~23.49 TB uncompressed; start small.

### Logging follow-up

After this handoff was first written, the user reported a failed run without a
saved console traceback and requested improved logging. Both benchmark and
production training now wrap setup/training in `RunLogging`. Each process writes
`<output>/logs/rank-<rank>-<timestamp>-pid<pid>.log` and `.status.json`, including
full Python tracebacks, process/host identity and final status. The benchmark's
legacy `failure.json` also includes a traceback. Inspect every rank and distinguish
`completed_nonzero` (overfit criterion failure) from `failed` (exception).
Native OS-level NCCL output still needs shell tee when required. Hard kills can
leave status at `running`; it is not a liveness check. See the workflow guide.

## Immediate objective

Tune **330M throughput first**, then validate overfit convergence using the chosen
settings. Proceed to **700M**, then **3B**, one model at a time. The user explicitly
agreed that throughput mode is appropriate for speed tuning. No measured H200
results for the newly proposed DDP settings have been received yet.

The user wants working commands and concrete fixes, rather than repeated
confirmation questions. They authorized creating this branch and pushing related
work to this repository. Preserve unrelated changes and existing experiments.

## Repository and environment

- Repository: https://github.com/nasa-nccs-hpda/satvision-pix4d-cloud
- Working branch: `codex/foundation-model-training`.
- Last implementation commit before this handoff: `d7d2251`.
- Training checkout: `/raid/jacaraba/satvision-pix4d-cloud` on `gs6n-dgx01`.
- Actual hardware: **8 NVIDIA H200**, approximately 140 GiB visible per GPU.
  The user initially said H100; the subsequent nvidia-smi output established H200.
- Reported driver: **570.211.01**, supporting CUDA 12.8.
- Runtime: Python 3.12, torch 2.7.1+cu128, torchvision 0.22.1, Lightning 2.6.6,
  timm 1.0.29, DeepSpeed 0.17.6, TensorBoard. See `pyproject.toml` and `uv.lock`.
- The uv environment and GPU preflight now work, per the user.
- Other workloads share some GPUs. At earlier snapshots GPUs 2, 4, 6, 7 had
  other allocations; 0, 1, 3, 5 were candidates for an isolated run. Recheck live
  allocation and utilization; these are not permanent reservations. A momentary
  0% utilization does not mean an existing workload will remain idle.

```bash
cd /raid/jacaraba/satvision-pix4d-cloud
git switch codex/foundation-model-training
git pull --ff-only
uv sync --locked --extra cu128
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -m satvision_pix4d.check_environment --require-deepspeed
```

DeepSpeed needs a discoverable real CUDA toolkit on GPU nodes, including for
import-time optional-operator probes. Initially the directory existed but nvcc
was absent; the user subsequently installed the toolkit and confirmed it worked.
The NVIDIA toolkit-only package recommended was `cuda-toolkit-12-8`, plus
`build-essential`; the driver did not need replacement. Do not repeat the earlier
incorrect claim that the PyTorch CUDA runtime alone suffices for this setup.

DeepSpeed's isolated build must exclude torch and use DS_BUILD_OPS=0, as currently
configured. Adding torch to its build dependencies causes another toolkit probe
during installation. Do not use a fake CUDA_HOME or bypass checks as a fix.
Use activated `python`, or `uv run --locked --extra cu128 ...`; plain `uv run`
can synchronize without the selected optional CUDA dependencies.

## Scope and architecture decisions

Original basis: https://github.com/sustainlab-group/SatMAE, modified in this repo.
The requested figures were a temporal masked autoencoder and a reconstruction
loss with weights **0.7 Charbonnier + 0.2 Sobel gradient + 0.1 MS-SSIM**.
MS-SSIM is applied to VIS/NIR channels only, not thermal brightness temperatures.

- Active sizes: 330M (336,485,888 parameters), 700M (705,144,064),
  3B (3,037,456,384). The initial assumption that the original model was ~700M
  was corrected by counting parameters: its architecture is the 330M preset.
- `--model 330M`, `700M`, `3B` are the preferred CLI labels. Lowercase remains
  accepted. `300M`/`300m` is only a compatibility alias for 330M, not a fourth model.
  YAML filenames and internal preset identifiers remain lowercase.
- Experimental 25B preset exists (25,016,330,752 parameters) at the user's request.
  It is meta-shape/count checked only, not a current training target.
- Inputs for these experiments: `[B,7,16,512,512]`, ordered ABI C01–C16.
- Patch size 16; dynamic 2D spatial sine/cosine encoding; temporal sine/cosine
  features for year/month/day/hour/minute; learned temporal/spatial projection.
- 75% random masking before the encoder; masked reconstruction in original
  channel space. Temporal component ordering must remain consistent.
- timm SDPA enables FlashAttention when hardware/dtype conditions permit.
  There is no separate flash-attn package requirement.
- Presets disable norm-pixel loss and CNN refinement for the intended objective.
- Spatial resolution and sequence length can vary within architectural limits;
  input channel count is configured in the patch projection, not arbitrary at
  runtime for one fixed checkpoint. Do not overstate the figure's flexibility.
- This is a reconstruction foundation model. World-model training was discussed
  as a possible future direction, but no predictive/rollout world-model pipeline
  was implemented or validated. Scaling beyond 3B has no demonstrated benefit
  here; neither 25B quality nor compute feasibility has been established.

## Data, normalization, tracking, and production training

- Feed raw imagery values in the units expected by `DATA.MEAN` and `DATA.STD`.
  Apply fixed per-channel standardization once in the pipeline. This z-score
  approach was already present before the initial changes; no per-chip min/max
  normalization is required. Validate the configured statistics on the real data.
- Fixed channel min/max bounds used by the SSIM term are separate loss scaling,
  not a replacement for input normalization. If input arrays are already
  standardized, configure mean=0/std=1 and consistent SSIM bounds.
- NumPy/Zarr ingestion, timestamp validation, synthetic generation, TensorBoard
  scalar metrics and reconstructions, per-epoch checkpoints and resume support
  were added/checked. Production configs specify 100 epochs.
- Preferred NPZ members: `chip`/`chips` and `timestamps`; NPY timestamp sidecars
  use `chip.timestamps.npy`. Single chips `[7,16,512,512]` or channels-last;
  batched shards add a leading N. No pickle/object arrays.
- Datetime timestamps are preferred. Numeric components are UTC
  `[year-2000, month-1, day-1, hour, minute]`. Real presets require timestamps.
- Nonfinite imagery fails validation. Missing-data masks/preprocessing still
  require a deliberate design if the real corpus contains missing pixels.
- Separate train/validation events, dates and regions; file-disjoint splits alone
  do not guarantee geographic/temporal independence. Real dataset paths have not
  been supplied in the conversation. See `docs/training-workflows.md` for details.

```bash
# Check data before allocating a large model (replace the paths).
python -m satvision_pix4d.satvision_pix4d_cli -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation --check-data-only

# Real pretraining: 100 epochs, epoch checkpoints and TensorBoard.
python -m satvision_pix4d.satvision_pix4d_cli -c configs/train/330m.yaml \
  --train-data /path/to/train --val-data /path/to/validation

# Separate synthetic 100-epoch pipeline with epoch checkpoints.
python -m satvision_pix4d.satvision_pix4d_cli \
  -c configs/benchmark/330m.yaml --epochs 100
```

The short `benchmark.py` throughput/overfit command uses optimizer STEPS, not
epochs, and deliberately does not save large model checkpoints. Do not conflate
it with the 100-epoch CLI workflows. Default strategy is ZeRO-3; defaults prioritize
memory savings rather than maximum H200 throughput.

## Known failures already fixed

| Commit | Failure and fix |
| --- | --- |
| `6e280d5` | Initial model review, presets, synthetic/real training workflows. |
| `4d99231` | Locked uv environments and GPU preflight. |
| `5265f7a` | Removed torch from isolated DeepSpeed build dependencies. |
| `8aad063` | Corrected runtime toolkit requirement and preflight diagnostic. |
| `6d6e738` | Uppercase CLI model labels with case-insensitive input. |
| `da759c3` | `deepspeed.zero` is an attribute, not an importable submodule. Use `import deepspeed; deepspeed.zero.GatheredParameters(...)`. |
| `05efddf` | ZeRO-3 skips whole-model `.to(device)` after partitioned initialization. Explicitly move normalization/loss buffers and TorchMetrics states to each rank's root device, without moving sharded parameters. |
| `2f83b7b` | Float32 normalized images reached BF16 convolution weights without autocast. Cast encoder input to patch-projection weight dtype while retaining original fp32 reconstruction targets. |
| `d7d2251` | Added `--no-activation-checkpointing` / `--activation-checkpointing` and `--cpu-threads`, and recorded tuning settings in benchmark metadata. |

NCCL/TCPStore shutdown warnings followed rank crashes; diagnose the FIRST Python
exception rather than assuming a networking problem. A missing VMware PVRDMA
provider warning appeared but did not prevent the subsequent successful run.
Loader worker-count, logging-interval, and LitLogger suggestions were nonfatal.
The benchmark records its own metrics each optimizer update; it evaluates probes
separately, so its skipped Lightning validation-loop warning was expected.

## What has actually been verified

Local CPU tests exercise architecture, losses, NumPy inputs, TensorBoard,
checkpoints/resume, and tiny synthetic convergence. BF16 input mismatch was
reproduced locally before the fix; inference and backward without autocast then
passed. Device-state tests check a separate meta device without relocating model
parameters. These are useful checks but not substitutes for CUDA/ZeRO tests.

After the dtype fix, the user reported a successful 8-H200 330M run:

```text
step=1  loss=0.442671 seconds=4.058 samples/s=1.972
step=10 loss=0.308466 seconds=1.022 samples/s=7.829
```

Command: 330M, overfit, eight GPUs, 500 steps, 10 warmup steps, four fixed samples
and four probe samples, default batch one/ZeRO-3/checkpointing/workers zero.
This proves initial training progress, not completion of 500 steps or the fixed
probe convergence criterion. Step 10 is still warmup; 7.829 sequences/sec is one
observed step, not a final averaged benchmark result. No final summary received.

Screenshot showed roughly 6.5 GiB per training process, uneven GPU utilization,
and other workloads on some GPUs. This suggests testing larger batches, less
recomputation and DDP, but does not establish the actual performance bottleneck.

The latest tuning controls passed a tiny two-process CPU DDP run with background
workers, batch two and checkpointing disabled, plus 13 training-workflow tests.
No H200 DDP speedup has been measured in this conversation. 700M/3B full-size GPU
training and 25B execution are not validated here. No full real-data pretraining
or scientific/downstream model quality claim is supported yet.

## Next experiment: throughput mode, then overfit

Inspect local `benchmark_runs/*/summary.json`, `metadata.json`, `steps.jsonl` and
`failure.json`, current processes and GPU allocation first. Existing remote files
may contain results the desktop conversation has not seen. Avoid overlapping
new runs with the user's current experiment. Use fresh output directories.

Proposed starting configuration (use GPU IDs only if available in allocation):

```bash
export CUDA_VISIBLE_DEVICES=0,1,3,5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m satvision_pix4d.benchmark \
  --model 330M --mode throughput --strategy ddp --devices 4 \
  --batch-size 2 --workers 2 --cpu-threads 4 \
  --no-activation-checkpointing --steps 100 --warmup-steps 10 \
  --output "benchmark_runs/330M-ddp-b2-$(date +%Y%m%d-%H%M%S)"
```

Four GPUs x batch two preserves the original global batch eight. This is a
practical candidate, not a controlled attribution of gains: for controlled
comparisons keep the GPU allocation fixed and change one factor at a time.
Try DDP vs ZeRO-3, checkpointing on/off, workers, then batch size. Compare final
aggregate sequences/sec, step time and peak allocated/reserved GiB. Larger batch
changes global batch and samples seen per optimizer step; do not equate it with
faster convergence. Do not blindly use 27 loader workers on EACH of eight ranks.

After selecting a stable fast configuration, retain those resource flags and
run `--mode overfit --steps 500 --fixed-samples 4 --probe-samples 4`.
Fixed before/after probe masks and samples test learning more reliably than
individual training losses. Synthetic learnability is not real generalization.
Then tune 700M and 3B individually rather than extrapolating memory requirements.

## TensorBoard and useful entry points

```bash
# On training node, within the activated environment:
tensorboard --logdir benchmark_runs --host 127.0.0.1 --port 6006
# On local computer:
ssh -N -L 6006:127.0.0.1:6006 jacaraba@gs6n-dgx01
# Open http://localhost:6006 locally.
```

Each short run writes `benchmark_runs/<run>/tensorboard/`. The 100-epoch
pretraining workflows write under `outputs`; use `tensorboard --logdir outputs`.

Read next:

- `docs/training-workflows.md`: data contracts, benchmark semantics, tuning.
- `docs/uv-environment.md`: environment and toolkit setup.
- `docs/model-architecture-review.md`: architecture review and exact counts.
- `satvision_pix4d/benchmark.py`: benchmark CLI, probes, timing, reports.
- `satvision_pix4d/models/encoders/models_mae_temporal.py`: model and ZeRO initialization.
- `satvision_pix4d/models/reconstruction_loss.py`: pixel/gradient/MS-SSIM loss.
- `satvision_pix4d/models/utils/device_state.py`: buffer and metric placement.
- `satvision_pix4d/pipelines/satvision_pix4d_pretrain.py`: production Lightning module.

Targeted verification:

```bash
python -m pytest tests/test_mae_architecture.py tests/test_abi_temporal_chips.py \
  tests/test_training_workflows.py -q
```

Record commands, code revision, GPU allocation, effective batch, throughput,
memory and convergence outcomes as work continues. Keep measured results distinct
from hypotheses and proposed settings.
