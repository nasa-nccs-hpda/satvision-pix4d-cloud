# Architecture and pretraining review

The existing SatMAE-derived encoder/decoder structure is a reasonable basis for
this experiment. Architecture correctness does not establish scientific quality:
reconstruction quality, temporal usefulness, and downstream transfer still need
training ablations on held-out storms/dates/regions.

## Sizes

The model versions are **330M**, **700M**, **3B**, and an experimental
**25B** capacity preset. These are
rounded size labels; the 330M preset retains the existing encoder architecture.

Exact trainable counts, using 16 input channels, patch size 16, decoder width
512/depth 8/16 heads, MLP ratio 4, no pixel refinement:

| Configuration | Encoder width / blocks / heads | Temporal fields | Encoder parameters | Total parameters |
| --- | --- | --- | ---: | ---: |
| Existing `tests/configs/test_satmae_dev_dgx.yaml` | 1024 / 24 / 16 | 3 | 307,950,592 | 336,158,208 |
| `configs/pretrain/330m.yaml` | 1024 / 24 / 16 | 5 | 308,212,736 | 336,485,888 |
| `configs/pretrain/700m.yaml` | 1280 / 34 / 20 | 5 | 676,739,840 | 705,144,064 |
| `configs/pretrain/3b.yaml` | 2560 / 38 / 32 | 5 | 3,008,396,800 | 3,037,456,384 |
| `configs/pretrain/25b.yaml` | 6144 / 55 / 48 | 5 | 24,985,436,160 | 25,016,330,752 |

Encoder counts include the patch embedding, class token, transformer, final norm,
and temporal/spatial projection. Decoder counts include the mask token. These
are parameter counts, not checkpoint file sizes. Large presets were instantiated
on PyTorch's meta device, which verifies tensor shapes without allocating weights.

The 25B preset keeps the same decoder and objective, with 128 dimensions per
attention head. It inherits bf16 mixed precision, activation checkpointing, and
ZeRO stage 3. Only construction and parameter counts have been checked on the
meta device; initialization memory, distributed execution, learning-rate tuning,
and numerical stability at this scale remain unvalidated. It is an optional
future experiment, not a validated training recipe.

## Matching the supplied figures

- Patch projection uses all configured channels, with nonoverlapping patches.
- Spatial sine/cosine embeddings are generated for the actual H/P × W/P grid.
  A single model can process 128, 256, or 512 square inputs and rectangular inputs;
  both dimensions must divide evenly by the patch size.
- Temporal sine/cosine embeddings are concatenated per configured field, then
  concatenated with spatial embeddings and projected through registered learned
  layers. New presets use year/month/day/hour/minute. Existing configs retain
  year/month/hour. Component order must remain consistent with the checkpoint.
- Random masking precedes encoder blocks. The new presets use 75% masking;
  SAME_MASK can reuse one spatial mask across every timestep. Decoder tokens
  are restored to original temporal/spatial order before reconstruction.
- Pre-norm timm blocks use PyTorch scaled-dot-product attention in encoder and
  decoder. This permits FlashAttention on supported CUDA devices/dtypes and
  falls back on other devices. It does not guarantee a particular CUDA kernel.
- Activation checkpointing now actually checkpoints transformer blocks when
  TRAIN.USE_CHECKPOINT is enabled.
- New presets disable the optional CNN refinement head, matching the figure.

Channels are configurable **when building the model**. A trained patch projection
and output head have a fixed channel count and order; this design does not let a
single checkpoint accept arbitrary channel subsets. Temporal length and spatial
size may vary between forward calls. Dense batches still require common T/H/W;
the data loader currently validates one configured square chip size per run.
Changing temporal granularity requires a matching configured field list and
projection shape. Missing timestamps use the existing synthetic hourly fallback;
use real timestamps for scientifically meaningful temporal pretraining.

## Reconstruction objective

`LOSS_TYPE: pixel_structure` implements
`0.7 * Charbonnier + 0.2 * Sobel-L1 + 0.1 * (1 - MS-SSIM)`.

Charbonnier (epsilon 1e-3, with its constant floor subtracted) and normalized Sobel
kernels operate on all channel-standardized bands. Masked pixels are supervised;
visible target pixels provide context at boundaries, with no gradient to visible
predictions. Sobel support includes neighboring pixels touched by a masked region.
MS-SSIM uses five scales, 7×7 Gaussian windows (sigma 1.5), standard scale weights,
and mask-weighted local score averages. This window size supports the figure's
128-pixel chips; the implementation requires both dimensions >=112. This is an
explicit masked MS-SSIM variant, not an exact reproduction of an unspecified
implementation behind the figure. The five scale factors are clamped positive
before fractional powers. Reductions run in float32 under mixed precision.

VISNIR_CHANNELS uses zero-based indices. Presets assume ABI bands C01–C16 in
order and select C01–C06; thermal C07–C16 never enter MS-SSIM. DATA.MIN/MAX are
fixed raw per-channel bounds transformed into standardized space by the builder.
Validate these inherited dataset statistics for the actual training corpus.
Predictions are not clipped, to preserve gradients. The weighted coefficients
match the figure but do not guarantee 70/20/10 percent of gradient magnitude.

Old configs retain MSE. NORM_PIX_LOSS must be false for pixel-space loss or the
physical-unit reconstruction pipeline; otherwise the decoder target and reported
physical values have different meanings.

## Bugs corrected

Fixed resolution assertions and square-only position grids; unwired temporal
field selection; unwired activation checkpointing; missing timestamp/mask shape
validation; mixed-precision loss reductions; masked PSNR's missing channel divisor;
cumulative-average loss logging; zero-warmup scheduler initialization; pixel
refinement overwriting visible predictions needed for visible supervision; and
imports of nonexistent dataset modules. Dataset loading now rejects a mismatched
channel count instead of silently truncating a spatial axis/channel array.
Optional fused optimizers are imported only when requested.

The active CLI now passes configured gradient accumulation/clipping, validation
frequency, fast-dev mode, and checkpoint frequency to Lightning. Resume uses
Trainer.fit(ckpt_path=...), restoring optimizer/scheduler/training progress rather
than just loading model weights. These CLI changes need distributed GPU validation.

## Compatibility and usage

The existing three-field configuration retains model parameter names and shapes.
For an old checkpoint, use its original configuration; the new five-field presets
change projection shapes and are for fresh pretraining. The 700M, 3B, and 25B sizes also
require fresh weights. `SIZE: custom` (the default) honors explicit dimensions;
named sizes override encoder width/depth/heads only. Decoder sizes remain configurable.

Create a local YAML next to the presets (or adjust the BASE path):

```yaml
BASE: [700m.yaml]  # or 3b.yaml, 330m.yaml, or experimental 25b.yaml
DATA:
  TRAIN_DATA_PATHS: [/path/to/train/chips]
  VAL_DATA_PATHS: [/path/to/held-out/chips]
  IMG_SIZE: 128
  BATCH_SIZE: 1
OUTPUT: /path/to/checkpoints
TRAIN:
  ACCUMULATION_STEPS: 16
```

From the repository root:

```bash
python -m satvision_pix4d.satvision_pix4d_cli -c configs/pretrain/local.yaml
python -m pytest tests/test_mae_architecture.py tests/test_abi_temporal_chips.py -q
```

Use the repository's training container (with modern timm/PyTorch, Lightning,
DeepSpeed and MLflow). The old `view/satvision_pix4d_pretrain_cli.py` is a legacy
SatMAE script with obsolete imports and is not the active training entry point.
The presets use ZeRO stage 3 and bf16 mixed precision. GPU count, memory, effective
batch size, and learning rate need to be selected for your cluster. BASE_LR is the
actual optimizer learning rate; it is not automatically scaled by batch size.
At 7 timesteps and 512×512 resolution, the decoder sees 7,168 patches plus CLS.
FlashAttention reduces attention memory, but attention compute still scales
quadratically with token count. Begin with a small real-data smoke run.

## Verification and remaining validation

CPU tests cover dynamic resolution, rectangular grids, patch round trips,
variable temporal length/fields, same-spatial masking, explicit mask restoration,
invalid inputs, no/full masking, optimizer updates to projection layers,
checkpoint state loading, refinement supervision, VIS/NIR selection, masked loss
gradients, bf16 mixed-precision backward, exact large-model parameter counts,
dataset loading, masked PSNR, and a two-step Lightning train/validation smoke run.

Not validated here: CUDA FlashAttention kernel selection, distributed ZeRO-3
initialization/resume, full 700M/3B/25B backward passes, real-data training convergence,
and downstream transfer. No pretrained weights were produced. Aggregate legacy
PSNR/SSIM metrics use a common raw range across bands; use per-band validation
metrics when comparing radiance/temperature channels with different scales.

References: [upstream SatMAE temporal model](https://github.com/sustainlab-group/SatMAE/blob/main/models_mae_temporal.py)
and [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
Upstream uses fixed spatial positions and a three-frame temporal path; this
repository's dynamic projections and combined reconstruction objective are extensions.
