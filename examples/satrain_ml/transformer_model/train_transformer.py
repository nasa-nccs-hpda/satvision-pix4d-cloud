import argparse
import os
import time

# Must be set BEFORE `import torch` -- torch._inductor's async Triton compile-worker pool
# can be forked/spawned as a side effect of the first torch.compile() call, and those worker
# subprocesses do not reliably pick up an os.environ change made after that point. See the
# audit: this cluster's `module load nvidia` already EXPORTS CC/CXX pointing at NVIDIA's
# `nvc` (confirmed via env inspection), so a plain os.environ.setdefault(...) is a no-op --
# it must be a direct assignment to actually override it. Triton's cuda_utils host-compiler
# build under `nvc` fails intermittently (nvc-Error-Unknown switch: -Wno-psabi); `gcc`
# (confirmed present at /usr/bin/gcc, succeeded every time it was tried) replaces it. Forcing
# single-threaded (in-process) Triton compilation additionally sidesteps the worker-subprocess
# path entirely.
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import torch
import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger

from model_transformer import IPWGTransformer, FastSatRainTemporal, N_TIMESTEPS, N_ABI_CH, STATIC_CH


class EpochTimer(Callback):
    """Records wall-clock duration of each full training epoch. A 50-batch slice under-
    counts one-time persistent_workers startup and mixes it with steady-state throughput;
    per-epoch timing over >=2 real epochs separates the one-time cost (epoch 1, cold
    filesystem cache) from steady-state (epoch 2+, partially warm cache)."""

    def __init__(self):
        self.epoch_times = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_times.append(time.time() - self._t0)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true",
    help="2-epoch smoke test: max_epochs=2, full training epochs (no train batch limit), "
         "limit_val_batches=20. Measures real per-epoch time separately for epoch 1 (cold "
         "filesystem cache) and epoch 2 (partially warm) before committing to the full "
         "200-epoch job.",
)
parser.add_argument(
    "--no-compile", action="store_true",
    help="Skip torch.compile entirely and run eager. Useful to isolate whether a training "
         "instability (e.g. NaN divergence) is specific to the compiled graph.",
)
parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs (default: 200, or 2 with --smoke-test).")
parser.add_argument("--run-name", type=str, default="transformer_v2", help="Checkpoint/log run name (dirpath filename prefix, CSVLogger name). Defaults to v2 -- v1's logs/checkpoints are from the run that diverged to NaN by epoch 2; keep them separate rather than overwrite.")
parser.add_argument("--base-lr", type=float, default=1e-3, help="Base learning rate passed to IPWGTransformer.")
parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient_clip_val for the Trainer.")
parser.add_argument("--fp32-loss", action="store_true", help="Compute the loss in fp32 instead of the bf16-mixed autocast dtype.")
parser.add_argument("--fp32-attn", action="store_true", help="Compute attention (softmax) in fp32 instead of the bf16-mixed autocast dtype.")
parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers (default: min(64, cpu_count)).")
parser.add_argument("--detect-anomaly", action="store_true", help="Enable Trainer(detect_anomaly=True) for pinpointing NaN sources. Significantly slows training -- debugging only.")
args = parser.parse_args()

print("SLURM job initialized.")
PP_ROOT = "/explore/nobackup/projects/pix4dcloud/sanumolu/satrain_ml/satrain_data/preprocessed/satrain/gmi/{split}/l/on_swath"
os.environ["SATRAIN_DATA_PATH"] = "/explore/nobackup/projects/pix4dcloud/sanumolu/"
print("Data Path Set.")

# --- Resource detection (Part C.1): print, then adapt -- never hardcode blind. ---
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {device_name}, total VRAM: {total_vram_gb:.1f} GB")
else:
    print("WARNING: no CUDA device visible.")
cpu_count = os.cpu_count() or 32
print(f"CPU count: {cpu_count}")

# Unlock H100 Tensor Cores for massive speedup
torch.set_float32_matmul_precision("high")
# Fixed 64x64 input every batch -- safe to let cuDNN autotune its fastest conv algorithms.
torch.backends.cudnn.benchmark = True
# NaN-divergence fix: the epoch-2/3 NaN reproduced even under --no-compile and --fp32-attn
# (job 37889585) -- detect_anomaly pinpointed it to ScaledDotProductCudnnAttentionBackward0,
# i.e. the cuDNN SDPA backend's backward kernel itself, not a bf16-precision or compile issue.
# Disable only the cuDNN backend; flash/efficient/math remain available and still fast.
torch.backends.cuda.enable_cudnn_sdp(False)

# Optimal settings for a single H100 GPU (identical to baseline for a fair comparison).
# num_workers is capped at the CPU count actually available rather than assumed.
# Part C.6 diagnostic (jobs 37888442/37888449) measured the real bottleneck as random-access
# I/O against the network filesystem (774ms/sample single-threaded cold-read; augmentation
# itself is only ~3-4ms/sample and GPU forward+backward is only 0.116s/batch) -- num_workers=64
# gave substantially higher throughput than 32 by parallelizing more of these I/O waits.
batch_size = 128
num_workers = args.num_workers if args.num_workers is not None else min(64, cpu_count)

training_data = FastSatRainTemporal(PP_ROOT.format(split="training"), augment=True)
validation_data = FastSatRainTemporal(PP_ROOT.format(split="validation"))

training_loader = DataLoader(
    training_data, shuffle=True, batch_size=batch_size, num_workers=num_workers,
    pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True,
)
validation_loader = DataLoader(
    validation_data, shuffle=False, batch_size=batch_size, num_workers=num_workers,
    pin_memory=True, persistent_workers=True, prefetch_factor=4,
)

print("Data Loaded!")

# Distinct checkpoint/log names so baseline CNN artifacts (and other named runs) are never
# overwritten -- run_name defaults to "transformer_v1" but is overridable via --run-name.
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename=args.run_name + "-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")

csv_logger = CSVLogger("logs/", name=args.run_name)

if __name__ == "__main__":
    (geo_t0, static0), _ = training_data[0]
    print(f"Loader gives geo_t {tuple(geo_t0.shape)}, static {tuple(static0.shape)}")
    assert geo_t0.shape == (N_TIMESTEPS, N_ABI_CH, 64, 64), f"geo_t shape mismatch: {geo_t0.shape}"
    assert static0.shape == (STATIC_CH, 64, 64), f"static shape mismatch: {static0.shape}"

    n_epochs = args.epochs if args.epochs is not None else (2 if args.smoke_test else 200)
    transformer = IPWGTransformer(
        n_epochs=n_epochs, base_lr=args.base_lr, fp32_loss=args.fp32_loss, attn_fp32=args.fp32_attn,
    )

    # torch.compile (Part C.3): applied to the underlying nn.Module, not the LightningModule
    # wrapper, so saved checkpoints stay portable (no "_orig_mod." key prefix). CC/CXX/
    # TORCHINDUCTOR_COMPILE_THREADS are forced at the top of this file, before `import torch`
    # (see comment there for why). We still run an actual pre-flight compiled forward+backward
    # pass before trainer.fit() and fall back to eager on ANY failure, so a bad run never wastes
    # walltime discovering the failure mid-training. --no-compile skips this entirely (used to
    # isolate whether a training instability is specific to the compiled graph).
    original_model = transformer.model
    compiled = False
    if args.no_compile:
        print("--no-compile passed; running eager.")
    elif torch.cuda.is_available():
        try:
            transformer.model = torch.compile(original_model)
            transformer.to("cuda")
            dummy_geo_t = torch.randn(2, N_TIMESTEPS, N_ABI_CH, 64, 64, device="cuda")
            dummy_static = torch.randn(2, STATIC_CH, 64, 64, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = transformer.model(dummy_geo_t, dummy_static)
            (out["surface_precip"].float().sum()
             + out["probability_of_precip"].float().sum()
             + out["probability_of_heavy_precip"].float().sum()).backward()
            compiled = True
        except Exception as e:
            print(f"torch.compile pre-flight failed, falling back to eager: {e}")
            transformer.model = original_model
        transformer.zero_grad(set_to_none=True)
    else:
        print("No CUDA device visible; skipping torch.compile.")
    print(f"torch.compile actually usable this run (verified via real compiled forward+backward): {compiled}")

    epoch_timer = EpochTimer()
    trainer_kwargs = dict(
        max_epochs=transformer.n_epochs,
        precision="bf16-mixed",  # <-- Optimized for H100!
        accelerator="gpu",
        devices=1,
        detect_anomaly=args.detect_anomaly,
        callbacks=[checkpoint_callback, early_stopping, epoch_timer],
        logger=csv_logger,
        gradient_clip_val=args.grad_clip,  # NaN-divergence fix: went nan by epoch 7 without clipping.
    )
    if args.smoke_test:
        # Part C.8: 2-epoch smoke test -- full train epochs (see EpochTimer docstring for why
        # capping at a small batch slice undercounts one-time startup/cache-warming effects).
        trainer_kwargs.update(limit_val_batches=20)

    trainer = L.Trainer(**trainer_kwargs)

    print("Starting Training on H100 (temporal ABI + transformer, 'l' subset)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    trainer.fit(
        model=transformer,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )
    elapsed = time.time() - t_start

    if args.smoke_test:
        print(f"\n=== SMOKE TEST TIMING ({len(training_loader)} batches/epoch, "
              f"batch_size={batch_size}) ===")
        print(f"Total wall time (2 train epochs + validation): {elapsed:.1f}s")
        for i, t in enumerate(epoch_timer.epoch_times, start=1):
            print(f"  epoch {i}: {t:.1f}s ({t / 60:.2f} min), {t / len(training_loader):.3f}s/batch")
        if len(epoch_timer.epoch_times) >= 2:
            warm_epoch_s = epoch_timer.epoch_times[-1]
            print(f"Extrapolated 200-epoch total using steady-state (epoch 2) time: "
                  f"{warm_epoch_s * 200 / 3600:.2f} h (walltime budget: 8h)")
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak VRAM allocated: {peak_gb:.2f} GB")
        print(f"torch.compile actually used this run: {compiled}")
    else:
        print(f"Training complete in {elapsed / 3600:.2f}h! Model saved in the 'checkpoints' directory.")
