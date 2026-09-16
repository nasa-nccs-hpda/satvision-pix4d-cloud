"""Synthetic MAE training benchmark: python -m satvision_pix4d.benchmark --help."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

import lightning.pytorch as pl
import torch

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.datamodules.abi_temporal_benchmark_datamodule import ABITemporalBenchmarkDataModule
from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.optimizers.build import build_optimizer
from satvision_pix4d.models.utils.device_state import move_non_parameter_state


class SyntheticMAEBenchmark(pl.LightningModule):
    """Build inside Lightning's sharded initialization context for ZeRO-3."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = None
        self.register_buffer("mean", torch.tensor(config.DATA.MEAN).view(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(config.DATA.STD).view(1, 1, -1, 1, 1), persistent=False)

    def configure_model(self):
        if self.model is None:
            self.model = build_satmae_model(self.config)
        if self._trainer is not None:
            move_non_parameter_state(self, self.trainer.strategy.root_device)

    def forward(self, raw, timestamps, permutation=None):
        z = (raw.float() - self.mean) / self.std
        return self.model(z, timestamps, self.config.DATA.MASK_RATIO, mask=permutation)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self(*batch)
        # Reduce before raising so every distributed rank exits together.
        finite = self.trainer.strategy.reduce(torch.isfinite(loss).float(), reduce_op="min")
        if finite.item() != 1:
            raise FloatingPointError("Non-finite training loss")
        return {"loss": loss, "raw_loss": loss.detach()}

    def configure_optimizers(self):
        return build_optimizer(self.config, self.model, is_pretrain=True)


class BenchmarkReport(pl.Callback):
    def __init__(self, config, output, metadata):
        self.config, self.output, self.metadata = config, Path(output), metadata
        self.records = []
        self.last_step = 0
        self.loss_sum = 0.0
        self.microbatches = 0
        self.samples = 0

    @staticmethod
    def synchronize(module):
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)

    def write(self, name, value):
        path = self.output / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.output.mkdir(parents=True, exist_ok=True)
            if (self.output / "metadata.json").exists():
                raise FileExistsError(f"Choose a fresh output directory: {self.output}")
            self.write("metadata.json", dict(self.metadata, world_size=trainer.world_size,
                       device=str(pl_module.device),
                       device_name=(torch.cuda.get_device_name(pl_module.device)
                                    if pl_module.device.type == "cuda" else "CPU")))
            (self.output / "config.yaml").write_text(self.config.dump())

    @torch.no_grad()
    def probe(self, trainer, module, dataset):
        """Use identical samples and explicit masks before/after optimization."""
        was_training = module.training
        module.eval()
        losses, mean_baselines = [], []
        for i in range(self.config.BENCHMARK.PROBE_SAMPLES):
            raw, timestamps = dataset[i]
            raw = raw.unsqueeze(0).to(module.device)
            timestamps = timestamps.unsqueeze(0).to(module.device)
            tokens = self.config.BENCHMARK.TIMESTEPS * (self.config.DATA.IMG_SIZE // self.config.MODEL.MAE_VIT.PATCH_SIZE) ** 2
            generator = torch.Generator().manual_seed(self.config.SEED + 1_000_000 + i)
            if self.config.MODEL.MAE_VIT.SAME_MASK:
                spatial = tokens // self.config.BENCHMARK.TIMESTEPS
                order = torch.randperm(spatial, generator=generator)
                keep = int(spatial * (1 - self.config.DATA.MASK_RATIO))
                offsets = torch.arange(self.config.BENCHMARK.TIMESTEPS)[:, None] * spatial
                permutation = torch.cat([(order[:keep] + offsets).flatten(),
                                         (order[keep:] + offsets).flatten()])
            else:
                permutation = torch.randperm(tokens, generator=generator)
            with trainer.precision_plugin.forward_context():
                loss, pred, mask = module(raw, timestamps, permutation.unsqueeze(0).to(module.device))
                z = (raw.float() - module.mean) / module.std
                baseline = module.model.forward_loss(z, torch.zeros_like(pred), mask)
            losses.append(loss.float())
            mean_baselines.append(baseline.float())
        if trainer.is_global_zero:
            from satvision_pix4d.models.utils.reconstruction_logging import reconstruction_grid
            reconstruction = module.model.unpatchify(pred, raw.shape[1], raw.shape[-2], raw.shape[-1])
            pixel_mask = module.model.tokens_to_pixel_mask(mask, raw.shape[1], raw.shape[-2], raw.shape[-1])
            merged = torch.where(pixel_mask.bool(), reconstruction * module.std + module.mean, raw)
            split = "held_out" if dataset is trainer.datamodule.validset else "fixed_probe"
            trainer.logger.experiment.add_image(f"benchmark/{split}_target_masked_reconstruction",
                reconstruction_grid(raw, merged, pixel_mask), trainer.global_step)
        module.train(was_training)
        values = torch.stack([torch.stack(losses).mean(), torch.stack(mean_baselines).mean()])
        values = trainer.strategy.reduce(values, reduce_op="mean")
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite probe loss")
        return {"loss": values[0].item(), "channel_mean_baseline_loss": values[1].item()}

    def on_train_start(self, trainer, pl_module):
        self.initial = self.probe(trainer, pl_module, trainer.datamodule.trainset)
        if trainer.is_global_zero:
            trainer.logger.experiment.add_scalar("benchmark/fixed_probe_loss", self.initial["loss"], 0)
        self.synchronize(pl_module)
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.tick = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.loss_sum += outputs["raw_loss"].detach().float().item()
        self.microbatches += 1
        self.samples += batch[0].shape[0]
        step = trainer.global_step
        if step == self.last_step:
            return
        self.synchronize(pl_module)
        elapsed = time.perf_counter() - self.tick
        # Slowest rank defines throughput. Each record covers an optimizer update,
        # including its accumulated microbatches and data-loading/transfer time.
        elapsed = trainer.strategy.reduce(torch.tensor(elapsed, device=pl_module.device), reduce_op="max").item()
        loss = trainer.strategy.reduce(torch.tensor(self.loss_sum / self.microbatches, device=pl_module.device), reduce_op="mean").item()
        samples = trainer.strategy.reduce(torch.tensor(float(self.samples), device=pl_module.device), reduce_op="sum").item()
        row = dict(step=step, loss=loss, seconds=elapsed, samples=int(samples),
                   samples_per_second=samples / elapsed,
                   frames_per_second=samples * self.config.BENCHMARK.TIMESTEPS / elapsed,
                   measured=step > self.config.BENCHMARK.WARMUP_STEPS)
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite recorded training loss")
        self.records.append(row)
        if trainer.is_global_zero:
            for name in ("loss", "seconds", "samples_per_second", "frames_per_second"):
                trainer.logger.experiment.add_scalar(f"benchmark/{name}", row[name], step)
            with (self.output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 10 == 0:
                print(f"step={step} loss={loss:.6f} seconds={elapsed:.3f} samples/s={samples/elapsed:.3f}", flush=True)
        if step == self.config.BENCHMARK.WARMUP_STEPS and pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.last_step, self.loss_sum, self.microbatches, self.samples = step, 0.0, 0, 0
        self.tick = time.perf_counter()

    def on_train_end(self, trainer, pl_module):
        self.synchronize(pl_module)
        peak = torch.tensor([
            torch.cuda.max_memory_allocated(pl_module.device) if pl_module.device.type == "cuda" else 0,
            torch.cuda.max_memory_reserved(pl_module.device) if pl_module.device.type == "cuda" else 0,
        ], device=pl_module.device, dtype=torch.float64)
        peak = trainer.strategy.reduce(peak, reduce_op="max").tolist()
        final = self.probe(trainer, pl_module, trainer.datamodule.trainset)
        held_out = self.probe(trainer, pl_module, trainer.datamodule.validset)
        reduction = (self.initial["loss"] - final["loss"]) / max(abs(self.initial["loss"]), 1e-12)
        measured = [r for r in self.records if r["measured"]]
        seconds = sum(r["seconds"] for r in measured)
        samples = sum(r["samples"] for r in measured)
        passed = reduction >= self.config.BENCHMARK.MIN_RELATIVE_IMPROVEMENT
        self.summary = dict(
            status=("passed" if passed else "convergence_not_met")
                   if self.config.BENCHMARK.MODE == "overfit" else "completed",
            mode=self.config.BENCHMARK.MODE, optimizer_steps=trainer.global_step,
            measured_steps=len(measured), initial_probe=self.initial, final_probe=final,
            held_out_probe=held_out, relative_loss_reduction=reduction,
            minimum_relative_improvement=self.config.BENCHMARK.MIN_RELATIVE_IMPROVEMENT,
            samples_per_second=samples / seconds, frames_per_second=samples * self.config.BENCHMARK.TIMESTEPS / seconds,
            mean_step_seconds=seconds / len(measured),
            median_step_seconds=statistics.median(r["seconds"] for r in measured),
            peak_allocated_gib=peak[0] / 2**30 if pl_module.device.type == "cuda" else None,
            peak_reserved_gib=peak[1] / 2**30 if pl_module.device.type == "cuda" else None,
            timing="Post-warmup training updates including data generation/transfer; probes and report writes excluded",
        )
        if trainer.is_global_zero:
            trainer.logger.experiment.add_scalar("benchmark/fixed_probe_loss", final["loss"], trainer.global_step)
            self.write("summary.json", self.summary)
            for name, value in {"initial_probe_loss": self.initial["loss"], "final_probe_loss": final["loss"],
                                "held_out_loss": held_out["loss"], "relative_loss_reduction": reduction}.items():
                trainer.logger.experiment.add_scalar(f"benchmark/{name}", value, trainer.global_step)
            print(json.dumps(self.summary, indent=2), flush=True)

    def on_exception(self, trainer, pl_module, exception):
        if trainer.is_global_zero and self.output.exists():
            self.write("failure.json", dict(status="failed", exception=type(exception).__name__,
                       message=str(exception), completed_steps=trainer.global_step))


def prepare_config(args):
    root = Path(__file__).resolve().parents[1]
    model = args.model.lower()
    model = "330m" if model == "300m" else model
    config = _C.clone()
    _update_config_from_file(config, str(root / "configs" / "benchmark" / f"{model}.yaml"))
    config.defrost()
    config.BENCHMARK.MODE = args.mode
    config.BENCHMARK.STEPS = args.steps
    config.BENCHMARK.WARMUP_STEPS = args.warmup_steps
    config.BENCHMARK.FIXED_SAMPLES = args.fixed_samples
    config.BENCHMARK.PROBE_SAMPLES = args.probe_samples
    config.BENCHMARK.MIN_RELATIVE_IMPROVEMENT = args.min_relative_improvement
    config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    config.DATA.BATCH_SIZE = args.batch_size
    config.DATA.NUM_WORKERS = args.workers
    config.DATA.IMG_SIZE = args.image_size
    config.SEED = args.seed
    if args.lr is not None:
        config.TRAIN.BASE_LR = args.lr
    if args.precision is not None:
        config.PRECISION = args.precision
    if args.tiny:
        config.MODEL.NAME = "satvision_tiny_test_only"
        config.MODEL.MAE_VIT.SIZE = "custom"
        config.MODEL.MAE_VIT.EMBED_DIM = 32
        config.MODEL.MAE_VIT.DEPTHS = 1
        config.MODEL.MAE_VIT.NUM_HEADS = 4
        config.MODEL.MAE_VIT.DECODER_EMBED_DIM = 32
        config.MODEL.MAE_VIT.DECODER_DEPTH = 1
        config.MODEL.MAE_VIT.DECODER_NUM_HEADS = 4
    # Avoid a short final accumulation group; keep dataset length divisible by
    # batch_size * world_size * accumulation, even when training crosses epochs.
    group = args.devices * args.num_nodes * args.batch_size * args.accumulation_steps
    config.DATA.LENGTH = max(256, args.probe_samples, args.fixed_samples)
    config.DATA.LENGTH = math.ceil(config.DATA.LENGTH / group) * group
    config.freeze()
    return config


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str.upper, choices=["300M", "330M", "700M", "3B"],
                   default="330M", help="Model size (case-insensitive); 300M is an alias for 330M")
    p.add_argument("--mode", choices=["throughput", "overfit"], default="throughput")
    p.add_argument("--output", type=Path, required=True, help="Fresh artifact directory")
    p.add_argument("--steps", type=int, default=100, help="Total optimizer updates, including timing warmup")
    p.add_argument("--warmup-steps", type=int, default=10, help="Updates excluded from throughput statistics")
    p.add_argument("--batch-size", type=int, default=1, help="Sequences per GPU/microbatch")
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--devices", type=int, default=1, help="Devices per node")
    p.add_argument("--num-nodes", type=int, default=1)
    p.add_argument("--strategy", choices=["auto", "ddp", "deepspeed"], default="deepspeed")
    p.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--precision", choices=["32-true", "bf16-mixed", "16-mixed"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--fixed-samples", type=int, default=4)
    p.add_argument("--probe-samples", type=int, default=1)
    p.add_argument("--min-relative-improvement", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="Meta-device parameter/shape check, no training")
    p.add_argument("--tiny", action="store_true", help="Small test model; NOT a capacity benchmark")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.steps <= args.warmup_steps or args.warmup_steps < 0
            or min(args.batch_size, args.accumulation_steps, args.devices, args.num_nodes,
                   args.fixed_samples, args.probe_samples) < 1
            or args.workers < 0 or args.image_size < 112 or args.image_size % 16
            or not 0 <= args.min_relative_improvement < 1
            or (args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0))):
        p.error("Require steps>warmup>=0, positive sizes/counts/LR, image size>=112 divisible by 16, and improvement in [0,1)")
    config = prepare_config(args)
    pl.seed_everything(config.SEED, workers=True)
    with torch.device("meta"):
        model = build_satmae_model(config)
    metadata = dict(model=config.MODEL.NAME, parameters=sum(p.numel() for p in model.parameters()),
                    shape=[args.batch_size, 7, 16, args.image_size, args.image_size],
                    tokens_per_sequence=7 * (args.image_size // 16) ** 2,
                    precision=config.PRECISION, strategy=args.strategy,
                    batch_size_per_device=args.batch_size, accumulation_steps=args.accumulation_steps,
                    expected_effective_batch_size=args.batch_size * args.accumulation_steps * args.devices * args.num_nodes,
                    seed=config.SEED, tiny=args.tiny, torch_version=torch.__version__, lightning_version=pl.__version__)
    del model
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return 0
    if args.accelerator == "cpu" and args.strategy == "deepspeed":
        p.error("Use --strategy auto or ddp for CPU checks")
    if (args.output / "metadata.json").exists():
        p.error(f"Choose a fresh output directory: {args.output}")
    strategy = args.strategy
    if strategy == "deepspeed":
        from lightning.pytorch.strategies import DeepSpeedStrategy
        strategy = DeepSpeedStrategy(stage=3, reduce_bucket_size=50_000_000,
                                     allgather_bucket_size=50_000_000)
    callback = BenchmarkReport(config, args.output, metadata)
    from lightning.pytorch.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=str(args.output), name="tensorboard", version="")
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes,
                         strategy=strategy, precision=config.PRECISION,
                         max_steps=config.BENCHMARK.STEPS, max_epochs=-1,
                         accumulate_grad_batches=config.TRAIN.ACCUMULATION_STEPS,
                         gradient_clip_val=config.TRAIN.CLIP_GRAD,
                         callbacks=[callback], logger=logger, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False,
                         num_sanity_val_steps=0, limit_val_batches=0)
    trainer.fit(SyntheticMAEBenchmark(config), datamodule=ABITemporalBenchmarkDataModule(config))
    return 2 if callback.summary["status"] == "convergence_not_met" else 0


if __name__ == "__main__":
    raise SystemExit(main())
