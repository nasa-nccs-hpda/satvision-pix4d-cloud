"""Epoch-level wall time and GPU memory diagnostics for TensorBoard."""
import time

from lightning.pytorch import Callback
import torch


class EpochPerformanceMonitor(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self.start = time.perf_counter()
        self.samples = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.samples += batch[0].shape[0]

    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        elapsed = trainer.strategy.reduce(torch.tensor(time.perf_counter() - self.start,
            device=pl_module.device), reduce_op="max")
        samples = trainer.strategy.reduce(torch.tensor(float(self.samples), device=pl_module.device), reduce_op="sum")
        pl_module.log("performance/epoch_wall_seconds", elapsed, sync_dist=False)
        pl_module.log("performance/samples_per_wall_second", samples / elapsed, sync_dist=False)
        if pl_module.device.type == "cuda":
            peak = trainer.strategy.reduce(torch.tensor(torch.cuda.max_memory_allocated(pl_module.device) / 2**30,
                device=pl_module.device), reduce_op="max")
            pl_module.log("performance/peak_allocated_gib", peak, sync_dist=False)
