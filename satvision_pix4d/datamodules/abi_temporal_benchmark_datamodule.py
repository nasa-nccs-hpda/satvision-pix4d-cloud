"""Synthetic data module usable by both the benchmark and the normal CLI."""
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from satvision_pix4d.datasets.abi_temporal_benchmark_dataset import ABITemporalBenchmarkDataset


class ABITemporalBenchmarkDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        c = self.config
        kwargs = dict(img_size=c.DATA.IMG_SIZE, in_chans=c.MODEL.MAE_VIT.IN_CHANS,
                      num_timesteps=c.BENCHMARK.TIMESTEPS, length=c.DATA.LENGTH,
                      seed=c.SEED, temporal_embeddings=c.DATA.TEMPORAL_COMPONENTS,
                      mean=c.DATA.MEAN, std=c.DATA.STD)
        self.trainset = ABITemporalBenchmarkDataset(
            split="train", fixed_samples=(c.BENCHMARK.FIXED_SAMPLES
                                         if c.BENCHMARK.MODE == "overfit" else 0), **kwargs)
        self.validset = ABITemporalBenchmarkDataset(split="valid", **kwargs)

    def _loader(self, dataset):
        c = self.config
        kwargs = dict(batch_size=c.DATA.BATCH_SIZE, num_workers=c.DATA.NUM_WORKERS,
                      pin_memory=c.DATA.PIN_MEMORY, shuffle=False, drop_last=True)
        if c.DATA.NUM_WORKERS > 0:
            kwargs.update(persistent_workers=c.DATA.PERSISTENT_WORKERS, prefetch_factor=1)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._loader(self.trainset)

    def val_dataloader(self):
        return self._loader(self.validset)
