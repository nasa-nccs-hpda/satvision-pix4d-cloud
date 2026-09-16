"""Deterministic, learnable synthetic ABI sequences; no files are required."""
from datetime import datetime, timedelta

import torch
from torch.utils.data import Dataset


class ABITemporalBenchmarkDataset(Dataset):
    def __init__(self, data_paths=(), split="train", img_size=512, in_chans=16,
                 num_timesteps=7, transform=None, length=256, seed=42,
                 fixed_samples=0, temporal_embeddings=("year", "month", "hour"),
                 mean=None, std=None):
        if min(img_size, in_chans, num_timesteps, length) < 1 or fixed_samples < 0:
            raise ValueError("Synthetic dimensions/length must be positive; fixed_samples >= 0")
        self.img_size, self.in_chans = img_size, in_chans
        self.num_timesteps, self.length = num_timesteps, length
        self.seed = seed + (0 if split == "train" else 10_000_000)
        self.fixed_samples = fixed_samples
        self.temporal_embeddings = tuple(temporal_embeddings)
        self.transform = transform
        self.mean = torch.tensor(mean if mean is not None else [0.] * in_chans).view(1, -1, 1, 1)
        self.std = torch.tensor(std if std is not None else [1.] * in_chans).view(1, -1, 1, 1)
        if self.mean.shape[1] != in_chans or self.std.shape[1] != in_chans or (self.std <= 0).any():
            raise ValueError("Synthetic channel statistics must match in_chans with positive std")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not 0 <= idx < self.length:
            raise IndexError(idx)
        sample_id = idx % self.fixed_samples if self.fixed_samples else idx
        generator = torch.Generator().manual_seed(self.seed + sample_id)
        phase, cx, cy, vx, vy = torch.rand(5, generator=generator).tolist()
        axis = torch.linspace(-1, 1, self.img_size)
        y, x = axis.view(1, -1, 1), axis.view(1, 1, -1)
        time = torch.arange(self.num_timesteps).view(-1, 1, 1).float()
        # Smooth moving structures with channel-specific responses. Unlike fresh
        # white noise, masked pixels can be inferred from their visible context.
        x_shift = x - (cx - 0.5) - (vx - 0.5) * time * 0.08
        y_shift = y - (cy - 0.5) - (vy - 0.5) * time * 0.08
        cloud = torch.exp(-(x_shift.square() + y_shift.square()) / 0.18)
        wave = torch.sin(3 * x_shift + 4 * y_shift + phase * 6.283185)
        band = torch.linspace(0, 1, self.in_chans).view(1, -1, 1, 1)
        z = (0.4 + 0.4 * band) * wave[:, None] + (1.2 - 0.4 * band) * cloud[:, None] - 0.3
        raw = (z * self.std + self.mean).contiguous().float()
        if self.transform is not None:
            raw = torch.stack([self.transform(frame.permute(1, 2, 0).numpy()) for frame in raw])
        start = datetime(2020, 1, 1) + timedelta(minutes=10 * sample_id)
        timestamps = []
        for step in range(self.num_timesteps):
            dt = start + timedelta(minutes=10 * step)
            values = dict(year=dt.year - 2000, month=dt.month - 1, day=dt.day - 1,
                          hour=dt.hour, minute=dt.minute)
            timestamps.append([values[field] for field in self.temporal_embeddings])
        return raw, torch.tensor(timestamps, dtype=torch.int32)
