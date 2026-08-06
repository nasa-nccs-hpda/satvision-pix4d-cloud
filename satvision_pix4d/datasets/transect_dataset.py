"""
Dataset and DataModule for 1D CloudSat/ABI transect data.

The updatedShape preprocessing pipeline produces `.npz` files where each
sample contains:

    ABI/chip              (7, 512, 1, 16)   ABI radiance at 7 temporal
                                            offsets for 512 CloudSat
                                            footprints × 1 nearest ABI
                                            pixel × 16 channels.

    CloudSat/cloud_class  (512, 40)         Cloud-type labels per footprint
                                            in 40 vertical bins (500 m
                                            intervals up to 20 km).
                                            Values: 0=clear, 1–8=cloud type.

    CloudSat/cloud_binary_mask (512, 40)    Binary cloud/no-cloud mask.

The adapted SatMAE ViT temporal encoder expects input shaped
``(B, C, T, H, W)`` = ``(B, 16, 7, 512, 1)``.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # type: ignore[no-redef]

LOG = logging.getLogger(__name__)

# ─── NPZ key discovery helpers ────────────────────────────────────────────────
_ABI_CHIP_KEYS = ("ABI/chip", "chip")
_CLOUD_CLASS_KEYS = ("CloudSat/cloud_class",)
_CLOUD_BINARY_KEYS = ("CloudSat/cloud_binary_mask",)


def _resolve_key(
    npz_file: np.lib.npyio.NpzFile,
    candidates: tuple[str, ...],
    name: str,
) -> str:
    """Return the first key from *candidates* found in *npz_file*."""
    for key in candidates:
        if key in npz_file.files:
            return key
    raise KeyError(
        f"Cannot find {name} array; tried {candidates}, "
        f"available keys: {npz_file.files}"
    )


# ─── Normalization ─────────────────────────────────────────────────────────────

class TransectMinMaxScale:
    """Per-channel min-max normalization using statistics from the
    updatedModelSummer2026 pipeline.

    Expects an array whose *last* axis is the 16-channel dimension.
    Output is clipped to [0, 1] and NaNs are replaced with 0.
    """

    def __init__(self) -> None:
        # Taken from examples/abi_3d_reconstruction/updatedModelSummer2026/transforms.py
        self.min_vals = np.array(
            [-25.936647, -20.289911, -12.037643, -4.522368,
             -3.059614,  -0.960951, -0.037600,   0.144772,
             -0.823600,  -0.956100, -1.302200,  -1.539400,
             -1.644300,   5.903100, -1.755800,  -5.239200],
            dtype=np.float32,
        )
        self.max_vals = np.array(
            [804.036072, 628.987244, 373.166992, 140.193420,
              94.848030,  29.789471,  25.589600,   9.452010,
              45.291401,  81.092896, 135.264603, 109.844803,
             185.569885, 200.902390, 214.301407, 174.692612],
            dtype=np.float32,
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        range_vals = self.max_vals - self.min_vals
        range_vals[range_vals == 0] = 1.0
        img = (img - self.min_vals) / range_vals
        img = np.clip(img, 0.0, 1.0)
        img = np.nan_to_num(img, nan=0.0)
        return img


class TransectPerChannelMinMaxScale:
    """Per-sample, per-channel min-max normalization.

    Each channel is independently scaled to [0, 1] based on its own
    min/max within the sample.  This mirrors the per-channel approach
    used in the CloudHeight notebook ``CloudSatBinaryDataset``.
    """

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Normalize *img* in-place. Last axis must be channels."""
        for c in range(img.shape[-1]):
            ch = img[..., c]
            mn, mx = float(np.nanmin(ch)), float(np.nanmax(ch))
            if mx > mn:
                img[..., c] = (ch - mn) / (mx - mn)
            else:
                img[..., c] = 0.0
        img = np.nan_to_num(img, nan=0.0)
        return img


# ─── Dataset ───────────────────────────────────────────────────────────────────

class TransectDataset(Dataset):
    """PyTorch Dataset for 1D CloudSat/ABI transect samples.

    Each ``__getitem__`` call returns a dictionary::

        {
            "chip":  (C, T, H, W) float32 tensor, e.g. (16, 7, 512, 1),
            "mask":  (H, num_bins) int64 tensor, e.g. (512, 40),
            "path":  str — source file path,
        }

    Parameters
    ----------
    file_paths : list[str]
        Paths to ``.npz`` files produced by the ``updatedShape`` pipeline.
    label_key : str
        Which CloudSat label array to load. ``"cloud_class"`` for the 9-class
        segmentation task, ``"cloud_binary_mask"`` for binary cloud detection.
    normalization : {"global", "per_sample", None}
        - ``"global"``     — use ``TransectMinMaxScale`` with pre-computed
                             per-channel statistics.
        - ``"per_sample"`` — use ``TransectPerChannelMinMaxScale`` which
                             normalizes each sample independently.
        - ``None``         — no normalization (raw radiance values).
    """

    # Accepted label_key values → NPZ key lookup tuples
    _LABEL_KEY_MAP = {
        "cloud_class":       _CLOUD_CLASS_KEYS,
        "cloud_binary_mask": _CLOUD_BINARY_KEYS,
    }

    def __init__(
        self,
        file_paths: list[str],
        label_key: str = "cloud_class",
        normalization: str | None = "global",
    ) -> None:
        self.file_paths = file_paths
        self.label_key = label_key

        if label_key not in self._LABEL_KEY_MAP:
            raise ValueError(
                f"Unknown label_key {label_key!r}; "
                f"choose from {list(self._LABEL_KEY_MAP)}"
            )
        self._label_candidates = self._LABEL_KEY_MAP[label_key]

        if normalization == "global":
            self.transform = TransectMinMaxScale()
        elif normalization == "per_sample":
            self.transform = TransectPerChannelMinMaxScale()
        elif normalization is None:
            self.transform = None
        else:
            raise ValueError(
                f"Unknown normalization {normalization!r}; "
                f"choose from 'global', 'per_sample', or None"
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, object] | None:
        path = self.file_paths[idx]
        try:
            with np.load(path, allow_pickle=True) as data:
                # ── ABI chip ──────────────────────────────────────────
                chip_key = _resolve_key(data, _ABI_CHIP_KEYS, "ABI chip")
                chip = data[chip_key].astype(np.float32)

                # The pipeline writer may output (7, 512, 16) or
                # (7, 512, 1, 16).  Normalise to (T, H, W, C).
                if chip.ndim == 3:
                    # (T, H, C) → (T, H, 1, C)
                    chip = chip[:, :, np.newaxis, :]
                # Now chip is (T, H, W, C) = (7, 512, 1, 16)

                # Normalisation operates on the channels (last axis).
                if self.transform is not None:
                    chip = self.transform(chip)

                # Permute to (C, T, H, W) for the SatMAE ViT encoder.
                # (T, H, W, C) → (C, T, H, W) = (16, 7, 512, 1)
                chip = np.transpose(chip, (3, 0, 1, 2))

                # ── Label mask ────────────────────────────────────────
                label_key = _resolve_key(
                    data, self._label_candidates, self.label_key
                )
                mask = data[label_key]
                if self.label_key == "cloud_binary_mask":
                    mask = mask.astype(np.float32)
                else:
                    mask = mask.astype(np.int64)

            return {
                "chip": torch.from_numpy(chip),
                "mask": torch.from_numpy(mask),
                "path": path,
            }

        except Exception as exc:
            LOG.warning("Error loading %s: %s", path, exc)
            return None


# ─── Lightning DataModule ──────────────────────────────────────────────────────

class TransectDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for 1D transect finetuning.

    Parameters
    ----------
    data_dir : str | list[str]
        Directory (or list of directories) containing ``.npz`` files.
        Subdirectories one level deep are also searched
        (e.g. ``data_dir/jan_chips/*.npz``).
    label_key : str
        ``"cloud_class"`` or ``"cloud_binary_mask"``.
    normalization : str | None
        Forwarded to :class:`TransectDataset`.
    batch_size : int
        Batch size for all data loaders.
    num_workers : int
        Number of data-loading worker processes.
    train_val_test_split : tuple[float, float, float]
        Fraction of files for train / val / test.
    """

    def __init__(
        self,
        data_dir: str | list[str],
        label_key: str = "cloud_class",
        normalization: str | None = "global",
        batch_size: int = 16,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dirs = data_dir
        self.label_key = label_key
        self.normalization = normalization
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split

    # ── File discovery ────────────────────────────────────────────────

    @staticmethod
    def _discover_files(dirs: list[str]) -> list[str]:
        """Find ``.npz`` files in each directory and one level of subdirs."""
        files: list[str] = []
        for d in dirs:
            files.extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
            files.extend(sorted(glob.glob(os.path.join(d, "*", "*.npz"))))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    # ── Lightning hooks ───────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        file_paths = self._discover_files(self.data_dirs)
        if not file_paths:
            raise RuntimeError(
                f"No .npz files found in {self.data_dirs}"
            )

        n = len(file_paths)
        n_train = int(n * self.train_val_test_split[0])
        n_val = int(n * self.train_val_test_split[1])

        train_paths = file_paths[:n_train]
        val_paths = file_paths[n_train: n_train + n_val]
        test_paths = file_paths[n_train + n_val:]

        ds_kwargs = dict(
            label_key=self.label_key,
            normalization=self.normalization,
        )
        self.train_dataset = TransectDataset(train_paths, **ds_kwargs)
        self.val_dataset = TransectDataset(val_paths, **ds_kwargs)
        self.test_dataset = TransectDataset(test_paths, **ds_kwargs)

        LOG.info(
            "TransectDataModule: %d train / %d val / %d test files",
            len(train_paths), len(val_paths), len(test_paths),
        )

    @staticmethod
    def _collate_fn(batch: list[dict | None]) -> dict | None:
        """Drop ``None`` samples (from load errors) before collating."""
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
