
import os
import glob
import torch
import logging
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from satvision_pix4d.transforms.min_max import PerTileMinMaxNormalize
from satvision_pix4d.transforms.z_score import PerChannelStandardize


CHIP_KEYS = ("chip", "chips", "rad", "data", "image", "images", "arr_0")
TIMESTAMP_KEYS = ("timestamps", "timestamp", "times", "time", "t")


class ABITemporalDataset(Dataset):
    """
    ABITemporalDataset for temporal ABI chips.

    Supported inputs are .zarr directories shaped [time, band, y, x] and
    .npy/.npz chip files shaped either [time, band, y, x] or
    [time, y, x, band].
    """

    def __init__(
        self,
        data_paths: list,
        img_size: int = 512,
        in_chans: int = 16,
        temporal_embeddings: list = None,
        transform=None,
        require_timestamps=False,
        num_timesteps=0,
    ):
        self.min_year = 2000
        self.require_timestamps = require_timestamps
        self.num_timesteps = num_timesteps
        self._warned_missing_timestamps = False
        self.img_size = img_size
        self.in_chans = in_chans
        
        self.transform = transform

        if temporal_embeddings is None:
            self.temporal_embeddings = ["year", "month", "hour"]
        else:
            self.temporal_embeddings = temporal_embeddings

        self.files = self._discover_files(data_paths)
        self.samples = self._index_samples(self.files)

        if not self.samples:
            raise RuntimeError(
                "No ABI temporal chips found. Expected .zarr, .npy, or .npz "
                "files in DATA.TRAIN_DATA_PATHS / DATA.VAL_DATA_PATHS."
            )

        logging.info(f"Loaded {len(self.samples)} samples from {len(self.files)} files.")

    def __len__(self):
        return len(self.samples)

    def _discover_files(self, data_paths):
        files = []
        for p in data_paths:
            if os.path.isfile(p) and p.endswith((".npy", ".npz")):
                files.append(p)
                continue
            if os.path.isdir(p) and p.endswith(".zarr"):
                files.append(p)
                continue
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.zarr"))))
                files.extend(sorted(glob.glob(os.path.join(p, "*.npy"))))
                files.extend(sorted(glob.glob(os.path.join(p, "*.npz"))))
        return sorted(set(p for p in files if not p.endswith(".timestamps.npy")))

    def _numpy_chip_shape(self, path):
        loaded = np.load(path, allow_pickle=False, mmap_mode="r" if path.endswith(".npy") else None)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            chip_key = next((key for key in CHIP_KEYS if key in loaded.files), None)
            if chip_key is None:
                loaded.close()
                raise ValueError(
                    f"{path} does not contain a chip array. Expected one of {CHIP_KEYS}; "
                    f"found {loaded.files}."
                )
            shape = loaded[chip_key].shape
            loaded.close()
            return shape
        return loaded.shape

    def _index_samples(self, files):
        samples = []
        for path in files:
            if path.endswith(".zarr"):
                samples.append((path, None))
                continue

            shape = self._numpy_chip_shape(path)
            if len(shape) == 5:
                samples.extend((path, i) for i in range(shape[0]))
            elif len(shape) == 4:
                samples.append((path, None))
            else:
                raise ValueError(f"Expected a 4D or 5D temporal chip in {path}, got shape {shape}")
        return samples

    def _coerce_chip_layout(self, array, path):
        rad = np.asarray(array, dtype=np.float32)
        if rad.ndim != 4:
            raise ValueError(f"Expected a 4D temporal chip in {path}, got shape {rad.shape}")

        if rad.shape[1] == self.in_chans:
            pass
        elif rad.shape[-1] == self.in_chans:
            rad = np.transpose(rad, (0, 3, 1, 2))
        else:
            raise ValueError(
                f"Could not identify channel axis for {path}; expected {self.in_chans} "
                f"channels in shape {rad.shape}."
            )

        if rad.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected chip size ({self.img_size}, {self.img_size}) in {path}, "
                f"got {rad.shape[-2:]}."
            )

        if not np.isfinite(rad).all():
            raise ValueError(f"Non-finite pixels in {path}; clean or mask missing observations before training")
        if rad.shape[0] == 0 or (self.num_timesteps and rad.shape[0] != self.num_timesteps):
            raise ValueError(f"Expected {self.num_timesteps or 'nonzero'} timesteps in {path}, got {rad.shape[0]}")
        return np.array(rad, dtype=np.float32, copy=True, order="C")

    def _parse_timestamp_array(self, values):
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == len(self.temporal_embeddings) and np.issubdtype(arr.dtype, np.number):
            if not np.isfinite(arr).all():
                raise ValueError("Timestamp components must be finite")
            return arr.astype(np.int32)
        if arr.ndim != 1:
            raise ValueError("Timestamps must be (T,) dates/hours or (T,K) configured numeric components")
        if np.issubdtype(arr.dtype, np.number):
            if not np.isfinite(arr).all():
                raise ValueError("Timestamp hours must be finite")
            emb_map = {
                "year": np.zeros(arr.shape[0], dtype=np.int32),
                "month": np.zeros(arr.shape[0], dtype=np.int32),
                "day": np.zeros(arr.shape[0], dtype=np.int32),
                "hour": arr.astype(np.int32),
                "minute": np.zeros(arr.shape[0], dtype=np.int32),
            }
            return np.stack([emb_map[k] for k in self.temporal_embeddings], axis=1).astype(np.int32)

        pd_timestamps = pd.to_datetime(arr, utc=True)
        if pd_timestamps.isna().any():
            raise ValueError("Timestamps cannot contain NaT")
        emb_map = {
            "year": pd_timestamps.year - self.min_year,
            "month": pd_timestamps.month - 1,
            "day": pd_timestamps.day - 1,
            "hour": pd_timestamps.hour,
            "minute": pd_timestamps.minute
        }
        return np.stack([emb_map[k] for k in self.temporal_embeddings], axis=1).astype(np.int32)

    def _fallback_timestamps(self, n_times):
        if self.require_timestamps:
            raise ValueError("Missing timestamps: use NPZ timestamps or a <chip>.timestamps.npy sidecar")
        if not self._warned_missing_timestamps:
            logging.warning("Missing observation times: using synthetic hourly timestamps; not suitable for temporal pretraining")
            self._warned_missing_timestamps = True
        emb_map = {
            "year": np.zeros(n_times, dtype=np.int32),
            "month": np.zeros(n_times, dtype=np.int32),
            "day": np.zeros(n_times, dtype=np.int32),
            "hour": np.arange(n_times, dtype=np.int32),
            "minute": np.zeros(n_times, dtype=np.int32),
        }
        return np.stack([emb_map[k] for k in self.temporal_embeddings], axis=1).astype(np.int32)

    def _load_zarr(self, path):
        ds = xr.open_zarr(path)
        rad = ds["__xarray_dataarray_variable__"].values
        rad = self._coerce_chip_layout(rad, path)

        if "t" in ds:
            timestamps = self._parse_timestamp_array(ds["t"].values)
        else:
            timestamps = self._fallback_timestamps(rad.shape[0])
        ds.close()
        return rad, timestamps

    def _select_timestamps(self, values, sample_idx, n_samples, n_times):
        values = np.asarray(values)
        if sample_idx is None or values.ndim == 1:
            return values
        # Shared numeric (T,K) metadata vs per-sample (N,T) dates/hours.
        shared = (values.ndim == 2 and values.shape == (n_times, len(self.temporal_embeddings))
                  and np.issubdtype(values.dtype, np.number))
        batched = ((values.ndim == 3 and values.shape[:2] == (n_samples, n_times))
                   or (values.ndim == 2 and values.shape == (n_samples, n_times)))
        if shared and batched:
            raise ValueError("Ambiguous timestamps; use explicit (N,T,K) component arrays")
        return values[sample_idx] if batched else values

    def _load_numpy_chip(self, path, sample_idx=None):
        loaded = np.load(path, allow_pickle=False, mmap_mode="r" if path.endswith(".npy") else None)
        timestamps = None
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                chip_key = next((key for key in CHIP_KEYS if key in loaded.files), None)
                if chip_key is None:
                    raise ValueError(f"No chip array in {path}; expected one of {CHIP_KEYS}")
                all_rad = loaded[chip_key]
                n_samples = all_rad.shape[0] if sample_idx is not None else 1
                rad = all_rad[sample_idx] if sample_idx is not None else all_rad
                ts_key = next((key for key in TIMESTAMP_KEYS if key in loaded.files), None)
                if ts_key is not None:
                    timestamps = self._parse_timestamp_array(self._select_timestamps(
                        loaded[ts_key], sample_idx, n_samples, rad.shape[0]))
        else:
            n_samples = loaded.shape[0] if sample_idx is not None else 1
            rad = loaded[sample_idx] if sample_idx is not None else loaded
            sidecar = os.path.splitext(path)[0] + ".timestamps.npy"
            if os.path.isfile(sidecar):
                timestamps = self._parse_timestamp_array(self._select_timestamps(
                    np.load(sidecar, allow_pickle=False), sample_idx, n_samples, rad.shape[0]))
        rad = self._coerce_chip_layout(rad, path)
        if timestamps is None:
            timestamps = self._fallback_timestamps(rad.shape[0])
        return rad, timestamps

    def __getitem__(self, idx):

        path, sample_idx = self.samples[idx]

        if path.endswith(".zarr"):
            rad, timestamps = self._load_zarr(path)
        else:
            rad, timestamps = self._load_numpy_chip(path, sample_idx)

        if timestamps.shape != (rad.shape[0], len(self.temporal_embeddings)):
            raise ValueError(f"Timestamp shape {timestamps.shape} does not match chip time axis in {path}")
        # Convert to torch tensor (T, C, H, W)
        tensor = torch.from_numpy(rad)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, timestamps


if __name__ == "__main__":
    # Example directory list
    train_dirs = ["/home/jacaraba/tiles_pix4d"]

    # Train dataset
    train_ds = ABITemporalDataset(
        data_paths=train_dirs,
        img_size=512,
        in_chans=14
    )

    # DataLoader example
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # First batch
    for imgs, ts in train_loader:
        print("Images:", imgs.shape)   # (B, T, C, H, W)
        print("Timestamps:", ts.shape) # (B, T, n_components)
        print(ts)
        break
