
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
    ):
        self.min_year = 2000
        self.img_size = img_size
        self.in_chans = in_chans
        
        # self.transform = transform
        # self.transform = PerTileMinMaxNormalize()
        # /home/jacaraba/satvision-pix4d/satvision_pix4d/transforms/min_max.py
        self.transform = None

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
        return sorted(files)

    def _numpy_chip_shape(self, path):
        loaded = np.load(path, allow_pickle=True, mmap_mode="r" if path.endswith(".npy") else None)
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
        elif rad.shape[1] > self.in_chans:
            rad = rad[:, :self.in_chans, :, :]
        elif rad.shape[-1] > self.in_chans:
            rad = np.transpose(rad[..., :self.in_chans], (0, 3, 1, 2))
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

        return np.nan_to_num(rad, nan=0.0)

    def _parse_timestamp_array(self, values):
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == len(self.temporal_embeddings):
            return arr.astype(np.int32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        if np.issubdtype(arr.dtype, np.number):
            emb_map = {
                "year": np.zeros(arr.shape[0], dtype=np.int32),
                "month": np.zeros(arr.shape[0], dtype=np.int32),
                "day": np.zeros(arr.shape[0], dtype=np.int32),
                "hour": arr.astype(np.int32),
                "minute": np.zeros(arr.shape[0], dtype=np.int32),
            }
            return np.stack([emb_map[k] for k in self.temporal_embeddings], axis=1).astype(np.int32)

        pd_timestamps = pd.to_datetime(arr)
        emb_map = {
            "year": pd_timestamps.year - self.min_year,
            "month": pd_timestamps.month - 1,
            "day": pd_timestamps.day - 1,
            "hour": pd_timestamps.hour,
            "minute": pd_timestamps.minute
        }
        return np.stack([emb_map[k] for k in self.temporal_embeddings], axis=1).astype(np.int32)

    def _fallback_timestamps(self, n_times):
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

    def _load_numpy_chip(self, path, sample_idx=None):
        loaded = np.load(path, allow_pickle=True)
        timestamps = None

        if isinstance(loaded, np.lib.npyio.NpzFile):
            chip_key = next((key for key in CHIP_KEYS if key in loaded.files), None)
            if chip_key is None:
                raise ValueError(
                    f"{path} does not contain a chip array. Expected one of {CHIP_KEYS}; "
                    f"found {loaded.files}."
                )
            rad = loaded[chip_key]
            if sample_idx is not None:
                rad = rad[sample_idx]
            ts_key = next((key for key in TIMESTAMP_KEYS if key in loaded.files), None)
            if ts_key is not None:
                ts_values = loaded[ts_key]
                if sample_idx is not None and np.asarray(ts_values).ndim >= 2:
                    ts_values = ts_values[sample_idx]
                timestamps = self._parse_timestamp_array(ts_values)
            loaded.close()
        else:
            rad = loaded
            if sample_idx is not None:
                rad = rad[sample_idx]

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
