"""Shared GOES ABI L1b radiance reader and common-grid crop helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_l1b_rad(path: str | Path) -> np.ndarray:
    """Read the ABI L1b ``Rad`` variable as float32, preserving raw units."""
    nc = _require_netcdf4()
    with nc.Dataset(path) as dataset:
        raw = dataset.variables["Rad"][:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    return np.asarray(raw, dtype=np.float32)


def crop_l1b_rad_to_common_grid(
    path: str | Path,
    center_row: int,
    center_column: int,
    size: int,
    common_grid_shape: tuple[int, int],
) -> np.ndarray:
    """Crop one ABI L1b channel onto the 1 km common grid used by SATVISION.

    The output is still raw L1b radiance from ``Rad``. The only normalization
    performed here is spatial harmonization to the 10848 x 10848 common ABI
    grid:

    - native 0.5 km channels are sampled every two native pixels
    - native 1 km channels are unchanged
    - native 2 km channels are repeated to 1 km by nearest-neighbor indexing
    """
    nc = _require_netcdf4()
    with nc.Dataset(path) as dataset:
        variable = dataset.variables["Rad"]
        scale = variable.shape[0] / common_grid_shape[0]
        if (
            not np.isclose(scale, round(scale))
            and not np.isclose(1 / scale, round(1 / scale))
        ):
            raise ValueError(
                f"Unsupported ABI channel resolution {variable.shape} in {path}"
            )

        common_row_start = center_row - size // 2
        common_row_stop = center_row + size // 2
        common_column_start = center_column - size // 2
        common_column_stop = center_column + size // 2
        native_rows = common_to_native_indices(
            common_row_start, common_row_stop, scale
        )
        native_columns = common_to_native_indices(
            common_column_start, common_column_stop, scale
        )
        row_slice = slice(int(native_rows[0]), int(native_rows[-1]) + 1)
        column_slice = slice(int(native_columns[0]), int(native_columns[-1]) + 1)
        if (
            row_slice.start < 0
            or column_slice.start < 0
            or row_slice.stop > variable.shape[0]
            or column_slice.stop > variable.shape[1]
        ):
            raise ValueError(
                f"Chip centered at {(center_row, center_column)} extends "
                "outside ABI grid"
            )
        raw = variable[row_slice, column_slice]
        if np.ma.isMaskedArray(raw):
            raw = raw.filled(np.nan)
        chip = np.asarray(raw, dtype=np.float32)

    row_index = native_rows - native_rows[0]
    column_index = native_columns - native_columns[0]
    return chip[np.ix_(row_index, column_index)]


def common_to_native_indices(start: int, stop: int, scale: float) -> np.ndarray:
    """Map common 1 km ABI grid indices to native channel pixel indices."""
    return np.floor(np.arange(start, stop) * scale).astype(int)


def normalize_abi_l1b_for_model(
    chip: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
) -> np.ndarray:
    """Apply explicit z-score normalization to an ABI chip.

    This is intentionally not applied by the CloudSat cropper. Callers should
    pass the exact training mean/std they want Leah and everyone else to share.
    """
    return (chip.astype(np.float32) - mean) / np.maximum(std, 1e-6)


def _require_netcdf4():
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required for ABI L1b data") from exc
    return netCDF4
