"""Small dependency and coordinate helpers."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np


def require_netcdf4():
    try:
        import netCDF4
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required for ABI geometry and MERRA-2 data") from exc
    return netCDF4


def require_pyhdf():
    try:
        import pyhdf

        # HDF.vstart() dereferences pyhdf.VS at runtime. Some pyhdf releases do
        # not import or expose that submodule when pyhdf.HDF is imported alone.
        pyhdf.VS = importlib.import_module("pyhdf.VS")

        from pyhdf.HDF import HDF
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise RuntimeError("pyhdf is required for CloudSat HDF4 products") from exc
    return HDF, SD, SDC


def datetime_from_year_doy(
    year: int | str, doy: int | str, utc_hour: float
) -> datetime:
    return (
        datetime(int(year), 1, 1, tzinfo=timezone.utc)
        + timedelta(days=int(doy) - 1, hours=float(utc_hour))
    )


def as_datetime(value: Any) -> datetime:
    return datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        int(value.second),
        tzinfo=timezone.utc,
    )


def normalize_longitude(values: np.ndarray | float, use_360: bool):
    array = np.asarray(values)
    if use_360:
        return np.mod(array, 360.0)
    return (array + 180.0) % 360.0 - 180.0
