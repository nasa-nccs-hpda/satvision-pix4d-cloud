"""Chip-gridded sampling from local MERRA-2 NetCDF collections."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.utils import (
    as_datetime,
    normalize_longitude,
    require_netcdf4,
)


MERRA2_VARIABLES = {
    "Pressure": ("lev", "PL"),
    "Temperature": ("T",),
    "WV": ("QV",),
    "Geopotential_height": ("H", "Height"),
    "Dem_elevation": ("PHIS",),
    "T2m": ("T2M",),
    "U10m": ("U10M",),
    "V10m": ("V10M",),
}

DEFAULT_MERRA2_OUTPUTS = tuple(MERRA2_VARIABLES)
PRESSURE_LEVEL_OUTPUTS = frozenset(
    {"Pressure", "Temperature", "WV", "Geopotential_height"}
)
TWO_D_OUTPUTS = frozenset({"T2m", "U10m", "V10m"})


class MERRA2Reader:
    """Find same-date MERRA-2 files and sample them onto an ABI chip grid."""

    def __init__(self, root: Path, variables: Sequence[str] = ()):
        self.root = Path(root)
        self.outputs = self._normalize_outputs(variables)
        self._dated_file_cache: dict[str, list[Path]] = {}
        self._constant_file_cache: Path | None = None

    def sample_chip(
        self,
        times: Sequence[datetime],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], list[Path]]:
        outputs: dict[str, np.ndarray] = {}
        sources: list[Path] = []

        if "Pressure" in self.outputs:
            pressure, path = self._read_pressure_levels(times[0])
            outputs["Pressure"] = pressure.astype(np.float32)
            sources.append(path)

        three_d = [
            output
            for output in ("Temperature", "WV", "Geopotential_height")
            if output in self.outputs
        ]
        if three_d:
            values, paths = self._sample_time_stack(
                times, latitudes, longitudes, three_d
            )
            outputs.update(values)
            sources.extend(paths)

        two_d = [
            output
            for output in ("T2m", "U10m", "V10m")
            if output in self.outputs
        ]
        if two_d:
            values, paths = self._sample_time_stack(
                [times[len(times) // 2]], latitudes, longitudes, two_d
            )
            for name, value in values.items():
                outputs[name] = value[0]
            sources.extend(paths)

        if "Dem_elevation" in self.outputs:
            dem, path = self._sample_dem(latitudes, longitudes)
            outputs["Dem_elevation"] = dem
            sources.append(path)

        return outputs, self._unique_paths(sources)

    def _sample_time_stack(
        self,
        times: Sequence[datetime],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        output_names: Sequence[str],
    ) -> tuple[dict[str, np.ndarray], list[Path]]:
        nc = require_netcdf4()
        stacked: dict[str, list[np.ndarray]] = {name: [] for name in output_names}
        sources: list[Path] = []

        for when in times:
            remaining = set(output_names)
            for path in self.files_for_date(when, output_names):
                with nc.Dataset(path) as dataset:
                    lat_var = self._coordinate(dataset, ("lat", "latitude"))
                    lon_var = self._coordinate(dataset, ("lon", "longitude"))
                    time_var = self._coordinate(dataset, ("time",))
                    dates = nc.num2date(
                        time_var[:],
                        time_var.units,
                        getattr(time_var, "calendar", "standard"),
                    )
                    time_index = min(
                        range(len(dates)),
                        key=lambda index: abs(as_datetime(dates[index]) - when),
                    )
                    lat_index, lon_index = self._nearest_indices(
                        np.asarray(lat_var[:]),
                        np.asarray(lon_var[:]),
                        latitudes,
                        longitudes,
                    )
                    found = []
                    for output_name in list(remaining):
                        variable = self._find_variable(dataset, output_name)
                        if variable is None:
                            continue
                        stacked[output_name].append(
                            self._sample_variable(
                                variable, time_index, lat_index, lon_index
                            )
                        )
                        remaining.remove(output_name)
                        found.append(output_name)
                    if found:
                        sources.append(path)
                if not remaining:
                    break
            if remaining:
                raise KeyError(
                    f"MERRA-2 variable(s) {sorted(remaining)} not found for "
                    f"{when.date()} below {self.root}"
                )

        return (
            {name: np.stack(values).astype(np.float32) for name, values in stacked.items()},
            sources,
        )

    def files_for_date(
        self, when: datetime, output_names: Sequence[str] = ()
    ) -> list[Path]:
        requested = tuple(sorted(output_names))
        date_token = when.strftime("%Y%m%d")
        cache_key = f"{date_token}:{','.join(requested)}"
        if cache_key in self._dated_file_cache:
            return self._dated_file_cache[cache_key]
        month_root = self.root / f"Y{when:%Y}" / f"M{when:%m}"
        search_roots = [month_root] if month_root.is_dir() else [self.root]
        daily = self._matching_files(search_roots, date_token)
        month_key = when.strftime("%Y%m")
        monthly = [
            path
            for path in self._matching_files(search_roots, month_key)
            if self._monthly_file(path, month_key)
        ]
        daily_set = set(daily)
        candidates = daily + [
            path
            for path in monthly
            if path not in daily_set
            and date_token not in path.name
        ]
        matches = self._filter_collection(candidates, requested)
        if not matches:
            raise FileNotFoundError(
                f"No MERRA-2 NetCDF file for {when.date()} below {self.root}"
            )
        self._dated_file_cache[cache_key] = matches
        return matches

    @staticmethod
    def _matching_files(search_roots: Sequence[Path], token: str) -> list[Path]:
        return sorted(
            path
            for search_root in search_roots
            for path in search_root.glob(f"*{token}*.nc*")
            if path.suffix.lower() in {".nc", ".nc4"}
        )

    @staticmethod
    def _monthly_file(path: Path, month_key: str) -> bool:
        date_match = re.search(r"\.(\d{6,8})\.nc", path.name)
        return date_match is not None and date_match.group(1) == month_key

    @staticmethod
    def _filter_collection(
        paths: Sequence[Path], output_names: Sequence[str]
    ) -> list[Path]:
        requested = set(output_names)
        if requested & PRESSURE_LEVEL_OUTPUTS:
            return [path for path in paths if "_Np." in path.name]
        if requested & TWO_D_OUTPUTS:
            return [path for path in paths if "_2d_" in path.name]
        return list(paths)

    def constant_file(self) -> Path:
        if self._constant_file_cache is not None:
            return self._constant_file_cache
        direct = self.root / "MERRA2.const_2d_asm_Nx.00000000.nc4"
        if direct.exists():
            self._constant_file_cache = direct
            return direct
        matches = sorted(
            path
            for path in self.root.rglob("*")
            if path.suffix.lower() in {".nc", ".nc4"}
            and "const_2d_asm_Nx" in path.name
        )
        if not matches:
            raise FileNotFoundError(
                f"No MERRA-2 const_2d_asm_Nx NetCDF file below {self.root}"
            )
        self._constant_file_cache = matches[0]
        return matches[0]

    def _read_pressure_levels(self, when: datetime) -> tuple[np.ndarray, Path]:
        nc = require_netcdf4()
        for path in self.files_for_date(when, ("Pressure",)):
            with nc.Dataset(path) as dataset:
                for name in MERRA2_VARIABLES["Pressure"]:
                    if name in dataset.variables:
                        return np.asarray(dataset.variables[name][:]), path
        raise KeyError(f"No MERRA-2 pressure coordinate found for {when.date()}")

    def _sample_dem(
        self, latitudes: np.ndarray, longitudes: np.ndarray
    ) -> tuple[np.ndarray, Path]:
        nc = require_netcdf4()
        path = self.constant_file()
        with nc.Dataset(path) as dataset:
            variable = self._find_variable(dataset, "Dem_elevation")
            if variable is None:
                raise KeyError(f"No MERRA-2 PHIS variable found in {path}")
            lat_var = self._coordinate(dataset, ("lat", "latitude"))
            lon_var = self._coordinate(dataset, ("lon", "longitude"))
            lat_index, lon_index = self._nearest_indices(
                np.asarray(lat_var[:]),
                np.asarray(lon_var[:]),
                latitudes,
                longitudes,
            )
            value = self._sample_variable(variable, None, lat_index, lon_index)
        return (value / 9.80665).astype(np.float32), path

    @staticmethod
    def _coordinate(dataset: Any, names: Sequence[str]):
        variables = {name.lower(): name for name in dataset.variables}
        for candidate in names:
            if candidate.lower() in variables:
                return dataset.variables[variables[candidate.lower()]]
        raise KeyError(f"None of coordinate variables {names} found")

    def _find_variable(self, dataset: Any, output_name: str):
        for source_name in MERRA2_VARIABLES[output_name]:
            if source_name in dataset.variables:
                return dataset.variables[source_name]
        return None

    @staticmethod
    def _nearest_indices(
        lat_values: np.ndarray,
        lon_values: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat_index = np.searchsorted(lat_values, latitudes)
        lat_index = np.clip(lat_index, 1, len(lat_values) - 1)
        lower_lat = lat_values[lat_index - 1]
        upper_lat = lat_values[lat_index]
        lat_index = np.where(
            np.abs(latitudes - lower_lat) <= np.abs(latitudes - upper_lat),
            lat_index - 1,
            lat_index,
        )

        use_360 = bool(np.nanmax(lon_values) > 180)
        target_lon = normalize_longitude(longitudes, use_360)
        normalized_lon = normalize_longitude(lon_values, use_360)
        lon_index = np.searchsorted(normalized_lon, target_lon)
        lon_index = np.clip(lon_index, 1, len(normalized_lon) - 1)
        lower_delta = np.abs(target_lon - normalized_lon[lon_index - 1])
        upper_delta = np.abs(target_lon - normalized_lon[lon_index])
        lower_delta = np.minimum(lower_delta, 360.0 - lower_delta)
        upper_delta = np.minimum(upper_delta, 360.0 - upper_delta)
        lon_index = np.where(lower_delta <= upper_delta, lon_index - 1, lon_index)
        return lat_index.astype(np.int64), lon_index.astype(np.int64)

    @staticmethod
    def _sample_variable(
        variable: Any,
        time_index: int | None,
        lat_index: np.ndarray,
        lon_index: np.ndarray,
    ) -> np.ndarray:
        index: list[int | slice] = []
        remaining_dimensions: list[str] = []
        for dimension in variable.dimensions:
            lower = dimension.lower()
            if lower == "time":
                if time_index is None:
                    index.append(0)
                else:
                    index.append(time_index)
            else:
                index.append(slice(None))
                remaining_dimensions.append(lower)
        value = variable[tuple(index)]
        if np.ma.isMaskedArray(value):
            value = value.filled(np.nan)
        array = np.asarray(value)
        lat_axis = next(
            index
            for index, dimension in enumerate(remaining_dimensions)
            if dimension in {"lat", "latitude"}
        )
        lon_axis = next(
            index
            for index, dimension in enumerate(remaining_dimensions)
            if dimension in {"lon", "longitude"}
        )
        array = np.moveaxis(array, (lat_axis, lon_axis), (0, 1))
        sampled = array[lat_index, lon_index]
        return sampled.astype(np.float32)

    @staticmethod
    def _normalize_outputs(variables: Sequence[str]) -> tuple[str, ...]:
        if not variables:
            return DEFAULT_MERRA2_OUTPUTS
        aliases = {
            output.lower().replace(" ", "_"): output
            for output in DEFAULT_MERRA2_OUTPUTS
        }
        for output, source_names in MERRA2_VARIABLES.items():
            aliases[output.lower()] = output
            for source_name in source_names:
                aliases[source_name.lower()] = output
        normalized = []
        for variable in variables:
            key = variable.lower().replace("-", "_").replace(" ", "_")
            try:
                normalized.append(aliases[key])
            except KeyError as exc:
                raise ValueError(f"Unsupported MERRA-2 variable {variable!r}") from exc
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _unique_paths(paths: Sequence[Path]) -> list[Path]:
        seen: set[Path] = set()
        unique: list[Path] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            unique.append(path)
        return unique
