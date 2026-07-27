"""CloudSat 2B-CLDCLASS-LIDAR transect access."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.utils import require_pyhdf


ORBIT_RE = re.compile(r"_(?P<orbit>\d{5})_")


@dataclass(frozen=True)
class CloudSatOrbitFile:
    day: int
    orbit: str
    path: Path


@dataclass
class CloudSatTransect:
    """CloudSat profiles restricted to a latitude transect."""

    source: Path
    latitude: np.ndarray
    longitude: np.ndarray
    utc_hour: np.ndarray
    quality: np.ndarray
    cloud_layer_base: np.ndarray
    cloud_layer_top: np.ndarray
    cloud_layer_type: np.ndarray
    cloud_layer_count: np.ndarray
    _cached_cloud_mask: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    def __len__(self) -> int:
        return len(self.latitude)

    def profile_window(self, center: int, count: int) -> np.ndarray:
        start = center - count // 2
        stop = start + count
        if start < 0 or stop > len(self):
            raise IndexError("CloudSat profile window extends outside the transect")
        return np.arange(start, stop)

    def cloud_mask(self, levels: int = 40, resolution_km: float = 0.5) -> np.ndarray:
        if (
            levels == 40
            and resolution_km == 0.5
            and self._cached_cloud_mask is not None
        ):
            return self._cached_cloud_mask
        valid_profiles = self.profile_validity()
        mask = np.full((len(self), levels), -1, dtype=np.int8)
        mask[valid_profiles] = 0
        for profile in range(len(self)):
            if not valid_profiles[profile]:
                continue
            layer_count = min(
                int(self.cloud_layer_count[profile]),
                self.cloud_layer_base.shape[1],
            )
            for layer in range(layer_count):
                base = self.cloud_layer_base[profile, layer]
                if base < 0:
                    continue
                top = self.cloud_layer_top[profile, layer]
                start = max(0, int(np.floor(base / resolution_km)))
                stop = min(levels, int(np.floor(top / resolution_km)) + 1)
                if stop > start:
                    mask[profile, start:stop] = self.cloud_layer_type[profile, layer]
        if levels == 40 and resolution_km == 0.5:
            self._cached_cloud_mask = mask
        return mask

    def profile_validity(self) -> np.ndarray:
        """Distinguish valid clear profiles from fill-only retrievals."""
        # Cloudlayer is 0 for valid clear profiles, positive for cloudy
        # profiles, and -9 for missing retrievals.
        return self.cloud_layer_count >= 0

    def metadata_arrays(self, center: int, count: int) -> dict[str, np.ndarray]:
        indices = self.profile_window(center, count)
        return self.metadata_arrays_for_indices(indices)

    def metadata_arrays_for_indices(
        self, indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        mask = self.cloud_mask()[indices]
        result = {
            "latitude": self.latitude[indices],
            "longitude": self.longitude[indices],
            "utc_hour": self.utc_hour[indices],
            "quality": self.quality[indices],
            "cloud_layer_base": self.cloud_layer_base[indices],
            "cloud_layer_top": self.cloud_layer_top[indices],
            "cloud_layer_type": self.cloud_layer_type[indices],
            "cloud_layer_count": self.cloud_layer_count[indices],
            "cloud_class": mask,
            "cloud_binary_mask": np.where(mask < 0, -1, mask != 0).astype(
                np.int8
            ),
            "profile_valid": self.profile_validity()[indices].astype(np.int8),
        }
        return {f"cloudsat_{name}": value for name, value in result.items()}


@dataclass
class CloudSatAuxiliaryTransect:
    """CloudSat ECMWF-AUX profiles restricted to the same latitude transect."""

    source: Path
    pressure: np.ndarray
    dem_elevation: np.ndarray
    temperature: np.ndarray
    specific_humidity: np.ndarray
    ec_height: np.ndarray
    temperature_2m: np.ndarray
    skin_temperature: np.ndarray
    surface_pressure: np.ndarray
    u10_velocity: np.ndarray
    v10_velocity: np.ndarray
    utc_time: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray

    def metadata_arrays_for_indices(
        self, indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        result = {
            "pressure": self.pressure[indices],
            "dem_elevation": self.dem_elevation[indices],
            "temperature": self.temperature[indices],
            "specific_humidity": self.specific_humidity[indices],
            # "ec_height": self._ec_height_for_indices(indices),
            "temperature_2m": self.temperature_2m[indices],
            "skin_temperature": self.skin_temperature[indices],
            "surface_pressure": self.surface_pressure[indices],
            "u10_velocity": self.u10_velocity[indices],
            "v10_velocity": self.v10_velocity[indices],
            # "aux_utc_time": self.utc_time[indices],
            # "aux_latitude": self.latitude[indices],
            # "aux_longitude": self.longitude[indices],
        }
        return {f"cloudsat_{name}": value for name, value in result.items()}

    def _ec_height_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.ec_height.ndim == 1:
            return np.broadcast_to(
                self.ec_height, (len(indices), self.ec_height.shape[0])
            ).copy()
        return self.ec_height[indices]


class CloudSatReader:
    """Discover and read local CloudSat cloud-classification products."""

    def __init__(self, root: Path, index_root: Path | None = None):
        self.root = Path(root)
        self.index_root = Path(index_root or self.root / "2B-CLDCLASS-LIDAR")

    def discover_orbits(
        self,
        year: int,
        day_start: int = 1,
        day_end: int | None = None,
        orbit: str | None = None,
    ) -> Iterable[CloudSatOrbitFile]:
        year_root = self.index_root / str(year)
        if not year_root.is_dir():
            raise FileNotFoundError(
                f"CloudSat year directory does not exist: {year_root}"
            )
        final_day = day_end if day_end is not None else 366
        seen: set[tuple[int, str]] = set()
        for day_dir in sorted(year_root.iterdir()):
            if not day_dir.is_dir() or not day_dir.name.isdigit():
                continue
            day = int(day_dir.name)
            if not day_start <= day <= final_day:
                continue
            for path in sorted(day_dir.glob("*.hdf")):
                match = ORBIT_RE.search(path.name)
                if match is None:
                    continue
                orbit_id = match.group("orbit")
                key = (day, orbit_id)
                if key in seen or (orbit is not None and orbit_id != orbit):
                    continue
                seen.add(key)
                yield CloudSatOrbitFile(day, orbit_id, path)

    def read(
        self, path: Path, latitude_bounds: tuple[float, float]
    ) -> CloudSatTransect:
        HDF, SD, SDC = require_pyhdf()
        sd = SD(str(path), SDC.READ)
        try:
            arrays = {
                "cloud_layer_base": np.asarray(sd.select("CloudLayerBase")[:]),
                "cloud_layer_top": np.asarray(sd.select("CloudLayerTop")[:]),
                "cloud_layer_type": np.asarray(sd.select("CloudLayerType")[:]),
            }
        finally:
            sd.end()

        hdf = HDF(str(path), SDC.READ)
        vs = hdf.vstart()
        try:
            arrays.update(
                quality=self._read_vdata(vs, "Data_quality"),
                latitude=self._read_vdata(vs, "Latitude"),
                longitude=self._read_vdata(vs, "Longitude"),
                profile_time=self._read_vdata(vs, "Profile_time"),
                utc_start=self._read_vdata(vs, "UTC_start"),
                cloud_layer_count=self._read_first_vdata(
                    vs, ("Cloudlayer", "CloudLayer")
                ),
            )
        finally:
            vs.end()
            hdf.close()

        low, high = latitude_bounds

        profile_count = len(arrays["latitude"])
        profile_time = self._as_profile_array(
            arrays.pop("profile_time"), profile_count, "Profile_time"
        )
        utc_start = self._as_profile_array(
            arrays.pop("utc_start"), profile_count, "UTC_start"
        )

        indices = np.flatnonzero(
            (arrays["latitude"] >= low) & (arrays["latitude"] <= high)
        )
        utc_hour = (utc_start[indices] + profile_time[indices]) / 3600.0
        selected = {name: value[indices] for name, value in arrays.items()}
        return CloudSatTransect(source=Path(path), utc_hour=utc_hour, **selected)

    @staticmethod
    def _read_vdata(vs: Any, name: str) -> np.ndarray:
        handle = vs.attach(name)
        try:
            return np.squeeze(np.asarray(handle[:]))
        finally:
            handle.detach()

    @classmethod
    def _read_first_vdata(
        cls, vs: Any, names: tuple[str, ...]
    ) -> np.ndarray:
        last_error = None
        for name in names:
            try:
                return cls._read_vdata(vs, name)
            except Exception as exc:
                last_error = exc
        raise KeyError(f"None of the CloudSat Vdata fields {names} exist") from last_error

    @staticmethod
    def _as_profile_array(
        value: np.ndarray, profile_count: int, field_name: str
    ) -> np.ndarray:
        """Broadcast scalar orbit metadata or validate profile-length fields."""
        array = np.asarray(value)
        if array.ndim == 0 or array.size == 1:
            return np.full(profile_count, array.reshape(-1)[0])
        if array.shape[0] != profile_count:
            raise ValueError(
                f"CloudSat {field_name} has {array.shape[0]} values for "
                f"{profile_count} profiles"
            )
        return array


class CloudSatAuxiliaryReader:
    """Discover and read local CloudSat ECMWF-AUX products."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path_for_orbit(self, year: int, day: int, orbit: str) -> Path:
        day_root = self.root / str(year) / f"{day:03d}"
        matches = sorted(
            day_root.glob(
                f"{year}{day:03d}*{orbit}*ECMWF-AUX*P1_R05*.hdf"
            )
        )
        if not matches:
            raise FileNotFoundError(
                f"No ECMWF-AUX file found for orbit {orbit} in {day_root}"
            )
        return matches[0]

    def read(
        self, path: Path, latitude_bounds: tuple[float, float]
    ) -> CloudSatAuxiliaryTransect:
        HDF, SD, SDC = require_pyhdf()
        sd = SD(str(path), SDC.READ)
        try:
            pressure = np.asarray(sd.select("Pressure")[:])
            temperature = np.asarray(sd.select("Temperature")[:])
            specific_humidity = np.asarray(sd.select("Specific_humidity")[:])
        finally:
            sd.end()

        hdf = HDF(str(path), SDC.READ)
        vs = hdf.vstart()
        try:
            ec_height = np.asarray(CloudSatReader._read_vdata(vs, "EC_height"))
            profile_time = np.asarray(
                CloudSatReader._read_vdata(vs, "Profile_time")
            )
            utc_start = np.asarray(
                CloudSatReader._read_vdata(vs, "UTC_start")
            )
            latitude = np.asarray(CloudSatReader._read_vdata(vs, "Latitude"))
            longitude = np.asarray(CloudSatReader._read_vdata(vs, "Longitude"))
            dem_elevation = np.asarray(
                CloudSatReader._read_vdata(vs, "DEM_elevation")
            )
            temperature_2m = np.asarray(
                CloudSatReader._read_vdata(vs, "Temperature_2m")
            )
            skin_temperature = self._read_optional_profile_vdata(
                vs, "Skin_temperature", len(latitude)
            )
            surface_pressure = self._read_optional_profile_vdata(
                vs, "Surface_pressure", len(latitude)
            )
            u10_velocity = np.asarray(
                CloudSatReader._read_vdata(vs, "U10_velocity")
            )
            v10_velocity = np.asarray(
                CloudSatReader._read_vdata(vs, "V10_velocity")
            )
        finally:
            vs.end()
            hdf.close()

        profile_count = len(latitude)
        profile_time = CloudSatReader._as_profile_array(
            profile_time, profile_count, "Profile_time"
        )
        utc_start = CloudSatReader._as_profile_array(
            utc_start, profile_count, "UTC_start"
        )
        utc_time = (utc_start + profile_time) / 3600.0

        low, high = latitude_bounds
        indices = np.flatnonzero((latitude >= low) & (latitude <= high))
        selected_ec_height = (
            ec_height[indices]
            if ec_height.ndim > 1 and ec_height.shape[0] == profile_count
            else ec_height
        )
        return CloudSatAuxiliaryTransect(
            source=Path(path),
            pressure=self._interp_to_cloudsat_grid(
                pressure[indices], selected_ec_height
            ),
            dem_elevation=dem_elevation[indices],
            temperature=self._interp_to_cloudsat_grid(
                temperature[indices], selected_ec_height
            ),
            specific_humidity=self._interp_to_cloudsat_grid(
                specific_humidity[indices], selected_ec_height
            ),
            ec_height=selected_ec_height,
            temperature_2m=temperature_2m[indices],
            skin_temperature=skin_temperature[indices],
            surface_pressure=surface_pressure[indices],
            u10_velocity=u10_velocity[indices],
            v10_velocity=v10_velocity[indices],
            utc_time=utc_time[indices],
            latitude=latitude[indices],
            longitude=longitude[indices],
        )

    @staticmethod
    def _read_optional_profile_vdata(
        vs: Any, name: str, profile_count: int
    ) -> np.ndarray:
        try:
            return np.asarray(CloudSatReader._read_vdata(vs, name))
        except Exception:
            return np.full(profile_count, np.nan, dtype=np.float32)

    @staticmethod
    def _interp_to_cloudsat_grid(
        values: np.ndarray,
        ec_height: np.ndarray,
        levels: int = 40,
        resolution_km: float = 0.5,
    ) -> np.ndarray:
        z_grid = np.arange(levels, dtype=np.float32) * resolution_km
        output = np.zeros((values.shape[0], levels), dtype=np.float32)
        for profile in range(values.shape[0]):
            profile_values = np.squeeze(values[profile])
            profile_heights = (
                np.squeeze(ec_height[profile])
                if ec_height.ndim > 1
                else np.squeeze(ec_height)
            )
            valid = np.flatnonzero(
                np.isfinite(profile_values)
                & np.isfinite(profile_heights)
                & (profile_values > 0)
            )
            if len(valid) < 2:
                continue
            output[profile] = np.interp(
                z_grid,
                np.flip(profile_heights[valid] / 1000.0),
                np.flip(profile_values[valid]),
            )
        return output
