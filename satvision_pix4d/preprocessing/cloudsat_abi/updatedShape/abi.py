"""Local GOES ABI archive and geometry access."""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.config import SatelliteSpec
from satvision_pix4d.preprocessing.cloudsat_abi.utils import (
    datetime_from_year_doy,
    normalize_longitude,
    require_netcdf4,
)


LOG = logging.getLogger(__name__)

ABI_FILENAME_RE = re.compile(
    r"C(?P<channel>\d{2})_(?P<platform>G\d{2})_s"
    r"(?P<year>\d{4})(?P<doy>\d{3})(?P<hour>\d{2})"
    r"(?P<minute>\d{2})(?P<second>\d{2})"
)


@dataclass(frozen=True)
class ABIFileInfo:
    timestamp: datetime
    channel: int
    platform: str

    @classmethod
    def from_path(cls, path: str | Path) -> "ABIFileInfo | None":
        match = ABI_FILENAME_RE.search(Path(path).name)
        if match is None:
            return None
        timestamp = datetime_from_year_doy(
            match.group("year"), match.group("doy"), int(match.group("hour"))
        ) + timedelta(
            minutes=int(match.group("minute")),
            seconds=int(match.group("second")),
        )
        return cls(
            timestamp=timestamp,
            channel=int(match.group("channel")),
            platform=match.group("platform"),
        )


class ABIGeometry:
    """Find the nearest pixel in an East or West ABI geolocation grid."""

    WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
    WGS84_SEMI_MINOR_AXIS_M = 6356752.31414
    SATELLITE_HEIGHT_M = 35786023.0

    def __init__(self, path: Path, coarse_target_size: int = 256):
        self.path = Path(path)
        self.coarse_target_size = coarse_target_size
        nc = require_netcdf4()
        with nc.Dataset(self.path) as dataset:
            self.latitude = np.asarray(
                dataset.variables["Latitude"][:], dtype=np.float32
            )
            self.longitude = np.asarray(
                dataset.variables["Longitude"][:], dtype=np.float32
            )

        invalid = (~np.isfinite(self.latitude)) | (~np.isfinite(self.longitude))
        invalid |= (np.abs(self.latitude) > 90) | (np.abs(self.longitude) > 360)
        self.valid = ~invalid
        if not np.any(self.valid):
            raise ValueError(f"No valid coordinates in {self.path}")
        self.use_360 = bool(np.nanmax(self.longitude[self.valid]) > 180)
        self.longitude = normalize_longitude(
            self.longitude, self.use_360
        ).astype(np.float32)
        self.lat_min = float(np.min(self.latitude[self.valid]))
        self.lat_max = float(np.max(self.latitude[self.valid]))

    def nearest(self, latitude: float, longitude: float) -> tuple[int, int]:
        longitude = float(normalize_longitude(longitude, self.use_360))
        if not self.lat_min <= latitude <= self.lat_max:
            raise ValueError(f"Latitude {latitude:.3f} is outside ABI coverage")

        stride = max(1, min(self.latitude.shape) // self.coarse_target_size)
        coarse_lat = self.latitude[::stride, ::stride]
        coarse_lon = self.longitude[::stride, ::stride]
        coarse_valid = self.valid[::stride, ::stride]
        distance = self._distance(coarse_lat, coarse_lon, latitude, longitude)
        distance[~coarse_valid] = np.inf
        coarse_row, coarse_column = np.unravel_index(
            int(np.argmin(distance)), distance.shape
        )

        row = int(coarse_row * stride)
        column = int(coarse_column * stride)
        radius = 2 * stride
        row_start = max(0, row - radius)
        row_stop = min(self.latitude.shape[0], row + radius + 1)
        column_start = max(0, column - radius)
        column_stop = min(self.latitude.shape[1], column + radius + 1)
        selection = np.s_[row_start:row_stop, column_start:column_stop]
        distance = self._distance(
            self.latitude[selection],
            self.longitude[selection],
            latitude,
            longitude,
        )
        distance[~self.valid[selection]] = np.inf
        local_row, local_column = np.unravel_index(
            int(np.argmin(distance)), distance.shape
        )
        return row_start + int(local_row), column_start + int(local_column)

    def valid_fraction(self, row: int, column: int, size: int) -> float:
        """Return the fraction of a chip covered by valid Earth coordinates."""
        half = size // 2
        row_start, row_stop = row - half, row + half
        column_start, column_stop = column - half, column + half
        if (
            row_start < 0
            or column_start < 0
            or row_stop > self.valid.shape[0]
            or column_stop > self.valid.shape[1]
        ):
            return 0.0
        return float(
            np.mean(self.valid[row_start:row_stop, column_start:column_stop])
        )

    def crop_latlon(
        self, row: int, column: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return latitude and longitude arrays for one ABI chip footprint."""
        half = size // 2
        row_start, row_stop = row - half, row + half
        column_start, column_stop = column - half, column + half
        if (
            row_start < 0
            or column_start < 0
            or row_stop > self.latitude.shape[0]
            or column_stop > self.latitude.shape[1]
        ):
            raise ValueError("ABI chip footprint extends outside geometry grid")
        selection = np.s_[row_start:row_stop, column_start:column_stop]
        return self.latitude[selection], self.longitude[selection]

    def solar_zenith_angle(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        timestamp: datetime,
    ) -> np.ndarray:
        """Approximate per-pixel solar zenith angle in degrees."""
        utc_hour = (
            timestamp.hour
            + timestamp.minute / 60
            + timestamp.second / 3600
            + timestamp.microsecond / 3.6e9
        )
        year = timestamp.year
        days_in_year = 366 if self._leap_year(year) else 365
        day_of_year = int(timestamp.strftime("%j"))
        fractional_year = (
            2 * np.pi / days_in_year * (day_of_year - 1 + (utc_hour - 12) / 24)
        )
        equation_of_time = 229.18 * (
            0.000075
            + 0.001868 * np.cos(fractional_year)
            - 0.032077 * np.sin(fractional_year)
            - 0.014615 * np.cos(2 * fractional_year)
            - 0.040849 * np.sin(2 * fractional_year)
        )
        declination = (
            0.006918
            - 0.399912 * np.cos(fractional_year)
            + 0.070257 * np.sin(fractional_year)
            - 0.006758 * np.cos(2 * fractional_year)
            + 0.000907 * np.sin(2 * fractional_year)
            - 0.002697 * np.cos(3 * fractional_year)
            + 0.00148 * np.sin(3 * fractional_year)
        )
        true_solar_minutes = (
            timestamp.hour * 60
            + timestamp.minute
            + timestamp.second / 60
            + equation_of_time
            + 4 * longitudes
        ) % 1440
        hour_angle = np.deg2rad(true_solar_minutes / 4 - 180)
        latitude_rad = np.deg2rad(latitudes)
        cosine_zenith = (
            np.sin(latitude_rad) * np.sin(declination)
            + np.cos(latitude_rad)
            * np.cos(declination)
            * np.cos(hour_angle)
        )
        angle = np.rad2deg(np.arccos(np.clip(cosine_zenith, -1, 1)))
        return self._mask_invalid_angle(angle, latitudes, longitudes)

    def view_zenith_angle(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        satellite_longitude: float,
    ) -> np.ndarray:
        """Compute GOES satellite/view zenith angle in degrees."""
        latitude_rad = np.deg2rad(latitudes)
        longitude_rad = np.deg2rad(longitudes)
        satellite_longitude_rad = np.deg2rad(satellite_longitude)
        semi_major = self.WGS84_SEMI_MAJOR_AXIS_M
        semi_minor = self.WGS84_SEMI_MINOR_AXIS_M
        eccentricity_sq = 1 - (semi_minor ** 2 / semi_major ** 2)

        sin_latitude = np.sin(latitude_rad)
        prime_vertical_radius = semi_major / np.sqrt(
            1 - eccentricity_sq * sin_latitude ** 2
        )
        observer_x = (
            prime_vertical_radius
            * np.cos(latitude_rad)
            * np.cos(longitude_rad)
        )
        observer_y = (
            prime_vertical_radius
            * np.cos(latitude_rad)
            * np.sin(longitude_rad)
        )
        observer_z = (
            prime_vertical_radius
            * (1 - eccentricity_sq)
            * sin_latitude
        )

        satellite_radius = semi_major + self.SATELLITE_HEIGHT_M
        satellite_x = satellite_radius * np.cos(satellite_longitude_rad)
        satellite_y = satellite_radius * np.sin(satellite_longitude_rad)
        line_x = satellite_x - observer_x
        line_y = satellite_y - observer_y
        line_z = -observer_z
        line_norm = np.sqrt(line_x ** 2 + line_y ** 2 + line_z ** 2)

        up_x = np.cos(latitude_rad) * np.cos(longitude_rad)
        up_y = np.cos(latitude_rad) * np.sin(longitude_rad)
        up_z = np.sin(latitude_rad)
        cosine_zenith = (
            line_x * up_x + line_y * up_y + line_z * up_z
        ) / line_norm
        angle = np.rad2deg(np.arccos(np.clip(cosine_zenith, -1, 1)))
        return self._mask_invalid_angle(angle, latitudes, longitudes)

    def inside_inner_disk(self, row: int, column: int, margin: int) -> bool:
        """Match the original conservative square inner-disk center bounds."""
        return (
            margin <= row <= self.latitude.shape[0] - margin
            and margin <= column <= self.latitude.shape[1] - margin
        )

    @staticmethod
    def _distance(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        latitude: float,
        longitude: float,
    ) -> np.ndarray:
        longitude_delta = np.abs(longitudes - longitude)
        longitude_delta = np.minimum(longitude_delta, 360.0 - longitude_delta)
        return np.abs(latitudes - latitude) + longitude_delta

    @staticmethod
    def _mask_invalid_angle(
        angle: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        angle = angle.astype(np.float32, copy=True)
        invalid = (
            ~np.isfinite(latitudes)
            | ~np.isfinite(longitudes)
            | (np.abs(latitudes) > 90)
            | (np.abs(longitudes) > 360)
        )
        angle[invalid] = np.nan
        return angle

    @staticmethod
    def _leap_year(year: int) -> bool:
        return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


class ABIArchive:
    """Locate complete local ABI scans and extract pixel values along transects."""

    CHANNELS = tuple(range(1, 17))

    # Each ABI channel has a native spatial resolution. The geometry grid is
    # on a 1 km "common" grid. To find the corresponding native-resolution
    # pixel for a common-grid coordinate, we multiply the row/column index
    # by this scale factor.
    #   - 0.5 km channels → scale 2  (2 native pixels per 1 km grid cell)
    #   - 1.0 km channels → scale 1  (1:1 mapping)
    #   - 2.0 km channels → scale 0.5 (every other common-grid pixel maps
    #                                   to the same native pixel)
    CHANNEL_SCALES: dict[int, float] = {
        1: 1.0,    # 1.0 km
        2: 2.0,    # 0.5 km
        3: 1.0,    # 1.0 km
        4: 0.5,    # 2.0 km
        5: 1.0,    # 1.0 km
        6: 0.5,    # 2.0 km
        7: 0.5,    # 2.0 km
        8: 0.5,    # 2.0 km
        9: 0.5,    # 2.0 km
        10: 0.5,   # 2.0 km
        11: 0.5,   # 2.0 km
        12: 0.5,   # 2.0 km
        13: 0.5,   # 2.0 km
        14: 0.5,   # 2.0 km
        15: 0.5,   # 2.0 km
        16: 0.5,   # 2.0 km
    }

    def __init__(
        self,
        root: Path,
        geometry: ABIGeometry,
        satellite: SatelliteSpec,
        max_delta_minutes: float = 8.0,
        min_valid_fraction: float = 1.0,
        inner_disk_margin: int = 1600,
    ):
        self.root = Path(root)
        self.geometry = geometry
        self.satellite = satellite
        self.max_delta = timedelta(minutes=max_delta_minutes)
        self.min_valid_fraction = min_valid_fraction
        self.inner_disk_margin = inner_disk_margin
        self._scan_cache: dict[
            tuple[int, int, int], dict[datetime, dict[int, Path]]
        ] = {}
        self._unreadable_scans: set[datetime] = set()

    def scans_for_hour(self, when: datetime) -> dict[datetime, dict[int, Path]]:
        key = (when.year, int(when.strftime("%j")), when.hour)
        if key in self._scan_cache:
            return self._scan_cache[key]

        directory = self.root / str(key[0]) / f"{key[1]:03d}" / f"{key[2]:02d}"
        scans: dict[datetime, dict[int, Path]] = defaultdict(dict)
        if directory.is_dir():
            for path in directory.iterdir():
                info = ABIFileInfo.from_path(path)
                if info is None or info.platform != self.satellite.platform_code:
                    continue
                scans[info.timestamp][info.channel] = path
        result = dict(scans)
        self._scan_cache[key] = result
        return result

    def nearest_scan(self, requested: datetime) -> tuple[datetime, dict[int, Path]]:
        return self.candidate_scans(requested)[0]

    def candidate_scans(
        self, requested: datetime
    ) -> list[tuple[datetime, dict[int, Path]]]:
        """Return complete readable-candidate scans ordered by time distance."""
        candidates: dict[datetime, dict[int, Path]] = {}
        for hour_delta in (-1, 0, 1):
            candidates.update(
                self.scans_for_hour(requested + timedelta(hours=hour_delta))
            )
        required = set(self.CHANNELS)
        complete = {
            timestamp: files
            for timestamp, files in candidates.items()
            if required.issubset(files)
            and timestamp not in self._unreadable_scans
        }
        if not complete:
            raise FileNotFoundError(
                f"No complete 16-channel {self.satellite.name} ABI scan near "
                f"{requested.isoformat()}"
            )
        ordered = sorted(complete, key=lambda value: abs(value - requested))
        within_tolerance = [
            timestamp
            for timestamp in ordered
            if abs(timestamp - requested) <= self.max_delta
        ]
        if not within_tolerance:
            scan_time = ordered[0]
            difference = abs(scan_time - requested)
            raise FileNotFoundError(
                f"Nearest ABI scan is {difference.total_seconds() / 60:.1f} minutes "
                f"from {requested.isoformat()} (limit "
                f"{self.max_delta.total_seconds() / 60:g})"
            )
        return [
            (timestamp, complete[timestamp])
            for timestamp in within_tolerance
        ]

    def extract_transect(
        self,
        requested: datetime,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> tuple[np.ndarray, datetime]:
        """Extract 16-channel ABI pixel values at specific (row, col) positions.

        Instead of cropping a 2D square chip, this method pulls out the
        radiance values at exactly the pixel locations specified by `rows`
        and `columns` (which are indices on the 1 km common grid, typically
        512 CloudSat footprint positions).

        Parameters
        ----------
        requested : datetime
            Target time; the nearest valid ABI scan is used.
        rows : np.ndarray
            1D array of row indices on the common 1 km grid (length N).
        columns : np.ndarray
            1D array of column indices on the common 1 km grid (length N).

        Returns
        -------
        values : np.ndarray
            Float32 array of shape (N, 16) — one radiance value per
            footprint per ABI channel.
        scan_time : datetime
            Actual timestamp of the ABI scan that was used.
        """
        nc = require_netcdf4()
        n_points = len(rows)

        # Try candidate scans in order of closeness, skipping any that are
        # unreadable (e.g. corrupt files).
        failures: list[tuple[datetime, Exception]] = []
        while True:
            try:
                candidates = self.candidate_scans(requested)
            except FileNotFoundError:
                if failures:
                    break
                raise

            scan_time, paths = candidates[0]
            try:
                # We will fill this array with one column per ABI channel.
                # Final shape: (N, 16), where N is the number of footprints.
                result = np.empty((n_points, 16), dtype=np.float32)

                for channel in self.CHANNELS:
                    scale = self.CHANNEL_SCALES[channel]

                    # Convert common-grid (1 km) row/col indices to the
                    # native resolution of this channel.
                    #
                    # Example: Channel 2 is 0.5 km resolution, so scale=2.
                    #   common-grid row 100 → native row 200.
                    # Example: Channel 7 is 2.0 km resolution, so scale=0.5.
                    #   common-grid row 100 → native row 50.
                    native_rows = np.floor(rows * scale).astype(np.intp)
                    native_cols = np.floor(columns * scale).astype(np.intp)

                    # Open the NetCDF file for this channel and read the
                    # radiance variable. netCDF4 automatically applies
                    # scale_factor and add_offset, giving us physical
                    # radiance values (not packed integers).
                    with nc.Dataset(str(paths[channel])) as ds:
                        rad = ds.variables["Rad"]

                        min_r, max_r = native_rows.min(), native_rows.max()
                        min_c, max_c = native_cols.min(), native_cols.max()
                        bbox_data = np.asarray(
                            rad[min_r:max_r + 1, min_c:max_c + 1], 
                            dtype=np.float32
                        )
                        pixel_values = bbox_data[
                            native_rows - min_r, 
                            native_cols - min_c
                        ]

                    # Store this channel's values in the corresponding
                    # column of the output array. Channels are 1-indexed
                    # (1..16), so we subtract 1 for 0-based column indexing.
                    result[:, channel - 1] = pixel_values

            except OSError as exc:
                self._unreadable_scans.add(scan_time)
                failures.append((scan_time, exc))
                LOG.warning(
                    "Quarantining unreadable ABI scan %s: %s",
                    scan_time.isoformat(),
                    exc,
                )
                continue

            return result, scan_time

        # All candidate scans were unreadable.
        details = "; ".join(
            f"{timestamp.isoformat()}: {error}"
            for timestamp, error in failures
        )
        raise FileNotFoundError(
            f"No readable complete ABI scan near {requested.isoformat()}"
            + (f" ({details})" if details else "")
        )
