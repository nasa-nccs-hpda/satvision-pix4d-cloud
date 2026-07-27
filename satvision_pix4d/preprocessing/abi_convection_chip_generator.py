#!/usr/bin/env python3
"""Generate ABI convection chips from local GOES-16 and GOES-17 archives."""

import argparse
import json
import logging
import multiprocessing
import os
import re
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


DEFAULT_GOES16_ABI_ROOT = Path(
    "/css/geostationary/NonOptimized/L1/GOES-16-ABI-L1B-FULLD"
)
DEFAULT_GOES17_ABI_ROOT = Path(
    "/css/geostationary/NonOptimized/L1/GOES-17-ABI-L1B-FULLD"
)
DEFAULT_METADATA_GLOB = (
    "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/"
    "1-metadata/convection-filtered/*_DCS_number_monthly_metadata.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/"
    "3-tiles/convection"
)
DEFAULT_GOES16_GEO_FILE = Path(
    "/explore/nobackup/projects/pix4dcloud/jgong/"
    "ABI_EAST_GEO_TOPO_LOMSK.nc"
)
DEFAULT_GOES17_GEO_FILE = Path(
    "/explore/nobackup/projects/pix4dcloud/jgong/"
    "ABI_WEST_GEO_TOPO_LOMSK.nc"
)


class LocalABIArchive:
    """Find complete ABI scans in the local year/day/hour archive."""

    def __init__(
        self,
        root: Path,
        satellite_number: int,
        channels=range(1, 17),
        max_time_difference_minutes: float = 10.0,
    ):
        self.root = Path(root)
        self.satellite_number = int(satellite_number)
        self.channels = tuple(channels)
        self.max_time_difference = pd.Timedelta(
            minutes=max_time_difference_minutes
        )
        self._directory_cache = {}
        self._filename_re = re.compile(
            r"OR_ABI-L1b-RadF-M\dC(?P<band>\d{2})_"
            rf"G{self.satellite_number}_s(?P<start>\d{{14}})_"
        )

    @staticmethod
    def _scan_time(start_token: str) -> pd.Timestamp:
        return pd.Timestamp(
            datetime.strptime(start_token[:13], "%Y%j%H%M%S")
        )

    def _hour_directory(self, timestamp: pd.Timestamp) -> Path:
        return (
            self.root
            / timestamp.strftime("%Y")
            / timestamp.strftime("%j")
            / timestamp.strftime("%H")
        )

    def _scan_groups(self, directory: Path):
        cache_key = str(directory)
        if cache_key in self._directory_cache:
            return self._directory_cache[cache_key]

        groups = defaultdict(dict)
        if directory.is_dir():
            for path in directory.glob(
                f"OR_ABI-L1b-RadF-*_G{self.satellite_number}_s*.nc"
            ):
                match = self._filename_re.match(path.name)
                if match is None:
                    continue
                groups[match.group("start")][int(match.group("band"))] = path

        self._directory_cache[cache_key] = groups
        return groups

    def nearest_complete_scan(self, timestamp):
        requested_time = pd.Timestamp(timestamp)
        required_bands = set(self.channels)
        candidates = []

        for hour_offset in (-1, 0, 1):
            directory = self._hour_directory(
                requested_time + pd.Timedelta(hours=hour_offset)
            )
            for start_token, band_files in self._scan_groups(directory).items():
                if not required_bands.issubset(band_files):
                    continue
                scan_time = self._scan_time(start_token)
                difference = abs(scan_time - requested_time)
                candidates.append((difference, scan_time, band_files))

        if not candidates:
            raise FileNotFoundError(
                f"No complete ABI scan found near {requested_time}."
            )

        difference, scan_time, band_files = min(candidates, key=lambda item: item[0])
        if difference > self.max_time_difference:
            raise FileNotFoundError(
                f"Nearest complete ABI scan to {requested_time} is {scan_time}, "
                f"{difference.total_seconds() / 60:.1f} minutes away."
            )

        return scan_time, band_files


class ABIConvectionChipGenerator:
    """Create local ABI time-series chips centered on convection metadata."""

    COMMON_GRID_SIZE = 10848
    WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
    WGS84_SEMI_MINOR_AXIS_M = 6356752.31414
    SATELLITE_HEIGHT_M = 35786023.0

    def __init__(
        self,
        satellite_label,
        satellite_number,
        satellite_longitude,
        abi_root,
        geo_file,
        output_dir=DEFAULT_OUTPUT_DIR,
        tile_size=512,
        n_timesteps=7,
        step_minutes=20,
        convection_index=3,
        crop_pad=1600,
        channels=range(1, 17),
        max_time_difference_minutes=10.0,
    ):
        if tile_size % 2:
            raise ValueError("tile_size must be even.")
        if not 0 <= convection_index < n_timesteps:
            raise ValueError("convection_index must be within the time sequence.")

        self.satellite_label = satellite_label
        self.satellite_number = int(satellite_number)
        self.satellite_name = f"GOES-{self.satellite_number}"
        self.satellite_longitude = float(satellite_longitude)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geo_file = Path(geo_file)
        self.tile_size = tile_size
        self.n_timesteps = n_timesteps
        self.step_minutes = step_minutes
        self.convection_index = convection_index
        self.crop_pad = crop_pad
        self.channels = tuple(channels)
        self.archive = LocalABIArchive(
            abi_root,
            satellite_number=self.satellite_number,
            channels=self.channels,
            max_time_difference_minutes=max_time_difference_minutes,
        )
        self._geo_dataset = None
        self._coarse_stride = None
        self._coarse_latitude = None
        self._coarse_longitude = None

    def _load_coarse_geolocation(self, coarse_stride):
        if (
            self._geo_dataset is not None
            and self._coarse_stride == coarse_stride
        ):
            return

        self._geo_dataset = xr.open_dataset(self.geo_file)
        self._coarse_stride = coarse_stride
        self._coarse_latitude = self._geo_dataset["Latitude"].isel(
            ydim=slice(None, None, coarse_stride),
            xdim=slice(None, None, coarse_stride),
        ).values
        self._coarse_longitude = self._geo_dataset["Longitude"].isel(
            ydim=slice(None, None, coarse_stride),
            xdim=slice(None, None, coarse_stride),
        ).values

    @staticmethod
    def _normalize_longitude(longitude):
        longitude = np.asarray(longitude)
        return np.where(longitude > 180, longitude - 360, longitude)

    def _nearest_grid_pixel(self, latitude, longitude, coarse_stride=16):
        """Use a coarse search followed by a local full-resolution search."""
        self._load_coarse_geolocation(coarse_stride)
        longitude = longitude - 360 if longitude > 180 else longitude

        coarse_lat = self._coarse_latitude
        coarse_lon = self._normalize_longitude(self._coarse_longitude)
        valid = (
            np.isfinite(coarse_lat)
            & np.isfinite(coarse_lon)
            & (coarse_lat > -90)
            & (coarse_lat < 90)
        )
        distance = (
            (coarse_lat - latitude) ** 2
            + ((coarse_lon - longitude) * np.cos(np.deg2rad(latitude))) ** 2
        )
        distance[~valid] = np.inf

        coarse_y, coarse_x = np.unravel_index(
            np.argmin(distance),
            distance.shape,
        )
        if not np.isfinite(distance[coarse_y, coarse_x]):
            raise ValueError(
                f"Could not locate ABI pixel for lat={latitude}, "
                f"lon={longitude}."
            )

        approximate_y = coarse_y * coarse_stride
        approximate_x = coarse_x * coarse_stride
        radius = coarse_stride * 2
        y_start = max(0, approximate_y - radius)
        y_stop = min(
            self._geo_dataset.sizes["ydim"],
            approximate_y + radius + 1,
        )
        x_start = max(0, approximate_x - radius)
        x_stop = min(
            self._geo_dataset.sizes["xdim"],
            approximate_x + radius + 1,
        )

        local_lat = self._geo_dataset["Latitude"].isel(
            ydim=slice(y_start, y_stop),
            xdim=slice(x_start, x_stop),
        ).values
        local_lon = self._normalize_longitude(
            self._geo_dataset["Longitude"].isel(
                ydim=slice(y_start, y_stop),
                xdim=slice(x_start, x_stop),
            ).values
        )
        valid = (
            np.isfinite(local_lat)
            & np.isfinite(local_lon)
            & (local_lat > -90)
            & (local_lat < 90)
        )
        distance = (
            (local_lat - latitude) ** 2
            + ((local_lon - longitude) * np.cos(np.deg2rad(latitude))) ** 2
        )
        distance[~valid] = np.inf
        local_y, local_x = np.unravel_index(np.argmin(distance), distance.shape)
        if not np.isfinite(distance[local_y, local_x]):
            raise ValueError(
                f"Could not locate local ABI pixel for lat={latitude}, "
                f"lon={longitude}."
            )
        return y_start + local_y, x_start + local_x

    def _extract_normalized_tile(self, abi_path, center_y, center_x):
        """Read only the source window needed for a tile on the 10848 grid."""
        with xr.open_dataset(abi_path) as dataset:
            rad = dataset["Rad"]
            source_height, source_width = rad.shape
            if source_height != source_width:
                raise ValueError(f"Unexpected non-square ABI grid in {abi_path}.")

            scale = source_height / self.COMMON_GRID_SIZE
            if scale not in (0.5, 1.0, 2.0, 4.0):
                raise ValueError(
                    f"Unsupported ABI grid shape {rad.shape} in {abi_path}."
                )

            source_tile_size = int(self.tile_size * scale)
            source_center_y = int(round(center_y * scale))
            source_center_x = int(round(center_x * scale))
            source_half = source_tile_size // 2
            y_slice = slice(
                source_center_y - source_half,
                source_center_y + source_half,
            )
            x_slice = slice(
                source_center_x - source_half,
                source_center_x + source_half,
            )
            source_tile = rad.isel(y=y_slice, x=x_slice).values

            if source_tile.shape != (source_tile_size, source_tile_size):
                raise ValueError(f"Tile falls outside ABI grid in {abi_path}.")

            if scale == 0.5:
                tile = np.repeat(
                    np.repeat(source_tile, 2, axis=0),
                    2,
                    axis=1,
                )
            elif scale == 1.0:
                tile = source_tile
            else:
                step = int(scale)
                tile = source_tile[::step, ::step]

            if tile.shape != (self.tile_size, self.tile_size):
                raise ValueError(
                    f"Normalized tile has shape {tile.shape}, expected "
                    f"{(self.tile_size, self.tile_size)}."
                )
            return tile.astype(np.float32, copy=False)

    def _extract_geolocation_tile(self, center_y, center_x):
        half = self.tile_size // 2
        indexers = {
            "ydim": slice(center_y - half, center_y + half),
            "xdim": slice(center_x - half, center_x + half),
        }
        latitude = self._geo_dataset["Latitude"].isel(indexers).values
        longitude = self._geo_dataset["Longitude"].isel(indexers).values
        longitude = self._normalize_longitude(longitude)

        if latitude.shape != (self.tile_size, self.tile_size):
            raise ValueError("Geolocation tile has an unexpected shape.")

        invalid = (
            ~np.isfinite(latitude)
            | ~np.isfinite(longitude)
            | (latitude <= -90)
            | (latitude >= 90)
        )
        latitude = latitude.astype(np.float32, copy=True)
        longitude = longitude.astype(np.float32, copy=True)
        latitude[invalid] = np.nan
        longitude[invalid] = np.nan
        return latitude, longitude

    @staticmethod
    def _solar_zenith_angle(latitude, longitude, timestamp):
        """Approximate per-pixel solar zenith angle in degrees."""
        timestamp = pd.Timestamp(timestamp)
        utc_hour = (
            timestamp.hour
            + timestamp.minute / 60
            + timestamp.second / 3600
            + timestamp.microsecond / 3.6e9
        )
        days_in_year = 366 if timestamp.is_leap_year else 365
        fractional_year = (
            2
            * np.pi
            / days_in_year
            * (timestamp.dayofyear - 1 + (utc_hour - 12) / 24)
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
            + 4 * longitude
        ) % 1440
        hour_angle = np.deg2rad(true_solar_minutes / 4 - 180)
        latitude_rad = np.deg2rad(latitude)
        cosine_zenith = (
            np.sin(latitude_rad) * np.sin(declination)
            + np.cos(latitude_rad)
            * np.cos(declination)
            * np.cos(hour_angle)
        )
        return np.rad2deg(
            np.arccos(np.clip(cosine_zenith, -1, 1))
        ).astype(np.float32)

    def _view_zenith_angle(self, latitude, longitude):
        """Compute GOES-16 satellite/view zenith angle in degrees."""
        latitude_rad = np.deg2rad(latitude)
        longitude_rad = np.deg2rad(longitude)
        satellite_longitude = np.deg2rad(self.satellite_longitude)
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
        satellite_x = satellite_radius * np.cos(satellite_longitude)
        satellite_y = satellite_radius * np.sin(satellite_longitude)
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
        return np.rad2deg(
            np.arccos(np.clip(cosine_zenith, -1, 1))
        ).astype(np.float32)

    def _read_scan(self, requested_time, center_y, center_x):
        scan_time, band_files = self.archive.nearest_complete_scan(requested_time)
        band_tiles = [
            self._extract_normalized_tile(
                band_files[band],
                center_y,
                center_x,
            )
            for band in self.channels
        ]
        source_files = [str(band_files[band]) for band in self.channels]
        return scan_time, np.stack(band_tiles, axis=0), source_files

    def _output_path(self, record):
        event_time = pd.Timestamp(record["datetime"])
        satellite_token = self.satellite_name.replace("-", "")
        return self.output_dir / (
            f"{satellite_token}_abi_{event_time.strftime('%Y%m%dT%H%M')}_"
            f"sys{int(record['system_id'])}.npz"
        )

    def generate_chip(self, record):
        if record["satellite"] != self.satellite_label:
            return False

        output_path = self._output_path(record)
        if output_path.exists():
            logging.info("Exists, skipping: %s", output_path)
            return False

        center_y, center_x = self._nearest_grid_pixel(
            float(record["latitude"]),
            float(record["longitude"]),
        )
        half = self.tile_size // 2
        inner_min = self.crop_pad + half
        inner_max = self.COMMON_GRID_SIZE - self.crop_pad - half
        if not (
            inner_min <= center_y < inner_max
            and inner_min <= center_x < inner_max
        ):
            logging.warning(
                "System %s is too close to the ABI inner-disk edge.",
                record["system_id"],
            )
            return False

        event_time = pd.Timestamp(record["datetime"])
        start_time = event_time - timedelta(
            minutes=self.step_minutes * self.convection_index
        )
        requested_times = [
            start_time + timedelta(minutes=self.step_minutes * index)
            for index in range(self.n_timesteps)
        ]

        actual_times = []
        frames = []
        source_files = []
        try:
            for requested_time in requested_times:
                actual_time, frame, frame_source_files = self._read_scan(
                    requested_time,
                    center_y,
                    center_x,
                )
                actual_times.append(actual_time)
                frames.append(frame)
                source_files.append(frame_source_files)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logging.warning(
                "Skipping system %s at %s: %s",
                record["system_id"],
                event_time,
                exc,
            )
            return False

        latitude, longitude = self._extract_geolocation_tile(center_y, center_x)
        solar_zenith_angle = np.stack(
            [
                self._solar_zenith_angle(latitude, longitude, timestamp)
                for timestamp in actual_times
            ],
            axis=0,
        )
        static_view_zenith_angle = self._view_zenith_angle(latitude, longitude)
        view_zenith_angle = np.broadcast_to(
            static_view_zenith_angle,
            (len(actual_times), self.tile_size, self.tile_size),
        ).copy()

        metadata = {
            "system_id": int(record["system_id"]),
            "convection_datetime": event_time.isoformat(),
            "convection_timestep_index": self.convection_index,
            "center_latitude": float(record["latitude"]),
            "center_longitude": float(record["longitude"]),
            "abi_center_y": int(center_y),
            "abi_center_x": int(center_x),
            "tile_size": self.tile_size,
            "n_timesteps": self.n_timesteps,
            "step_minutes": self.step_minutes,
            "satellite": self.satellite_name,
            "satellite_region": self.satellite_label,
            "satellite_longitude": self.satellite_longitude,
            "source_product": "ABI-L1b-RadF",
            "source_root": str(self.archive.root),
            "solar_zenith_units": "degrees",
            "view_zenith_units": "degrees",
            "metadata_source": {
                key: (
                    value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in record.to_dict().items()
            },
        }

        temporary_path = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_path,
            chip=np.stack(frames, axis=0).astype(np.float32),
            timestamps=np.asarray(
                [timestamp.isoformat() for timestamp in actual_times]
            ),
            requested_timestamps=np.asarray(
                [timestamp.isoformat() for timestamp in requested_times]
            ),
            bands=np.asarray(self.channels, dtype=np.int16),
            latitude=latitude,
            longitude=longitude,
            solar_zenith_angle=solar_zenith_angle,
            view_zenith_angle=view_zenith_angle,
            source_files=np.asarray(source_files),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary_path, output_path)
        logging.info("Saved %s", output_path)
        return True

    def generate_from_dataframe(self, metadata, max_chips=None):
        saved = 0
        for _, record in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc="Generating ABI convection chips",
        ):
            if self.generate_chip(record):
                saved += 1
                if max_chips is not None and saved >= max_chips:
                    break
        return saved


_WORKER_GENERATOR_CONFIGS = {}
_WORKER_GENERATORS = {}


def _create_generator(config):
    return ABIConvectionChipGenerator(**config)


def _initialize_worker(generator_configs):
    global _WORKER_GENERATOR_CONFIGS
    global _WORKER_GENERATORS
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(processName)s %(levelname)s %(message)s"
        ),
    )
    _WORKER_GENERATOR_CONFIGS = generator_configs
    _WORKER_GENERATORS = {}


def _worker_generate_chip(record):
    try:
        satellite_label = record["satellite"]
        generator = _WORKER_GENERATORS.get(satellite_label)
        if generator is None:
            generator = _create_generator(
                _WORKER_GENERATOR_CONFIGS[satellite_label]
            )
            _WORKER_GENERATORS[satellite_label] = generator
        return generator.generate_chip(pd.Series(record))
    except Exception:
        logging.exception(
            "Unexpected failure for system %s at %s.",
            record.get("system_id"),
            record.get("datetime"),
        )
        return False


def _record_dicts(metadata):
    columns = list(metadata.columns)
    for values in metadata.itertuples(index=False, name=None):
        yield dict(zip(columns, values))


def generate_parallel(executor, metadata, workers):
    """Generate chips with a bounded number of in-flight process tasks."""
    records = iter(_record_dicts(metadata))
    pending = set()
    max_pending = max(workers * 2, 1)

    for _ in range(max_pending):
        try:
            pending.add(executor.submit(_worker_generate_chip, next(records)))
        except StopIteration:
            break

    saved = 0
    progress = tqdm(
        total=len(metadata),
        desc="Generating ABI convection chips",
    )
    while pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            try:
                if future.result():
                    saved += 1
            except Exception:
                logging.exception("Chip worker process failed.")
            progress.update(1)

            try:
                pending.add(
                    executor.submit(_worker_generate_chip, next(records))
                )
            except StopIteration:
                pass

    progress.close()
    return saved


def load_metadata(csv_path, one_per_system=True):
    columns = [
        "datetime",
        "system_id",
        "latitude",
        "longitude",
        "satellite",
        "inside_inner_disk",
    ]
    metadata = pd.read_csv(
        csv_path,
        usecols=columns,
        parse_dates=["datetime"],
    )
    inside_inner_disk = metadata["inside_inner_disk"]
    if inside_inner_disk.dtype != bool:
        inside_inner_disk = (
            inside_inner_disk.astype(str).str.strip().str.lower() == "true"
        )

    metadata = metadata[
        metadata["satellite"].isin(["GOES-East", "GOES-West"])
        & inside_inner_disk
    ].sort_values(["system_id", "datetime"])

    if one_per_system:
        group_columns = ["satellite", "system_id"]
        middle_index = metadata.groupby(group_columns).cumcount()
        group_size = metadata.groupby(group_columns)["system_id"].transform("size")
        metadata = metadata[middle_index == group_size // 2]

    return metadata.reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate convection-centered chips from local GOES-16 and "
            "GOES-17 ABI data."
        )
    )
    parser.add_argument(
        "metadata_csv",
        nargs="*",
        help="One or more convection metadata CSV files.",
    )
    parser.add_argument("--metadata-glob", default=DEFAULT_METADATA_GLOB)
    parser.add_argument(
        "--goes16-abi-root",
        type=Path,
        default=DEFAULT_GOES16_ABI_ROOT,
    )
    parser.add_argument(
        "--goes17-abi-root",
        type=Path,
        default=DEFAULT_GOES17_ABI_ROOT,
    )
    parser.add_argument(
        "--goes16-geo-file",
        type=Path,
        default=DEFAULT_GOES16_GEO_FILE,
    )
    parser.add_argument(
        "--goes17-geo-file",
        type=Path,
        default=DEFAULT_GOES17_GEO_FILE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--n-timesteps", type=int, default=7)
    parser.add_argument("--step-minutes", type=int, default=20)
    parser.add_argument("--convection-index", type=int, default=3)
    parser.add_argument("--max-time-difference-minutes", type=float, default=10)
    parser.add_argument("--max-chips", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help=(
            "Number of chip-generation processes. Use 1 for serial debugging; "
            "2-8 is recommended for shared storage."
        ),
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Generate a chip for every metadata row instead of one per system.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )

    csv_files = [Path(path) for path in args.metadata_csv]
    if not csv_files:
        csv_files = [Path(path) for path in sorted(glob(args.metadata_glob))]
    if not csv_files:
        raise FileNotFoundError("No convection metadata CSV files were found.")

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    workers = args.workers
    if args.max_chips is not None and workers > 1:
        logging.info(
            "Using one worker with --max-chips to preserve the exact chip limit."
        )
        workers = 1

    common_generator_args = {
        "output_dir": args.output_dir,
        "tile_size": args.tile_size,
        "n_timesteps": args.n_timesteps,
        "step_minutes": args.step_minutes,
        "convection_index": args.convection_index,
        "max_time_difference_minutes": args.max_time_difference_minutes,
    }
    generator_configs = {
        "GOES-East": {
            "satellite_label": "GOES-East",
            "satellite_number": 16,
            "satellite_longitude": -75.0,
            "abi_root": args.goes16_abi_root,
            "geo_file": args.goes16_geo_file,
            **common_generator_args,
        },
        "GOES-West": {
            "satellite_label": "GOES-West",
            "satellite_number": 17,
            "satellite_longitude": -137.0,
            "abi_root": args.goes17_abi_root,
            "geo_file": args.goes17_geo_file,
            **common_generator_args,
        },
    }
    generators = None
    executor = None
    if workers == 1:
        generators = {
            label: _create_generator(config)
            for label, config in generator_configs.items()
        }
    else:
        logging.info("Using %s chip-generation workers.", workers)
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(generator_configs,),
        )

    total_saved = 0
    try:
        for csv_path in csv_files:
            logging.info("Loading metadata: %s", csv_path)
            metadata = load_metadata(
                csv_path,
                one_per_system=not args.all_records,
            )
            logging.info(
                "Selected %s records: %s.",
                len(metadata),
                metadata["satellite"].value_counts().to_dict(),
            )
            for satellite_label in generator_configs:
                satellite_metadata = metadata[
                    metadata["satellite"] == satellite_label
                ].reset_index(drop=True)
                if satellite_metadata.empty:
                    continue

                remaining = (
                    None
                    if args.max_chips is None
                    else max(args.max_chips - total_saved, 0)
                )
                if remaining == 0:
                    break

                if workers == 1:
                    total_saved += generators[
                        satellite_label
                    ].generate_from_dataframe(
                        satellite_metadata,
                        remaining,
                    )
                else:
                    total_saved += generate_parallel(
                        executor,
                        satellite_metadata,
                        workers,
                    )

            if args.max_chips is not None and total_saved >= args.max_chips:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    logging.info("Finished. Saved %s chips.", total_saved)


if __name__ == "__main__":
    main()
