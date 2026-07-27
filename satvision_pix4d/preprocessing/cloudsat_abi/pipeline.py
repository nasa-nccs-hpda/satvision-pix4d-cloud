"""Orchestration for CloudSat and multitemporal ABI collocation."""

from __future__ import annotations

import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np

from satvision_pix4d.preprocessing.cloudsat_abi.abi import ABIArchive, ABIGeometry
from satvision_pix4d.preprocessing.cloudsat_abi.cloudsat import (
    CloudSatAuxiliaryReader,
    CloudSatAuxiliaryTransect,
    CloudSatOrbitFile,
    CloudSatReader,
    CloudSatTransect,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import CropConfig
from satvision_pix4d.preprocessing.cloudsat_abi.merra2 import MERRA2Reader
from satvision_pix4d.preprocessing.cloudsat_abi.utils import datetime_from_year_doy
from satvision_pix4d.preprocessing.cloudsat_abi.writer import (
    CollocatedChip,
    NPZChipWriter,
)


LOG = logging.getLogger(__name__)


class CloudSatLabelError(ValueError):
    """A candidate lacks the requested valid CloudSat labels."""


@dataclass(frozen=True)
class OrbitResult:
    day: int
    orbit: str
    candidates: int
    written: int
    skipped: dict[str, int]


def _progress(
    iterable,
    *,
    enabled: bool,
    **kwargs,
):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


class CloudSatABICollocationPipeline:
    """Build and write ABI time-series chips along CloudSat transects."""

    def __init__(
        self,
        config: CropConfig,
        abi_archive: ABIArchive | None = None,
        cloudsat_reader: CloudSatReader | None = None,
        cloudsat_aux_reader: CloudSatAuxiliaryReader | None = None,
        merra2_reader: MERRA2Reader | None = None,
        writer: NPZChipWriter | None = None,
    ):
        self.config = config
        if abi_archive is None:
            geometry = ABIGeometry(config.latlon_path)
            abi_archive = ABIArchive(
                config.abi_root,
                geometry,
                config.satellite,
                config.max_scan_delta_minutes,
                config.min_valid_fraction,
                config.inner_disk_margin,
            )
        self.abi = abi_archive
        self.cloudsat = cloudsat_reader or CloudSatReader(
            config.cloudsat_root, config.cloudsat_index_root
        )
        self.cloudsat_aux = cloudsat_aux_reader
        if "cloudsat_aux" in config.metadata and self.cloudsat_aux is None:
            self.cloudsat_aux = CloudSatAuxiliaryReader(
                config.cloudsat_aux_root
                or config.cloudsat_root / "ECMWF-AUX"
            )
        self.merra2 = merra2_reader
        if "merra2" in config.metadata and self.merra2 is None:
            assert config.merra2_root is not None
            self.merra2 = MERRA2Reader(
                config.merra2_root, config.merra2_variables
            )
        self.writer = writer or NPZChipWriter(
            config.output_dir, config.overwrite
        )
        self._profile_pixel_cache: dict[tuple[str, int], tuple[int, int]] = {}

    def run(self) -> int:
        written = 0
        for orbit_file in self.cloudsat.discover_orbits(
            self.config.year,
            self.config.day_start,
            self.config.day_end,
            self.config.orbit,
        ):
            remaining = (
                None
                if self.config.max_chips is None
                else self.config.max_chips - written
            )
            if remaining is not None and remaining <= 0:
                break
            result = self.process_orbit(orbit_file, max_new=remaining)
            written += result.written
        return written

    def process_orbit(
        self, orbit_file: CloudSatOrbitFile, max_new: int | None = None
    ) -> OrbitResult:
        LOG.info(
            "Processing CloudSat orbit %s on %d-%03d",
            orbit_file.orbit,
            self.config.year,
            orbit_file.day,
        )
        transect = self.cloudsat.read(orbit_file.path, self.config.transect)
        auxiliary_transect = None
        if "cloudsat_aux" in self.config.metadata:
            assert self.cloudsat_aux is not None
            try:
                auxiliary_path = self.cloudsat_aux.path_for_orbit(
                    self.config.year, orbit_file.day, orbit_file.orbit
                )
                auxiliary_transect = self.cloudsat_aux.read(
                    auxiliary_path, self.config.transect
                )
                if len(auxiliary_transect.latitude) != len(transect):
                    raise ValueError(
                        "CloudSat ECMWF-AUX transect length "
                        f"{len(auxiliary_transect.latitude)} does not match "
                        f"2B-CLDCLASS-LIDAR length {len(transect)}"
                    )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                LOG.warning(
                    "Skipping orbit %s: CloudSat ECMWF-AUX unavailable: %s",
                    orbit_file.orbit,
                    exc,
                )
                return OrbitResult(
                    orbit_file.day,
                    orbit_file.orbit,
                    0,
                    0,
                    {"cloudsat_aux": 1},
                )
        if (
            self.config.profile_selection == "fixed"
            and len(transect) < self.config.profiles_per_chip
        ):
            LOG.warning(
                "Skipping orbit %s: only %d profiles in transect",
                orbit_file.orbit,
                len(transect),
            )
            return OrbitResult(
                orbit_file.day, orbit_file.orbit, 0, 0,
                {"short_transect": 1},
            )

        skipped = Counter()
        orbit_written = 0
        candidates = 0
        centers = self._profile_centers(transect)
        progress = _progress(
            centers,
            enabled=self.config.progress,
            total=len(centers),
            desc=f"orbit {orbit_file.orbit}",
            unit="candidate",
            leave=False,
        )
        for candidates, center in enumerate(progress, start=1):
            if max_new is not None and orbit_written >= max_new:
                break
            try:
                sample = self.build_sample(
                    orbit_file, transect, center, auxiliary_transect
                )
                output, created = self.writer.write(sample)
            except (FileNotFoundError, ValueError, IndexError, KeyError) as exc:
                reason = self._skip_reason(exc)
                skipped[reason] += 1
                if skipped[reason] <= 2:
                    LOG.warning(
                        "Skipping orbit %s profile %d [%s]: %s",
                        orbit_file.orbit,
                        center,
                        reason,
                        exc,
                    )
            else:
                orbit_written += int(created)
                LOG.info("%s %s", "Saved" if created else "Exists", output)

            if candidates % 100 == 0:
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        new=orbit_written,
                        skipped=sum(skipped.values()),
                        refresh=False,
                    )
                LOG.info(
                    "Orbit %s progress: %d candidates, %d new, skipped=%s",
                    orbit_file.orbit,
                    candidates,
                    orbit_written,
                    dict(skipped),
                )

        result = OrbitResult(
            orbit_file.day,
            orbit_file.orbit,
            candidates,
            orbit_written,
            dict(skipped),
        )
        LOG.info(
            "Finished orbit %s: %d new chip(s), skipped=%s",
            result.orbit,
            result.written,
            result.skipped,
        )
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                new=orbit_written,
                skipped=sum(skipped.values()),
                refresh=False,
            )
        return result

    @staticmethod
    def _skip_reason(exc: Exception) -> str:
        if isinstance(exc, CloudSatLabelError):
            return "cloudsat_labels"
        if isinstance(exc, FileNotFoundError):
            return "abi_unavailable"
        if isinstance(exc, IndexError):
            return "profile_window"
        message = str(exc).lower()
        if (
            "inner disk" in message
            or "on-disk" in message
            or "coverage" in message
            or "footprint" in message
            or "geometry grid" in message
        ):
            return "abi_geometry"
        if "track" in message or "profiles cross" in message:
            return "cloudsat_track"
        return type(exc).__name__.lower()

    def _profile_centers(self, transect: CloudSatTransect) -> range:
        if self.config.profile_selection == "chip":
            return range(0, len(transect), self.config.profile_stride)
        margin = self.config.profiles_per_chip // 2
        return range(
            margin,
            len(transect) - margin,
            self.config.profile_stride,
        )

    def build_sample(
        self,
        orbit_file: CloudSatOrbitFile,
        transect: CloudSatTransect,
        center: int,
        auxiliary_transect: CloudSatAuxiliaryTransect | None = None,
    ) -> CollocatedChip:
        center_time = datetime_from_year_doy(
            self.config.year, orbit_file.day, transect.utc_hour[center]
        )
        center_latitude = float(transect.latitude[center])
        center_longitude = float(transect.longitude[center])
        if (
            "cloudsat" in self.config.metadata
            and self.config.min_cloudsat_valid_fraction > 0
            and not transect.profile_validity()[center]
        ):
            raise CloudSatLabelError(
                "Center CloudSat profile has no valid retrieval"
            )
        row, column = self.abi.geometry.nearest(
            center_latitude, center_longitude
        )
        valid_fraction = (
            self.abi.geometry.valid_fraction(row, column, self.config.chip_size)
            if hasattr(self.abi.geometry, "valid_fraction")
            else 1.0
        )
        auxiliary: dict[str, np.ndarray] = {}
        segment_len = 0
        if "cloudsat" in self.config.metadata:
            if self.config.profile_selection == "chip":
                profile_indices, profile_rows, profile_columns = (
                    self._chip_profile_positions(
                        transect, center, row, column
                    )
                )
                auxiliary.update(
                    transect.metadata_arrays_for_indices(profile_indices)
                )
                auxiliary["cloudsat_abi_row"] = profile_rows
                auxiliary["cloudsat_abi_column"] = profile_columns
                auxiliary["cloudsat_profile_index"] = profile_indices
            else:
                profile_indices = transect.profile_window(
                    center, self.config.profiles_per_chip
                )
                auxiliary.update(
                    transect.metadata_arrays_for_indices(profile_indices)
                )
                profile_pixels = np.asarray(
                    [
                        self._profile_pixel(transect, int(index))
                        for index in profile_indices
                    ],
                    dtype=np.int32,
                )
                auxiliary["cloudsat_abi_row"] = profile_pixels[:, 0]
                auxiliary["cloudsat_abi_column"] = profile_pixels[:, 1]
                auxiliary["cloudsat_profile_index"] = profile_indices.astype(
                    np.int32
                )
            segment_len = len(profile_indices)
            self._add_cloudsat_auxiliary(
                auxiliary, metadata=None,
                auxiliary_transect=auxiliary_transect,
                profile_indices=profile_indices,
            )

        timestamp = center_time.strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{self.config.satellite.filename_token}_"
            f"{self.config.satellite.region}_abi_cloudsat_{timestamp}_"
            f"orbit{orbit_file.orbit}_r{row}_c{column}_p{center}_len{segment_len}.npz"
        )
        metadata = self._metadata(
            orbit_file,
            center,
            center_time.isoformat(),
            center_latitude,
            center_longitude,
            row,
            column,
            valid_fraction,
        )

        if "cloudsat" in self.config.metadata:
            self._validate_cloudsat_labels(auxiliary, metadata)
        if "cloudsat_aux" in self.config.metadata:
            if auxiliary_transect is None:
                raise FileNotFoundError(
                    "CloudSat ECMWF-AUX metadata was requested"
                )
            metadata["cloudsat_aux_source"] = str(auxiliary_transect.source)

        chip_latitude, chip_longitude = self.abi.geometry.crop_latlon(
            row, column, self.config.chip_size
        )

        # CloudSat eligibility is deliberately checked before expensive ABI I/O.
        chips, scan_times, valid, requested_times, angle_times = [], [], [], [], []
        for offset in self.config.offsets:
            requested = center_time + timedelta(minutes=offset)
            requested_times.append(requested)
            try:
                chip, scan_time = self.abi.crop(
                    requested, row, column, self.config.chip_size
                )
                valid.append(1)
            except (FileNotFoundError, ValueError) as exc:
                if not self.config.allow_missing_timesteps:
                    raise
                LOG.warning("Missing timestep %s: %s", requested.isoformat(), exc)
                chip = np.full(
                    (self.config.chip_size, self.config.chip_size, 16),
                    np.nan,
                    dtype=np.float32,
                )
                scan_time = None
                valid.append(0)
            chips.append(chip)
            scan_times.append(scan_time.isoformat() if scan_time else "")
            angle_times.append(scan_time or requested)

        solar_zenith_angle = np.stack(
            [
                self.abi.geometry.solar_zenith_angle(
                    chip_latitude, chip_longitude, angle_time
                )
                for angle_time in angle_times
            ]
        )
        view_zenith_angle = self.abi.geometry.view_zenith_angle(
            chip_latitude,
            chip_longitude,
            self.config.satellite.subpoint_longitude,
        )
        auxiliary["abi_solar_zenith_angle"] = solar_zenith_angle
        auxiliary["abi_view_zenith_angle"] = np.broadcast_to(
            view_zenith_angle, solar_zenith_angle.shape
        ).copy()

        if "merra2" in self.config.metadata:
            assert self.merra2 is not None
            values, sources = self.merra2.sample_chip(
                requested_times, chip_latitude, chip_longitude
            )
            auxiliary.update(
                {f"merra2_{name}": value for name, value in values.items()}
            )
            metadata["merra2_sources"] = [str(path) for path in sources]

        return CollocatedChip(
            filename=filename,
            chip=np.stack(chips).astype(np.float32),
            offsets_minutes=np.asarray(self.config.offsets, dtype=np.int32),
            valid_mask=np.asarray(valid, dtype=np.int8),
            scan_times=np.asarray(scan_times),
            metadata=metadata,
            auxiliary_arrays=auxiliary,
        )

    def _add_cloudsat_auxiliary(
        self,
        arrays: dict[str, np.ndarray],
        metadata: dict | None,
        auxiliary_transect: CloudSatAuxiliaryTransect | None,
        profile_indices: np.ndarray,
    ) -> None:
        if "cloudsat_aux" not in self.config.metadata:
            return
        if auxiliary_transect is None:
            raise FileNotFoundError("CloudSat ECMWF-AUX metadata was requested")
        arrays.update(auxiliary_transect.metadata_arrays_for_indices(profile_indices))
        if metadata is not None:
            metadata["cloudsat_aux_source"] = str(auxiliary_transect.source)

    def _validate_cloudsat_labels(
        self, arrays: dict[str, np.ndarray], metadata: dict
    ) -> None:
        validity = arrays["cloudsat_profile_valid"].astype(bool)
        valid_fraction = float(np.mean(validity)) if len(validity) else 0.0
        metadata["cloudsat_valid_profile_fraction"] = valid_fraction
        if valid_fraction < self.config.min_cloudsat_valid_fraction:
            raise CloudSatLabelError(
                f"CloudSat labels are only {valid_fraction:.1%} valid; required "
                f"{self.config.min_cloudsat_valid_fraction:.1%}"
            )

        mask = arrays["cloudsat_cloud_class"]
        cloudy_fraction = (
            float(np.mean(mask[mask >= 0] > 0))
            if np.any(mask >= 0)
            else 0.0
        )
        metadata["cloudsat_cloud_pixel_fraction"] = cloudy_fraction
        metadata["cloudsat_cloud_pixel_percentage"] = cloudy_fraction * 100.0
        metadata["cloudsat_cloud_percentage"] = cloudy_fraction * 100.0

        if mask.ndim == 2 and len(validity):
            valid_profiles = validity[: mask.shape[0]]
            if np.any(valid_profiles):
                cloudy_profiles = np.any(mask[valid_profiles] > 0, axis=1)
                cloudy_profile_fraction = float(np.mean(cloudy_profiles))
            else:
                cloudy_profile_fraction = 0.0
        else:
            cloudy_profile_fraction = float(np.any(mask > 0))
        metadata["cloudsat_cloudy_profile_fraction"] = cloudy_profile_fraction
        metadata["cloudsat_cloudy_profile_percentage"] = (
            cloudy_profile_fraction * 100.0
        )
        if self.config.require_cloud and not np.any(mask > 0):
            raise CloudSatLabelError(
                "CloudSat segment is valid but contains no cloud"
            )

    def _chip_profile_positions(
        self,
        transect: CloudSatTransect,
        center: int,
        center_row: int,
        center_column: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the contiguous CloudSat track crossing the chip vertically."""
        half = self.config.chip_size // 2
        row_min, row_max = center_row - half, center_row + half - 1
        column_min = center_column - half
        column_max = center_column + half - 1

        positions: list[tuple[int, int, int]] = []
        center_pixel = self._profile_pixel(transect, center)
        if not self._inside_chip(
            center_pixel, row_min, row_max, column_min, column_max
        ):
            raise ValueError("Center CloudSat profile is outside the ABI chip")
        positions.append((center, *center_pixel))

        for direction in (-1, 1):
            index = center + direction
            while 0 <= index < len(transect):
                pixel = self._profile_pixel(transect, index)
                if not self._inside_chip(
                    pixel, row_min, row_max, column_min, column_max
                ):
                    break
                positions.append((index, *pixel))
                index += direction

        positions.sort(key=lambda value: (value[1], value[2]))
        indices = np.asarray([value[0] for value in positions], dtype=np.int32)
        rows = np.asarray([value[1] for value in positions], dtype=np.int32)
        columns = np.asarray([value[2] for value in positions], dtype=np.int32)
        if len(rows) < 2:
            raise ValueError("Too few CloudSat profiles cross the ABI chip")

        row_steps = np.abs(np.diff(rows))
        positive_steps = row_steps[row_steps > 0]
        typical_step = (
            float(np.median(positive_steps)) if len(positive_steps) else 1.0
        )
        edge_tolerance = max(2.0, 2.0 * typical_step)
        if (
            rows[0] - row_min > edge_tolerance
            or row_max - rows[-1] > edge_tolerance
        ):
            raise ValueError(
                "CloudSat track does not cross the ABI chip from top to bottom"
            )
        return indices, rows, columns

    def _profile_pixel(
        self, transect: CloudSatTransect, index: int
    ) -> tuple[int, int]:
        key = (str(transect.source), index)
        if key not in self._profile_pixel_cache:
            self._profile_pixel_cache[key] = self.abi.geometry.nearest(
                float(transect.latitude[index]),
                float(transect.longitude[index]),
            )
        return self._profile_pixel_cache[key]

    @staticmethod
    def _inside_chip(
        pixel: tuple[int, int],
        row_min: int,
        row_max: int,
        column_min: int,
        column_max: int,
    ) -> bool:
        row, column = pixel
        return (
            row_min <= row <= row_max
            and column_min <= column <= column_max
        )

    def _metadata(
        self,
        orbit_file: CloudSatOrbitFile,
        center: int,
        center_time: str,
        latitude: float,
        longitude: float,
        row: int,
        column: int,
        valid_fraction: float,
    ) -> dict:
        return {
            "schema_version": 1,
            "satellite": self.config.satellite.name,
            "satellite_region": self.config.satellite.region,
            "cloudsat_orbit": orbit_file.orbit,
            "cloudsat_center_profile": center,
            "cloudsat_center_time": center_time,
            "center_latitude": latitude,
            "center_longitude": longitude,
            "abi_row": row,
            "abi_column": column,
            "abi_valid_fraction": valid_fraction,
            "abi_inner_disk_margin": self.config.inner_disk_margin,
            "abi_common_grid_resolution_km": 1.0,
            "abi_solar_zenith_units": "degrees",
            "abi_view_zenith_units": "degrees",
            "chip_size": self.config.chip_size,
            "cloudsat_profile_selection": self.config.profile_selection,
            "cloudsat_profiles_per_chip": (
                self.config.profiles_per_chip
                if self.config.profile_selection == "fixed"
                else None
            ),
            "transect_latitude_min": self.config.transect[0],
            "transect_latitude_max": self.config.transect[1],
            "metadata_groups": sorted(self.config.metadata),
            "abi_root": str(self.config.abi_root),
            "abi_geometry_source": str(self.config.latlon_path),
            "cloudsat_source": str(orbit_file.path),
        }


_WORKER_PIPELINE: CloudSatABICollocationPipeline | None = None


def _initialize_worker(config: CropConfig) -> None:
    global _WORKER_PIPELINE
    _WORKER_PIPELINE = CloudSatABICollocationPipeline(config)


def _process_orbit_worker(orbit_file: CloudSatOrbitFile) -> OrbitResult:
    if _WORKER_PIPELINE is None:
        raise RuntimeError("CloudSat-ABI worker was not initialized")
    return _WORKER_PIPELINE.process_orbit(orbit_file)


def run_parallel(config: CropConfig, workers: int) -> int:
    """Process independent CloudSat orbits using persistent worker pipelines."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return CloudSatABICollocationPipeline(config).run()
    if config.max_chips is not None:
        raise ValueError(
            "max_chips requires workers=1 to preserve the exact output limit"
        )

    reader = CloudSatReader(config.cloudsat_root, config.cloudsat_index_root)
    orbit_files = list(
        reader.discover_orbits(
            config.year,
            config.day_start,
            config.day_end,
            config.orbit,
        )
    )
    if not orbit_files:
        return 0

    LOG.info(
        "Processing %d CloudSat orbit(s) with %d workers. Each worker loads "
        "its own ABI geometry grid.",
        len(orbit_files),
        workers,
    )
    written = 0
    context = multiprocessing.get_context("spawn")
    worker_config = replace(config, progress=False)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(worker_config,),
    ) as executor:
        futures = {
            executor.submit(_process_orbit_worker, orbit_file): orbit_file
            for orbit_file in orbit_files
        }
        completed = _progress(
            as_completed(futures),
            enabled=config.progress,
            total=len(futures),
            desc="CloudSat orbits",
            unit="orbit",
        )
        for future in completed:
            orbit_file = futures[future]
            try:
                result = future.result()
            except Exception:
                LOG.exception(
                    "Worker failed for orbit %s on day %03d",
                    orbit_file.orbit,
                    orbit_file.day,
                )
                continue
            written += result.written
            if hasattr(completed, "set_postfix"):
                completed.set_postfix(chips=written, refresh=False)
            LOG.info(
                "Parallel progress: orbit %s complete, total new chips=%d",
                result.orbit,
                written,
            )
    return written
