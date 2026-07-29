"""Orchestration for CloudSat and multitemporal ABI collocation."""

from __future__ import annotations

import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np

from .abi import ABIArchive, ABIGeometry
from .cloudsat import (
    CloudSatAuxiliaryReader,
    CloudSatAuxiliaryTransect,
    CloudSatOrbitFile,
    CloudSatReader,
    CloudSatTransect,
)
from .config import CropConfig

from .utils import datetime_from_year_doy
from .writer import (
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
        self.writer = writer or NPZChipWriter(
            config.output_dir, config.overwrite
        )

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
        if len(transect) < self.config.profiles_per_chip:
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
        if "abi pixel" in message or "abi valid" in message:
            return "abi_quality"
        if "valid timesteps" in message:
            return "abi_temporal"
        if "track" in message or "profiles cross" in message:
            return "cloudsat_track"
        return type(exc).__name__.lower()

    def _profile_centers(self, transect: CloudSatTransect) -> range:
        """Return candidate center indices for sliding-window sampling.

        Each center must have enough profiles on both sides to form a
        complete window of profiles_per_chip contiguous footprints.
        """
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
        """Build a single collocated sample from 512 contiguous CloudSat profiles.

        The spatial footprint is determined once from the CloudSat overpass
        (the middle timestep, offset=0). Those same 512 ABI pixel locations
        are then reused to extract radiance values at all 7 temporal offsets.
        """

        # ── Step 1: Anchor on the center CloudSat profile ──────────────────
        # The center profile defines the reference time (middle timestep).
        center_time = datetime_from_year_doy(
            self.config.year, orbit_file.day, transect.utc_hour[center]
        )
        center_latitude = float(transect.latitude[center])
        center_longitude = float(transect.longitude[center])

        # Quick validity check at the center before any expensive work.
        if (
            "cloudsat" in self.config.metadata
            and self.config.min_cloudsat_valid_fraction > 0
            and not transect.profile_validity()[center]
        ):
            raise CloudSatLabelError(
                "Center CloudSat profile has no valid retrieval"
            )

        # ── Step 2: Select a contiguous window of 512 CloudSat profiles ────
        # profile_window returns an array of indices centered on `center`,
        # covering exactly `profiles_per_chip` consecutive footprints.
        profile_indices = transect.profile_window(
            center, self.config.profiles_per_chip
        )
        segment_len = len(profile_indices)

        # Extract the latitude and longitude for each of the 512 profiles.
        profile_lats = np.asarray(
            transect.latitude[profile_indices], dtype=np.float64
        )
        profile_lons = np.asarray(
            transect.longitude[profile_indices], dtype=np.float64
        )

        # ── Step 3: Gather CloudSat auxiliary metadata ─────────────────────
        # This is done before the expensive ABI I/O so we can bail out early
        # if the CloudSat labels are invalid.
        auxiliary: dict[str, np.ndarray] = {}
        if "cloudsat" in self.config.metadata:
            auxiliary.update(
                transect.metadata_arrays_for_indices(profile_indices)
            )

        # ── Step 4: Map each CloudSat footprint to its nearest ABI pixel ───
        # For each of the 512 lat/lon pairs, find the closest pixel on the
        # ABI 1 km common grid. This gives us two 1D arrays (rows, columns)
        # of length 512 that we will reuse for every temporal offset.
        profile_pixels = np.asarray(
            [
                self.abi.geometry.nearest(float(lat), float(lon))
                for lat, lon in zip(profile_lats, profile_lons)
            ],
            dtype=np.int32,
        )
        abi_rows = profile_pixels[:, 0]
        abi_cols = profile_pixels[:, 1]

        # ── Step 4b: Inner disk margin check ───────────────────────────────
        # Verify that ALL 512 footprint pixels are well within the ABI grid,
        # not just the center. This filters out transects that clip the limb.
        if self.config.inner_disk_margin > 0:
            grid_rows, grid_cols = self.abi.geometry.latitude.shape
            margin = self.config.inner_disk_margin
            outside = (
                (abi_rows < margin)
                | (abi_rows >= grid_rows - margin)
                | (abi_cols < margin)
                | (abi_cols >= grid_cols - margin)
            )
            n_outside = int(np.sum(outside))
            if n_outside > 0:
                raise ValueError(
                    f"{n_outside} of {segment_len} footprints are outside "
                    f"the inner disk margin of {margin} pixels"
                )

        if "cloudsat" in self.config.metadata:
            auxiliary["cloudsat_abi_row"] = abi_rows
            auxiliary["cloudsat_abi_column"] = abi_cols
            auxiliary["cloudsat_profile_index"] = profile_indices.astype(
                np.int32
            )
            self._add_cloudsat_auxiliary(
                auxiliary,
                metadata=None,
                auxiliary_transect=auxiliary_transect,
                profile_indices=profile_indices,
            )

        # The center profile's ABI pixel coordinates (for metadata only).
        center_row = int(abi_rows[segment_len // 2])
        center_col = int(abi_cols[segment_len // 2])

        # ── Step 5: Build filename and metadata ────────────────────────────
        timestamp = center_time.strftime("%Y%m%dT%H%M%SZ")
        filename = (
            f"{self.config.satellite.filename_token}_"
            f"{self.config.satellite.region}_abi_cloudsat_{timestamp}_"
            f"orbit{orbit_file.orbit}_r{center_row}_c{center_col}"
            f"_p{center}_len{segment_len}.npz"
        )
        metadata = self._metadata(
            orbit_file,
            center,
            center_time.isoformat(),
            center_latitude,
            center_longitude,
            center_row,
            center_col,
        )

        if "cloudsat" in self.config.metadata:
            self._validate_cloudsat_labels(auxiliary, metadata)
        if "cloudsat_aux" in self.config.metadata:
            if auxiliary_transect is None:
                raise FileNotFoundError(
                    "CloudSat ECMWF-AUX metadata was requested"
                )
            metadata["cloudsat_aux_source"] = str(auxiliary_transect.source)

        # ── Step 6: Extract ABI radiance at each temporal offset ───────────
        # We loop over the 7 offsets (-60 .. +60 min) and call
        # extract_transect with the SAME 512 (row, col) positions each time.
        # The spatial footprint is fixed by CloudSat; only the ABI
        # observations at those locations vary across time.
        chips, scan_times, valid, requested_times, angle_times = (
            [], [], [], [], [],
        )
        for offset in self.config.offsets:
            requested = center_time + timedelta(minutes=offset)
            requested_times.append(requested)
            try:
                # extract_transect returns shape (512, 16): one radiance
                # value per footprint per ABI channel.
                chip, scan_time = self.abi.extract_transect(
                    requested, abi_rows, abi_cols
                )
                valid.append(1)
            except (FileNotFoundError, ValueError) as exc:
                if not self.config.allow_missing_timesteps:
                    raise
                LOG.warning(
                    "Missing timestep %s: %s", requested.isoformat(), exc
                )
                # Fill with NaN: shape (512, 16) matching extract_transect.
                chip = np.full(
                    (segment_len, 16), np.nan, dtype=np.float32,
                )
                scan_time = None
                valid.append(0)
            chips.append(chip)
            scan_times.append(scan_time.isoformat() if scan_time else "")
            angle_times.append(scan_time or requested)

        # ── Step 7: Compute solar and view zenith angles ───────────────────
        # Pass the 512 CloudSat lat/lons directly into the angle functions.
        # Each returns a 1D array of length 512 (one angle per footprint).
        solar_zenith_angle = np.stack(
            [
                self.abi.geometry.solar_zenith_angle(
                    profile_lats, profile_lons, angle_time
                )
                for angle_time in angle_times
            ]
        )
        view_zenith_angle = self.abi.geometry.view_zenith_angle(
            profile_lats,
            profile_lons,
            self.config.satellite.subpoint_longitude,
        )
        auxiliary["abi_solar_zenith_angle"] = solar_zenith_angle
        auxiliary["abi_view_zenith_angle"] = view_zenith_angle[:, np.newaxis]

        # ── Step 8: Timestep-level validation ──────────────────────────────
        # After pixel-level masking (done in extract_transect), check each
        # timestep. If every pixel across all channels is NaN (entirely
        # out-of-bounds or fill), mark that timestep invalid and ensure
        # all its values are explicitly NaN.
        #
        # Additionally, check that each timestep has enough valid (non-NaN)
        # ABI pixels to meet the min_abi_valid_fraction threshold.
        stacked = np.stack(chips).astype(np.float32)
        valid_arr = np.asarray(valid, dtype=np.int8)
        for t in range(stacked.shape[0]):
            finite_fraction = float(np.mean(np.isfinite(stacked[t])))
            if finite_fraction < self.config.min_abi_valid_fraction:
                valid_arr[t] = 0
                stacked[t] = np.nan

        # Check that enough timesteps survived validation.
        n_valid = int(valid_arr.sum())
        if n_valid < self.config.min_valid_timesteps:
            raise ValueError(
                f"Only {n_valid} of {len(valid_arr)} timesteps have valid "
                f"ABI data; required {self.config.min_valid_timesteps}"
            )

        # ── Step 9: Assemble and return the collocated sample ──────────────
        # stacked has shape (7, 512, 16). The dummy spatial dimension for
        # the 3D U-Net is added later in the writer.
        return CollocatedChip(
            filename=filename,
            chip=stacked,
            offsets_minutes=np.asarray(self.config.offsets, dtype=np.int32),
            valid_mask=valid_arr,
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


    def _metadata(
        self,
        orbit_file: CloudSatOrbitFile,
        center: int,
        center_time: str,
        latitude: float,
        longitude: float,
        row: int,
        column: int,
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
            "abi_inner_disk_margin": self.config.inner_disk_margin,
            "abi_common_grid_resolution_km": 1.0,
            "abi_solar_zenith_units": "degrees",
            "abi_view_zenith_units": "degrees",
            "chip_size": self.config.chip_size,
            "cloudsat_profiles_per_chip": self.config.profiles_per_chip,
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
