#!/usr/bin/env python3
"""Generate causal ABI inputs with future A3DVOCADO nowcasting targets."""

import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from satvision_pix4d.preprocessing.a3dvocado_label_generator import (
    DEFAULT_STAGE_BINS,
    DEFAULT_TRUTH_ROOT,
    MonthlyConvectionTruth,
    classify_life_stage,
    nearest_truth_indices,
    sample_truth_grid,
    target_class_sequence,
    target_sequence_summary,
)
from satvision_pix4d.preprocessing.abi_convection_chip_generator import (
    DEFAULT_GOES16_ABI_ROOT,
    DEFAULT_GOES16_GEO_FILE,
    DEFAULT_GOES17_ABI_ROOT,
    DEFAULT_GOES17_GEO_FILE,
    DEFAULT_METADATA_GLOB,
    ABIConvectionChipGenerator,
    load_metadata,
)


DEFAULT_OUTPUT_DIR = Path(
    "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/"
    "3-tiles/a3dvocado-nowcasting"
)
LIFECYCLE_ANCHOR_FRACTIONS = (0.125, 0.375, 0.625, 0.875)


def load_nowcasting_metadata(csv_path, sampling="balanced"):
    """Load records using lifecycle-balanced anchors by default."""
    metadata = load_metadata(csv_path, one_per_system=False)
    if sampling == "all":
        return metadata
    if sampling == "middle":
        group_columns = ["satellite", "system_id"]
        position = metadata.groupby(group_columns).cumcount()
        group_size = metadata.groupby(group_columns)["system_id"].transform("size")
        return metadata[position == group_size // 2].reset_index(drop=True)
    if sampling != "balanced":
        raise ValueError(f"Unsupported metadata sampling mode: {sampling}")

    group_columns = ["satellite", "system_id"]
    position = metadata.groupby(group_columns).cumcount()
    group_size = metadata.groupby(group_columns)["system_id"].transform("size")
    lifecycle_bin = metadata["system_id"].astype(np.int64) % len(
        LIFECYCLE_ANCHOR_FRACTIONS
    )
    fraction = lifecycle_bin.map(
        dict(enumerate(LIFECYCLE_ANCHOR_FRACTIONS))
    )
    target_position = np.rint(fraction * (group_size - 1)).astype(np.int64)
    return metadata[position == target_position].reset_index(drop=True)


class A3DVOCADONowcastingGenerator(ABIConvectionChipGenerator):
    """Create 7-frame causal ABI inputs and N future truth targets."""

    def __init__(
        self,
        truth_root=DEFAULT_TRUTH_ROOT,
        forecast_steps=3,
        forecast_step_minutes=20,
        stage_bins=DEFAULT_STAGE_BINS,
        require_target_at_analysis=True,
        **kwargs,
    ):
        input_steps = kwargs.get("n_timesteps", 7)
        kwargs["convection_index"] = input_steps - 1
        super().__init__(**kwargs)
        self.truth = MonthlyConvectionTruth(truth_root)
        self.forecast_steps = int(forecast_steps)
        self.forecast_step_minutes = int(forecast_step_minutes)
        self.stage_bins = tuple(float(value) for value in stage_bins)
        self.require_target_at_analysis = require_target_at_analysis

    def _output_path(self, record):
        event_time = pd.Timestamp(record["datetime"])
        satellite_token = self.satellite_name.replace("-", "")
        return self.output_dir / (
            f"{satellite_token}_nowcast_"
            f"{event_time.strftime('%Y%m%dT%H%M')}_"
            f"sys{int(record['system_id'])}.npz"
        )

    def _sample_truth_sequence(
        self,
        timestamps,
        latitude,
        longitude,
        system_id,
    ):
        label_lists = {
            "dcs_id_mask": [],
            "local_stage_index": [],
            "local_duration_steps": [],
            "parent_stage": [],
            "parent_duration_hours": [],
        }
        mapping_cache = {}
        truth_sources = {}
        truth_positions = []

        for timestamp in timestamps:
            truth_grid = self.truth.grid_for_timestep(timestamp)
            month = truth_grid["month"]
            if month not in mapping_cache:
                mapping_cache[month] = nearest_truth_indices(
                    latitude,
                    longitude,
                    truth_grid["latitude"],
                    truth_grid["longitude"],
                )
            latitude_index, longitude_index, valid = mapping_cache[month]
            if not valid.any():
                raise ValueError("ABI chip does not intersect the truth grid.")

            latitude_start = int(latitude_index[valid].min())
            latitude_stop = int(latitude_index[valid].max()) + 1
            longitude_start = int(longitude_index[valid].min())
            longitude_stop = int(longitude_index[valid].max()) + 1
            truth = self.truth.read_window(
                timestamp,
                slice(latitude_start, latitude_stop),
                slice(longitude_start, longitude_stop),
            )
            local_latitude_index = latitude_index - latitude_start
            local_longitude_index = longitude_index - longitude_start

            values = {
                "dcs_id_mask": sample_truth_grid(
                    truth["mask"],
                    local_latitude_index,
                    local_longitude_index,
                    valid,
                    fill_value=0,
                    dtype=np.int32,
                ),
                "local_stage_index": sample_truth_grid(
                    truth["local_stage"],
                    local_latitude_index,
                    local_longitude_index,
                    valid,
                    fill_value=-1,
                    dtype=np.int16,
                ),
                "local_duration_steps": sample_truth_grid(
                    truth["local_duration"],
                    local_latitude_index,
                    local_longitude_index,
                    valid,
                    fill_value=-1,
                    dtype=np.int16,
                ),
                "parent_stage": sample_truth_grid(
                    truth["parent_stage"],
                    local_latitude_index,
                    local_longitude_index,
                    valid,
                    fill_value=np.nan,
                    dtype=np.float32,
                ),
                "parent_duration_hours": sample_truth_grid(
                    truth["parent_duration"],
                    local_latitude_index,
                    local_longitude_index,
                    valid,
                    fill_value=np.nan,
                    dtype=np.float32,
                ),
            }
            values["dcs_id_mask"][values["dcs_id_mask"] < 0] = 0
            values["parent_stage"][values["parent_stage"] < 0] = np.nan
            values["parent_duration_hours"][
                values["parent_duration_hours"] < 0
            ] = np.nan
            for key, value in values.items():
                label_lists[key].append(value)

            truth_sources[month] = truth["paths"]
            truth_positions.append(
                [truth["day_index"], truth["time_index"]]
            )

        labels = {
            key: np.stack(values, axis=0)
            for key, values in label_lists.items()
        }
        labels["convection_mask"] = (
            labels["dcs_id_mask"] > 0
        ).astype(np.uint8)
        labels["target_system_mask"] = (
            labels["dcs_id_mask"] == system_id
        ).astype(np.uint8)
        labels["life_stage_class"] = classify_life_stage(
            labels["parent_stage"],
            labels["convection_mask"].astype(bool),
            self.stage_bins,
        )
        labels["target_life_stage_class_mask"] = np.where(
            labels["target_system_mask"].astype(bool),
            labels["life_stage_class"],
            0,
        ).astype(np.uint8)

        target_mask = labels["target_system_mask"].astype(bool)
        labels["target_stage_class_sequence"] = target_class_sequence(
            labels["life_stage_class"],
            target_mask,
        )
        labels["target_parent_stage_sequence"] = target_sequence_summary(
            labels["parent_stage"],
            target_mask,
            np.nan,
        )
        labels["target_duration_hours_sequence"] = target_sequence_summary(
            labels["parent_duration_hours"],
            target_mask,
            np.nan,
        )
        return labels, truth_sources, truth_positions

    def generate_chip(self, record):
        if record["satellite"] != self.satellite_label:
            return False

        output_path = self._output_path(record)
        if output_path.exists():
            return False

        event_time = pd.Timestamp(record["datetime"])
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
            return False

        input_requested_times = [
            event_time
            - timedelta(
                minutes=self.step_minutes * (self.n_timesteps - 1 - index)
            )
            for index in range(self.n_timesteps)
        ]
        future_times = [
            event_time
            + timedelta(
                minutes=self.forecast_step_minutes * (index + 1)
            )
            for index in range(self.forecast_steps)
        ]

        actual_input_times = []
        input_frames = []
        source_files = []
        try:
            for requested_time in input_requested_times:
                actual_time, frame, frame_source_files = self._read_scan(
                    requested_time,
                    center_y,
                    center_x,
                )
                actual_input_times.append(actual_time)
                input_frames.append(frame)
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
        input_truth_labels, input_truth_sources, input_truth_positions = (
            self._sample_truth_sequence(
                input_requested_times,
                latitude,
                longitude,
                int(record["system_id"]),
            )
        )
        target_present = bool(
            input_truth_labels["target_system_mask"][-1].any()
        )
        if self.require_target_at_analysis and not target_present:
            return False

        future_labels, future_sources, future_positions = (
            self._sample_truth_sequence(
                future_times,
                latitude,
                longitude,
                int(record["system_id"]),
            )
        )
        solar_zenith_angle = np.stack(
            [
                self._solar_zenith_angle(latitude, longitude, timestamp)
                for timestamp in actual_input_times
            ],
            axis=0,
        )
        view_zenith = self._view_zenith_angle(latitude, longitude)
        view_zenith_angle = np.broadcast_to(
            view_zenith,
            (self.n_timesteps, self.tile_size, self.tile_size),
        ).copy()

        metadata = {
            "task": "a3dvocado_nowcasting",
            "system_id": int(record["system_id"]),
            "analysis_time": event_time.isoformat(),
            "input_steps": self.n_timesteps,
            "input_step_minutes": self.step_minutes,
            "forecast_steps": self.forecast_steps,
            "forecast_step_minutes": self.forecast_step_minutes,
            "satellite": self.satellite_name,
            "satellite_region": self.satellite_label,
            "center_latitude": float(record["latitude"]),
            "center_longitude": float(record["longitude"]),
            "abi_center_y": int(center_y),
            "abi_center_x": int(center_x),
            "stage_bins": list(self.stage_bins),
            "input_truth_sources": input_truth_sources,
            "input_truth_positions": input_truth_positions,
            "future_truth_sources": future_sources,
            "future_truth_positions": future_positions,
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

        arrays = {
            "chip": np.stack(input_frames, axis=0).astype(np.float32),
            "timestamps": np.asarray(
                [timestamp.isoformat() for timestamp in actual_input_times]
            ),
            "requested_timestamps": np.asarray(
                [timestamp.isoformat() for timestamp in input_requested_times]
            ),
            "future_timestamps": np.asarray(
                [timestamp.isoformat() for timestamp in future_times]
            ),
            "bands": np.asarray(self.channels, dtype=np.int16),
            "latitude": latitude,
            "longitude": longitude,
            "solar_zenith_angle": solar_zenith_angle,
            "view_zenith_angle": view_zenith_angle,
            "source_files": np.asarray(source_files),
            "analysis_target_system_mask": input_truth_labels[
                "target_system_mask"
            ][-1],
            "analysis_life_stage_class": input_truth_labels[
                "life_stage_class"
            ][-1],
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        arrays.update(
            {
                f"input_truth_{key}": value
                for key, value in input_truth_labels.items()
            }
        )
        arrays.update(
            {
                f"future_{key}": value
                for key, value in future_labels.items()
            }
        )

        temporary_path = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, output_path)
        logging.info("Saved %s", output_path)
        return True


_WORKER_CONFIGS = {}
_WORKER_GENERATORS = {}


def _initialize_worker(configs):
    global _WORKER_CONFIGS
    global _WORKER_GENERATORS
    _WORKER_CONFIGS = configs
    _WORKER_GENERATORS = {}


def _process_record(record):
    try:
        satellite = record["satellite"]
        generator = _WORKER_GENERATORS.get(satellite)
        if generator is None:
            generator = A3DVOCADONowcastingGenerator(
                **_WORKER_CONFIGS[satellite]
            )
            _WORKER_GENERATORS[satellite] = generator
        return generator.generate_chip(pd.Series(record))
    except Exception:
        logging.exception(
            "Failed nowcasting sample for system %s.",
            record.get("system_id"),
        )
        return False


def _records(metadata):
    columns = list(metadata.columns)
    for values in metadata.itertuples(index=False, name=None):
        yield dict(zip(columns, values))


def generate_parallel(metadata, configs, workers):
    record_iterator = iter(_records(metadata))
    pending = set()
    max_pending = max(2 * workers, 1)
    saved = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(configs,),
    ) as executor:
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(_process_record, next(record_iterator)))
            except StopIteration:
                break

        with tqdm(total=len(metadata), desc="Generating nowcasting samples") as progress:
            while pending:
                completed, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    if future.result():
                        saved += 1
                    progress.update(1)
                    try:
                        pending.add(
                            executor.submit(
                                _process_record,
                                next(record_iterator),
                            )
                        )
                    except StopIteration:
                        pass
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate causal ABI inputs and future A3DVOCADO targets."
    )
    parser.add_argument("metadata_csv", nargs="*")
    parser.add_argument("--metadata-glob", default=DEFAULT_METADATA_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT)
    parser.add_argument("--goes16-abi-root", type=Path, default=DEFAULT_GOES16_ABI_ROOT)
    parser.add_argument("--goes17-abi-root", type=Path, default=DEFAULT_GOES17_ABI_ROOT)
    parser.add_argument("--goes16-geo-file", type=Path, default=DEFAULT_GOES16_GEO_FILE)
    parser.add_argument("--goes17-geo-file", type=Path, default=DEFAULT_GOES17_GEO_FILE)
    parser.add_argument("--input-steps", type=int, default=7)
    parser.add_argument("--input-step-minutes", type=int, default=20)
    parser.add_argument("--forecast-steps", type=int, default=3)
    parser.add_argument("--forecast-step-minutes", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stage-bins", type=float, nargs=3, default=DEFAULT_STAGE_BINS)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sampling",
        choices=("balanced", "middle", "all"),
        default="balanced",
        help=(
            "Choose one lifecycle-balanced record per system, the middle "
            "record, or every record."
        ),
    )
    parser.add_argument("--all-records", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    csv_files = [Path(path) for path in args.metadata_csv]
    if not csv_files:
        csv_files = [Path(path) for path in sorted(glob(args.metadata_glob))]
    if not csv_files:
        raise FileNotFoundError("No convection metadata CSV files were found.")

    common = {
        "output_dir": args.output_dir,
        "truth_root": args.truth_root,
        "tile_size": args.tile_size,
        "n_timesteps": args.input_steps,
        "step_minutes": args.input_step_minutes,
        "forecast_steps": args.forecast_steps,
        "forecast_step_minutes": args.forecast_step_minutes,
        "stage_bins": args.stage_bins,
    }
    configs = {
        "GOES-East": {
            **common,
            "satellite_label": "GOES-East",
            "satellite_number": 16,
            "satellite_longitude": -75.0,
            "abi_root": args.goes16_abi_root,
            "geo_file": args.goes16_geo_file,
        },
        "GOES-West": {
            **common,
            "satellite_label": "GOES-West",
            "satellite_number": 17,
            "satellite_longitude": -137.0,
            "abi_root": args.goes17_abi_root,
            "geo_file": args.goes17_geo_file,
        },
    }

    sampling = "all" if args.all_records else args.sampling
    metadata_parts = [
        load_nowcasting_metadata(path, sampling=sampling)
        for path in csv_files
    ]
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if args.max_samples is not None:
        metadata = metadata.iloc[: args.max_samples].copy()
    logging.info(
        "Selected %s nowcasting records: %s",
        len(metadata),
        metadata["satellite"].value_counts().to_dict(),
    )

    if args.workers == 1:
        generators = {
            label: A3DVOCADONowcastingGenerator(**config)
            for label, config in configs.items()
        }
        saved = 0
        for _, record in tqdm(
            metadata.iterrows(),
            total=len(metadata),
            desc="Generating nowcasting samples",
        ):
            if generators[record["satellite"]].generate_chip(record):
                saved += 1
    else:
        saved = generate_parallel(metadata, configs, args.workers)

    logging.info("Finished. Saved %s nowcasting samples.", saved)


if __name__ == "__main__":
    main()
