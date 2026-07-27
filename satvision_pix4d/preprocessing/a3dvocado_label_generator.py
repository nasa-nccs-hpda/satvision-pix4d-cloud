#!/usr/bin/env python3
"""Add time-aligned convection masks and life-stage labels to ABI NPZ chips."""

import argparse
import atexit
import json
import logging
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CHIP_GLOB = (
    "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/"
    "3-tiles/convection/*.npz"
)
DEFAULT_TRUTH_ROOT = Path(
    "/explore/nobackup/projects/pix4dcloud/data/cloudsystem_2019-2020"
)
DEFAULT_OUTPUT_DIR = Path(
    "/explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d/"
    "3-tiles/a3dvocado"
)
DEFAULT_STAGE_BINS = (0.25, 0.5, 0.75)
STAGE_CLASS_NAMES = {
    0: "background",
    1: "initiating",
    2: "developing",
    3: "mature",
    4: "decaying",
}


def classify_life_stage(parent_stage, valid_mask, stage_bins):
    """Convert normalized parent stage to background plus four life stages."""
    classes = np.zeros(parent_stage.shape, dtype=np.uint8)
    valid = valid_mask & np.isfinite(parent_stage) & (parent_stage >= 0)
    classes[valid] = (
        np.digitize(parent_stage[valid], stage_bins, right=False) + 1
    ).astype(np.uint8)
    return classes


def nearest_truth_indices(
    chip_latitude,
    chip_longitude,
    truth_latitude,
    truth_longitude,
):
    """Map ABI chip coordinates to the regular 0.12-degree truth grid."""
    latitude = np.asarray(chip_latitude)
    longitude = np.asarray(chip_longitude)
    truth_latitude = np.asarray(truth_latitude)
    truth_longitude = np.asarray(truth_longitude)

    longitude_360 = np.where(longitude < truth_longitude[0], longitude + 360, longitude)
    latitude_step = truth_latitude[1] - truth_latitude[0]
    longitude_step = truth_longitude[1] - truth_longitude[0]

    safe_latitude = np.where(np.isfinite(latitude), latitude, truth_latitude[0])
    safe_longitude = np.where(
        np.isfinite(longitude_360),
        longitude_360,
        truth_longitude[0],
    )
    latitude_index = np.rint(
        (safe_latitude - truth_latitude[0]) / latitude_step
    ).astype(np.int32)
    longitude_index = np.rint(
        (safe_longitude - truth_longitude[0]) / longitude_step
    ).astype(np.int32)

    in_bounds = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude_index >= 0)
        & (latitude_index < truth_latitude.size)
        & (longitude_index >= 0)
        & (longitude_index < truth_longitude.size)
    )
    clipped_latitude_index = np.clip(
        latitude_index,
        0,
        truth_latitude.size - 1,
    )
    clipped_longitude_index = np.clip(
        longitude_index,
        0,
        truth_longitude.size - 1,
    )

    latitude_error = np.abs(
        truth_latitude[clipped_latitude_index] - latitude
    )
    longitude_error = np.abs(
        truth_longitude[clipped_longitude_index] - longitude_360
    )
    tolerance = max(abs(latitude_step), abs(longitude_step)) * 0.51
    valid = in_bounds & (latitude_error <= tolerance) & (longitude_error <= tolerance)
    return clipped_latitude_index, clipped_longitude_index, valid


def sample_truth_grid(
    truth_array,
    latitude_index,
    longitude_index,
    valid,
    fill_value,
    dtype,
):
    sampled = np.full(valid.shape, fill_value, dtype=dtype)
    sampled[valid] = truth_array[
        latitude_index[valid],
        longitude_index[valid],
    ].astype(dtype, copy=False)
    return sampled


def target_sequence_summary(values, target_mask, fill_value):
    summary = np.full(values.shape[0], fill_value, dtype=np.float32)
    for time_index in range(values.shape[0]):
        selected = values[time_index][target_mask[time_index]]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            summary[time_index] = np.median(selected)
    return summary


def target_class_sequence(stage_classes, target_mask):
    sequence = np.zeros(stage_classes.shape[0], dtype=np.uint8)
    for time_index in range(stage_classes.shape[0]):
        selected = stage_classes[time_index][target_mask[time_index]]
        selected = selected[selected > 0]
        if selected.size:
            sequence[time_index] = np.bincount(selected).argmax()
    return sequence


class MonthlyConvectionTruth:
    """Cache monthly DCS, local-stage, and parent-stage truth files."""

    def __init__(self, truth_root):
        self.truth_root = Path(truth_root)
        self._months = {}
        atexit.register(self.close)

    def _paths(self, month):
        return {
            "mask": (
                self.truth_root
                / "mask"
                / f"{month}_DCS_number_monthly.nc"
            ),
            "local_stage": self.truth_root / "local" / f"{month}_stage.nc",
            "local_duration": self.truth_root / "local" / f"{month}_duration.nc",
            "parent_stage": self.truth_root / "parent" / f"{month}_stage.nc",
            "parent_duration": self.truth_root / "parent" / f"{month}_duration.nc",
        }

    def _open_month(self, month):
        if month in self._months:
            return self._months[month]

        paths = self._paths(month)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing monthly convection truth files: {missing}"
            )

        handles = {
            name: h5py.File(path, "r")
            for name, path in paths.items()
        }
        parent_stage = handles["parent_stage"]
        month_data = {
            "handles": handles,
            "paths": paths,
            "latitude": parent_stage["lat"][...],
            "longitude": parent_stage["lon"][...],
        }
        self._months[month] = month_data
        return month_data

    @staticmethod
    def _truth_position(timestamp):
        timestamp = pd.Timestamp(timestamp)
        seconds = (
            timestamp.hour * 3600
            + timestamp.minute * 60
            + timestamp.second
            + timestamp.microsecond / 1e6
        )
        time_index = int(np.floor(seconds / 1200 + 0.5))
        date = timestamp.normalize()
        if time_index == 72:
            date += pd.Timedelta(days=1)
            time_index = 0
        return date.strftime("%Y%m"), date.day - 1, time_index

    def grid_for_timestep(self, timestamp):
        month, day_index, time_index = self._truth_position(timestamp)
        month_data = self._open_month(month)
        return {
            "month": month,
            "day_index": day_index,
            "time_index": time_index,
            "latitude": month_data["latitude"],
            "longitude": month_data["longitude"],
            "paths": {
                name: str(path)
                for name, path in month_data["paths"].items()
            },
        }

    def read_window(self, timestamp, latitude_slice, longitude_slice):
        month, day_index, time_index = self._truth_position(timestamp)
        month_data = self._open_month(month)
        handles = month_data["handles"]
        return {
            "month": month,
            "day_index": day_index,
            "time_index": time_index,
            "mask": handles["mask"]["DCS_number"][
                day_index,
                time_index,
                latitude_slice,
                longitude_slice,
            ],
            "local_stage": handles["local_stage"]["stage"][
                day_index,
                time_index,
                latitude_slice,
                longitude_slice,
            ],
            "local_duration": handles["local_duration"]["duration"][
                day_index,
                time_index,
                latitude_slice,
                longitude_slice,
            ],
            "parent_stage": handles["parent_stage"]["stage"][
                day_index,
                time_index,
                latitude_slice,
                longitude_slice,
            ],
            "parent_duration": handles["parent_duration"]["duration"][
                day_index,
                time_index,
                latitude_slice,
                longitude_slice,
            ],
            "paths": {
                name: str(path)
                for name, path in month_data["paths"].items()
            },
        }

    def close(self):
        for month_data in self._months.values():
            for handle in month_data["handles"].values():
                try:
                    handle.close()
                except Exception:
                    pass
        self._months.clear()


class A3DVOCADOLabelGenerator:
    """Create paired ABI and convection-label NPZ files."""

    def __init__(
        self,
        truth_root=DEFAULT_TRUTH_ROOT,
        output_dir=DEFAULT_OUTPUT_DIR,
        stage_bins=DEFAULT_STAGE_BINS,
        require_target=True,
    ):
        self.truth = MonthlyConvectionTruth(truth_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_bins = tuple(float(value) for value in stage_bins)
        self.require_target = require_target

    def _output_path(self, chip_path):
        return self.output_dir / Path(chip_path).name

    def generate(self, chip_path):
        chip_path = Path(chip_path)
        output_path = self._output_path(chip_path)
        if output_path.exists():
            return "exists"

        with np.load(chip_path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}

        metadata = json.loads(str(arrays["metadata"].item()))
        system_id = int(metadata["system_id"])
        timestamps = pd.to_datetime(
            arrays.get("requested_timestamps", arrays["timestamps"])
        )
        latitude = arrays["latitude"]
        longitude = arrays["longitude"]

        label_lists = {
            "dcs_id_mask": [],
            "local_stage_index": [],
            "local_duration_steps": [],
            "parent_stage": [],
            "parent_duration_hours": [],
        }
        truth_sources = {}
        truth_positions = []
        mapping_cache = {}

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
                raise ValueError(
                    f"ABI chip {chip_path} does not intersect the truth grid."
                )

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

            dcs_id_mask = sample_truth_grid(
                truth["mask"],
                local_latitude_index,
                local_longitude_index,
                valid,
                fill_value=0,
                dtype=np.int32,
            )
            dcs_id_mask[dcs_id_mask < 0] = 0
            local_stage = sample_truth_grid(
                truth["local_stage"],
                local_latitude_index,
                local_longitude_index,
                valid,
                fill_value=-1,
                dtype=np.int16,
            )
            local_duration = sample_truth_grid(
                truth["local_duration"],
                local_latitude_index,
                local_longitude_index,
                valid,
                fill_value=-1,
                dtype=np.int16,
            )
            parent_stage = sample_truth_grid(
                truth["parent_stage"],
                local_latitude_index,
                local_longitude_index,
                valid,
                fill_value=np.nan,
                dtype=np.float32,
            )
            parent_duration = sample_truth_grid(
                truth["parent_duration"],
                local_latitude_index,
                local_longitude_index,
                valid,
                fill_value=np.nan,
                dtype=np.float32,
            )
            parent_stage[parent_stage < 0] = np.nan
            parent_duration[parent_duration < 0] = np.nan

            label_lists["dcs_id_mask"].append(dcs_id_mask)
            label_lists["local_stage_index"].append(local_stage)
            label_lists["local_duration_steps"].append(local_duration)
            label_lists["parent_stage"].append(parent_stage)
            label_lists["parent_duration_hours"].append(parent_duration)
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
        target_stage_sequence = target_class_sequence(
            labels["life_stage_class"],
            target_mask,
        )
        target_parent_stage_sequence = target_sequence_summary(
            labels["parent_stage"],
            target_mask,
            np.nan,
        )
        target_duration_hours_sequence = target_sequence_summary(
            labels["parent_duration_hours"],
            target_mask,
            np.nan,
        )
        event_index = int(metadata["convection_timestep_index"])
        target_present = bool(target_mask[event_index].any())
        if self.require_target and not target_present:
            logging.warning(
                "Skipping %s: target system %s is absent at event timestep.",
                chip_path,
                system_id,
            )
            return "missing_target"

        metadata.update(
            {
                "a3dvocado_labels": {
                    "truth_root": str(self.truth.truth_root),
                    "truth_sources": truth_sources,
                    "truth_positions_day_time": truth_positions,
                    "stage_bins": list(self.stage_bins),
                    "stage_classes": STAGE_CLASS_NAMES,
                    "target_present_at_event": target_present,
                    "label_grid": "ABI 2-km chip grid, nearest-neighbor from 0.12-degree truth",
                    "label_dimensions": "time,y,x",
                }
            }
        )
        arrays.update(labels)
        arrays.update(
            {
                "target_stage_class_sequence": target_stage_sequence,
                "target_parent_stage_sequence": target_parent_stage_sequence,
                "target_duration_hours_sequence": target_duration_hours_sequence,
                "target_stage_class": np.asarray(
                    target_stage_sequence[event_index],
                    dtype=np.uint8,
                ),
                "target_stage_class_index": np.asarray(
                    target_stage_sequence[event_index] - 1,
                    dtype=np.uint8,
                ),
                "target_parent_stage": np.asarray(
                    target_parent_stage_sequence[event_index],
                    dtype=np.float32,
                ),
                "target_duration_hours": np.asarray(
                    target_duration_hours_sequence[event_index],
                    dtype=np.float32,
                ),
                "truth_timestamps": np.asarray(
                    [timestamp.isoformat() for timestamp in timestamps]
                ),
                "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
            }
        )

        temporary_path = output_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, output_path)
        return "saved"


_WORKER_GENERATOR = None


def _initialize_worker(config):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = A3DVOCADOLabelGenerator(**config)


def _process_chip(chip_path):
    try:
        return _WORKER_GENERATOR.generate(chip_path)
    except Exception:
        logging.exception("Failed to label %s.", chip_path)
        return "failed"


def process_parallel(chip_paths, config, workers):
    counts = {
        "saved": 0,
        "exists": 0,
        "missing_target": 0,
        "failed": 0,
    }
    path_iterator = iter(chip_paths)
    pending = set()
    max_pending = max(workers * 2, 1)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(config,),
    ) as executor:
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(_process_chip, next(path_iterator)))
            except StopIteration:
                break

        with tqdm(total=len(chip_paths), desc="Pairing ABI chips and labels") as progress:
            while pending:
                completed, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    try:
                        counts[future.result()] += 1
                    except Exception:
                        counts["failed"] += 1
                        logging.exception("Label worker failed.")
                    progress.update(1)

                    try:
                        pending.add(
                            executor.submit(
                                _process_chip,
                                next(path_iterator),
                            )
                        )
                    except StopIteration:
                        pass

    return counts


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pair ABI convection NPZ chips with DCS masks and life-stage truth."
        )
    )
    parser.add_argument("chips", nargs="*", help="ABI convection NPZ files.")
    parser.add_argument("--chip-glob", default=DEFAULT_CHIP_GLOB)
    parser.add_argument("--truth-root", type=Path, default=DEFAULT_TRUTH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage-bins",
        type=float,
        nargs=3,
        default=DEFAULT_STAGE_BINS,
        metavar=("DEVELOPING", "MATURE", "DECAYING"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--max-chips", type=int)
    parser.add_argument(
        "--allow-missing-target",
        action="store_true",
        help="Save chips even if the target system is absent at the event time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not (
        0 < args.stage_bins[0] < args.stage_bins[1] < args.stage_bins[2] < 1
    ):
        raise ValueError("--stage-bins must be increasing values between 0 and 1.")

    chip_paths = [Path(path) for path in args.chips]
    if not chip_paths:
        chip_paths = [Path(path) for path in sorted(glob(args.chip_glob))]
    if args.max_chips is not None:
        chip_paths = chip_paths[: args.max_chips]
    if not chip_paths:
        raise FileNotFoundError("No ABI convection NPZ chips were found.")

    config = {
        "truth_root": args.truth_root,
        "output_dir": args.output_dir,
        "stage_bins": args.stage_bins,
        "require_target": not args.allow_missing_target,
    }
    if args.workers == 1:
        generator = A3DVOCADOLabelGenerator(**config)
        counts = {
            "saved": 0,
            "exists": 0,
            "missing_target": 0,
            "failed": 0,
        }
        for chip_path in tqdm(chip_paths, desc="Pairing ABI chips and labels"):
            try:
                counts[generator.generate(chip_path)] += 1
            except Exception:
                counts["failed"] += 1
                logging.exception("Failed to label %s.", chip_path)
    else:
        counts = process_parallel(chip_paths, config, args.workers)

    logging.info("Finished label pairing: %s", counts)


if __name__ == "__main__":
    main()
