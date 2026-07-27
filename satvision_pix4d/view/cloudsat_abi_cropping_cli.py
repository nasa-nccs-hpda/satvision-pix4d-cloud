"""Command-line interface for CloudSat and ABI collocation."""

from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from typing import Sequence

from satvision_pix4d.preprocessing.cloudsat_abi import (
    CropConfig,
    get_satellite,
)
from satvision_pix4d.preprocessing.cloudsat_abi.config import (
    SATELLITES,
    DEFAULT_GEOMETRY_DIR
)
from satvision_pix4d.preprocessing.cloudsat_abi.pipeline import run_parallel

LOG = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop multitemporal GOES ABI chips colocated with a CloudSat "
            "transect."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  python scripts/crop_abi_multitemporal_rewriting.py \\
    --abi-root /data/GOES-16-ABI-L1B-FULLD \\
    --cloudsat-root /data/cloudsat \\
    --latlon-dir /data/geometry \\
    --output-dir /data/chips --year 2019 --day-start 335 --day-end 336 \\
    --transect -30 30 --satellite goes16 \\
    --metadata cloudsat cloudsat_aux merra2 \\
    --merra2-root /data/MERRA2 \\
    --merra2-variables Pressure Temperature WV Geopotential_height Dem_elevation T2m U10m V10m
""",
    )
    parser.add_argument(
        "--abi-root", type=Path, required=True,
        help="ABI root organized as YYYY/DDD/HH",
    )
    parser.add_argument(
        "--cloudsat-root", type=Path, required=True,
        help="Root containing the 2B-CLDCLASS-LIDAR product directory",
    )
    parser.add_argument(
        "--cloudsat-index-root", type=Path,
        help="Direct 2B-CLDCLASS-LIDAR path; defaults below --cloudsat-root",
    )
    parser.add_argument(
        "--latlon-path", type=Path,
        help=(
            "Exact ABI Latitude/Longitude NetCDF file. Overrides "
            "--latlon-dir."
        ),
    )
    parser.add_argument(
        "--latlon-dir", type=Path, default=DEFAULT_GEOMETRY_DIR,
        help=(
            "Directory containing ABI_EAST_GEO_TOPO_LOMSK.nc and/or "
            "ABI_WEST_GEO_TOPO_LOMSK.nc. The file is selected from "
            "--satellite."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--day-start", type=int, default=1)
    parser.add_argument("--day-end", type=int)
    parser.add_argument("--orbit", help="Process only this five-digit CloudSat orbit")
    parser.add_argument(
        "--transect", type=float, nargs=2,
        metavar=("LAT_MIN", "LAT_MAX"), required=False, default=(-90.0, 90.0),
        help="Inclusive CloudSat latitude bounds",
    )
    parser.add_argument(
        "--satellite", choices=sorted(SATELLITES), required=True,
        help="Physical GOES spacecraft; East/West is inferred and recorded",
    )
    parser.add_argument(
        "--offsets", type=int, nargs="+", default=[-40, -20, 0, 20, 40]
    )
    parser.add_argument(
        "--chip-size", type=int, default=128,
        help=(
            "Chip width and height in common 1 km ABI-grid pixels. "
            "Defaults to 128; use 512 for a 512x512 chip."
        ),
    )
    parser.add_argument("--profile-stride", type=int, default=45)
    parser.add_argument("--profiles-per-chip", type=int, default=91)
    parser.add_argument(
        "--profile-selection", choices=("fixed", "chip"), default="fixed",
        help=(
            "Use a fixed profile count, or select the CloudSat track crossing "
            "the ABI chip from top to bottom."
        ),
    )
    parser.add_argument(
        "--metadata", nargs="*", choices=("cloudsat", "cloudsat_aux", "merra2"),
        default=["cloudsat"],
        help="Metadata groups to store; pass with no values for none",
    )
    parser.add_argument(
        "--cloudsat-aux-root", type=Path,
        help=(
            "Direct ECMWF-AUX product path; defaults to "
            "--cloudsat-root/ECMWF-AUX when --metadata includes cloudsat_aux."
        ),
    )
    parser.add_argument("--merra2-root", type=Path)
    parser.add_argument(
        "--merra2-variables", nargs="*", default=[],
        help=(
            "MERRA-2 outputs to save. Empty means all spreadsheet minimums: "
            "Pressure, Temperature, WV, Geopotential_height, Dem_elevation, "
            "T2m, U10m, V10m. Raw aliases like T, QV, H, T2M are accepted."
        ),
    )
    parser.add_argument("--max-scan-delta-minutes", type=float, default=8.0)
    parser.add_argument(
        "--min-valid-fraction", type=float, default=1.0,
        help=(
            "Minimum fraction of the chip with valid Earth geolocation. "
            "Defaults to 1.0, rejecting limb-crossing chips."
        ),
    )
    parser.add_argument(
        "--inner-disk-margin", type=int, default=1600,
        help=(
            "Margin on the 10848-pixel common ABI grid. Defaults to 1600 "
            "to match the original script; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--min-cloudsat-valid-fraction", type=float, default=1.0,
        help=(
            "Minimum fraction of CloudSat profiles with valid retrievals. "
            "Defaults to 1.0, excluding fill-only nighttime segments."
        ),
    )
    parser.add_argument(
        "--require-cloud", action="store_true",
        help="Also skip valid CloudSat segments containing no classified cloud.",
    )
    parser.add_argument("--allow-missing-timesteps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of orbit worker processes. Each worker loads a large ABI "
            "geometry grid; 2-4 is recommended. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--max-chips", type=int,
        help="Stop after this many new outputs (useful for validation)",
    )
    parser.add_argument(
        "--progress", action="store_true",
        help=(
            "Show tqdm progress bars when tqdm is installed. In serial mode "
            "this tracks candidates per orbit; with --workers it tracks "
            "completed orbits."
        ),
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CropConfig:
    satellite = get_satellite(args.satellite)
    index_root = (
        args.cloudsat_index_root
        or args.cloudsat_root / "2B-CLDCLASS-LIDAR"
    )
    return CropConfig(
        abi_root=args.abi_root,
        cloudsat_root=args.cloudsat_root,
        cloudsat_index_root=index_root,
        latlon_path=(
            args.latlon_path
            if args.latlon_path is not None
            else satellite.geometry_path(args.latlon_dir)
        ),
        output_dir=args.output_dir,
        year=args.year,
        satellite=satellite,
        day_start=args.day_start,
        day_end=args.day_end,
        orbit=args.orbit,
        transect=tuple(args.transect),
        offsets=tuple(args.offsets),
        chip_size=args.chip_size,
        profile_stride=args.profile_stride,
        profiles_per_chip=args.profiles_per_chip,
        profile_selection=args.profile_selection,
        metadata=frozenset(args.metadata),
        cloudsat_aux_root=args.cloudsat_aux_root,
        merra2_root=args.merra2_root,
        merra2_variables=tuple(args.merra2_variables),
        max_scan_delta_minutes=args.max_scan_delta_minutes,
        min_valid_fraction=args.min_valid_fraction,
        inner_disk_margin=args.inner_disk_margin,
        min_cloudsat_valid_fraction=args.min_cloudsat_valid_fraction,
        require_cloud=args.require_cloud,
        allow_missing_timesteps=args.allow_missing_timesteps,
        overwrite=args.overwrite,
        max_chips=args.max_chips,
        progress=args.progress,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )

    try:
        config = config_from_args(args)
        workers = args.workers
        if config.max_chips is not None and workers > 1:
            LOG.warning(
                "Using workers=1 with --max-chips to preserve the exact limit"
            )
            workers = 1
        written = run_parallel(config, workers)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("Finished: %d chip(s) written", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
