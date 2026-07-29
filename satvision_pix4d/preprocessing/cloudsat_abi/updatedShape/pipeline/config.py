"""Configuration and satellite definitions for CloudSat-ABI collocation."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


DEFAULT_GEOMETRY_DIR = Path(
    "/explore/nobackup/projects/pix4dcloud/jgong"
)


@dataclass(frozen=True)
class SatelliteSpec:
    """A GOES spacecraft and its operational viewing region."""

    token: str
    name: str
    region: str
    number: int
    subpoint_longitude: float

    @property
    def platform_code(self) -> str:
        return f"G{self.number}"

    @property
    def filename_token(self) -> str:
        return self.name.replace("-", "")

    @property
    def geometry_filename(self) -> str:
        return f"ABI_{self.region.upper()}_GEO_TOPO_LOMSK.nc"

    def geometry_path(self, directory: Path = DEFAULT_GEOMETRY_DIR) -> Path:
        return Path(directory) / self.geometry_filename


SATELLITES = {
    "goes16": SatelliteSpec("goes16", "GOES-16", "east", 16, -75.0),
    "goes17": SatelliteSpec("goes17", "GOES-17", "west", 17, -137.0),
    "goes18": SatelliteSpec("goes18", "GOES-18", "west", 18, -137.0),
    "goes19": SatelliteSpec("goes19", "GOES-19", "east", 19, -75.0),
}


def get_satellite(token: str) -> SatelliteSpec:
    key = token.lower().replace("-", "")
    try:
        return SATELLITES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported satellite {token!r}") from exc


@dataclass(frozen=True)
class CropConfig:
    """Validated settings for one collocation run."""

    abi_root: Path
    cloudsat_root: Path
    cloudsat_index_root: Path
    latlon_path: Path
    output_dir: Path
    year: int
    satellite: SatelliteSpec
    day_start: int = 1
    day_end: int | None = None
    orbit: str | None = None
    transect: tuple[float, float] = (-90.0, 90.0)
    # 7 temporal offsets (minutes) relative to the CloudSat overpass time.
    offsets: tuple[int, ...] = (-60, -40, -20, 0, 20, 40, 60)
    # Each CloudSat footprint maps to a single ABI pixel (1x1 spatial).
    chip_size: int = 1
    profile_stride: int = 45
    # Number of contiguous CloudSat footprints per sample.
    profiles_per_chip: int = 512
    metadata: frozenset[str] = frozenset({"cloudsat"})
    cloudsat_aux_root: Path | None = None

    max_scan_delta_minutes: float = 8.0
    min_valid_fraction: float = 1.0
    inner_disk_margin: int = 1600
    min_cloudsat_valid_fraction: float = 1.0
    # Minimum fraction of non-NaN ABI pixels per timestep (across all
    # 512 footprints × 16 channels). Timesteps below this are marked
    # invalid. Chips where too few timesteps survive are rejected.
    min_abi_valid_fraction: float = 0.95
    # Minimum number of temporal offsets that must have valid ABI data.
    # With 7 offsets and min_valid_timesteps=7, all must succeed (matching
    # the old pipeline's behavior). Lower to e.g. 5 to tolerate gaps.
    min_valid_timesteps: int = 7
    require_cloud: bool = False
    # Tolerate individual missing ABI scans (fill with NaN) instead of
    # rejecting the entire sample. Combined with min_valid_timesteps to
    # control how many gaps are acceptable.
    allow_missing_timesteps: bool = True
    overwrite: bool = False
    max_chips: int | None = None
    progress: bool = False

    def __post_init__(self):
        if self.chip_size <= 0:
            raise ValueError("chip_size must be a positive integer")
        if self.profile_stride <= 0:
            raise ValueError("profile_stride must be positive")
        if self.profiles_per_chip <= 0:
            raise ValueError("profiles_per_chip must be a positive integer")
        if "cloudsat_aux" in self.metadata and "cloudsat" not in self.metadata:
            raise ValueError("cloudsat_aux metadata requires CloudSat metadata")
        if self.day_end is not None and self.day_end < self.day_start:
            raise ValueError("day_end must be greater than or equal to day_start")
        if not self.offsets:
            raise ValueError("at least one ABI offset is required")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if self.inner_disk_margin < 0:
            raise ValueError("inner_disk_margin cannot be negative")
        if not 0 <= self.min_cloudsat_valid_fraction <= 1:
            raise ValueError("min_cloudsat_valid_fraction must be in [0, 1]")
        if not 0 <= self.min_abi_valid_fraction <= 1:
            raise ValueError("min_abi_valid_fraction must be in [0, 1]")
        if self.min_valid_timesteps < 0:
            raise ValueError("min_valid_timesteps cannot be negative")
        if self.min_valid_timesteps > len(self.offsets):
            raise ValueError(
                f"min_valid_timesteps ({self.min_valid_timesteps}) cannot "
                f"exceed the number of offsets ({len(self.offsets)})"
            )
        unknown = self.metadata - {"cloudsat", "cloudsat_aux"}
        if unknown:
            raise ValueError(f"Unsupported metadata groups: {sorted(unknown)}")

        low, high = sorted(self.transect)
        if low < -90 or high > 90:
            raise ValueError("transect latitude bounds must be within [-90, 90]")
        object.__setattr__(self, "transect", (float(low), float(high)))
