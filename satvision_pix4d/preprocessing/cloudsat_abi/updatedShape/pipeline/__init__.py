"""Core pipeline modules for CloudSat and ABI collocation."""

from .config import (
    CropConfig,
    SatelliteSpec,
    get_satellite,
    SATELLITES,
    DEFAULT_GEOMETRY_DIR,
)
from .pipeline import (
    CloudSatABICollocationPipeline,
    run_parallel,
)

__all__ = [
    "CloudSatABICollocationPipeline",
    "CropConfig",
    "SatelliteSpec",
    "get_satellite",
    "SATELLITES",
    "DEFAULT_GEOMETRY_DIR",
    "run_parallel",
]
