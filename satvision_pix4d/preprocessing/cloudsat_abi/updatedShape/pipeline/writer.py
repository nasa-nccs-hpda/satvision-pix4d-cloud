"""Output models and writers for collocated chips."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CollocatedChip:
    """One serializable ABI and CloudSat collocation sample."""

    filename: str
    chip: np.ndarray
    offsets_minutes: np.ndarray
    valid_mask: np.ndarray
    scan_times: np.ndarray
    metadata: dict[str, Any]
    auxiliary_arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def arrays(self) -> dict[str, Any]:
        # Reshape the ABI chip from (time, sequence, channels) -> (time, sequence, 1, channels)
        # We explicitly include this dummy spatial dimension to satisfy standard 3D U-Net architectural requirements.
        chip_reshaped = (
            np.expand_dims(self.chip, axis=2)
            if self.chip.ndim == 3
            else self.chip
        )

        arrays: dict[str, Any] = {
            "ABI/chip": chip_reshaped,
            "ABI/offsets_minutes": self.offsets_minutes,
            "ABI/valid_mask": self.valid_mask,
            "ABI/scan_times": self.scan_times,
            "metadata_json": np.asarray(
                json.dumps(
                    self.metadata, sort_keys=True, separators=(",", ":")
                )
            ),
        }
        for name, value in self.auxiliary_arrays.items():
            if name.startswith("abi_"):
                arrays[f"ABI/{name.removeprefix('abi_')}"] = value
            elif name.startswith("cloudsat_"):
                arrays[f"CloudSat/{name.removeprefix('cloudsat_')}"] = value
            else:
                arrays[name] = value
        return arrays


class NPZChipWriter:
    """Write compressed NPZ samples without pickle-dependent object arrays."""

    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample: CollocatedChip) -> tuple[Path, bool]:
        output = self.output_dir / sample.filename
        if output.exists() and not self.overwrite:
            return output, False
        np.savez_compressed(output, **sample.arrays())
        return output, True
