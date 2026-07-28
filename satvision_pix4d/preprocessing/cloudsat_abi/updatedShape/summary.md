# CloudSat and GOES-ABI Collocation Pipeline (1D Transect-Aligned)

## Overview
This directory contains a data preprocessing pipeline that colocates CloudSat 2B-CLDCLASS-LIDAR profiles with GOES-ABI satellite imagery.

The pipeline is specifically designed to extract **1D curved sequences (transects) of 512 spatial footprints**, capturing ABI channel radiances along the CloudSat orbital track. The output is structured to satisfy the architectural requirements of a 3D U-Net.

## Core Components
* **`pipeline.py` (`CloudSatABICollocationPipeline`)**: 
  The main orchestrator. It sweeps through CloudSat overpasses, selects contiguous segments of exactly 512 profiles, and extracts the corresponding ABI radiances for 7 temporal offsets (-60, -40, -20, 0, +20, +40, +60 minutes relative to the CloudSat overpass).
* **`abi.py`**: 
  Handles the GOES-ABI (Advanced Baseline Imager) data access. It maps the geographic coordinates to native ABI pixels on a common 1km grid and utilizes advanced NumPy indexing to rapidly extract all 16 channels along the transect sequence in a single operation.
* **`cloudsat.py`**: 
  Parses the CloudSat HDF-EOS datasets and manages sliding-window footprint sampling along the orbit track.
* **`config.py`**: 
  Stores the configuration (defaults: `chip_size=1`, `profiles_per_chip=512`).
* **`writer.py`**: 
  Defines the `CollocatedChip` serialization process. It ensures the final `.npz` arrays feature the shape `(time, sequence, spatial, channels)` — strictly `(7, 512, 1, 16)` — retaining the spatial dummy dimension for U-Net compatibility.
* **`utils.py`**: 
  Lightweight utilities for datetime handling and geolocation normalization.

## Output Details
Each sample is saved as an `.npz` file containing:
* `ABI/chip`: `(7, 512, 1, 16)` float32 array (Radiance data).
* `ABI/offsets_minutes`: `(7,)` int32 array (Temporal offsets).
* `ABI/scan_times`: `(7,)` string array (Actual scan times).
* `ABI/valid_mask`: `(7,)` int8 array (Timestep availability mask).
* `CloudSat/*`: Assorted auxiliary and metadata arrays for the 512 footprints.

## Recent Changes
The pipeline was recently refactored to eliminate legacy 2D square-chip cropping logic (e.g., bounding boxes, uniform grid crops) in favor of direct 1D native-coordinate advanced indexing. Legacy integration with MERRA-2 data was completely stripped out to simplify the codebase.
