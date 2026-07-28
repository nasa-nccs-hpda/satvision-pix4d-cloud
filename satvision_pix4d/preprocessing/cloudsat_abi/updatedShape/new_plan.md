# Proposed Implementation Plan for 1D Transect-Aligned Collocation

## Overview
The goal is to simplify the preprocessing pipeline to align with a 1D sequence of CloudSat footprints rather than a 2D spatial ABI chip. The new output will correspond to a sequence of 512 CloudSat footprints, mapped to their nearest ABI pixels. 

Based on discussions and the use of a 3D U-Net, **the new ABI data shape will be strictly `(7, 512, 1, 16)`**. The dummy spatial dimension of 1 is intentionally kept to allow standard 3D convolutions to process the data as frames that are 512 pixels tall and 1 pixel wide.

## Key Changes by Module

### 1. `config.py`
* **Update configuration parameters:** 
  * Set default `profiles_per_chip = 512` (this dictates the 512 footprint length).
  * Set `chip_size = 1` (since we only want a 1x1 spatial footprint for each CloudSat point).
  * Update default `offsets` to cover the 7 time steps: `(-60, -40, -20, 0, 20, 40, 60)`.

### 2. `pipeline.py`
* **Simplified Windowing (`_profile_centers` and `build_sample`):**
  * Step through the CloudSat transect based on `profile_stride` (which defines the start of the next 512-profile window).
  * For a given starting point, select a contiguous window of exactly 512 CloudSat profiles.
  * **Short Transects:** If a transect runs out of footprints and a full 512-profile window cannot be formed, that window will be skipped (matching the existing pipeline behavior).
* **Coordinate Mapping:**
  * For each of the 512 profiles, use `self.abi.geometry.nearest(lat, lon)` to find the `row, column` coordinates on the ABI grid.
* **Extracting ABI Time Series:**
  * For each of the 7 temporal offsets, extract the 16-channel ABI pixel values for the 512 specific `(row, column)` pairs.
* **SZA and VZA Handling:**
  * We will pass the 512 CloudSat latitudes and longitudes directly into the `solar_zenith_angle` and `view_zenith_angle` functions newly introduced in the `dev` branch.

### 3. `abi.py`
* **Refactoring the Extraction Method:**
  * *Critical Change:* The `dev` branch currently uses an external function (`crop_l1b_rad_to_common_grid`) to crop square chips. Since we need to extract a non-square 1D array of 512 specific points, we must bypass this external function and write a custom method (e.g. `extract_transect`) that uses advanced NumPy indexing (like `variable[rows, columns]`) to pull out exactly the 512 pixels.

### 4. `writer.py`
* **Update the `CollocatedChip` Dataclass:**
  * The `dev` branch already added support for SZA and VZA serialization.
  * We just need to ensure the final array outputs correctly represent the new ABI shape `(7, 512, 1, 16)`.
