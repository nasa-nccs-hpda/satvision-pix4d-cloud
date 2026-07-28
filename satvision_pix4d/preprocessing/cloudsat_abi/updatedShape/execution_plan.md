# Execution Plan: 1D Transect-Aligned Collocation

## Core Directives
1. **Simplify & Clean Up:** Aggressively delete any code related to the old 2D "square cropping" logic (e.g., `_chip_profile_positions`, bounding box validations).
2. **Comment Liberally:** Add clear, plain-English comments to any new or modified functions explaining *why* the data is being sliced a certain way.
3. **Data Shape:** The final ABI output shape MUST be `(7, 512, 1, 16)` to satisfy 3D U-Net architectural requirements.

---

## Step 1: Update Configuration (`config.py`)
* **Modify Defaults:**
  * Change `profiles_per_chip` to `512` (this is the new sequence length).
  * Change `chip_size` to `1` (representing our 1x1 spatial footprint).
  * Update `offsets` to cover the 7 required timesteps: `(-60, -40, -20, 0, 20, 40, 60)`.
* **Clean Up:** Remove any configuration parameters that were strictly used for 2D square bounding box logic if they are no longer needed.

---

## Step 2: Refactor ABI Extraction (`abi.py`)
* **Remove External Dependency:** Delete the `_crop_channel` method and remove the import for `crop_l1b_rad_to_common_grid`.
* **Create `extract_transect`:** Write a new method `extract_transect(self, requested: datetime, rows: np.ndarray, columns: np.ndarray)` that:
  1. Finds the nearest valid ABI scan.
  2. Opens the NetCDF file directly.
  3. Uses NumPy advanced indexing (e.g., `variable[rows, columns]`) to extract the 16-channel values for the 512 specific coordinates.
  4. Returns the stacked `(512, 16)` array for that specific timestep.

---

## Step 3: Streamline the Pipeline (`pipeline.py`)
* **Clean Up:** Delete the `_chip_profile_positions` and `_inside_chip` methods. We are no longer checking if CloudSat footprints fit inside a square chip.
* **Windowing Logic:** In `_profile_centers` and `process_orbit`, continue to step through the CloudSat transect based on `profile_stride`. Select a contiguous window of exactly 512 CloudSat profiles. If a transect has fewer than 512 footprints remaining, skip it.
* **Coordinate Mapping:** For the 512 profiles in the window, loop over their latitudes/longitudes and use `abi.geometry.nearest(lat, lon)` to generate arrays of 512 `rows` and `columns`.
* **Extract Time Series:** Loop through the 7 `offsets`. For each offset, call the new `abi.extract_transect(time, rows, columns)` method.
* **Calculate SZA/VZA:** Pass the 512 CloudSat latitudes and longitudes directly into `abi.geometry.solar_zenith_angle` and `abi.geometry.view_zenith_angle`. Store these in the auxiliary arrays.

---

## Step 4: Validate Output Writing (`writer.py`)
* **Update `CollocatedChip`:** Ensure that the dataclass can accept the new shape arrays.
* **Verify Shapes:** Before saving the `.npz` file, ensure the `ABI/chip` array is reshaped to explicitly include the dummy spatial dimension: `(7, 512, 1, 16)`. Add comments clarifying why the dummy dimension exists for the U-Net.
