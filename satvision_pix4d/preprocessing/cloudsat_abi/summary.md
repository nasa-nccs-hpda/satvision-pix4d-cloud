# Data Preprocessing Pipeline Summary: CloudSat & GOES-ABI Collocation

## Overview
This preprocessing pipeline is designed to generate spatially and temporally collocated datasets from two satellite sources:
1. **CloudSat**: Provides vertical cross-sections (transects) of cloud structures using an active cloud profiling radar (2B-CLDCLASS-LIDAR).
2. **GOES-ABI**: Provides geostationary, multi-spectral (16 channels) optical imagery over a wide area.

The primary goal is to pair the highly detailed, 1D vertical cloud profiles from CloudSat with the 2D spatio-temporal imagery from GOES-ABI. This results in "chips" that are formatted as inputs for machine learning models—likely for tasks such as estimating 3D cloud structures, cloud top heights, or cloud classifications from 2D optical imagery.

## Pipeline Workflow
1. **Orbit Discovery**: The pipeline iterates through CloudSat orbit files within a specified time range.
2. **Transect Reading**: For each orbit, it reads the 2B-CLDCLASS-LIDAR cloud classification transects and optionally ECMWF-AUX/MERRA-2 atmospheric data.
3. **Spatial Collocation**: CloudSat latitude and longitude coordinates are mapped to the GOES-ABI fixed grid using nearest-neighbor lookup (`ABIGeometry`).
4. **ABI Cropping**: The pipeline extracts a spatial "chip" (now targeting 512x512 pixels) around a central CloudSat footprint.
5. **Temporal Offsets**: The ABI imagery is sampled not just at the exact time of the CloudSat pass, but at multiple temporal offsets (e.g., 7 timesteps: -60, -40, -20, 0, 20, 40, 60 minutes) to capture cloud evolution dynamics.
6. **CloudSat Label Extraction**: It gathers all CloudSat footprints that cross the ABI chip and processes their vertical cloud layers into fixed 500m vertical bins (40 bins covering 0 to 20 km altitude).
7. **Serialization**: The ABI imagery, CloudSat labels, matching coordinates, and rich metadata are saved as compressed `.npz` files for efficient ML dataloading.

## Output Chip Structure
Based on `Chip_explainer.txt`, a single output chip `.npz` file contains the following key components:

### ABI Features (Inputs)
* **`ABI/chip`** `(7, 512, 512, 16)`: The main optical imagery. 7 time steps, 512x512 spatial extent, and 16 spectral channels. Resolution differences (e.g., 0.5km or 2km bands) are re-sampled to a common 1km grid.
* **`ABI/offsets_minutes`** `(7,)`: The time offsets relative to the CloudSat pass (e.g., `[-60, -40, -20, 0, 20, 40, 60]`).
* **`ABI/valid_mask`** `(7,)`: Binary indicator of whether ABI data was successfully retrieved for each timestep.
* **`ABI/scan_times`** `(7,)`: Explicit timestamp strings for each ABI scan.

### CloudSat Features (Labels / Ground Truth)
* **Spatial/Temporal Info**: `CloudSat/latitude`, `CloudSat/longitude`, `CloudSat/utc_hour`. The length of these arrays (`length_of_transect`) equals the number of CloudSat footprints that intersect the ABI chip.
* **Alignment Info**: `CloudSat/abi_row` and `CloudSat/abi_column` map exactly which pixel in the 512x512 ABI chip corresponds to each CloudSat footprint.
* **Raw Cloud Layers**: `cloud_layer_base`, `cloud_layer_top`, `cloud_layer_type`, `cloud_layer_count`. Each footprint can record up to 10 distinct cloud layers. Types include cirrus, altostratus, cumulus, deep convection, etc.
* **Binned Cloud Profiles**: 
  * `CloudSat/cloud_class` `(length_of_transect, 40)`: Cloud types mapped into 40 bins (500m each, up to 20km).
  * `CloudSat/cloud_binary_mask` `(length_of_transect, 40)`: A simplified 0 (clear) or 1 (cloud) mask for the 40 vertical bins.
* **Quality & Atmospheric Variables**: Includes radar quality flags, surface elevation (`dem_elevation`), pressure, temperature profiles, specific humidity, and wind velocities (10m).

### Metadata
* **`metadata_json`**: Contains configuration details like satellite ID, valid data fractions, footprint counts, bounding box coordinates, and various sanity check percentages.
