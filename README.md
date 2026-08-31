# SatVision-Pix4DCloud

SatVision-Pix4DCloud is a scalable data-generation and pre-training framework for geostationary satellite imagery, designed to support self-supervised learning and foundation-model development using GOES ABI Level-1B observations.

The framework supports large-scale spatiotemporal tile generation, stratified sampling of atmospheric phenomena, and foundation-model pre-training. It is optimized for execution on NASA NCCS HPC resources using containerized workflows.

## 🚀 Model Pre-Release

A pre-release checkpoint of the **SatVision-Pix4DCloud Base** foundation model is now available on Hugging Face:

**🤗 SatVision-Pix4DCloud Base:**
https://huggingface.co/nasa-cisto-data-science-group/satvision-pix4d-cloud-base

This checkpoint represents an early release of the SatVision-Pix4DCloud foundation model and is intended to support evaluation, experimentation, and development of downstream Earth science applications.

> **Pre-release notice**
>
> This model is under active development. Model architecture, checkpoints, preprocessing conventions, configuration files, and APIs may change as development continues. Additional documentation, downstream examples, and validated training recipes will be added in future releases.

---

## 1. Container Setup

### Download and Build Container (Singularity Sandbox)

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/satvision-pix4d \
  docker://nasanccs/satvision-pix4d:latest
```

> **Note**
> The sandbox format is recommended for development and debugging on NCCS GPU nodes. The container is OCI compliant and can be used with any container engine.

---

## 2. Tile Generation Pipelines

All pipelines are driven through the unified CLI:

```text
satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

Ensure `PYTHONPATH` points to the location where the repository was cloned when running inside the container.

During the current development phase, the source tree is imported directly through `PYTHONPATH`. A future version of the container will include the `satvision_pix4d` Python package directly.

### 2.1 ABI + CloudSat Tile Generator

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/$USER/development/satvision-pix4d \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m \
  /lscratch/$USER/container/satvision-pix4d \
  python /explore/nobackup/people/$USER/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_cropping_cli.py
```

---

### 2.2 Random ABI Tile Generator (Baseline)

Generates random spatial tiles without stratification.

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/$USER/development/satvision-pix4d \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
  /lscratch/$USER/container/satvision-pix4d \
  python /explore/nobackup/people/$USER/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

---

### 2.3 Convection-Stratified Tile Generator

Uses external cloud-system masks to target convective regions.

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css \
  /lscratch/jacaraba/container/satvision-pix4d \
  python /explore/nobackup/people/jacaraba/development/satvision-pix4d/satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py \
  --stratification convection \
  --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" \
  --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tests/tiles_pix4d
```

---

### 2.4 Convection Tiles with Local ABI Files (Experimental)

⚠️ **Known limitation:** some local ABI files may be missing or incomplete.

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d \
  --nv \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m \
  /lscratch/jacaraba/container/satvision-pix4d \
  python /explore/nobackup/people/jacaraba/development/satvision-pix4d/\
satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py \
  --stratification convection \
  --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" \
  --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d \
  --tile-size 512 \
  --channels 1 2 \
  --local-data-dir "/css/geostationary/BackStage/GOES-16-ABI-L1B-FULLD"
```

---

### 2.5 AWS-Only ABI Access

Uses on-the-fly downloads from AWS without requiring a local ABI archive.

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d \
  --nv \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css,/nfs4m \
  /lscratch/jacaraba/container/satvision-pix4d \
  python /explore/nobackup/people/jacaraba/development/satvision-pix4d/\
satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py \
  --stratification convection \
  --convection-regex "/explore/nobackup/projects/pix4dcloud/Jingbo/cloudsystem_mask_2019-2020/2020*.nc" \
  --output-dir /explore/nobackup/projects/pix4dcloud/jacaraba/tiles_pix4d \
  --tile-size 512 \
  --channels 1 2
```

---

## 3. Performance Notes / Metrics

Empirical measurements on NCCS GPU nodes:

* **16 ABI bands × single timestep**

  * ~40 GB RAM
  * ~3.5 minutes
* **Temporal windowing**

  * Sliding windows over 14 timesteps
  * Select best 7-timestep subsequence for pre-training

---

## 4. Stratified Tile Buckets

### Bucket 1: Convection Tiles

Default when `--stratification convection` is used.

### Bucket 2: Cloud Feature Tiles

Planned support for cloud-property-driven stratification, including cloud type, texture, and organization.

### Bucket 3: Land-Cover Tiles

Planned stratification using MODIS land-cover classes to improve geographic and surface-type balance.

---

## 5. Metadata-Only Generation

Generate tile metadata without extracting pixel data:

```bash
singularity exec \
  --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d \
  --nv \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects \
  /lscratch/jacaraba/container/satvision-pix4d \
  python /explore/nobackup/people/jacaraba/development/satvision-pix4d/\
satvision_pix4d/readers/convection_reader.py
```

---

## 6. Pre-Training Workflows

### 6.1 Development Mode (Interactive)

```bash
singularity shell \
  --env PYTHONPATH=/explore/nobackup/people/jacaraba/development/satvision-pix4d \
  --nv \
  -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/lscratch \
  /lscratch/jacaraba/container/satvision-pix4d
```

---

### 6.2 Testing SatMAE Configuration

```bash
TRITON_CACHE_DIR="/lscratch/jacaraba/triton_cache" \
python /explore/nobackup/people/jacaraba/development/satvision-pix4d/\
satvision_pix4d/satvision_pix4d_cli.py \
  -c /explore/nobackup/people/jacaraba/development/satvision-pix4d/\
tests/configs/test_satmae_dev.yaml
```

---

### 6.3 Production Runs

🚧 To be documented, including Slurm orchestration, distributed training recipes, and checkpoint management.

---

## 7. Model Releases

| Model                         | Status         | Checkpoint                                                                                      |
| ----------------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| **SatVision-Pix4DCloud Base** | 🧪 Pre-release | [Hugging Face](https://huggingface.co/nasa-cisto-data-science-group/satvision-pix4d-cloud-base) |

Additional checkpoints and downstream fine-tuned models will be released as development progresses.

---

## 8. Status Summary

* ✅ ABI L1B ingestion (AWS + local)
* ✅ Convection-based stratification
* ✅ Large-scale spatiotemporal tile generation
* ✅ SatVision-Pix4DCloud Base pre-release
* ✅ Cloud-feature stratification
* ✅ Land-cover stratification
* ✅ Production-scale pre-training recipes
* ✅ Downstream fine-tuning and evaluation examples
