# SatVision-Pix4DCloud

SatVision-PIX4D is a scalable data generation and pre-training pipeline for geostationary satellite imagery, designed to support self-supervised and foundation-model development using ABI L1 data. The system is optimized for execution on NASA NCCS HPC resources using Singularity containers and supports stratified tile generation (e.g., convection, cloud systems, land cover).

## 1. Container Setup

### Download and Build Container (Singularity Sandbox)

```bash
module load singularity
singularity build --sandbox /lscratch/$USER/container/satvision-pix4d \
  docker://nasanccs/satvision-pix4d:latest
````

> **Note**
> The sandbox format is recommended for development and debugging on NCCS GPU nodes. The container is OCI compliant and can be used with any container engine.

## 2. Tile Generation Pipelines

All pipelines are driven through the unified CLI:

```
satvision_pix4d/view/abi_tiles_generator_pipeline_cli.py
```

Ensure `PYTHONPATH` is set to the path where the code was cloned when running inside the container.
In the future version of this software the Python package will be installed as part of the container.
Right now during development is easier to import the PYTHONPATH.

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

⚠️ **Known limitation**: some local ABI files may be missing or incomplete.

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

Uses on-the-fly downloads from AWS (no local ABI dependency).

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

(Default when `--stratification convection` is used.)

### Bucket 2: Cloud Feature Tiles

Planned support for cloud-property-driven stratification
(e.g., cloud type, texture, organization).

### Bucket 3: Land-Cover Tiles

Planned stratification using MODIS land-cover classes for global balance.

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

🚧 To be documented (Slurm orchestration, training recipes, checkpoints).

---

## 7. Status Summary

* ✅ ABI L1 ingestion (AWS + local)
* ✅ Convection-based stratification
* ✅ Large-scale tile generation
* 🚧 Cloud feature stratification
* 🚧 Land-cover stratification
* 🚧 End-to-end pre-training recipes

## Model architecture and size presets

See [the architecture review](docs/model-architecture-review.md) for verified
parameter counts, fixes, figure-aligned pretraining loss, checkpoint compatibility,
and validation limits. Presets are available in `configs/pretrain/` for the
330M, 700M, and 3B model versions (approximately 336.49M, 705.14M, and
3.037B total parameters with the preset settings). An experimental `25b.yaml`
preset adds 25.016B parameters; construction/counts are verified on the meta
device, while full-scale training remains unvalidated.

## Synthetic benchmarks and 100-epoch NumPy training

See [training workflows](docs/training-workflows.md) for synthetic
`[B,7,16,512,512]` throughput/overfit benchmarks, NumPy array and timestamp formats,
and 100-epoch training with TensorBoard reconstructions and epoch checkpoints.
Configurations cover the 330M, 700M, and 3B models in `configs/benchmark/` and
`configs/train/`.

## uv environment (H100)

Use `uv sync --locked --extra cu126`, then `source .venv/bin/activate`.
Run `python -m satvision_pix4d.check_environment --require-deepspeed` on the GPU
node before training. See [uv setup](docs/uv-environment.md) for the locked
Python/CUDA dependencies, CPU checks, and CUDA 12.8 alternative.
