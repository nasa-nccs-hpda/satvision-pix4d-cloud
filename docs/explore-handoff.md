# Continue dataset preparation on Explore

## Scope of this handoff

The user requested preparing the repo for continuation on Explore, not extracting
the dataset from the desktop machine. Source discovery, product annotation and
real extraction are the next work there. The branch is
`codex/foundation-model-training`. Preserve local changes before pulling.

Read `docs/stratified-training-data.md` and `docs/codex-handoff.md` next. Key facts:

- Contract: seven ABI frames, **20-minute spacing**, 16 bands, 512x512 pixels;
  two hours from first to last frame. Preserve actual scan times and raw units.
- Provisional pilot: 1K total (800/100/100 train/validation/test), then 10K, then
  an eventual ~200K total. Float32 arrays alone are ~23.5 TB at 200K.
- Targeted training pools: convection duration, cloud-top height, MODIS land cover,
  plus background. Fractions and label thresholds remain provisional.
- Metadata selector, candidate QA flag, group-disjoint splitting, source/class
  balancing, quotas and distribution reports are implemented and tested.
- CTH/MODIS annotation, a background candidate generator and a general manifest
  materializer are not implemented. No real pilot chips have been created here.
- Existing local convection extraction is available in
  `satvision_pix4d/preprocessing/abi_convection_chip_generator.py`.
  Its historical defaults are leads, not confirmation of current accessibility.
- Build leakage groups from events/overlapping space-time windows before splitting.
  The selector cannot infer overlap from arbitrary group IDs. Namespace track IDs
  and handle event lifetimes censored at catalog boundaries.

## CPU environment (no CUDA/DeepSpeed needed)

From the repository root after checking out the branch:

```bash
git switch codex/foundation-model-training
git pull --ff-only

# Requires uv >=0.9; use a site-provided uv if available.
uv sync --project environments/explore --locked
source environments/explore/.venv/bin/activate

# Stay at the repository root so Python can import the source tree.
python -m satvision_pix4d.preprocessing.stratified_manifest --help
python -m satvision_pix4d.preprocessing.abi_convection_chip_generator --help
python -m pytest tests/test_stratified_manifest.py -q
```

This separate locked Python 3.12 environment contains NumPy, pandas, xarray,
PyYAML, tqdm, NetCDF4 and HDF5 readers. It does not install torch, CUDA or DeepSpeed
and does not change the DGX training environment. Initial sync needs access to
package/Python downloads; follow site policy on where installation may run.

The environment supports the current local ABI extractor and catalog selector.
It is **not** a complete GDAL/Satpy/HDF4 environment. In particular, MODIS HDF4
reading/reprojection needs a suitable site module/container or additional reader
integration once the actual product format is known. HDF5 support is not HDF4
support. Do not advertise this environment as supporting every geospatial format.

Use activated `python` for commands. Unqualified `uv run` from the repo root
targets the training project; alternatively use
`uv run --project environments/explore --locked python -m ...` explicitly.

## Locate sources before extraction

```bash
cp configs/data/explore_sources.example.yaml configs/data/explore_sources.local.yaml
# Edit the local YAML with verified paths for all intended source products.
python -m satvision_pix4d.preprocessing.explore_preflight \
  --config configs/data/explore_sources.local.yaml \
  --output dataset_manifests/explore-preflight.json
```

Preflight checks paths/access and package versions only. It does not read imagery,
download data, traverse archives recursively, or certify schemas/coverage/quality.
For metadata globs it reports up to five example matching files, not a total count.
Exit code 2 means sources/dependencies need setup; null CTH/MODIS/lifetime paths in
the template are expected to trigger this until located. Local source config and
generated reports are gitignored. The YAML is only a discovery inventory: the
existing extractor does not automatically consume it.

On Explore, inspect a few actual metadata CSV and NetCDF/HDF headers next. Verify
coverage years, satellite IDs, physical units, projections/grids, QA meanings,
availability of seven distinct scans at the required cadence, and ID namespaces.
Keep small summaries in the handoff, not large raw datasets in git.

## Work sequence after discovery

1. Inventory ABI plus track/lifetime, CTH/QA and MODIS sources; inspect actual schemas.
2. Agree on duration bins, cloud-height statistic/bins, surface class policy,
   available years/regions, QA acceptance and blocked holdouts. Do not infer an
   observed full storm lifetime from one two-hour sequence.
3. Build a candidate metadata catalog covering background as well as enriched
   scenes. Retain geolocation/footprint, source paths and actual times, fractions,
   QA reasons, provenance and globally consistent leakage groups.
4. Run the existing selector and inspect shortages/distributions before extracting:

   ```bash
   python -m satvision_pix4d.preprocessing.stratified_manifest \
     --catalog /path/to/annotated_candidates.csv \
     --config configs/data/stratified_pilot.yaml \
     --output dataset_manifests/pilot-v1
   ```

5. Implement/connect a manifest materializer, reusing local scan/window helpers.
   Validate 10–20 chips before materializing the selected 1K. Avoid accidentally
   extracting every event or starting a 200K job as a setup test.
6. Audit saved sample shape, unique timestamps, georegistration, units, QA,
   split leakage and label coverage. Estimate normalization from training only.
7. Transfer or expose the split chip directories to the DGX; run training data
   checks and a short real-data learning test before scaling extraction.

The selector produces CSV manifests, not training-ready arrays. Do not pass those
CSVs to the current `--train-data` argument. Preserve the distinction between
implemented functionality and pending source integration in status reports.

## Start Codex CLI on Explore

From the checkout root with the CPU environment activated:

```bash
codex "Read docs/explore-handoff.md, docs/stratified-training-data.md and docs/codex-handoff.md. Continue preparing the stratified ABI dataset on Explore. First inspect source availability and schemas, record findings, and identify the remaining annotation/materialization work. Keep the confirmed 7-frame, 20-minute cadence. Start with a small pilot and update the handoff as you work."
```

The separate DGX throughput-tuning work remains documented in `codex-handoff.md`;
it is not a prerequisite for metadata discovery or CPU data preparation here.
