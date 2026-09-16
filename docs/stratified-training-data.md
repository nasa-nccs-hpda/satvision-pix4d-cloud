# Stratified ABI pretraining data: pilot before 200K sequences

## Agreed contract and current status

Each sample is **7 x 16 x 512 x 512**, with **20 minutes between frames**: seven
observations spanning two hours, not seven independent samples. Use ordered ABI
C01-C16 channels in consistent physical units on a common georeferenced grid.
The 20-minute cadence was explicitly confirmed by the user on 2026-09-16.

The active model sizes are 330M, 700M and 3B. Use the same versioned splits for
capacity comparisons. A target of 200K sequences is a planning budget, not evidence
that any of these capacities will be well trained. Unique events, dates, regions,
and held-out quality matter more than redundant overlapping windows.

The new `stratified_manifest` CLI selects metadata only. It does **not** download
or extract imagery, infer cloud/land-cover labels, or verify geographic overlap.
Source archives are not mounted in the desktop workspace, and their current
server paths and product availability have not yet been supplied. Real pilot
chips have therefore not been created. The next integration step is catalog
construction/annotation on the machine with the archives, followed by extraction.

## Existing repository components

- `preprocessing/abi_convection_chip_generator.py` provides local ABI scan lookup,
  windowed extraction/resampling, geolocation and viewing/solar angles, source
  provenance, timestamped NPZ output, and configurable 7-frame/20-minute sequences.
- `readers/convection_reader.py` produces event records and duration metadata.
- The older `pipelines/abi_tiles_generator_pipeline.py` has unimplemented cloud,
  land-cover and random methods. Its old convection path hardcodes 14 timestamps
  and uses only the first input metadata file. It is not the new production path.
- Existing preprocessing is a different dependency scope from the uv training
  environment. NetCDF/HDF/geospatial access may require the preprocessing image
  or additional compatible reader dependencies on the extraction system.

## Sampling design: overlapping attributes, unique sequences

A scene can be convective, high-cloud and forest-covered simultaneously. Do not
treat those as exclusive physical categories or extract three copies. Annotate
each candidate with all attributes, then select it once with `sampled_via`
recording why it was chosen. MODIS/CTH are selection annotations, not extra input
channels for the current 16-band model.

Provisional pilot configuration: **1,000 total sequences = 800 train + 100
validation + 100 test**. The training selection has four equal pools:

| Pool | Training sequences | Within-pool balancing |
| --- | ---: | --- |
| Convection | 200 | Source dataset, then short/mid/long duration |
| Cloud height | 200 | Source dataset, then annotated height class |
| MODIS land cover | 200 | Source dataset, then annotated surface class |
| Background/random | 200 | Random candidates remaining after enriched pools |

These percentages are a starting proposal, not scientifically established optimal
weights. The selector balances targeted pools hierarchically by source then class,
without demanding every combination of convection x height x cover x season.
Earlier pools get first choice of overlapping candidates; the report exposes the
resulting distributions. If a class/source is exhausted, other available buckets
can fill that pool, but a pool shortfall is never silently filled from a different
pool. Failed quotas produce `status: shortfall` and exit code 2. No replacement or
duplicate rows are used. Missing/unknown labels can enter background sampling but
do not satisfy a targeted pool. Scarce classes need deliberate quota adjustment.

Validation/test are random samples of their quality-screened candidate splits,
subject to the group cap. They do not reproduce training enrichment. This is only
representative if the underlying candidate universe and grouping are representative;
a convection-only catalog cannot produce a representative background split.
For dedicated balanced diagnostic evaluation, reserve a separate held-out panel
after examining per-class availability. One hundred holdout sequences is a pipeline
pilot, not enough for precise estimates across many rare classes.

Keep cross-tabs of dataset/satellite, month/season, local solar time, view zenith,
geographic region, cloud fractions and surface fractions. Equalizing the three
label axes does not automatically balance all those other factors. The current
selector reports source and label marginals; richer geographic/time QA belongs
in the catalog audit before approving the final extraction manifest.

## Labels and quality

**Convection:** join a namespaced track ID and full event duration from tracking
metadata. The diagram's long/mid/short labels are provisionally interpreted as
duration; thresholds still need agreement. Consider train-catalog quantiles for a
pilot, then publish frozen physical-duration boundaries. Preserve duration as a
number as well as the bin. Do not estimate storm lifetime from the two-hour chip.
The older parser uses `number_of_steps * 20`; gaps and tracks cut at daily/monthly
file boundaries can bias this. Reconcile track boundaries, namespace reused IDs,
and label censored durations as unknown rather than confidently short.

**Cloud height:** use the matching GOES ABI Level-2 height product, scan time,
units and quality flags. Version/product resolution can vary: inspect metadata.
Store height-bin area fractions, cloudy fraction and unknown fraction, not only
the maximum height. A draft set of low/mid/high-mid/high boundaries could be
2/6/10 km, but those are tunable project choices, not an asserted standard or
implemented default. Define whether the selector's class is the dominant cloudy
area bin or a percentile criterion, and freeze that policy. Treat clear sky and
failed retrievals separately. The NOAA product guide explicitly distinguishes
good retrievals from clear/probably-clear and other invalid retrieval flags.

**Land cover:** MODIS MCD12Q1 Collection 6.1 provides yearly global 500 m classes.
Use a declared classification (e.g. LC_Type1/IGBP) for the matching imagery year
and retain product version/year/QA. Reproject categorical maps with appropriate
categorical handling, never bilinear interpolation of class IDs. Prefer area
fractions for mixed 512x512 footprints; preserve ocean, coast, snow/ice and unknown
coverage. Define a dominant/mixed class policy explicitly before annotation.
Land cover describes the surface even under clouds; it is not a cloud label.

**ABI QA:** all 16 bands, seven distinct strictly increasing scans, actual times
within a tight declared tolerance of requested times, correct common footprint,
valid geolocation, and acceptable pixel quality. Existing nearest-scan tolerance
is 10 minutes; for 20-minute cadence check unique scan IDs to prevent accidental
repeats at ties/gaps. Do not fill a missing frame with the same scan silently.
The training loader currently rejects nonfinite values, so extraction must screen
such chips or a validity-mask-aware training design must be implemented first.
Keep day/night/twilight represented intentionally; VIS/NIR information differs.

The existing generator reads `Rad` (radiance); its method name "normalized" refers
to the spatial grid, not per-channel z-scoring. Do not mix radiance and calibrated
reflectance/brightness-temperature arrays under the same statistics. Estimate
mean/std and inspect SSIM bounds from training data only, and record the units.

## Splits before selection/extraction

Build `group_id` for leakage control before sampling. Connect candidates sharing
the same tracked event, overlapping/near-duplicate space-time windows, or duplicate
observations across catalogs. Buffer split boundaries in space/time using footprint
and two-hour sequence extent. Cross-satellite observations of one event stay in
the same split. A filename or daily system number alone is not a robust group ID.

The selector enforces group-disjoint splits and checks event IDs do not span group
IDs. It cannot infer physical overlap from an arbitrary group label. A bad group
assignment remains a leakage risk. Freeze and audit groups with source provenance.

Prefer explicit `split` values for a planned held-out year/region/event study. The
selector preserves them and rejects groups crossing splits. If absent, a stable
seeded SHA-256 hash assigns entire groups to train/validation/test; row ordering
does not change assignments. Hash splitting is not itself a geographic or future-
time generalization test. Large groups can make available split proportions differ
from requested counts, so the selector reports capacity and shortages.

## Candidate CSV contract and pilot command

Required columns (one row per unique physical sequence):

| Column | Meaning |
| --- | --- |
| `tile_id` | Stable ID for satellite/grid window/actual scan sequence; deduplicated upstream |
| `group_id` | Audited space-time/event leakage group |
| `source_dataset` | Source domain to balance, e.g. GOES16-ABI or GOES17-ABI |
| `eligible` | `true` or `false`, based on upstream sequence/imagery QA |
| `convection_bin` | Annotated short/mid/long, or blank/unknown |
| `cloud_height_bin` | Annotated low/mid/high_mid/high, or blank/unknown |
| `landcover_class` | Declared class/group label, or blank/unknown |

Optional `event_id` must be globally namespaced, and optional `split` must be
train/validation/test for every row. All additional columns are retained: include
coordinates, target grid/pixel window, requested/actual times, source paths,
cloud/surface fractions, product versions, event duration, quality/rejection
reasons and an existing chip path if materialized. `sampled_via` is reserved for
selector output. The selector never invents environmental labels from ABI values.

```bash
python -m satvision_pix4d.preprocessing.stratified_manifest \
  --catalog /path/to/annotated_candidates.csv \
  --config configs/data/stratified_pilot.yaml \
  --output dataset_manifests/pilot-v1
```

Outputs: `train.csv`, `validation.csv`, `test.csv`, full
`catalog_with_splits.csv`, resolved `config.yaml`, and `report.json` with quotas,
shortfalls, distributions and the catalog SHA-256. These are extraction manifests,
**not arrays and not direct inputs to the current training loader**. A materializer
must consume them and write separate flat split directories with compatible NPZ
chips/timestamps, or an explicit manifest-backed dataset adapter must be added.
Do not pass these CSV files to `--train-data`.

Audit the metadata first, extract only selected rows, then audit actual saved
chips: shapes, units, scan intervals, georegistration, examples of every stratum,
train-only stats, split disjointness and failed extractions. Backfill only from
the same frozen split/pool with an auditable new manifest revision. Reuse the
existing convection extractor's scan/window helpers where appropriate; random,
CTH and MODIS annotation adapters still need source-product integration.

## Size and staged expansion

One float32 sequence is **117,440,512 bytes = 112 MiB**. Array-only uncompressed
storage is approximately:

| Total sequences | Decimal storage |
| --- | ---: |
| 1,000 pilot | 117.4 GB |
| 10,000 intermediate | 1.17 TB |
| 200,000 target | 23.49 TB (21.36 TiB) |

Compression may reduce disk usage but must be measured on real chips and can cost
loading CPU time. Latitude/longitude, angle arrays, metadata, temporary files and
source archives add storage. Do not change to float16 simply to halve disk use
without assessing physical-value precision and the resulting preprocessing.

Scale 1K -> 10K -> 200K after checking extraction throughput, representation,
unique-event count, data-loader throughput and model learning curves. At 200K
total with this 80/10/10 policy, training has 160K sequences, not 200K. If the
intention is 200K training sequences plus holdouts, increase the total accordingly.
The same seed with a larger count preserves split assignments, but pool allocation
can change individual selections; manifests are not guaranteed nested. Freeze an
explicit inclusion set if nested scaling experiments are required.

## Primary product references

- [NASA MCD12Q1 Collection 6.1](https://doi.org/10.5067/MODIS/MCD12Q1.061)
- [NASA MODIS land-cover user guide](https://www.earthdata.nasa.gov/s3fs-public/2025-04/MCD12_User_Guide_V6.pdf)
- [NOAA GOES-R Level-2 product guide, cloud-height QA](https://www.ospo.noaa.gov/resources/documents/PUG/GS%20Series%20416-R-PUG-L2%20Plus-0349%20Vol%205%20v3.0%20final.pdf)
