Scripts that crawled and downloaded the raw SatRain dataset from
rain.atmos.colostate.edu/ipwgml/satrain/. Only needed if the raw data ever needs to be
re-fetched from scratch - normally you'd just use ../satrain/, which already has it.

- crawl.py — walks the source site and lists every .nc file URL it finds. Run as
  `python crawl.py > urls.txt` (progress/errors go to stderr, only file URLs go to stdout).
  urls.txt (2,835,012 lines) is the full crawl result: both gridded/ and on_swath/
  geometry variants, for every sensor/split.
- download_urls.txt (1,371,610 lines) — urls.txt filtered down to just the on_swath/
  variant, which is the one actually used by the training pipeline. This is the list
  download.sbatch downloads from.
- download.sbatch — SLURM job (account s2911, compute partition, satrain_env env) that
  runs download_urls.txt through wget in parallel (16 workers) into $NOBACKUP/satrain_data.
