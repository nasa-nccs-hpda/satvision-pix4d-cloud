"""Read-only source-path discovery; never extracts chips or scans archives recursively."""
import argparse
from datetime import datetime, timezone
from glob import iglob
from importlib.metadata import version, PackageNotFoundError
from itertools import islice
import json
import os
from pathlib import Path
import platform

import yaml


def inspect_sources(config):
    sources = config.get('sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Config requires a nonempty sources mapping')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  host=platform.node(), python=platform.python_version(), packages={}, sources={})
    for package in ('numpy', 'pandas', 'xarray', 'PyYAML', 'tqdm', 'netCDF4', 'h5netcdf', 'h5py'):
        try:
            report['packages'][package] = version(package)
        except PackageNotFoundError:
            report['packages'][package] = None
    for name, source in sources.items():
        kind, raw = source['kind'], source.get('path')
        if kind not in ('file', 'directory', 'glob'):
            raise ValueError(f'Unsupported source kind for {name}: {kind}')
        record = dict(kind=kind, path=raw, status='needs_configuration')
        if raw:
            path = os.path.expandvars(os.path.expanduser(raw))
            record['path'] = path
            try:
                if kind == 'glob':
                    matches = list(islice((p for p in iglob(path, recursive=False) if Path(p).is_file()), 5))
                    record['example_matches'] = matches
                    present = bool(matches) and all(os.access(p, os.R_OK) for p in matches)
                else:
                    item = Path(path)
                    present = (item.is_file() if kind == 'file' else item.is_dir())
                    present = present and os.access(path, os.R_OK | (os.X_OK if kind == 'directory' else 0))
                record['status'] = 'accessible' if present else 'missing_or_unreadable'
            except OSError as exc:
                record.update(status='error', error=str(exc))
        report['sources'][name] = record
    report['status'] = ('paths_accessible' if all(v['status'] == 'accessible' for v in report['sources'].values())
                        and all(report['packages'].values()) else 'needs_setup')
    report['limitations'] = 'Path/package discovery only; no coverage, product schema, QA, alignment or chip validation.'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    args = parser.parse_args(argv)
    report = inspect_sources(yaml.safe_load(args.config.read_text()))
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0 if report['status'] == 'paths_accessible' else 2


if __name__ == '__main__':
    raise SystemExit(main())
