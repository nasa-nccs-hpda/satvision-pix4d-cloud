"""Select metadata manifests before extracting large ABI sequence arrays.

This selector consumes an already annotated, quality-screened candidate catalog.
It does not infer labels, detect overlapping footprints, or download imagery.
"""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


POOL_COLUMNS = {
    'convection': 'convection_bin',
    'cloud_height': 'cloud_height_bin',
    'landcover': 'landcover_class',
    'random': None,
}


def allocate(total, weights):
    if not weights or any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('Weights must be finite, nonnegative and have positive total')
    norm = sum(weights.values())
    if norm <= 0:
        raise ValueError('Weights must have positive total')
    raw = {key: total * value / norm for key, value in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    order = sorted(raw, key=lambda key: -(raw[key] - result[key]))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    return result


def group_split(group, seed, fractions):
    digest = hashlib.sha256(f'{seed}:{group}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / 2**64
    cumulative = 0.
    for split, fraction in fractions.items():
        cumulative += fraction
        if value < cumulative:
            return split
    return next(reversed(fractions))


def validate_catalog(catalog):
    required = {'tile_id', 'group_id', 'source_dataset', 'eligible', *filter(None, POOL_COLUMNS.values())}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f'Missing catalog columns: {sorted(missing)}')
    catalog = catalog.copy().fillna('')
    for column in required:
        catalog[column] = catalog[column].astype(str).str.strip()
    for column in ('tile_id', 'group_id', 'source_dataset'):
        if catalog[column].eq('').any():
            raise ValueError(f'{column} must be nonempty')
    if catalog.tile_id.duplicated().any():
        raise ValueError('tile_id must uniquely identify a physical sequence; deduplicate first')
    if 'sampled_via' in catalog:
        raise ValueError('sampled_via is reserved for selector output')
    if 'split' in catalog:
        if not catalog['split'].isin(['train', 'validation', 'test']).all():
            raise ValueError('Preassigned split must be train, validation or test for every row')
        if (catalog.groupby('group_id')['split'].nunique() > 1).any():
            raise ValueError('A leakage group spans multiple preassigned splits')
    eligibility = catalog.eligible.str.lower()
    if not eligibility.isin(['true', 'false']).all():
        raise ValueError('eligible must be true or false, with QA computed upstream')
    catalog['eligible'] = eligibility.eq('true')
    # A namespaced event_id must not have independently assigned groups.
    if 'event_id' in catalog:
        events = catalog[catalog.event_id.ne('')]
        if (events.groupby('event_id').group_id.nunique() > 1).any():
            raise ValueError('An event spans multiple group_id values; merge leakage groups first')
    return catalog.sort_values('tile_id').reset_index(drop=True)


def _balanced_order(frame, column, rng):
    """Round-robin sources, then classes per source; randomize within each cell."""
    if column is None:
        yield from rng.permutation(frame.index).tolist()
        return
    frame = frame[~frame[column].str.lower().isin(['', 'unknown', 'missing', 'none'])]
    sources = deque()
    for _, source in frame.groupby('source_dataset', sort=True):
        cells = [deque(rng.permutation(cell.index).tolist())
                 for _, cell in source.groupby(column, sort=True)]
        rng.shuffle(cells)
        sources.append(deque(cells))
    rng.shuffle(sources)
    while sources:
        cells = sources.popleft()
        cell = cells.popleft()
        yield cell.popleft()
        if cell:
            cells.append(cell)
        if cells:
            sources.append(cells)


def select(catalog, config):
    catalog = validate_catalog(catalog)
    total = config['total_sequences']
    cap = config['max_sequences_per_group']
    if type(total) is not int or type(cap) is not int or min(total, cap) < 1:
        raise ValueError('total_sequences and max_sequences_per_group must be positive integers')
    seed = config['seed']
    fractions = config['split_fractions']
    if set(fractions) != {'train', 'validation', 'test'}:
        raise ValueError('Provide train, validation and test split fractions')
    if any(not np.isfinite(v) or v <= 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError('Split fractions must be positive and sum to one')
    weights = config['train_pool_weights']
    if set(weights) != set(POOL_COLUMNS):
        raise ValueError(f'train_pool_weights must specify {list(POOL_COLUMNS)}')
    allocate(total, weights)  # validate before selecting
    if 'split' not in catalog:
        splits = {group: group_split(group, seed, fractions) for group in catalog.group_id.unique()}
        catalog['split'] = catalog.group_id.map(splits)
    targets = allocate(total, fractions)
    selected, used, group_counts = [], set(), Counter()
    report = dict(requested=total, candidate_rows=len(catalog),
                  qa_rejected=int((~catalog.eligible).sum()), splits={})
    rng = np.random.default_rng(seed)
    for split, target in targets.items():
        available = catalog[catalog.eligible & catalog.split.eq(split)]
        # Holdouts sample background prevalence rather than training enrichment.
        quotas = allocate(target, weights) if split == 'train' else {'random': target}
        counts = {}
        for pool, quota in quotas.items():
            count = 0
            if quota:
                for index in _balanced_order(available, POOL_COLUMNS[pool], rng):
                    group = catalog.at[index, 'group_id']
                    if index in used or group_counts[group] >= cap:
                        continue
                    used.add(index)
                    group_counts[group] += 1
                    selected.append(dict(catalog.loc[index], sampled_via=pool))
                    count += 1
                    if count == quota:
                        break
            counts[pool] = dict(requested=quota, selected=count, shortfall=quota-count)
        report['splits'][split] = dict(requested=target, available=len(available), pools=counts)
    result = pd.DataFrame(selected, columns=[*catalog.columns, 'sampled_via'])
    report['selected'] = len(result)
    report['status'] = 'complete' if len(result) == total else 'shortfall'
    report['uncompressed_float32_bytes'] = len(result) * 7 * 16 * 512 * 512 * 4
    for split in targets:
        subset = result[result.split.eq(split)]
        report['splits'][split]['marginals'] = {
            column: {str(k): int(v) for k, v in subset[column].value_counts(dropna=False).items()}
            for column in ('source_dataset', 'group_id', *filter(None, POOL_COLUMNS.values()))
        }
    return result, catalog, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New manifest directory')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    catalog = pd.read_csv(args.catalog, dtype=str, keep_default_na=False)
    selected, inventory, report = select(catalog, config)
    args.output.mkdir(parents=True, exist_ok=False)
    for split in config['split_fractions']:
        selected[selected.split.eq(split)].to_csv(args.output / f'{split}.csv', index=False)
    inventory.to_csv(args.output / 'catalog_with_splits.csv', index=False)
    report['catalog_sha256'] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({key: report[key] for key in ('status', 'requested', 'selected', 'qa_rejected')}, indent=2))
    return 0 if report['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
