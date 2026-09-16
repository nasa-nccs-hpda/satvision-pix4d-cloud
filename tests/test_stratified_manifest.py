import json

import pandas as pd
import pytest
import yaml

from satvision_pix4d.preprocessing.stratified_manifest import select, main


def config():
    return dict(total_sequences=100, seed=42, max_sequences_per_group=2,
                split_fractions=dict(train=.8, validation=.1, test=.1),
                train_pool_weights=dict(convection=.25, cloud_height=.25, landcover=.25, random=.25))


def candidates():
    return pd.DataFrame([
        dict(tile_id=f'tile-{i:04d}', group_id=f'group-{i//3:04d}',
             source_dataset=['GOES16', 'GOES17'][i % 2], eligible='true',
             convection_bin=['short', 'mid', 'long'][i % 3],
             cloud_height_bin=['low', 'mid', 'high_mid', 'high'][i % 4],
             landcover_class=['forest', 'cropland', 'ocean'][i % 3])
        for i in range(3000)
    ])


def test_deterministic_disjoint_capped_sampling_and_quotas():
    frame = candidates()
    selected, inventory, report = select(frame, config())
    reordered, _, _ = select(frame.sample(frac=1, random_state=9), config())
    pd.testing.assert_frame_equal(selected, reordered)
    assert selected.tile_id.is_unique
    assert selected.groupby('group_id').size().max() <= 2
    assert inventory.groupby('group_id').split.nunique().max() == 1
    assert report['status'] == 'complete'
    assert selected.split.value_counts().to_dict() == dict(train=80, validation=10, test=10)
    assert selected[selected.split.eq('train')].sampled_via.value_counts().to_dict() == dict(
        convection=20, cloud_height=20, landcover=20, random=20)
    assert selected[selected.split.ne('train')].sampled_via.eq('random').all()


def test_quality_rejection_and_explicit_shortfall():
    frame = candidates()
    frame['eligible'] = 'false'
    selected, _, report = select(frame, config())
    assert selected.empty and report['status'] == 'shortfall'
    assert report['qa_rejected'] == len(frame)
    frame['eligible'] = 'true'
    frame['convection_bin'] = 'unknown'
    selected, _, report = select(frame, config())
    assert report['splits']['train']['pools']['convection']['shortfall'] == 20
    assert report['selected'] == 80  # No silent redistribution or duplicates.


def test_event_leakage_and_duplicate_ids_rejected():
    frame = candidates()
    frame['event_id'] = 'same-event'
    with pytest.raises(ValueError, match='merge leakage groups'):
        select(frame, config())
    with pytest.raises(ValueError, match='deduplicate'):
        select(pd.concat([candidates(), candidates().iloc[:1]]), config())


def test_preassigned_blocked_splits_are_preserved():
    frame = candidates()
    frame['split'] = ['train' if i < 2400 else 'validation' if i < 2700 else 'test'
                      for i in range(len(frame))]
    selected, inventory, report = select(frame, config())
    assert report['status'] == 'complete'
    assert inventory['split'].tolist() == frame['split'].tolist()
    frame.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError, match='leakage group'):
        select(frame, config())


def test_manifest_cli_outputs_are_reproducible_and_not_overwritten(tmp_path):
    csv, cfg, out = tmp_path / 'catalog.csv', tmp_path / 'config.yaml', tmp_path / 'out'
    candidates().to_csv(csv, index=False)
    cfg.write_text(yaml.safe_dump(config(), sort_keys=False))
    args = ['--catalog', str(csv), '--config', str(cfg), '--output', str(out)]
    assert main(args) == 0
    assert len(pd.read_csv(out / 'train.csv')) == 80
    assert len(json.loads((out / 'report.json').read_text())['catalog_sha256']) == 64
    with pytest.raises(FileExistsError):
        main(args)
