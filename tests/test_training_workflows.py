import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from satvision_pix4d.benchmark import main as benchmark_main
from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.datasets.abi_temporal_benchmark_dataset import ABITemporalBenchmarkDataset
from satvision_pix4d.datasets.abi_temporal_dataset import ABITemporalDataset
from satvision_pix4d.satvision_pix4d_cli import main as train_main, check_data

FIELDS = ['year', 'month', 'day', 'hour', 'minute']


@pytest.mark.parametrize('workflow', ['benchmark', 'pretrain'])
def test_configure_model_places_buffers_without_moving_parameters(workflow):
    from satvision_pix4d.benchmark import SyntheticMAEBenchmark, parser, prepare_config
    from satvision_pix4d.pipelines.satvision_pix4d_pretrain import SatVisionPix4DSatMAEPretrain

    config = prepare_config(parser().parse_args(['--tiny', '--output', 'unused']))
    module = (SyntheticMAEBenchmark(config) if workflow == 'benchmark'
              else SatVisionPix4DSatMAEPretrain(config, defer_model=True))
    # Meta provides a distinct device without requiring GPUs or allocating a
    # large model. Parameters stand in for shards owned by DeepSpeed.
    module.model = torch.nn.Linear(2, 2)
    module.model.register_buffer('loss_bounds', torch.tensor([0., 1.]), persistent=False)
    weight = module.model.weight
    module._trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=torch.device('meta')))
    module.configure_model()
    assert all(buffer.device.type == 'meta' for buffer in module.buffers())
    assert module.model.weight is weight and weight.device.type == 'cpu'
    assert 'model.loss_bounds' not in module.state_dict()
    if workflow == 'pretrain':
        assert module.train_loss_avg.mean_value.device.type == 'meta'
        assert module.val_psnr.sum_squared_error.device.type == 'meta'


def test_synthetic_full_shape_and_reproducibility():
    data = ABITemporalBenchmarkDataset(length=4, fixed_samples=1, temporal_embeddings=FIELDS)
    raw, timestamps = data[0]
    assert raw.shape == (7, 16, 512, 512)
    assert timestamps.shape == (7, 5)
    assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
    assert torch.equal(raw, data[1][0])
    assert timestamps[:, -1].tolist() == [0, 10, 20, 30, 40, 50, 0]
    assert timestamps[-1, -2] == 1


def test_synthetic_split_and_sample_variation():
    train = ABITemporalBenchmarkDataset(img_size=128, length=2)
    val = ABITemporalBenchmarkDataset(img_size=128, length=2, split='valid')
    assert not torch.equal(train[0][0], train[1][0])
    assert not torch.equal(train[0][0], val[0][0])


def test_npy_sidecar_batched_components_and_shared_npz_timestamps(tmp_path):
    chips = np.zeros((2, 7, 16, 32, 32), dtype=np.float32)
    times = np.tile(np.array([20, 0, 0, 0, 0]), (2, 7, 1))
    times[1, :, -1] = 10
    np.save(tmp_path / 'chips.npy', chips)
    np.save(tmp_path / 'chips.timestamps.npy', times)
    dataset = ABITemporalDataset([str(tmp_path)], img_size=32, temporal_embeddings=FIELDS,
                                 require_timestamps=True, num_timesteps=7)
    assert len(dataset) == 2  # sidecars must not be discovered as chips
    assert dataset[1][1][:, -1].tolist() == [10] * 7
    np.savez(tmp_path / 'shared.npz', chips=chips, timestamps=times[0])
    shared = ABITemporalDataset([str(tmp_path / 'shared.npz')], img_size=32,
                                temporal_embeddings=FIELDS, require_timestamps=True)
    assert np.array_equal(shared[1][1], times[0])


@pytest.mark.parametrize('problem', ['missing_times', 'nan', 'wrong_time_count', 'wrong_components'])
def test_real_data_validation(tmp_path, problem):
    chips = np.zeros((7, 16, 32, 32), dtype=np.float32)
    times = np.tile([20, 0, 0, 0, 0], (7, 1))
    if problem == 'nan':
        chips[0, 0, 0, 0] = np.nan
    if problem == 'wrong_components':
        times = times[:, :3]
    if problem == 'missing_times':
        np.save(tmp_path / 'chip.npy', chips)
    else:
        np.savez(tmp_path / 'chip.npz', chip=chips, timestamps=times)
    dataset = ABITemporalDataset([str(tmp_path)], img_size=32, temporal_embeddings=FIELDS,
                                 require_timestamps=True,
                                 num_timesteps=6 if problem == 'wrong_time_count' else 7)
    with pytest.raises(ValueError):
        dataset[0]


def test_benchmark_overfit_and_tensorboard(tmp_path):
    out = tmp_path / 'benchmark'
    result = benchmark_main(['--tiny', '--image-size', '128', '--mode', 'overfit',
                             '--steps', '100', '--warmup-steps', '2', '--fixed-samples', '1',
                             '--lr', '0.001', '--accelerator', 'cpu', '--strategy', 'auto',
                             '--precision', '32-true', '--output', str(out)])
    summary = json.loads((out / 'summary.json').read_text())
    assert result == 0 and summary['status'] == 'passed'
    assert summary['relative_loss_reduction'] >= .1
    assert summary['measured_steps'] == 98
    assert summary['final_probe']['loss'] < summary['final_probe']['channel_mean_baseline_loss']
    events = EventAccumulator(str(out / 'tensorboard')).Reload()
    assert len(events.Scalars('benchmark/loss')) == 100
    assert len(events.Images('benchmark/fixed_probe_target_masked_reconstruction')) == 2
    assert len(events.Images('benchmark/held_out_target_masked_reconstruction')) == 1


def test_benchmark_accumulation_and_failure_status(tmp_path):
    out = tmp_path / 'benchmark'
    code = benchmark_main(['--tiny', '--image-size', '128', '--mode', 'overfit',
                           '--steps', '2', '--warmup-steps', '0', '--accumulation-steps', '2',
                           '--min-relative-improvement', '.99', '--lr', '0.000001',
                           '--accelerator', 'cpu', '--strategy', 'auto', '--precision', '32-true',
                           '--output', str(out)])
    records = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
    assert code == 2
    assert len(records) == 2
    assert all(row['samples'] == 2 for row in records)
    # Logged losses are unscaled by gradient accumulation.
    assert records[0]['loss'] > .2


def tiny_real_config(tmp_path):
    config = _C.clone()
    _update_config_from_file(config, 'configs/train/330m.yaml')
    config.defrost()
    config.MODEL.MAE_VIT.SIZE = 'custom'
    config.MODEL.MAE_VIT.IN_CHANS = 2
    config.MODEL.MAE_VIT.EMBED_DIM = 32
    config.MODEL.MAE_VIT.DEPTHS = 1
    config.MODEL.MAE_VIT.NUM_HEADS = 4
    config.MODEL.MAE_VIT.DECODER_EMBED_DIM = 32
    config.MODEL.MAE_VIT.DECODER_DEPTH = 1
    config.MODEL.MAE_VIT.DECODER_NUM_HEADS = 4
    config.MODEL.MAE_VIT.VISNIR_CHANNELS = [0]
    config.DATA.IMG_SIZE = 128
    config.DATA.NUM_TIMESTEPS = 2
    config.DATA.NUM_WORKERS = 0
    config.DATA.MEAN = [0., 0.]
    config.DATA.STD = [1., 1.]
    config.DATA.MIN = [-3., -3.]
    config.DATA.MAX = [3., 3.]
    config.TRAIN.ACCELERATOR = 'cpu'
    config.TRAIN.STRATEGY = 'auto'
    config.TRAIN.EPOCHS = 2
    config.TRAIN.WARMUP_EPOCHS = 0
    config.PRECISION = '32-true'
    config.PRINT_FREQ = 1
    for split in ['train', 'valid']:
        data = ABITemporalBenchmarkDataset(img_size=128, in_chans=2, num_timesteps=2,
                                           length=2, temporal_embeddings=FIELDS, split=split)
        pairs = [data[i] for i in range(2)]
        np.savez(tmp_path / f'{split}.npz', chips=torch.stack([p[0] for p in pairs]).numpy(),
                 timestamps=torch.stack([p[1] for p in pairs]).numpy())
    config.DATA.TRAIN_DATA_PATHS = [str(tmp_path / 'train.npz')]
    config.DATA.VAL_DATA_PATHS = [str(tmp_path / 'valid.npz')]
    return config


def test_numpy_train_tensorboard_reconstruction_epoch_checkpoints_resume(tmp_path):
    config = tiny_real_config(tmp_path)
    report = check_data(config)
    assert report['train']['samples'] == 2
    out = tmp_path / 'run'
    trainer = train_main(config, str(out))
    assert trainer.global_step == 4
    assert (out / 'epoch-000.ckpt').exists()
    assert (out / 'epoch-001.ckpt').exists()
    assert (out / 'last.ckpt').exists()
    events = EventAccumulator(str(out / 'tensorboard' / 'version_0')).Reload()
    scalars = events.Tags()['scalars']
    assert 'val_loss' in scalars
    assert 'val/masked_mae_band_00' in scalars
    assert 'val/charbonnier' in scalars
    assert 'performance/epoch_wall_seconds' in scalars
    images = events.Images('validation/target_masked_reconstruction')
    assert len(images) == 2
    assert len(events.Scalars('val_loss')) == 2
    assert all(np.isfinite(e.value) for e in events.Scalars('val_loss'))
    config.TRAIN.EPOCHS = 3
    config.MODEL.RESUME = str(out / 'last.ckpt')
    resumed = train_main(config, str(out))
    assert resumed.global_step == 6
    assert (out / 'epoch-002.ckpt').exists()


def test_production_presets_are_100_epochs():
    for size in ['330m', '700m', '3b']:
        c = _C.clone()
        _update_config_from_file(c, f'configs/train/{size}.yaml')
        assert c.TRAIN.EPOCHS == 100 and c.SAVE_FREQ == 1
        assert c.DATA.NUM_TIMESTEPS == 7 and c.DATA.IMG_SIZE == 512
        assert c.DATA.REQUIRE_TIMESTAMPS and c.TENSORBOARD.RECONSTRUCTIONS
