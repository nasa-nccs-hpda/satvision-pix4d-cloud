import numpy as np
import torch

from satvision_pix4d.datasets.abi_temporal_dataset import ABITemporalDataset
from satvision_pix4d.models.encoders.models_mae_temporal import MaskedAutoencoderViT


def test_temporal_dataset_loads_channel_first_npz_chip(tmp_path):
    chip = np.random.rand(7, 16, 32, 32).astype(np.float32)
    timestamps = np.array(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T01:00:00",
            "2020-01-01T02:00:00",
            "2020-01-01T03:00:00",
            "2020-01-01T04:00:00",
            "2020-01-01T05:00:00",
            "2020-01-01T06:00:00",
        ]
    )
    np.savez(tmp_path / "chip.npz", chip=chip, timestamps=timestamps)

    dataset = ABITemporalDataset([str(tmp_path)], img_size=32, in_chans=16)
    tensor, ts = dataset[0]

    assert tensor.shape == torch.Size([7, 16, 32, 32])
    assert ts.shape == (7, 3)
    assert ts[:, 2].tolist() == list(range(7))


def test_temporal_dataset_loads_channel_last_npy_chip(tmp_path):
    chip = np.random.rand(7, 32, 32, 16).astype(np.float32)
    np.save(tmp_path / "chip.npy", chip)

    dataset = ABITemporalDataset([str(tmp_path)], img_size=32, in_chans=16)
    tensor, ts = dataset[0]

    assert tensor.shape == torch.Size([7, 16, 32, 32])
    assert ts.shape == (7, 3)


def test_temporal_dataset_indexes_batched_npz_chip_file(tmp_path):
    chips = np.random.rand(20, 7, 16, 32, 32).astype(np.float32)
    timestamps = np.tile(np.arange(7).reshape(1, 7), (20, 1))
    np.savez(tmp_path / "chips.npz", chips=chips, timestamps=timestamps)

    dataset = ABITemporalDataset([str(tmp_path)], img_size=32, in_chans=16)
    tensor, ts = dataset[19]

    assert len(dataset) == 20
    assert tensor.shape == torch.Size([7, 16, 32, 32])
    assert ts.shape == (7, 3)


def test_tiny_temporal_mae_forward_supports_pix4d_chip_shape():
    model = MaskedAutoencoderViT(
        img_size=32,
        patch_size=16,
        in_chans=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        n_time_components=3,
    )
    imgs = torch.randn(1, 7, 16, 32, 32)
    timestamps = torch.zeros(1, 7, 3, dtype=torch.int32)

    loss, pred, mask = model(imgs, timestamps, mask_ratio=0.5)

    assert loss.ndim == 0
    assert pred.shape == torch.Size([1, 28, 16 * 16 * 16])
    assert mask.shape == torch.Size([1, 28])


def test_temporal_mae_same_mask_reuses_spatial_mask_across_time():
    model = MaskedAutoencoderViT(
        img_size=32,
        patch_size=16,
        in_chans=16,
        embed_dim=64,
        depth=1,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        n_time_components=3,
        same_mask=True,
    )
    imgs = torch.randn(1, 7, 16, 32, 32)
    timestamps = torch.zeros(1, 7, 3, dtype=torch.int32)

    _, _, mask = model(imgs, timestamps, mask_ratio=0.5)
    mask_by_time = mask.view(1, 7, 4)

    assert torch.equal(mask_by_time[:, 0:1, :].expand_as(mask_by_time), mask_by_time)
