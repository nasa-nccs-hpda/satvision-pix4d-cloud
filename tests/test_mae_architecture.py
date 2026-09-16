"""CPU regression tests; large presets are constructed on meta, never allocated."""
from types import SimpleNamespace

import pytest
import torch

from satvision_pix4d.configs.config import get_config
from satvision_pix4d.models.encoders.mae import build_satmae_model
from satvision_pix4d.models.encoders.models_mae_temporal import MaskedAutoencoderViT
from satvision_pix4d.models.reconstruction_loss import ReconstructionLoss


def tiny(**kwargs):
    args = dict(img_size=32, patch_size=16, in_chans=2, embed_dim=32, depth=1,
                num_heads=4, decoder_embed_dim=16, decoder_depth=1,
                decoder_num_heads=4, enc_ts_dim_per_comp=8, dec_ts_dim_per_comp=4)
    args.update(kwargs)
    return MaskedAutoencoderViT(**args)


@pytest.mark.parametrize('shape', [(1, 1, 2, 32, 48), (2, 3, 2, 64, 32)])
def test_dynamic_shapes_roundtrip_and_backward(shape):
    model = tiny(n_time_components=5, use_checkpoint=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    names_before = tuple(model.state_dict())
    x = torch.randn(shape)
    assert torch.equal(model.unpatchify(model.patchify(x), shape[1], *shape[-2:]), x)
    before = model.encoder_temporal_spatial_proj.weight.detach().clone()
    loss, pred, mask = model(x, torch.zeros(shape[:2] + (5,)), 0.75)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.encoder_temporal_spatial_proj.weight)
    assert names_before == tuple(model.state_dict())
    assert pred.shape == model.patchify(x).shape


def test_no_mask_and_full_mask_are_finite():
    model = tiny()
    for ratio in (0, 1):
        loss, _, mask = model(torch.randn(1, 2, 2, 32, 32), torch.zeros(1, 2, 3), ratio)
        assert torch.isfinite(loss)
        assert mask.eq(ratio).all()
        loss.backward()


def test_invalid_inputs_and_mask_permutations():
    model = tiny()
    x, ts = torch.randn(1, 1, 2, 32, 32), torch.zeros(1, 1, 3)
    with pytest.raises(ValueError, match='timestamps'):
        model(x, ts[..., :2])
    with pytest.raises(ValueError, match='mask_ratio'):
        model(x, ts, 1.1)
    with pytest.raises(ValueError, match='permutation'):
        model(x, ts, mask=torch.zeros(1, 4, dtype=torch.long))
    perm = torch.tensor([[3, 1, 0, 2]])
    _, _, mask = model(x, ts, 0.5, perm)
    assert mask.tolist() == [[1, 0, 1, 0]]


def test_pixel_structure_identity_masking_and_gradients():
    criterion = ReconstructionLoss('pixel_structure', 2, [0], [-3, -3], [3, 3])
    target = torch.randn(1, 1, 2, 128, 128)
    mask = torch.zeros(1, 1, 1, 128, 128)
    mask[..., :64, :] = 1
    exact = criterion(target, target, mask)
    assert exact.abs() < 1e-6
    pred = (target + 0.2 * torch.randn_like(target)).requires_grad_()
    loss = criterion(pred, target, mask)
    loss.backward()
    assert loss > exact
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[..., :64, :].abs().sum() > 0
    assert pred.grad[..., 64:, :].eq(0).all()
    assert criterion(pred, target, torch.zeros_like(mask)).item() == 0


def test_ssim_excludes_ir(monkeypatch):
    criterion = ReconstructionLoss('pixel_structure', 2, [0], [0, 0], [1, 1])
    seen = []
    original = criterion._ms_ssim
    def record(p, t, m):
        seen.append(p.detach().clone())
        return original(p, t, m)
    monkeypatch.setattr(criterion, '_ms_ssim', record)
    target = torch.rand(1, 1, 2, 128, 128)
    pred = target.clone()
    pred[:, :, 1] += 1
    criterion(pred, target, torch.ones(1, 1, 1, 128, 128))
    assert torch.equal(seen[0], target.flatten(0, 1)[:, :1])


def test_combined_model_loss_mixed_precision_backward():
    model = tiny(loss_type='pixel_structure', visnir_channels=[0],
                 ssim_min=[-3, -3], ssim_max=[3, 3], use_checkpoint=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _, _ = model(torch.randn(1, 1, 2, 128, 128), torch.zeros(1, 1, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('size,limits', [('330m', (330e6, 340e6)), ('700m', (690e6, 710e6)), ('3b', (3e9, 3.1e9)), ('25b', (24.9e9, 25.1e9))])
def test_preset_counts(size, limits):
    config = get_config(SimpleNamespace(cfg=f'configs/pretrain/{size}.yaml'))
    with torch.device('meta'):
        model = build_satmae_model(config)
    count = sum(p.numel() for p in model.parameters())
    assert limits[0] < count < limits[1]
    assert model.n_time_components == 5
    assert model.use_checkpoint
    assert all(b.attn.fused_attn for b in model.blocks)
    print(size, count)


def test_checkpoint_compatibility():
    first, second = tiny(), tiny()
    second.load_state_dict(first.state_dict(), strict=True)
    # Loss configuration introduces no persistent weights.
    third = tiny(loss_type='pixel_structure', visnir_channels=[0], ssim_min=[0, 0], ssim_max=[1, 1])
    third.load_state_dict(first.state_dict(), strict=True)


def test_pipeline_metrics_scheduler_and_training(tmp_path):
    from lightning.pytorch import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    from satvision_pix4d.configs.config import _C
    from satvision_pix4d.pipelines.satvision_pix4d_pretrain import SatVisionPix4DSatMAEPretrain
    cfg = _C.clone()
    cfg.MODEL.TYPE = 'satmae'
    cfg.MODEL.MAE_VIT.IN_CHANS = 2
    cfg.MODEL.MAE_VIT.EMBED_DIM = 32
    cfg.MODEL.MAE_VIT.DEPTHS = 1
    cfg.MODEL.MAE_VIT.NUM_HEADS = 4
    cfg.MODEL.MAE_VIT.DECODER_EMBED_DIM = 16
    cfg.MODEL.MAE_VIT.DECODER_DEPTH = 1
    cfg.MODEL.MAE_VIT.DECODER_NUM_HEADS = 4
    cfg.MODEL.MAE_VIT.LOSS_TYPE = 'pixel_structure'
    cfg.MODEL.MAE_VIT.VISNIR_CHANNELS = [0]
    cfg.DATA.IMG_SIZE = 128
    cfg.DATA.BATCH_SIZE = 1
    cfg.DATA.MEAN = [0., 0.]
    cfg.DATA.STD = [1., 1.]
    cfg.DATA.MIN = [-3., -3.]
    cfg.DATA.MAX = [3., 3.]
    cfg.TRAIN.EPOCHS = 1
    cfg.TRAIN.WARMUP_EPOCHS = 0
    model = SatVisionPix4DSatMAEPretrain(cfg)
    target = torch.zeros(1, 1, 2, 128, 128)
    metric = model._avg_over_time(model.train_psnr, target + 1, target, torch.ones(1, 1, 1, 128, 128))
    assert torch.allclose(metric, torch.tensor(10 * __import__('math').log10(36.)))
    loader = DataLoader(TensorDataset(torch.randn(2, 1, 2, 128, 128), torch.zeros(2, 1, 3)), batch_size=1)
    trainer = Trainer(accelerator='cpu', devices=1, max_steps=2, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0,
                      default_root_dir=str(tmp_path), limit_val_batches=1)
    before = model.model.patch_embed.proj.weight.detach().clone()
    trainer.fit(model, loader, loader)
    assert trainer.global_step == 2
    assert not torch.equal(before, model.model.patch_embed.proj.weight)
    assert torch.isfinite(trainer.callback_metrics['val_loss'])


def test_one_model_handles_all_figure_resolutions():
    model = tiny().eval()
    with torch.no_grad():
        for size in (128, 256, 512):
            loss, pred, mask = model(torch.randn(1, 1, 2, size, size), torch.zeros(1, 1, 3))
            assert pred.shape[1] == (size // 16) ** 2
            assert torch.isfinite(loss)


def test_refinement_preserves_visible_predictions_for_supervision():
    model = tiny(refine_pixels=True, visible_loss_weight=1.)
    imgs = torch.randn(1, 1, 2, 32, 32)
    pred = torch.randn_like(model.patchify(imgs), requires_grad=True)
    mask = torch.tensor([[1., 0., 1., 0.]])
    refined = model.refine_prediction_tokens(imgs, pred, mask)
    assert torch.equal(refined[:, [1, 3]], pred[:, [1, 3]])
    model.forward_loss(imgs, refined, mask).backward()
    assert pred.grad[:, [1, 3]].abs().sum() > 0
