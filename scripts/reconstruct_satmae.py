#!/usr/bin/env python3
"""
Run masked reconstructions for SatVision Pix4D SatMAE checkpoints.

Example:
    python scripts/reconstruct_satmae.py \
        --config tests/configs/test_satmae_dev.yaml \
        --checkpoint /path/to/best.ckpt \
        --data-dir /path/to/zarr_chips \
        --output-dir recon_eval \
        --max-samples 20 \
        --save-visualizations 5
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.datasets.abi_temporal_dataset import ABITemporalDataset
from satvision_pix4d.models.encoders.mae import build_satmae_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SatMAE masked reconstructions.")
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt or DeepSpeed model_states.pt.")
    parser.add_argument("--data-dir", nargs="+", default=None, help="Directory or directories containing chip .zarr files.")
    parser.add_argument("--output-dir", required=True, help="Where to save metrics and visualizations.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None, help="Override config.DATA.MASK_RATIO.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-visualizations", type=int, default=5, help="Number of samples to render as PNG grids.")
    parser.add_argument(
        "--save-reconstruction-arrays",
        type=int,
        default=None,
        help="Number of samples to save as .npz arrays. Defaults to --save-visualizations.",
    )
    parser.add_argument("--viz-bands", type=int, nargs="+", default=[0, 1, 2, 6, 10, 13, 15])
    parser.add_argument("--viz-timestep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys instead of failing.",
    )
    return parser.parse_args()


def load_config(path):
    config = _C.clone()
    _update_config_from_file(config, path)
    return config


def clean_state_dict(checkpoint):
    lightning_state = False
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        lightning_state = True
    elif isinstance(checkpoint, dict) and "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        if lightning_state and not (
            key.startswith("model.") or key.startswith("module.model.")
        ):
            continue
        new_key = key
        for prefix in ("module.model.", "model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_model(config, checkpoint_path, device, allow_partial=False):
    model = build_satmae_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = clean_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info("Loaded checkpoint with %d missing and %d unexpected keys.", len(missing), len(unexpected))
    if missing:
        logging.info("First missing keys: %s", missing[:10])
    if unexpected:
        logging.info("First unexpected keys: %s", unexpected[:10])
    if (missing or unexpected) and not allow_partial:
        raise RuntimeError(
            "Checkpoint did not exactly match the SatMAE model. "
            "Rerun with --allow-partial-checkpoint only if this is intentional."
        )
    model.to(device)
    model.eval()
    return model


def standardize(samples_raw, config, device):
    mean = torch.as_tensor(config.DATA.MEAN, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    std = torch.as_tensor(config.DATA.STD, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    return (samples_raw.float() - mean) / std.clamp_min(1e-6)


def inverse_standardize(samples_z, config, device):
    mean = torch.as_tensor(config.DATA.MEAN, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    std = torch.as_tensor(config.DATA.STD, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    return samples_z.float() * std + mean


def tokens_to_pixel_mask(model, mask_tokens, t, h, w):
    patch = int(model.patch_embed.patch_size[0])
    hp, wp = h // patch, w // patch
    mask = mask_tokens.view(mask_tokens.shape[0], t, hp, wp)
    mask = mask.unsqueeze(2)
    mask = mask.repeat_interleave(patch, dim=3).repeat_interleave(patch, dim=4)
    return mask.float()


def reconstruct_batch(model, samples_z, timestamps, mask_ratio):
    loss, pred_tokens, mask_tokens = model(samples_z, timestamps, mask_ratio)
    b, t, c, h, w = samples_z.shape

    gt_tokens = model.patchify(samples_z)
    mask_expanded = mask_tokens.bool().unsqueeze(-1)
    merged_tokens = torch.where(mask_expanded, pred_tokens, gt_tokens)

    recon_z = model.unpatchify(merged_tokens, t, h, w)
    pred_only_z = model.unpatchify(pred_tokens, t, h, w)
    pixel_mask = tokens_to_pixel_mask(model, mask_tokens, t, h, w)
    return loss, recon_z, pred_only_z, pixel_mask


def visible_copy_error(recon, target, mask):
    visible = 1.0 - mask
    count = visible.sum().clamp_min(1.0)
    err = (recon - target) * visible
    return {
        "visible_copy_mse": float((err.square().sum() / count).item()),
        "visible_copy_mae": float((err.abs().sum() / count).item()),
        "visible_copy_max_abs": float(err.abs().max().item()),
    }


def global_ssim(pred, target, data_range, mask=None):
    # pred/target: (N,1,H,W), mask: optional (N,1,H,W). This is a masked/global
    # SSIM variant, not a windowed SSIM implementation.
    pred = pred.float()
    target = target.float()
    if mask is None:
        weight = torch.ones_like(pred)
    else:
        weight = mask.float().expand_as(pred)

    count = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    pred_mean = (pred * weight).sum(dim=(1, 2, 3)) / count
    target_mean = (target * weight).sum(dim=(1, 2, 3)) / count

    pred_centered = pred - pred_mean.view(-1, 1, 1, 1)
    target_centered = target - target_mean.view(-1, 1, 1, 1)
    pred_var = (pred_centered.square() * weight).sum(dim=(1, 2, 3)) / count
    target_var = (target_centered.square() * weight).sum(dim=(1, 2, 3)) / count
    covariance = (pred_centered * target_centered * weight).sum(dim=(1, 2, 3)) / count

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2)
    denominator = (pred_mean.square() + target_mean.square() + c1) * (pred_var + target_var + c2)
    return numerator / denominator.clamp_min(1e-12)


class BandAccumulator:
    def __init__(self, num_bands):
        self.num_bands = num_bands
        self.masked_sse = np.zeros(num_bands, dtype=np.float64)
        self.masked_sae = np.zeros(num_bands, dtype=np.float64)
        self.masked_sum_err = np.zeros(num_bands, dtype=np.float64)
        self.masked_count = np.zeros(num_bands, dtype=np.float64)
        self.visible_sse = np.zeros(num_bands, dtype=np.float64)
        self.visible_sae = np.zeros(num_bands, dtype=np.float64)
        self.visible_sum_err = np.zeros(num_bands, dtype=np.float64)
        self.visible_count = np.zeros(num_bands, dtype=np.float64)
        self.full_sse = np.zeros(num_bands, dtype=np.float64)
        self.full_sae = np.zeros(num_bands, dtype=np.float64)
        self.full_sum_err = np.zeros(num_bands, dtype=np.float64)
        self.full_count = np.zeros(num_bands, dtype=np.float64)
        self.masked_ssim_sum = np.zeros(num_bands, dtype=np.float64)
        self.visible_ssim_sum = np.zeros(num_bands, dtype=np.float64)
        self.full_ssim_sum = np.zeros(num_bands, dtype=np.float64)
        self.ssim_count = np.zeros(num_bands, dtype=np.float64)
        self.target_sum = np.zeros(num_bands, dtype=np.float64)
        self.pred_sum = np.zeros(num_bands, dtype=np.float64)

    def update(self, pred, target, mask, data_range):
        # pred/target: (B,T,C,H,W), mask: (B,T,1,H,W)
        err = pred - target
        masked_err = err * mask
        visible_mask = 1.0 - mask
        visible_err = err * visible_mask
        count_masked = mask.sum(dim=(0, 1, 3, 4)).detach().cpu().numpy()
        count_visible = visible_mask.sum(dim=(0, 1, 3, 4)).detach().cpu().numpy()
        count_full = np.prod([pred.shape[0], pred.shape[1], pred.shape[3], pred.shape[4]])

        self.masked_sse += (masked_err.square().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.masked_sae += (masked_err.abs().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.masked_sum_err += (masked_err.sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.masked_count += count_masked

        self.visible_sse += (visible_err.square().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.visible_sae += (visible_err.abs().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.visible_sum_err += (visible_err.sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.visible_count += count_visible

        self.full_sse += (err.square().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.full_sae += (err.abs().sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.full_sum_err += (err.sum(dim=(0, 1, 3, 4))).detach().cpu().numpy()
        self.full_count += count_full
        self.target_sum += target.sum(dim=(0, 1, 3, 4)).detach().cpu().numpy()
        self.pred_sum += pred.sum(dim=(0, 1, 3, 4)).detach().cpu().numpy()

        b, t, c, h, w = pred.shape
        flat_mask = mask.reshape(b * t, 1, h, w)
        flat_visible_mask = visible_mask.reshape(b * t, 1, h, w)
        for band in range(self.num_bands):
            flat_pred = pred[:, :, band:band + 1].reshape(b * t, 1, h, w)
            flat_target = target[:, :, band:band + 1].reshape(b * t, 1, h, w)
            band_range = max(float(data_range[band]), 1e-6)
            self.masked_ssim_sum[band] += global_ssim(flat_pred, flat_target, band_range, flat_mask).sum().item()
            self.visible_ssim_sum[band] += global_ssim(
                flat_pred, flat_target, band_range, flat_visible_mask
            ).sum().item()
            self.full_ssim_sum[band] += global_ssim(flat_pred, flat_target, band_range).sum().item()
            self.ssim_count[band] += b * t

    def rows(self, data_range):
        rows = []
        for band in range(self.num_bands):
            masked_mse = self.masked_sse[band] / max(self.masked_count[band], 1.0)
            visible_mse = self.visible_sse[band] / max(self.visible_count[band], 1.0)
            full_mse = self.full_sse[band] / max(self.full_count[band], 1.0)
            masked_rmse = math.sqrt(masked_mse)
            visible_rmse = math.sqrt(visible_mse)
            full_rmse = math.sqrt(full_mse)
            band_range = max(float(data_range[band]), 1e-6)
            rows.append({
                "band": band,
                "masked_count": int(self.masked_count[band]),
                "masked_mse": masked_mse,
                "masked_rmse": masked_rmse,
                "masked_mae": self.masked_sae[band] / max(self.masked_count[band], 1.0),
                "masked_bias": self.masked_sum_err[band] / max(self.masked_count[band], 1.0),
                "masked_psnr": 10.0 * math.log10((band_range ** 2) / max(masked_mse, 1e-12)),
                "masked_ssim": self.masked_ssim_sum[band] / max(self.ssim_count[band], 1.0),
                "visible_count": int(self.visible_count[band]),
                "visible_mse": visible_mse,
                "visible_rmse": visible_rmse,
                "visible_mae": self.visible_sae[band] / max(self.visible_count[band], 1.0),
                "visible_bias": self.visible_sum_err[band] / max(self.visible_count[band], 1.0),
                "visible_psnr": 10.0 * math.log10((band_range ** 2) / max(visible_mse, 1e-12)),
                "visible_ssim": self.visible_ssim_sum[band] / max(self.ssim_count[band], 1.0),
                "full_mse": full_mse,
                "full_rmse": full_rmse,
                "full_mae": self.full_sae[band] / max(self.full_count[band], 1.0),
                "full_bias": self.full_sum_err[band] / max(self.full_count[band], 1.0),
                "full_psnr": 10.0 * math.log10((band_range ** 2) / max(full_mse, 1e-12)),
                "full_ssim": self.full_ssim_sum[band] / max(self.ssim_count[band], 1.0),
                "target_mean": self.target_sum[band] / max(self.full_count[band], 1.0),
                "prediction_mean": self.pred_sum[band] / max(self.full_count[band], 1.0),
            })
        return rows


def per_sample_rows(sample_ids, pred, target, mask, data_range):
    rows = []
    err = pred - target
    b, t, c, h, w = pred.shape
    visible_mask = 1.0 - mask
    for i in range(b):
        for band in range(c):
            band_err = err[i:i + 1, :, band:band + 1]
            band_mask = mask[i:i + 1]
            band_visible_mask = visible_mask[i:i + 1]
            masked_count = float(band_mask.sum().item())
            visible_count = float(band_visible_mask.sum().item())
            masked_mse = float((band_err.square() * band_mask).sum().item() / max(masked_count, 1.0))
            visible_mse = float((band_err.square() * band_visible_mask).sum().item() / max(visible_count, 1.0))
            full_mse = float(band_err.square().mean().item())
            band_range = max(float(data_range[band]), 1e-6)
            flat_pred = pred[i:i + 1, :, band:band + 1].reshape(t, 1, h, w)
            flat_target = target[i:i + 1, :, band:band + 1].reshape(t, 1, h, w)
            flat_mask = band_mask.reshape(t, 1, h, w)
            flat_visible_mask = band_visible_mask.reshape(t, 1, h, w)
            rows.append({
                "sample": sample_ids[i],
                "band": band,
                "masked_count": int(masked_count),
                "masked_mse": masked_mse,
                "masked_rmse": math.sqrt(masked_mse),
                "masked_mae": float((band_err.abs() * band_mask).sum().item() / max(masked_count, 1.0)),
                "masked_psnr": 10.0 * math.log10((band_range ** 2) / max(masked_mse, 1e-12)),
                "masked_ssim": float(global_ssim(flat_pred, flat_target, band_range, flat_mask).mean().item()),
                "visible_count": int(visible_count),
                "visible_mse": visible_mse,
                "visible_rmse": math.sqrt(visible_mse),
                "visible_mae": float((band_err.abs() * band_visible_mask).sum().item() / max(visible_count, 1.0)),
                "visible_psnr": 10.0 * math.log10((band_range ** 2) / max(visible_mse, 1e-12)),
                "visible_ssim": float(global_ssim(flat_pred, flat_target, band_range, flat_visible_mask).mean().item()),
                "full_mse": full_mse,
                "full_rmse": math.sqrt(full_mse),
                "full_mae": float(band_err.abs().mean().item()),
                "full_psnr": 10.0 * math.log10((band_range ** 2) / max(full_mse, 1e-12)),
                "full_ssim": float(global_ssim(flat_pred, flat_target, band_range).mean().item()),
            })
    return rows


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_for_display(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x, [2, 98])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def save_visualization(path, target, masked_input, pred_only, recon, abs_err, bands, timestep):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping visualization %s", path)
        return

    target = target.detach().cpu().numpy()
    masked_input = masked_input.detach().cpu().numpy()
    pred_only = pred_only.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    abs_err = abs_err.detach().cpu().numpy()

    bands = [b for b in bands if b < target.shape[2]]
    fig, axes = plt.subplots(len(bands), 5, figsize=(15, 3 * len(bands)), squeeze=False)
    col_titles = ["target", "masked input", "prediction", "merged recon", "abs error"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)
    for row, band in enumerate(bands):
        images = [
            target[0, timestep, band],
            masked_input[0, timestep, band],
            pred_only[0, timestep, band],
            recon[0, timestep, band],
            abs_err[0, timestep, band],
        ]
        for col, image in enumerate(images):
            axes[row, col].imshow(normalize_for_display(image), cmap="gray")
            axes[row, col].set_axis_off()
        axes[row, 0].set_ylabel(f"band {band}", rotation=0, labelpad=32, va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reconstruction_array(path, target, masked_input, pred_only, recon, mask):
    np.savez_compressed(
        path,
        target=target.detach().cpu().numpy().astype(np.float32),
        masked_input=masked_input.detach().cpu().numpy().astype(np.float32),
        prediction_only=pred_only.detach().cpu().numpy().astype(np.float32),
        merged_reconstruction=recon.detach().cpu().numpy().astype(np.float32),
        mask=mask.detach().cpu().numpy().astype(np.uint8),
    )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_array_count = (
        args.save_visualizations if args.save_reconstruction_arrays is None else args.save_reconstruction_arrays
    )
    if args.save_visualizations == 0 and reconstruction_array_count == 0:
        logging.warning("Both visualization and reconstruction array saving are disabled.")

    config = load_config(args.config)
    data_paths = args.data_dir if args.data_dir is not None else list(config.DATA.VAL_DATA_PATHS)
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else float(config.DATA.MASK_RATIO)
    device = torch.device(args.device)

    dataset = ABITemporalDataset(
        data_paths=data_paths,
        img_size=int(config.DATA.IMG_SIZE),
        in_chans=int(config.MODEL.MAE_VIT.IN_CHANS),
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(config, args.checkpoint, device, allow_partial=args.allow_partial_checkpoint)
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(
            {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "data_paths": list(data_paths),
                "samples": len(dataset),
                "mask_ratio": mask_ratio,
                "device": str(device),
                "batch_size": args.batch_size,
                "save_visualizations": args.save_visualizations,
                "save_reconstruction_arrays": reconstruction_array_count,
            },
            f,
            indent=2,
        )

    num_bands = int(config.MODEL.MAE_VIT.IN_CHANS)
    data_range = np.asarray(config.DATA.MAX[:num_bands], dtype=np.float64) - np.asarray(
        config.DATA.MIN[:num_bands], dtype=np.float64
    )
    accumulator = BandAccumulator(num_bands)
    sample_rows = []
    loss_values = []
    visible_copy_mse_values = []
    visible_copy_mae_values = []
    visible_copy_max_abs_values = []
    visualizations_saved = 0
    reconstruction_arrays_saved = 0
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, (samples_raw, timestamps) in enumerate(loader):
            samples_raw = samples_raw.to(device, non_blocking=True).float()
            timestamps = timestamps.to(device, non_blocking=True)
            z = standardize(samples_raw, config, device)

            loss, recon_z, pred_only_z, pixel_mask = reconstruct_batch(model, z, timestamps, mask_ratio)
            recon_raw = inverse_standardize(recon_z, config, device)
            pred_only_raw = inverse_standardize(pred_only_z, config, device)
            copy_stats = visible_copy_error(recon_raw, samples_raw, pixel_mask)
            visible_copy_mse_values.append(copy_stats["visible_copy_mse"])
            visible_copy_mae_values.append(copy_stats["visible_copy_mae"])
            visible_copy_max_abs_values.append(copy_stats["visible_copy_max_abs"])
            if copy_stats["visible_copy_max_abs"] > 1e-4:
                logging.warning(
                    "Visible patch copy check is not near zero: mse=%.6g, max_abs=%.6g",
                    copy_stats["visible_copy_mse"],
                    copy_stats["visible_copy_max_abs"],
                )

            accumulator.update(recon_raw, samples_raw, pixel_mask, data_range)
            sample_ids = list(range(sample_offset, sample_offset + samples_raw.shape[0]))
            sample_rows.extend(per_sample_rows(sample_ids, recon_raw, samples_raw, pixel_mask, data_range))
            loss_values.append(float(loss.item()))

            batch_end = sample_offset + samples_raw.shape[0]
            for global_idx in range(sample_offset, batch_end):
                local_idx = global_idx - sample_offset
                one_mask = pixel_mask[local_idx:local_idx + 1]
                masked_input = samples_raw[local_idx:local_idx + 1] * (1.0 - one_mask)

                if reconstruction_arrays_saved < reconstruction_array_count:
                    save_reconstruction_array(
                        output_dir / f"reconstruction_sample_{global_idx:04d}.npz",
                        samples_raw[local_idx:local_idx + 1],
                        masked_input,
                        pred_only_raw[local_idx:local_idx + 1],
                        recon_raw[local_idx:local_idx + 1],
                        one_mask,
                    )
                    reconstruction_arrays_saved += 1

                if visualizations_saved < args.save_visualizations:
                    abs_err = (
                        recon_raw[local_idx:local_idx + 1] - samples_raw[local_idx:local_idx + 1]
                    ).abs() * one_mask
                    save_visualization(
                        output_dir / f"reconstruction_sample_{global_idx:04d}.png",
                        samples_raw[local_idx:local_idx + 1],
                        masked_input,
                        pred_only_raw[local_idx:local_idx + 1],
                        recon_raw[local_idx:local_idx + 1],
                        abs_err,
                        args.viz_bands,
                        min(args.viz_timestep, samples_raw.shape[1] - 1),
                    )
                    visualizations_saved += 1

            sample_offset += samples_raw.shape[0]
            logging.info("Processed batch %d/%d, loss=%.6f", batch_idx + 1, len(loader), loss.item())

    band_rows = accumulator.rows(data_range)
    write_csv(output_dir / "per_band_metrics.csv", band_rows)
    write_csv(output_dir / "per_sample_band_metrics.csv", sample_rows)

    summary = {
        "samples": len(dataset),
        "mask_ratio": mask_ratio,
        "mean_loss": float(np.mean(loss_values)) if loss_values else float("nan"),
        "mean_visible_copy_mse": float(np.mean(visible_copy_mse_values)),
        "mean_visible_copy_mae": float(np.mean(visible_copy_mae_values)),
        "max_visible_copy_abs": float(np.max(visible_copy_max_abs_values)),
        "mean_masked_mse": float(np.mean([row["masked_mse"] for row in band_rows])),
        "mean_masked_psnr": float(np.mean([row["masked_psnr"] for row in band_rows])),
        "mean_masked_ssim": float(np.mean([row["masked_ssim"] for row in band_rows])),
        "mean_visible_mse": float(np.mean([row["visible_mse"] for row in band_rows])),
        "mean_visible_psnr": float(np.mean([row["visible_psnr"] for row in band_rows])),
        "mean_visible_ssim": float(np.mean([row["visible_ssim"] for row in band_rows])),
        "mean_full_mse": float(np.mean([row["full_mse"] for row in band_rows])),
        "mean_full_psnr": float(np.mean([row["full_psnr"] for row in band_rows])),
        "mean_full_ssim": float(np.mean([row["full_ssim"] for row in band_rows])),
        "visualizations_saved": visualizations_saved,
        "reconstruction_arrays_saved": reconstruction_arrays_saved,
    }
    write_csv(output_dir / "summary_metrics.csv", [summary])
    logging.info("Saved metrics to %s", output_dir)
    logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
