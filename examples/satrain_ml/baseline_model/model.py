from typing import Callable, List, Any, Dict
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import lightning as L
from torch import nn, optim
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as TF


class FastSatRain(Dataset):
    def __init__(self, dir_: str, use_gmi: bool = False,
                 augment: bool = False, input_nan_fill: float = -1.5):
        self.geo = np.load(f"{dir_}/geo.npy", mmap_mode="r")
        self.anc = np.load(f"{dir_}/anc.npy", mmap_mode="r")
        self.gmi = np.load(f"{dir_}/gmi.npy", mmap_mode="r") if use_gmi else None
        self.sp  = np.load(f"{dir_}/sp.npy",  mmap_mode="r")
        self.pm  = np.load(f"{dir_}/pm.npy",  mmap_mode="r")
        self.hp  = np.load(f"{dir_}/hp.npy",  mmap_mode="r")
        self.use_gmi = use_gmi
        self.augment = augment
        self.fill = input_nan_fill
        self.rng = np.random.default_rng()

    def __len__(self): return self.geo.shape[0]

    def __getitem__(self, i):
        geo = np.ascontiguousarray(self.geo[i])
        anc = np.ascontiguousarray(self.anc[i])
        if self.use_gmi:
            gmi = np.ascontiguousarray(self.gmi[i])
            arr = np.concatenate([gmi, geo, anc], axis=0)   # (35, H, W) baseline
        else:
            arr = np.concatenate([geo, anc], axis=0)        # (22, H, W) abi_only
        x = torch.from_numpy(arr.copy())
        sp = torch.from_numpy(np.ascontiguousarray(self.sp[i]).copy())
        pm = torch.from_numpy(np.ascontiguousarray(self.pm[i]).copy())
        hp = torch.from_numpy(np.ascontiguousarray(self.hp[i]).copy())

        if self.augment:
            angle = float(self.rng.uniform(-180, 180))
            scale = float(self.rng.uniform(0.8, 1.2))
            shear = float(self.rng.uniform(-30.0, 30.0))
            kw = dict(angle=angle, translate=[0.0, 0.0], scale=scale, shear=[shear])
            x  = TF.affine(x, **kw, fill=self.fill)
            sp = TF.affine(sp.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)
            pm = TF.affine(pm.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)
            hp = TF.affine(hp.unsqueeze(0), **kw, fill=float("nan")).squeeze(0)

        return x, {"surface_precip": sp, "precip_mask": pm, "heavy_precip_mask": hp}

class ResNetBlock(nn.Module):
    """
    Implments a basic ResNet block consisting of two convolutions each
    followed by a normalization and activation layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        normalization_layer: Callable[[int], nn.Module] = nn.BatchNorm2d
    ):
        """
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels within the layer and in the output tensor.
        """
        super().__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through block.
        """
        y = self.body(x)
        if self.projection is None:
            shortcut = x
        else:
            shortcut = self.projection(x)
        return y + shortcut


class UNet(nn.Module):
    """
    UNet encoder-decoder architecture for precipitation retrievals.

    The model provides scalar estimates of the 'surface_precip' as well as probabilistic estimates of the
    'probability_of_precip' and 'probability_of_heavy_precip'.

    The model has the following components:
        - Stem: The stem is applied directly to the input and maps the number of input features to
              the features of the first encoder stage.
        - Encoder: Applied to the output from the stem. Consists of multiple stages and performs 2x
              downsampling at the beginning of each stage.
        - Decoder: Each stage consists of bilinear upsampling followed by a convolution block
              that merges the upsampled features with the output from the corresponding encoder
              layer.
        - Heads: A separate head for the retrieval outputs each consisting of a single ResNetBlock
              followed by a fully-connected output layer.
    """
    def __init__(
        self,
        input_features: int,
        internal_features: List[int],
        **block_kwargs
    ):
        """
        Args:
            input_features: The number of input features.
            internal_features: A list containing the number of features/channels within each stage
                of the encoder.
            block_kwargs: Keyword arguments to forward to ResNetBlock factory.
        """
        super().__init__()

        self.stem = ResNetBlock(input_features, internal_features[0])
        chans_in = internal_features[0]
        encoder_stages = []
        for n_features in internal_features:
            encoder_stages.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResNetBlock(chans_in, n_features, **block_kwargs),
            ))
            chans_in = n_features
        self.encoder = nn.ModuleList(encoder_stages)
        
        decoder_stages = []
        for n_features in internal_features[-2::-1]:
            decoder_stages.append(nn.Sequential(
                ResNetBlock(chans_in + n_features, n_features, **block_kwargs),
            ))
            chans_in = n_features
        decoder_stages.append(ResNetBlock(n_features + n_features, n_features, **block_kwargs))
            
        self.decoder = nn.ModuleList(decoder_stages)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        
        heads = {}
        for output in ["surface_precip", "probability_of_precip", "probability_of_heavy_precip"]:
            heads[output] = nn.Sequential(
                ResNetBlock(n_features, n_features, kernel_size=1, **block_kwargs),
                nn.Conv2d(n_features, 1, kernel_size=1)
            )
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Propagate input through network.

        Args:
            x: The tensor containing the input data.

        Return:
            A dictionary containing mapping the keys 'surface_precip', 'probability_of_precip',
            'probability_of_heavy_precip' to the corresponding retrieval results.
        """
        y = self.stem(x)
        shortcuts = []
        for layer in self.encoder:
            shortcuts.append(y)
            y = layer(y)

        shortcuts.reverse()
        for shortcut, layer in zip(shortcuts, self.decoder):
            y = self.upsample(y)
            y = torch.cat([y, shortcut], dim=1)
            y = layer(y)

        return {
            name: head(y) for name, head in self.heads.items()
        }


OUTPUTS = [
    "surface_precip",
    "probability_of_precipitation",
    "probability_of_heavy_precipitation"
]


class IPWGUNet(L.LightningModule):
    def __init__(self, input_features: int, internal_features: List[int],
                 n_epochs: int = 20, base_lr: float = 1e-3,
                 input_nan_fill: float = -1.5, **block_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.n_epochs = n_epochs
        self.base_lr = base_lr
        self.input_nan_fill = input_nan_fill
        self.model = UNet(input_features, internal_features, **block_kwargs)

    def forward(self, retrieval_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(retrieval_input)

    def _compute_losses(self, inpt, target):
        pred = self(inpt)
        surface_precip = target["surface_precip"]
        precip_mask = target["precip_mask"]
        heavy_precip_mask = target["heavy_precip_mask"]
    
        valid = torch.isfinite(surface_precip)
        surface_precip = surface_precip[valid]
        precip_mask = precip_mask[valid]
        heavy_precip_mask = heavy_precip_mask[valid]
        surface_precip_pred = pred["surface_precip"][:, 0][valid]
        pop = pred["probability_of_precip"][:, 0][valid]
        pohp = pred["probability_of_heavy_precip"][:, 0][valid]
    
        # Raw mm/h MSE (matches SatRain reference + abi_only for fair comparison).
        loss_estim = ((surface_precip_pred - surface_precip) ** 2).mean()
        loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
        loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
        # return preds/true so validation can compute mm/h metrics
        return loss_estim, loss_detect, loss_detect_heavy, surface_precip_pred, surface_precip

    ###OLD COMPUTE LOSSES ABOVE & log1p below, kept for reference. 

    # def _compute_losses(self, inpt, target):
    #     pred = self(inpt)
    #     surface_precip = target["surface_precip"]
    #     precip_mask = target["precip_mask"]
    #     heavy_precip_mask = target["heavy_precip_mask"]

    #     valid = torch.isfinite(surface_precip)
    #     surface_precip = surface_precip[valid]
    #     precip_mask = precip_mask[valid]
    #     heavy_precip_mask = heavy_precip_mask[valid]
    #     surface_precip_pred = pred["surface_precip"][:, 0][valid]
    #     pop = pred["probability_of_precip"][:, 0][valid]
    #     pohp = pred["probability_of_heavy_precip"][:, 0][valid]

    #     # --- log1p transform: model predicts in log-space, compresses heavy tail ---
    #     sp_true_log = torch.log1p(surface_precip.clamp(min=0))
    #     loss_estim = ((surface_precip_pred - sp_true_log) ** 2).mean()
    #     # ---------------------------------------------------------------------------

    #     loss_detect = binary_cross_entropy_with_logits(pop, precip_mask)
    #     loss_detect_heavy = binary_cross_entropy_with_logits(pohp, heavy_precip_mask)
    #     # Return preds/true in MM/H space for metrics (invert the model's log output)
    #     sp_pred_mmh = torch.expm1(surface_precip_pred.clamp(min=0))
    #     return loss_estim, loss_detect, loss_detect_heavy, sp_pred_mmh, surface_precip


    def _log_precip_metrics(self, sp_pred, sp_true):
        """Read-only mm/h-space metrics on valid pixels. No effect on training.
        sp_pred, sp_true are already masked to finite pixels, in raw mm/h."""
        with torch.no_grad():
            err = sp_pred - sp_true
            mae  = err.abs().mean()
            bias = err.mean()
            # metrics conditioned on RAINING pixels (>= 0.1 mm/h) — the signal we care about
            rain = sp_true >= 0.1
            if rain.any():
                mae_rain  = (sp_pred[rain] - sp_true[rain]).abs().mean()
                bias_rain = (sp_pred[rain] - sp_true[rain]).mean()
            else:
                mae_rain = torch.tensor(0.0, device=sp_true.device)
                bias_rain = torch.tensor(0.0, device=sp_true.device)
            # correlation over all valid pixels
            if sp_true.numel() > 1:
                vp = sp_pred - sp_pred.mean(); vt = sp_true - sp_true.mean()
                denom = (vp.norm() * vt.norm()).clamp_min(1e-8)
                corr = (vp * vt).sum() / denom
            else:
                corr = torch.tensor(0.0, device=sp_true.device)
            self.log_dict({
                "val_mae_mmh": mae, "val_bias_mmh": bias,
                "val_mae_rain": mae_rain, "val_bias_rain": bias_rain,
                "val_corr": corr,
            }, on_epoch=True, prog_bar=False)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inpt, target = batch
        loss_estim, loss_detect, loss_detect_heavy, _, _ = self._compute_losses(inpt, target)
        tot_loss = loss_estim + loss_detect + loss_detect_heavy

        self.log_dict(
            {
                "train_loss": tot_loss,
                "train_loss_estim": loss_estim,
                "train_loss_detect": loss_detect,
                "train_loss_detect_heavy": loss_detect_heavy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return tot_loss

    def validation_step(self, batch, batch_idx) -> None:
        inpt, target = batch
        loss_estim, loss_detect, loss_detect_heavy, sp_pred, sp_true = self._compute_losses(inpt, target)
        tot_loss = loss_estim + loss_detect + loss_detect_heavy
        learning_rate = self.optimizers().param_groups[0]["lr"]
        self._log_precip_metrics(sp_pred, sp_true)
        self.log_dict(
            {
                "val_loss": tot_loss,
                "val_loss_estim": loss_estim,
                "val_loss_detect": loss_detect,
                "val_loss_detect_heavy": loss_detect_heavy,
                "learning_rate": learning_rate,
            },
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.Adam(self.parameters(), lr=self.base_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}