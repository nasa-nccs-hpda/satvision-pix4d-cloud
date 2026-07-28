import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex
import pytorch_lightning as L


class UNET3D(nn.Module):
    def __init__(self, in_channels=16, decoder_classes=64, num_classes=9, head_dropout=0.2, output_shape=(478, 40)):
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        # 3D down-convolutions
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])

        # 3D up-transpositions. We only downsample/upsample spatial dimensions, keeping time dimension intact.
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose3d(layer, layer_n, kernel_size=(1,2,2), stride=(1,2,2))
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])

        # 3D up-convolutions
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])

        # Pooling only on spatial dimensions (H, W) to preserve the 7 timesteps
        self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # Collapse the 7 timesteps down to 1 feature map
        self.time_collapse = nn.Conv3d(64, decoder_classes, kernel_size=(7, 1, 1))

        # 2D prediction head to output the final transect shape
        self.prediction_head = nn.Sequential(
            nn.Conv2d(decoder_classes, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Dropout(head_dropout),
            nn.Upsample(size=output_shape, mode='bilinear', align_corners=False)
        )

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        # down layers
        concat_layers = []

        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool(x)

        concat_layers = concat_layers[::-1]

        # up layers
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = F.interpolate(x, size=concat_layer.shape[2:], mode='trilinear', align_corners=False)

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        # Collapse time dimension
        x = self.time_collapse(x) # [B, 64, 1, H, W]
        x = x.squeeze(2) # [B, 64, H, W]

        # Final prediction head
        x = self.prediction_head(x) # [B, 1, 478, 40]

        return x


class LightningModel(L.LightningModule):
    def __init__(self, lr=1e-4, num_classes=9, in_channels=16, final_size=(478, 40), head_dropout=0.2):
        super().__init__()
        
        # We only use the 3D UNet baseline per your request
        self.model = UNET3D(in_channels=in_channels, decoder_classes=64, num_classes=num_classes, head_dropout=head_dropout, output_shape=final_size)

        self.save_hyperparameters()
        
        # Multi-class loss and metrics
        # ignore_index=-1 tells PyTorch to skip the "missing retrieval" pixels during training
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, stage):
        chips, masks = batch["chip"], batch["mask"]
        logits = self.forward(chips)  # (B, 9, H, W)
        
        loss = self.ce_loss(logits, masks)
        
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        output = {"loss": loss}
        
        if stage == 'val' or stage == 'test':
            preds = torch.argmax(logits, dim=1) # Get the class with highest probability
            iou = self.iou(preds, masks)
            
            self.log(f'{stage}_iou', iou, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
            if stage == 'test':
                output["iou"] = iou
        return output

    def training_step(self, b, i): return self._common_step(b, 'train')
    def validation_step(self, b, i): return self._common_step(b, 'val')
    def test_step(self, b, i): return self._common_step(b, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
