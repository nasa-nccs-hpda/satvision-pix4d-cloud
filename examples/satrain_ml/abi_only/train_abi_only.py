import os
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from model import FastSatRain, IPWGUNet
import lightning as L
from torch.utils.data import DataLoader

# calculate_input_features is only needed to confirm channel count (22)
from satrain.input import Geo, Ancillary, calculate_input_features

print("SLURM job initialized.")

# Unlock Tensor Cores for matmul speedup (safe, standard)
torch.set_float32_matmul_precision("high")

# =============================================================
# CONFIG
# =============================================================
# Path to the PREPROCESSED .npy files (created by preprocess.py).
# Structure mirrors: preprocessed/satrain/<sensor>/<split>/<subset>/<geometry>/
PP_ROOT = "/explore/nobackup/projects/pix4dcloud/sanumolu/satrain_ml/satrain_data/preprocessed/satrain/gmi/{split}/xl/on_swath"

inputs = [
    Geo(normalize="minmax", nan=-1.5),
    Ancillary(variables=["ten_meter_wind_u","ten_meter_wind_v","two_meter_temperature",
                          "two_meter_dew_point","surface_type","elevation"],
              normalize="minmax", nan=-1.5),
]
expected = calculate_input_features(inputs)

_geo = np.load(PP_ROOT.format(split="training") + "/geo.npy", mmap_mode="r")
_anc = np.load(PP_ROOT.format(split="training") + "/anc.npy", mmap_mode="r")
INPUT_FEATURES = _geo.shape[1] + _anc.shape[1]   # 16 + 6 = 22 for abi_only
del _geo, _anc

assert INPUT_FEATURES == expected, (
    f"Channel mismatch! Preprocessed data has {INPUT_FEATURES} channels "
    f"but input config expects {expected}. Did you re-run preprocess.py?"
)
print(f"Confirmed {INPUT_FEATURES} input channels (data matches input config).")
BATCH_SIZE = 128             # saturates V100 on 64x64 patches
NUM_WORKERS = 32             # enough to feed GPU from RAM; not "more is better"
N_EPOCHS = 200
BASE_LR = 1e-3           
INPUT_NAN_FILL = -1.5        # matches nan=-1.5 used in preprocessing

training_data   = FastSatRain(PP_ROOT.format(split="training"),  use_gmi=False, augment=True)
validation_data = FastSatRain(PP_ROOT.format(split="validation"), use_gmi=False)

training_loader = DataLoader(
    training_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)
validation_loader = DataLoader(
    validation_data,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
print("Data Loaded!")

# =============================================================
# CALLBACKS & LOGGER
# =============================================================
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="abi_only_cnn_v5log1p-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")

csv_logger = CSVLogger("logs/", name="abi_only_cnn_v5log1p")


# =============================================================
# TRAIN
# =============================================================
if __name__ == "__main__":
    # Sanity check: confirm channel count matches preprocessed data
    input_features = calculate_input_features(inputs) if "inputs" in globals() else INPUT_FEATURES
    print(f"Initializing UNet with {INPUT_FEATURES} input channels...")

    unet = IPWGUNet(
        input_features=INPUT_FEATURES,
        internal_features=[32, 64, 128, 256, 512],
        n_epochs=N_EPOCHS,
        base_lr=BASE_LR,
        input_nan_fill=INPUT_NAN_FILL,
    )

    trainer = L.Trainer(
        max_epochs=unet.n_epochs,
        precision="bf16-mixed",     # V100 = fp16. Use "bf16-mixed" ONLY on grace/H100.
        accelerator="gpu",
        devices=1,
        benchmark=True,           # cuDNN autotuning; safe since input size is fixed (22x64x64)
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
        profiler="simple",        # RUN #1 ONLY: confirm GPU is busy, then remove this line.
    )

    print(f"Starting training with {INPUT_FEATURES} input channels (subset='l')...")
    trainer.fit(
        model=unet,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
    )

    print("Training complete! Model saved in the 'checkpoints' directory.")