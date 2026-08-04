import os
import torch
import lightning as L
from torch.utils.data import DataLoader

# Import PyTorch Lightning callbacks and loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Import correct input classes
from satrain.input import GMI, Geo, Ancillary, calculate_input_features
from satrain.target import TargetConfig

# Import model
from model import IPWGUNet, FastSatRain

print("SLURM job initialized.")
# Set Data Path
PP_ROOT = "/explore/nobackup/projects/pix4dcloud/sanumolu/satrain_ml/satrain_data/preprocessed/satrain/gmi/{split}/xl/on_swath"
os.environ["SATRAIN_DATA_PATH"] = "/explore/nobackup/projects/pix4dcloud/sanumolu/"
print("Data Path Set.")
# Unlock H100 Tensor Cores for massive speedup
torch.set_float32_matmul_precision('high')

# 1. Configuration for ABI + PMW + Ancillary (on-swath)
target_config = TargetConfig()
inputs = [
    # PMW Data (13 channels)
    GMI(normalize="minmax", nan=-1.5, include_angles=False),
    
    # Standard ABI Data (16 channels, single timestep)
    Geo(normalize="minmax", nan=-1.5),
    
    # Ancillary Data (6 variables)
    Ancillary(
        variables=[
            "ten_meter_wind_u", 
            "ten_meter_wind_v", 
            "two_meter_temperature", 
            "two_meter_dew_point", 
            "surface_type", 
            "elevation"
        ],
        normalize="minmax", 
        nan=-1.5
    )
]

geometry = "on_swath"

# Optimal settings for H100 GPU
batch_size = 128
num_workers = 32

training_data = FastSatRain(PP_ROOT.format(split="training"),  use_gmi=True, augment=True)
validation_data = FastSatRain(PP_ROOT.format(split="validation"), use_gmi=True)

training_loader = DataLoader(
    training_data, shuffle=True, batch_size=batch_size, num_workers=num_workers,
    pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True,
)
validation_loader = DataLoader(
    validation_data, shuffle=False, batch_size=batch_size, num_workers=num_workers,
    pin_memory=True, persistent_workers=True, prefetch_factor=4,
)


print("Data Loaded!")

# 5. Callbacks and Logger for SLURM Training
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="baseline_cnn_v5-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

early_stopping = EarlyStopping(monitor="val_loss", patience=20, mode="min")

# Logs metrics to a CSV file you can read easily later
csv_logger = CSVLogger("logs/", name="baseline_cnn_v5")

# 6. Run Training
if __name__ == "__main__":
    # Automatically get 35 features
    input_features = calculate_input_features(inputs)
    print(f"Initializing UNet with {input_features} input channels...")
    unet = IPWGUNet(input_features=input_features, internal_features=[32, 64, 128, 256, 512], n_epochs=200)
    _x, _ = training_data[0]
    print(f"Loader gives {_x.shape[0]} channels (expected {input_features})")
    assert _x.shape[0] == input_features, "Channel mismatch — check use_gmi!"
    _x, _ = training_data[0]
    print(f"Input channels from loader: {_x.shape[0]} (expected {input_features})")
    assert _x.shape[0] == input_features, "Channel mismatch — check use_gmi flag!"
    trainer = L.Trainer(
        max_epochs=unet.n_epochs,
        precision="bf16-mixed",  # <-- Optimized for H100!
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger
    )
    
    print(f"Starting Training on H100 with {input_features} input channels on 'l' subset...")
    trainer.fit(
        model=unet,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader
    )
    
    print("Training complete! Model saved in the 'checkpoints' directory.")