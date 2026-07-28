import os
import torch
import time
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Try to import TensorBoardLogger - may not be available on all cluster environments
try:
    from pytorch_lightning.loggers import TensorBoardLogger
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("WARNING: TensorBoardLogger not available. Using CSVLogger only.")

from abidatamodule import AbiDataModule
from models import LightningModel


class SlurmProgressCallback(L.Callback):
    """
    A text-based progress callback that prints plain-text lines to stdout.
    
    Lightning's default TQDMProgressBar uses carriage returns (\r) which 
    get swallowed in Slurm log files, making it look like nothing is happening.
    This callback prints clean, newline-separated progress updates that 
    show up properly in .log files.
    """
    def __init__(self, log_every_n_batches=50):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        try:
            total_batches = len(trainer.train_dataloader)
        except (TypeError, AttributeError):
            total_batches = "?"
        print(
            f"\n{'='*60}\n"
            f"  Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"| {total_batches} batches\n"
            f"{'='*60}",
            flush=True
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.log_every_n_batches == 0:
            # Extract loss from outputs (Lightning returns dict from training_step)
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_val = outputs["loss"].item()
            elif isinstance(outputs, torch.Tensor):
                loss_val = outputs.item()
            else:
                loss_val = None

            elapsed = time.time() - self.epoch_start_time
            try:
                total_batches = len(trainer.train_dataloader)
            except (TypeError, AttributeError):
                total_batches = "?"

            loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
            print(
                f"  [Batch {batch_idx + 1}/{total_batches}] "
                f"train_loss={loss_str} | elapsed={elapsed:.1f}s",
                flush=True
            )

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", None)
        if isinstance(train_loss, torch.Tensor):
            train_loss = f"{train_loss.item():.4f}"
        else:
            train_loss = "N/A"
        print(
            f"  >>> Epoch {trainer.current_epoch + 1} finished in {elapsed:.1f}s "
            f"| train_loss={train_loss}",
            flush=True
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip the sanity check validation that runs before training
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss", None)
        val_iou = metrics.get("val_iou", None)

        val_loss_str = f"{val_loss.item():.4f}" if isinstance(val_loss, torch.Tensor) else "N/A"
        val_iou_str = f"{val_iou.item():.4f}" if isinstance(val_iou, torch.Tensor) else "N/A"
        print(
            f"  >>> Validation: val_loss={val_loss_str} | val_iou={val_iou_str}",
            flush=True
        )


"""
CONFIGURABLE PARAMETERS
"""
# Training config is read from environment variables set by the submit script.
# This allows the same Python file to work for both V100 (batch=1, 4 GPUs) and H100 (batch=4, 1 GPU).
# Defaults are for single-GPU V100 if no env vars are set.
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 1))
NUM_DEVICES = int(os.environ.get("TRAIN_NUM_DEVICES", 1))
DDP_STRATEGY = os.environ.get("TRAIN_STRATEGY", "auto")
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT", None)
LEARNING_RATE = 1e-4
EPOCHS = 100
SAVE_EVERY_N_EPOCHS = 5
DATALOADER_WORKERS = 8

# How often (in batches) to print a progress line to the Slurm log
LOG_EVERY_N_BATCHES = 100

# Update this path to where your new 2026 chips are located
datapath = '/explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/chips/allChips'
TRAINING_SPLIT = (0, 0.8)
VAL_SPLIT = (0.8, 0.9)
TEST_SPLIT = (0.9, 1.0)

checkpointpath = './checkpoints/'

"""
MAIN EXECUTION
"""
if __name__ == '__main__':
    # Initialize the new 3D model
    model = LightningModel(lr=LEARNING_RATE)

    # Initialize the updated Data Module
    datamodule = AbiDataModule(
        data_path=datapath,
        train_split=TRAINING_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + "unet3d_baseline", save_top_k=-1, every_n_epochs=SAVE_EVERY_N_EPOCHS)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpointpath + "unet3d_baseline", save_top_k=1, monitor="val_loss", mode="min", filename="best-{epoch:02d}-{val_loss:.2f}")

    # Slurm-friendly progress callback (prints plain text lines instead of tqdm bars)
    progress_callback = SlurmProgressCallback(log_every_n_batches=LOG_EVERY_N_BATCHES)

    # --- Loggers ---
    logger_csv = CSVLogger(checkpointpath, name="unet3d_baseline")
    loggers = [logger_csv]

    if TENSORBOARD_AVAILABLE:
        logger_tb = TensorBoardLogger(checkpointpath, name="unet3d_baseline")
        loggers.append(logger_tb)
        print("TensorBoard logging enabled. View with: tensorboard --logdir ./checkpoints/unet3d_baseline")
    else:
        print("TensorBoard not available - continuing with CSVLogger only.")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=NUM_DEVICES,
        strategy=DDP_STRATEGY,
        callbacks=[checkpoint_callback, best_checkpoint_callback, progress_callback],
        logger=loggers,
        default_root_dir=checkpointpath,
        enable_progress_bar=False,  # Disable tqdm - it doesn't work in Slurm log files
    )

    print(f"Training config: BATCH_SIZE={BATCH_SIZE}, DEVICES={NUM_DEVICES}, STRATEGY={DDP_STRATEGY}")
    if RESUME_CHECKPOINT:
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=RESUME_CHECKPOINT)
    else:
        trainer.fit(model=model, datamodule=datamodule)

