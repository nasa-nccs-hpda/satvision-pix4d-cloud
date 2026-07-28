import os
import torch
import pytorch_lightning as L
from models import LightningModel
from abidatamodule import AbiDataModule

# Configuration variables
datapath = '/explore/nobackup/projects/pix4dcloud/aliewehr/chipTests/chips/allChips'
TRAINING_SPLIT = (0, 0.8)
VAL_SPLIT = (0.8, 0.9)
TEST_SPLIT = (0.9, 1.0)
BATCH_SIZE = 1
DATALOADER_WORKERS = 8

if __name__ == '__main__':
    # We use the best checkpoint based on validation loss (Epoch 13)
    checkpoint_path = "./checkpoints/unet3d_baseline/best-epoch=13-val_loss=0.37.ckpt"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize datamodule with the exact same splits used in training
    datamodule = AbiDataModule(
        data_path=datapath,
        train_split=TRAINING_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=DATALOADER_WORKERS
    )
    
    # Load the model weights from the checkpoint
    model = LightningModel.load_from_checkpoint(checkpoint_path)
    
    # Initialize trainer for testing (no loggers or progress bars needed for a quick test)
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False
    )
    
    print("Starting evaluation on test set...")
    trainer.test(model=model, datamodule=datamodule)
    print("Evaluation finished!")
