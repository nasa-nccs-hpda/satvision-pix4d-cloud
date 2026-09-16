"""Train temporal foundation models from NumPy/Zarr or synthetic samples."""
import argparse
import json
import logging
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from satvision_pix4d.configs.config import _C, _update_config_from_file
from satvision_pix4d.utils import get_strategy, get_distributed_train_batches
from satvision_pix4d.pipelines import PIPELINES
from satvision_pix4d.datamodules import DATAMODULES
from satvision_pix4d.training_monitor import EpochPerformanceMonitor


def build_callbacks(config, output_dir):
    return [
        ModelCheckpoint(dirpath=output_dir, monitor="val_loss", mode="min", save_top_k=3,
                        filename="best-{epoch:03d}-{val_loss:.4f}", auto_insert_metric_name=False),
        ModelCheckpoint(dirpath=output_dir, every_n_epochs=config.SAVE_FREQ,
                        save_on_train_epoch_end=True, save_top_k=-1, save_last=True,
                        filename="epoch-{epoch:03d}", auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="step"), EpochPerformanceMonitor(),
    ]


def build_loggers(config, output_dir):
    loggers = []
    if config.TENSORBOARD.ENABLED:
        loggers.append(TensorBoardLogger(save_dir=config.TENSORBOARD.WRITER_DIR or output_dir,
                                        name="tensorboard", default_hp_metric=False))
    if config.MLFLOW.ENABLED:
        loggers.append(MLFlowLogger(experiment_name=config.TAG, tracking_uri=config.MLFLOW.URI,
                                   tags={"Model": config.MODEL.NAME, "Pipeline": config.PIPELINE}))
    if not loggers:
        raise ValueError("Enable TENSORBOARD.ENABLED or MLFLOW.ENABLED for training tracking")
    return loggers


def check_data(config, count=4):
    """Read real samples without constructing the large model."""
    module = DATAMODULES[config.DATAMODULE](config)
    module.setup("fit")
    report = {}
    for split, dataset in (("train", module.trainset), ("validation", module.validset)):
        checked = []
        for i in range(min(count, len(dataset))):
            raw, ts = dataset[i]
            checked.append(dict(shape=list(raw.shape), timestamps_shape=list(ts.shape),
                                min=float(raw.min()), max=float(raw.max())))
        report[split] = dict(samples=len(dataset), checked=checked)
    print(json.dumps(report, indent=2))
    return report


def main(config, output_dir):
    seed_everything(config.SEED, workers=True)
    os.makedirs(output_dir, exist_ok=True)
    # Delay weight allocation until Lightning enters its ZeRO-3 sharded context.
    pipeline = PIPELINES[config.PIPELINE](config, defer_model=True)
    loggers = build_loggers(config, output_dir)
    trainer = Trainer(
        accelerator=config.TRAIN.ACCELERATOR,
        devices=torch.cuda.device_count() if config.TRAIN.ACCELERATOR == "gpu" else 1,
        strategy=get_strategy(config), precision=config.PRECISION,
        max_epochs=config.TRAIN.EPOCHS, gradient_clip_val=config.TRAIN.CLIP_GRAD,
        accumulate_grad_batches=config.TRAIN.ACCUMULATION_STEPS,
        check_val_every_n_epoch=config.VALIDATION_FREQ, fast_dev_run=config.FAST_DEV_RUN,
        log_every_n_steps=config.PRINT_FREQ, default_root_dir=output_dir,
        callbacks=build_callbacks(config, output_dir), logger=loggers,
    )
    if trainer.is_global_zero:
        Path(output_dir, f"{config.TAG}.config.yaml").write_text(config.dump())
    if config.TRAIN.LIMIT_TRAIN_BATCHES:
        trainer.limit_train_batches = get_distributed_train_batches(config, trainer)
    datamodule = DATAMODULES[config.DATAMODULE](config) if config.DATA.DATAMODULE else None
    trainer.fit(model=pipeline, datamodule=datamodule, ckpt_path=config.MODEL.RESUME or None)
    return trainer


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config-path', required=True)
    parser.add_argument('--train-data', nargs='+', help='NPY/NPZ/Zarr paths or directories')
    parser.add_argument('--val-data', nargs='+', help='Separate held-out paths or directories')
    parser.add_argument('--resume', help='Lightning checkpoint file or DeepSpeed checkpoint directory')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--check-data-only', action='store_true')
    parser.add_argument('--check-samples', type=int, default=4)
    args = parser.parse_args(argv)
    if args.check_samples < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('Epoch and sample counts must be positive')
    config = _C.clone()
    _update_config_from_file(config, args.config_path)
    config.defrost()
    if args.train_data is not None:
        config.DATA.TRAIN_DATA_PATHS = args.train_data
    if args.val_data is not None:
        config.DATA.VAL_DATA_PATHS = args.val_data
    if args.resume is not None:
        config.MODEL.RESUME = args.resume
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    config.freeze()
    if args.check_data_only:
        return check_data(config, args.check_samples)
    output_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    return main(config, output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cli()
