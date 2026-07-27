import logging
import multiprocessing
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler

from lightning.pytorch import LightningDataModule

# placeholder for testing
from satvision_pix4d.datasets.abi_temporal_dataset import ABITemporalDataset


class ABITemporalDataModule(LightningDataModule):
    def __init__(self, config) -> None:

        super().__init__()

        self.config = config
        self.batch_size = config.DATA.BATCH_SIZE
        self.shuffle = config.DATA.SHUFFLE
        self.num_workers = config.DATA.NUM_WORKERS
        self.persistent_workers = config.DATA.PERSISTENT_WORKERS
        self.img_size = config.DATA.IMG_SIZE
        self.in_chans = config.MODEL.MAE_VIT.IN_CHANS
        self.train_data_paths = config.DATA.TRAIN_DATA_PATHS
        self.val_data_paths = config.DATA.VAL_DATA_PATHS
        self.train_data_length = config.DATA.LENGTH
        self.pin_memory = config.DATA.PIN_MEMORY
        self.drop_last = config.DATA.DROP_LAST

        self.transform = None

        #transforms.Compose(
        #    [
        #        transforms.ToTensor(),
        #        #transforms.RandomCrop(self.img_size),
        #    ]
        #)

        self.trainset = None
        self.validset = None

    def setup(self, stage=None):
        # This is called after Lightning sets up distributed
        logging.info("> Init datasets")
        self.trainset = ABITemporalDataset(
            self.train_data_paths,
            transform=self.transform,
            img_size=self.img_size,
            in_chans=self.in_chans
        )
        self.validset = ABITemporalDataset(
            self.val_data_paths,
            transform=self.transform,
            img_size=self.img_size,
            in_chans=self.in_chans
        )
        logging.info("Done init datasets")
        return

    def train_dataloader(
        self,
    ):
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = 4

        return DataLoader(
            self.trainset,
            **loader_kwargs
        )

    def val_dataloader(
        self,
    ):
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = 4

        return DataLoader(
            self.validset,
            **loader_kwargs
        )

    def plot(*args, **kwargs):
        return None


if __name__ == "__main__":

    toa_module = ABITemporalDataModule(data_path=[])
