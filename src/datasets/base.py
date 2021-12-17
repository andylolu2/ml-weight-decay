from argparse import ArgumentParser

from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, size: int, seed: int, **kwargs):
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        raise NotImplementedError()

    @staticmethod
    def dimensions():
        raise NotImplementedError()


class BaseDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_size: int,
                 val_size: int,
                 seed: int,
                 **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.save_hyperparameters(
            "batch_size", "train_size", "val_size", "seed")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("BaseDataModule")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--train_size", type=int, default=1_000_000)
        parser.add_argument("--val_size", type=int, default=10_000)
        parser.add_argument("--seed", type=int, default=0)
        return parent_parser

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    @staticmethod
    def dimensions():
        raise NotImplementedError()
