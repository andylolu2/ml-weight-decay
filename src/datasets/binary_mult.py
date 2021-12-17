from typing import Optional, List
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .base import BaseDataset, BaseDataModule


class BinaryMultDataset(BaseDataset):
    binary_size = 32

    def __init__(self, digit_range: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.digit_range = np.array(digit_range)

        p = 10 ** self.digit_range
        p = p / np.sum(p)
        num_digits = self.rng.choice(self.digit_range,
                                     size=(self.size),
                                     replace=True,
                                     p=p)
        num_digits = np.tile(num_digits, (2, 1)).T
        self.xs = self.rng.integers(low=10 ** (num_digits - 1),
                                    high=10 ** num_digits,
                                    dtype=np.uint32)

    def __getitem__(self, idx):
        x1, x2 = self.xs[idx]
        y = x1 * x2

        nums = np.array([x1, x2, y])
        bits = 1 << np.arange(self.binary_size)
        nums = (nums[:, None] & bits) > 0
        x, y = torch.from_numpy(nums[0:2].T), torch.from_numpy(nums[2:3].T)
        return x, y

    @classmethod
    def dimensions(cls):
        return (cls.binary_size, 2), (cls.binary_size, 1)


class BinaryMultDataModule(BaseDataModule):
    def __init__(self,
                 train_range: List[int],
                 val_range: List[int],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_range = train_range
        self.val_range = val_range
        self.save_hyperparameters("train_range", "val_range")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        BaseDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("BinaryMultDataModule")
        parser.add_argument("--train_range", nargs="+",
                            type=int, default=[2, 4])
        parser.add_argument("--val_range", nargs="+", type=int, default=[3])
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        dset = BinaryMultDataset(size=self.train_size + self.val_size,
                                 digit_range=self.train_range,
                                 seed=self.seed)
        self.train_dset, self.in_dom_val_dset = random_split(
            dset,
            lengths=[self.train_size, self.val_size]
        )
        self.out_dom_val_dset = BinaryMultDataset(size=self.val_size,
                                                  digit_range=self.val_range,
                                                  seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return [
            DataLoader(self.in_dom_val_dset,
                       batch_size=self.batch_size,
                       drop_last=True,
                       shuffle=True),
            DataLoader(self.out_dom_val_dset,
                       batch_size=self.batch_size,
                       drop_last=True,
                       shuffle=True)
        ]

    @staticmethod
    def dimensions():
        return BinaryMultDataset.dimensions()


if __name__ == "__main__":
    parser = ArgumentParser()

    # add model specific args
    parser = BinaryMultDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    dict_args = vars(args)
    print(dict_args)
