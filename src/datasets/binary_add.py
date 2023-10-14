from argparse import ArgumentParser
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .base import BaseDataModule, BaseDataset


class BinaryAddDataset(BaseDataset):
    binary_size = 32

    def __init__(self, digit_range: List[int], noise: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.digit_range = np.array(digit_range)
        self.noise = noise

        p = 10**self.digit_range
        p = p / np.sum(p)
        digit_counts = self.rng.multinomial(n=self.size, pvals=p)
        self.xs = []
        for num_digit, count in zip(digit_range, digit_counts):
            x1 = self.rng.choice(
                np.arange(10 ** (num_digit - 1), 10**num_digit),
                size=count,
                replace=False,
            )
            x2 = self.rng.choice(
                np.arange(10 ** (num_digit - 1), 10**num_digit),
                size=count,
                replace=False,
            )
            self.xs.append([x1, x2])

        self.xs = np.concatenate(self.xs, axis=-1).T
        self.ys = self.xs[:, 0] + self.xs[:, 1]

        # Turn into binary representation
        nums = np.array([self.xs[:, 0], self.xs[:, 1], self.ys])
        bits = 1 << np.arange(self.binary_size)
        nums = (nums[:, :, None] & bits) > 0
        self.xs, self.ys = torch.from_numpy(nums[0:2].T), torch.from_numpy(nums[2:3].T)

        # Add noise to y
        noisy = torch.rand(self.ys.shape) < self.noise
        self.ys[noisy] = torch.randint_like(self.ys[noisy], 0, 2)

    def __getitem__(self, idx):
        return self.xs[:, idx], self.ys[:, idx]

    @classmethod
    def dimensions(cls):
        return (cls.binary_size, 2), (cls.binary_size, 1)


class BinaryAddDataModule(BaseDataModule):
    def __init__(
        self,
        train_range: List[int],
        val_range: List[int],
        noise: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.train_range = train_range
        self.val_range = val_range
        self.noise = noise
        self.save_hyperparameters("train_range", "val_range")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        BaseDataModule.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("BinaryAddDataModule")
        parser.add_argument("--train_range", nargs="+", type=int, default=[2, 4])
        parser.add_argument("--val_range", nargs="+", type=int, default=[3])
        parser.add_argument("--noise", type=float, default=0)
        return parent_parser

    def setup(self, stage: Optional[str] = None) -> None:
        dset = BinaryAddDataset(
            size=self.train_size + self.val_size,
            digit_range=self.train_range,
            seed=self.seed,
            noise=self.noise,
        )
        self.train_dset, self.in_dom_val_dset = random_split(
            dset, lengths=[self.train_size, self.val_size]
        )
        self.out_dom_val_dset = BinaryAddDataset(
            size=self.val_size,
            digit_range=self.val_range,
            seed=self.seed,
            noise=0,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        return [
            DataLoader(
                self.in_dom_val_dset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            ),
            DataLoader(
                self.out_dom_val_dset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            ),
        ]

    @staticmethod
    def dimensions():
        return BinaryAddDataset.dimensions()


if __name__ == "__main__":
    dset = BinaryAddDataset(digit_range=[1, 2], size=10, seed=0, noise=0.1)
    for item in dset:
        print(item)
