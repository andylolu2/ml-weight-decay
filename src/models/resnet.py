from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn

from src.helpers import get_loss_fn, get_optimizer


class ResNetBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ff1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim, eps=1e-2)
        self.ff2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim, eps=1e-2)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        out = self.ff1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.ff2(out)
        out = self.bn2(out)
        out += x
        out = self.act(out)
        return out


class ResNetModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        output_dim: Tuple[int, int],
        hidden_dim: int,
        num_layers: int,
        loss: str,
        optimizer: str,
        lr: float,
        l2_norm: float,
        momentum: float,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            "input_dim",
            "output_dim",
            "hidden_dim",
            "num_layers",
            "loss",
            "optimizer",
            "lr",
            "l2_norm",
            "momentum",
        )
        self.optimizer = optimizer
        self.lr = lr
        self.l2_norm = l2_norm
        self.momentum = momentum
        self.flatten = torch.nn.Flatten()
        self.ff1 = torch.nn.Linear(input_dim[0] * input_dim[1], hidden_dim)
        self.encoder = torch.nn.ModuleList(
            [ResNetBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.ff2 = torch.nn.Linear(hidden_dim, output_dim[0] * output_dim[1])
        self.sigmoid = torch.nn.Sigmoid()
        self.unflatten = torch.nn.Unflatten(1, output_dim)
        self.loss_fn = get_loss_fn(loss)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ResNetModel")
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument(
            "--loss", type=str, choices=["mse", "cross_entropy"], default="mse"
        )
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--l2_norm", type=float, default=0.0)
        parser.add_argument(
            "--optimizer", type=str, choices=["adam", "sgd"], default="adam"
        )
        parser.add_argument("--momentum", type=float, default=0.95)
        return parent_parser

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.ff1(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.ff2(x)
        x = self.sigmoid(x)
        x = self.unflatten(x)
        return x

    def training_step(self, batch, batch_idx):
        # compute
        x, y = batch
        x, f_y = x.to(torch.float32), y.to(torch.float32)
        y_hat = self(x)
        loss = self.loss_fn(f_y, y_hat)

        y_pred = y_hat > 0.5
        accuracy = torch.sum(torch.all(y == y_pred, dim=1)) / y.size(0)

        # log
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        l2_reg = None
        for param in self.parameters():
            if l2_reg is None:
                l2_reg = torch.norm(param, 2)
            else:
                l2_reg += torch.norm(param, 2)
        self.log("train/l2_norm", l2_reg)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # compute
        x, y = batch
        x, f_y = x.to(torch.float32), y.to(torch.float32)
        y_hat = self(x)
        val_loss = self.loss_fn(f_y, y_hat)

        y_pred = y_hat > 0.5
        accuracy = torch.sum(torch.all(y == y_pred, dim=1)) / y.size(0)

        # log
        self.log("val/loss", val_loss)
        self.log("val/accuracy", accuracy)

        return val_loss

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.optimizer,
            self.parameters(),
            lr=self.lr,
            weight_decay=self.l2_norm,
            momentum=self.momentum,
        )
        return optimizer
