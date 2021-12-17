from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn
import pytorch_lightning as pl
from transformers import BertModel, BertConfig

from src.helpers import get_loss_fn, get_optimizer


class TransformerModel(pl.LightningModule):
    def __init__(self,
                 input_dim: Tuple[int, int],
                 output_dim: Tuple[int, int],
                 hidden_dim: int,
                 num_layers: int,
                 num_heads: int,
                 loss: str,
                 optimizer: str,
                 lr: float,
                 l2_norm: float,
                 momentum: float,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters("input_dim",
                                  "output_dim",
                                  "hidden_dim",
                                  "num_layers",
                                  "num_heads",
                                  "loss",
                                  "optimizer",
                                  "lr",
                                  "l2_norm",
                                  "momentum")
        self.optimizer = optimizer
        self.lr = lr
        self.l2_norm = l2_norm
        self.momentum = momentum
        self.config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=4 * hidden_dim,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            hidden_act="relu",
        )
        self.ff1 = torch.nn.Linear(input_dim[1], hidden_dim)
        self.encoder = BertModel(self.config)
        self.ff2 = torch.nn.Linear(hidden_dim, output_dim[1])
        self.act = torch.nn.Sigmoid()
        self.loss_fn = get_loss_fn(loss)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("TransformerModel")
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--loss", type=str,
                            choices=["mse", "cross_entropy"], default="mse")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--l2_norm", type=float, default=0.0)
        parser.add_argument("--optimizer", type=str,
                            choices=["adam", "sgd"], default="adam")
        parser.add_argument("--momentum", type=float, default=0.95)
        return parent_parser

    def forward(self, x: torch.Tensor):
        x = self.ff1(x)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        x = self.ff2(x)
        x = self.act(x)
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
        optimizer = get_optimizer(self.optimizer,
                                  self.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.l2_norm,
                                  momentum=self.momentum
                                  )
        return optimizer
