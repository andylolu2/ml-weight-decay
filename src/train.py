from argparse import ArgumentParser

import pytorch_lightning as pl

from src.datasets import BinaryAddDataModule, BinaryMultDataModule
from src.models import ResNetModel, TransformerModel

if __name__ == "__main__":
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument("--global_seed", type=int, default=1234)
    parser.add_argument(
        "--dataset", choices=["binary_add", "binary_mult"], default="binary_add"
    )
    parser.add_argument("--model", choices=["resnet", "transformer"], default="resnet")
    parser = pl.Trainer.add_argparse_args(parser)

    temp_args, _ = parser.parse_known_args()

    if temp_args.dataset == "binary_add":
        dm_class = BinaryAddDataModule
    elif temp_args.dataset == "binary_mult":
        dm_class = BinaryMultDataModule
    else:
        raise ValueError(f"Unknown dataset {temp_args.dataset}")

    if temp_args.model == "resnet":
        model_class = ResNetModel
    elif temp_args.model == "transformer":
        model_class = TransformerModel
    else:
        raise ValueError(f"Unknown model {temp_args.model}")

    parser = dm_class.add_model_specific_args(parser)
    parser = model_class.add_model_specific_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.global_seed)

    # ------------
    # data
    # ------------
    datamodule = dm_class(**dict_args)

    # ------------
    # model
    # ------------
    model = model_class(
        input_dim=datamodule.dimensions()[0],
        output_dim=datamodule.dimensions()[1],
        **dict_args,
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)
