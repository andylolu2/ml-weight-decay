import torch
import torch.nn


def get_optimizer(name: str, *args, **kwargs):
    if name == "adam":
        return torch.optim.AdamW(
            *args,
            lr=kwargs["lr"],
            eps=1e-3,
            betas=(0.9, 0.98),
            weight_decay=kwargs["weight_decay"])
    elif name == "sgd":
        return torch.optim.SGD(
            *args,
            lr=kwargs["lr"],
            weight_decay=kwargs["weight_decay"],
            momentum=kwargs["momentum"]
        )
    else:
        raise ValueError()


def get_scheduler(name: str, *args, **kwargs):
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            *args,
            **kwargs
        )
    else:
        raise ValueError()
