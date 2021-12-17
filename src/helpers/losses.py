import torch.nn


def get_loss_fn(loss_name: str):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    elif loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError()
