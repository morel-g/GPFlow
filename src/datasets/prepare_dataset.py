import numpy as np

from ..precision import torch_float_precision
from .distributions_toy import inf_train_gen

import torch
from sklearn.model_selection import train_test_split
from src.dataset import Dataset
from src.data_helpers.data_type import (
    toy_data_type,
    latent_data_type,
)


def prepare_toy_data(data_type, n_samples, path="outputs"):
    X = inf_train_gen(data_type, batch_size=n_samples, path=path)
    X = torch.tensor(X, dtype=torch_float_precision)
    x_y = list(train_test_split(X, test_size=0.20, random_state=42))

    return x_y


def prepare_latent_data(path):
    X = np.load(path)
    X = torch.tensor(X, dtype=torch_float_precision)
    x_y = list(train_test_split(X, test_size=0.20, random_state=42))

    return x_y


def prepare_dataset(data, log_dir=None):
    """Load / construct the dataset.

    Args:
        data: A Data object.
        log_dir: Logger directory to store the labels in 2d.

    Raises:
        RuntimeError: Unknown data.data_type.

    Returns:
        A tuple made of a training and a validation dataset.
    """
    data.seed = torch.seed()

    # Prepare data
    if data.data_type in toy_data_type:
        x_train, x_val = prepare_toy_data(
            data.data_type, data.n_samples, path=log_dir
        )
        x_train, x_val = Dataset(x_train), Dataset(x_val)
    elif data.data_type in latent_data_type:
        x_train, x_val = prepare_latent_data(
            data.load_dict["latent_data_path"]
        )
        x_train, x_val = Dataset(x_train), Dataset(x_val)
    else:
        raise RuntimeError("Unknown data type.")

    return (x_train, x_val)
