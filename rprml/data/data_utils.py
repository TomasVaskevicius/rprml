import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset, random_split


# Directory for storing downloaded data.
data_path = './downloaded_data'


class TensorDataset(Dataset):
    """ A dataset representation simply containing two tensors X and y. """

    def __init__(self, X, y, device: torch.device):
        self.X = X.to(device)
        self.y = y.to(device)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.y)


def get_preloaded_tensor_dataset(dataset: Dataset, device: torch.device):
    """ Preloads a given dataset object as a TensorDataset. """
    X, y = dataset[0]

    # Try to convert X and y to torch tensors.
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)

    # Create empty tensors for pre-loading the dataset.
    n = len(dataset)
    X = torch.zeros([n] + list(X.shape), device=device, dtype=X.dtype)
    y = torch.zeros([n] + list(y.shape), device=device, dtype=y.dtype)

    # Fill in the tensors.
    for idx, (_X, _y) in enumerate(dataset):
        X[idx], y[idx] = _X, _y

    return TensorDataset(X, y, device)


def split_by_count(dataset: Dataset, n_train, n_valid):
    """ A function for randomly splitting a torch Dataset object into
    non-overlapping training and validation datasets of specified sizes.

    :dataset: A torch Dataset object.
    :n_train: Size of training dataset.
    :n_valid: Size of validation dataset.
    :returns: The given dataset split randomly into non-overlapping datasets
            for training and validation of the given sizes.

    """
    n_data_points = len(dataset)
    assert n_train + n_valid <= n_data_points

    rand_idx = np.random.permutation(n_data_points)
    subset_dataset = Subset(dataset, rand_idx[:n_train + n_valid])
    return random_split(subset_dataset, [n_train, n_valid])
