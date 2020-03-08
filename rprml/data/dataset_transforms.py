import numpy as np
import torch
from typing import Callable
from torch.utils.data.dataloader import DataLoader

from .data_utils import TensorDataset
from ..core.executor import Executor


class TransformedTensorDataset(TensorDataset):
    """ Implements a TensorDataset, where a transform function is applied once
    for every datapoint. This class provides functionality for returning
    clean and noisy splits of the data. """

    def __init__(self, dataset: TensorDataset, transform: Callable,
                 **transform_kwargs):
        """ The transform function must take X, y as input and return
        TensorDataset, transformed_idx as output,
        where transformed_idx is an iterable of indexes of affected data
        points. """
        self.X, self.y, self.transformed_idx = transform(dataset,
                                                         **transform_kwargs)
        self.clean_idx = np.arange(len(self.y))
        self.clean_idx = np.setdiff1d(self.clean_idx, self.transformed_idx)

    def get_clean_dataset(self):
        return TensorDataset(self.X[self.clean_idx],
                             self.y[self.clean_idx],
                             self.X.device)

    def get_noisy_dataset(self):
        return TensorDataset(self.X[self.transformed_idx],
                             self.y[self.transformed_idx],
                             self.X.device)


def randomize_labels_transform(dataset: TensorDataset, p):
    """ A function for randomly resampling labels for a given fraction
    of the dataset.

    :dataset: A TensorDataset object.
    :p: A fraction of the dataset for which labels should be resampled.
    :returns: X, y, transformed_idx.
    """
    X, y = dataset.X, dataset.y
    n = len(y)
    n_resample = int(n * p)  # Number of data points to resample.
    transform_idx = np.random.permutation(n)[:n_resample]
    n_classes = len(torch.unique(y))
    new_labels = np.random.randint(low=0, high=n_classes,
                                   size=n_resample)
    y[transform_idx] = torch.tensor(new_labels, device=y.device, dtype=y.dtype)

    return X, y, transform_idx


def register_metric_tracking_on_transformed_dataset(
        executor: Executor, dataset: TransformedTensorDataset,
        dataset_split: str, metric: str, handler_name: str):
    """ Registers the desired handler to the given executor.

        :executor: A core.Executor object to which the handler will be
            attached.
        :dataset: A dataset of type TransformedTensorDataset which will be
            used by the executors evaluator to compute metrics.
        :dataset_split: Either 'clean' or 'noisy'.
        :metric: Either 'loss' or 'accuracy'.
        :handler_name: The name of the handler, which will be visible in the
            history computed by the executor.
    """

    if metric == 'accuracy':
        # Need to register accuracy metric tracking for the evaluator to be
        # used by the handler defined below.
        executor.register_accuracy_handlers()
    elif metric != 'loss':
        raise ValueError(
            'Parameter metric has to be \"accuracy\" or \"loss\".')

    clean_dataset = dataset.get_clean_dataset()
    noisy_dataset = dataset.get_noisy_dataset()

    if len(clean_dataset) * len(noisy_dataset) == 0:
        # Nothing to do, since all data is noisy or clean.
        return

    if dataset_split == 'clean':
        data_loader = DataLoader(clean_dataset, batch_size=len(clean_dataset))
    elif dataset_split == 'noisy':
        data_loader = DataLoader(noisy_dataset, batch_size=len(noisy_dataset))
    else:
        raise ValueError(
            'Parameter dataset_split has to be \"clean\" or \"noisy\".')

    # Set up the handler tracking the desired metric.
    def handler(engine, _executor):
        evaluator = _executor.simulation.evaluator
        evaluator.run(data_loader)
        return evaluator.state.metrics[metric]

    # Finally, register the handler to the executor.
    executor.register_not_printable_metric(handler, handler_name)
