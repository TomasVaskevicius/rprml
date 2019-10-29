import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from .data_utils import get_preloaded_tensor_dataset, split_by_count, data_path


def __get_torchvision_datasets(dataset_class, transforms, n_train, n_valid,
                               device=torch.device('cpu')):
    """ A helper for returining torchvisino datasets, split into training,
    validation and test sets and preloaded as tensor datasets.

    :dataset_class: A class from torchvision.datasets.
    :transforms: Transforms to be performed on the data points.
    :n_train: Number of randomly selected data points for training.
    :n_valid: Number of randomly selected data_points for validation.
    :device: A torch.device specifying where the data should be loaded.
    :returns: The requested torchvision train, valid and test datasets.

    """

    train_dataset_full = dataset_class(download=True, root=data_path,
                                       transform=transforms, train=True)
    test_dataset = dataset_class(download=True, root=data_path,
                                 transform=transforms, train=False)

    train_dataset, valid_dataset = \
        split_by_count(train_dataset_full, n_train, n_valid)

    train_dataset = get_preloaded_tensor_dataset(train_dataset, device)
    valid_dataset = get_preloaded_tensor_dataset(valid_dataset, device)
    test_dataset = get_preloaded_tensor_dataset(test_dataset, device)

    return train_dataset, valid_dataset, test_dataset


def get_mnist_datasets(n_train, n_valid, device=torch.device('cpu')):
    """ A function for getting MNIST training, validation and testing
    datasets.

    :n_train: Number of randomly selected data points for training.
    :n_valid: Number of randomly selected data_points for validation.
    :device: A torch.device specifying where the data should be loaded.
    :returns: A tuple of Dataset objects, one for each split.

    """

    dataset_class = torchvision.datasets.MNIST
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    return __get_torchvision_datasets(dataset_class, data_transform,
                                      n_train, n_valid, device)


def get_cifar10_datasets(n_train, n_valid, device=torch.device('cpu')):
    """ A function for getting CIFAR10 training, validation and testing
    datasets.

    :n_train: Number of randomly selected data points for training.
    :n_valid: Number of randomly selected data_points for validation.
    :seed: A random seed for selecting subsets of data.
    :device: A torch.device specifying where the data should be loaded.
    :returns: A tuple of Dataset objects, one for each split.

    """

    dataset_class = torchvision.datasets.CIFAR10
    data_transform = Compose([ToTensor()])
    return __get_torchvision_datasets(dataset_class, data_transform,
                                      n_train, n_valid, device)
