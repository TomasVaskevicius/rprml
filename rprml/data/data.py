from ._torchvision_data import get_mnist_datasets, get_cifar10_datasets


# Defines available datasets and factory method names for creating data
# loaders.
dataset_factory_methods = {
    'mnist': get_mnist_datasets,
    'cifar10': get_cifar10_datasets
}
