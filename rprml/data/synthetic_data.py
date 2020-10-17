import numpy as np
import torch

from .data_utils import TensorDataset


default_dtype = torch.float32


def __generate_synthetic_regression_dataset(X, noise_std, w_star, device):
    """
    :X: A covariates matrix as a torch tensor.
    :noise_std: Standard deviation of Gaussian additive noise.
    :w_star: The true parameter vector, used to generate the labels.
    :device: A torch.device on which the dataset will be stored.
    """

    if X.shape[1] != len(w_star):
        raise ValueError('Dimensionality of the data and true parameter'
                         ' vector does not match.')

    n = X.shape[0]
    d = len(w_star)
    # Sample Gaussian noise.
    xi = torch.randn(n, 1, device=device) * noise_std

    # Try converting w_star to torch tensor.
    if not isinstance(w_star, torch.Tensor):
        w_star = torch.tensor(w_star, device=device)
    w_star = w_star.to(device, dtype=default_dtype).view(d, 1)

    # Generate labels
    X = X.to(device, dtype=default_dtype)
    y = torch.mm(X, w_star) + xi

    dataset = TensorDataset(X, y, device)
    # Save the noise vector in the dataset object.
    setattr(dataset, 'xi', xi)
    setattr(dataset, 'w_star', w_star)
    return dataset


def __generate_multivariate_gaussian_covariates(n, d, covariance_matrix,
                                                device):
    if covariance_matrix is None:
        # Assume identity covariance matrix.
        return torch.randn(n, d, device=device)

    mean = np.zeros(d)
    X = np.random.multivariate_normal(mean=mean, cov=covariance_matrix, size=n)

    return torch.tensor(X, dtype=default_dtype, device=device)


def get_gaussian_datasets(n_train, n_valid, device=torch.device('cpu'),
                          w_star=None, noise_std: float = 0.0,
                          covariance_matrix=None):
    """ A function for getting synthetic training and validation datasets
    with Gaussian covariates.

    :n_train: Number of randomly selected data points for training.
    :n_valid: Number of randomly selected data_points for validation.
    :device: A torch.device specifying where the data should be loaded.
    :w_star: The true parameter vector, used to generate the labels.
    :noise_std: Standard deviation of Gaussian additive noise.
    :covariance_matrix: The covariance matrix for i.i.d. Gaussian data points.
        If None, identity is assumed.
    :returns: A tuple of Dataset objects, one for training and one
            for validation.
    """
    d = len(w_star)
    datasets = []
    for n in [n_train, n_valid]:
        X = __generate_multivariate_gaussian_covariates(
            n, d, covariance_matrix, device)
        datasets.append(__generate_synthetic_regression_dataset(
            X, noise_std, w_star, device))
        setattr(datasets[-1], 'covariance_matrix', covariance_matrix)

    return tuple(datasets)


dataset_factory_methods.update({
    'gaussian': get_gaussian_datasets,
})
