from .simulation import Simulation
from .simulation_factory import SimulationFactory
from .executor import _iteration_level_event

import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from glmnet import ElasticNet

from ..models.linear import LinearModelFactory
from .synthetic_data_event_handlers import \
    log_linear_model_complexity_parameters


@dataclass
class ElasticNetSimulation(Simulation):
    """ A simulation object that computes regularization path for elastic net
    penalties, which include ridge and lasso as special cases.

    This class internally uses the glmnet implementation of
    https://github.com/civisanalytics/python-glmnet
    """

    # Trade-off between ridge and lasso penalties:
    # Should be a value in [0, 1], with alpha=0 reducing to ridge and
    # alpha=1 reducing to lasso.
    alpha: float = 0.0
    # A list of regularization strength parameters lambda, for which models
    # and metrics should be computed.
    lambdas: List[float] = None

    def __post_init__(self):
        super().__post_init__()
        # The following configuration of batch sizes and loss function is
        # needed for the correct tracking of metrics via the
        # rprml.core.Executor class.
        self.batch_size = self.n_train
        self.loss_function = torch.nn.MSELoss()
        self.simulation_name = 'ElasticNetSimulation'

    def run(self, epochs: int):
        print("Starting ElasticNet simulation.")

        # First, set up the data loaders, that will be used for computing
        # metrics via the self.executor object.
        self._reset_data_loaders()

        # The epochs parameter will be ignored but it is needed for
        # compatibility of the rprml.core.Executor class.
        X = self.train_dataset.X.cpu().detach().numpy()
        y = self.train_dataset.y.cpu().detach().numpy()

        # Lambda path has to be supplied in decreasing order.
        lambda_path = np.array(self.lambdas)
        lambda_path = -np.sort(-lambda_path)
        # Append infinity to the front of lambda_path.
        # We do this because the glmnet package modifies the first lambda.
        lambda_path = np.insert(lambda_path, 0, 1e10000, axis=0)

        glmnet = ElasticNet(
            alpha=self.alpha,
            n_splits=0,
            fit_intercept=False,
            standardize=False,
            lambda_path=lambda_path)
        glmnet.fit(X, y.squeeze())

        lambdas, coefs = glmnet.lambda_path_, glmnet.coef_path_
        # Swap axes for parameters learned by lasso path, so that lambda
        # corresponds to the first axis.
        coefs = np.swapaxes(coefs, 0, 1)
        # Remove the first value of lambda (because glmnet modifies its value)
        # and remove the associated fitted vector.
        lambdas = np.array(lambdas).flatten()[1:]
        coefs = np.array(coefs)[1:, :]

        # Save lambdas and alpha to the executor's history.
        self.executor.history['lambdas'] = lambdas
        self.executor.history['alpha'] = self.alpha

        # For each fitted model, compute the metrics registered to
        # self.executor.
        for lambda_id in range(coefs.shape[0]):
            w_lambda = coefs[lambda_id, :]
            w_lambda = torch.tensor(w_lambda, dtype=torch.float32,
                                    device=self.device)
            self.model.set_w(w_lambda)
            # Compute the metrics associated to w_lambda.
            self.trainer.fire_event(_iteration_level_event)


class ElasticNetSimulationFactory(SimulationFactory):
    """ A simulation factory for running elastic net for d-dimensional
    linear models. """

    def __init__(self, d, alpha, lambdas, **simulation_kwargs):
        """ :d: Dimensionality of covariates.
            :alpha: Trade-off between ridge and lasso penalties:
                Should be a value in [0, 1], with alpha=0 reducing to ridge and
                alpha=1 reducing to lasso.
            :lambdas: A list of regularization strength parameters lambda for
                which the regularization path is to be computed.
        """
        simulation_kwargs['alpha'] = alpha
        simulation_kwargs['lambdas'] = lambdas
        simulation_kwargs['loss_function'] = torch.nn.MSELoss()
        simulation_kwargs['model_factory'] = LinearModelFactory(d)
        simulation_kwargs['batch_size'] = simulation_kwargs['n_train']
        super().__init__(simulation_class=ElasticNetSimulation,
                         **simulation_kwargs)

    def get_simulation(self, **kwargs):
        simulation = super().get_simulation(**kwargs)
        log_linear_model_complexity_parameters(simulation)
        return simulation
