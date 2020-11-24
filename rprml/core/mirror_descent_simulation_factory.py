import torch

from . import Simulation, SimulationFactory
from .synthetic_data_event_handlers import \
    log_linear_model_complexity_parameters
from ..models.linear import LinearModelFactory
from ..optimizers.mirror_descent_optimizer import MirrorDescentOptimizer


class MirrorDescentSimulationFactory(SimulationFactory):
    """ A simulation factory for running mirror descent simulations on
    d-dimensional linear models. """

    def __init__(self, d, mirror_map, **simulation_kwargs):
        """ :d: Dimensionality of covariates.
            :mirror_map: An object of type MirrorMap.
        """
        simulation_kwargs['loss_function'] = torch.nn.MSELoss()
        if 'model_factory' not in simulation_kwargs:
            simulation_kwargs['model_factory'] = LinearModelFactory(d)
        if 'batch_size' not in simulation_kwargs:
            simulation_kwargs['batch_size'] = simulation_kwargs['n_train']
        super().__init__(simulation_class=Simulation, **simulation_kwargs)
        self.mirror_map = mirror_map

    def get_simulation(self, **kwargs):
        simulation = super().get_simulation(**kwargs)
        # Replace the optimizer.
        simulation.optimizer = MirrorDescentOptimizer(
            self.mirror_map, simulation.model.parameters(),
            lr=simulation.learning_rate,
            momentum=0, dampening=0, weight_decay=0, nesterov=False)
        # Register complexity parameters logging.
        log_linear_model_complexity_parameters(simulation)
        return simulation
