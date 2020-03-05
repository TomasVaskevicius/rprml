import copy
import torch

from .simulation import Simulation


# Default simulation parameters.
_default_params_dict = {
    'seed': 0,
    'device': torch.device('cpu'),
    'data': 'mnist',
    'loss_function': torch.nn.CrossEntropyLoss(),
    'batch_size': 100,
    '_learning_rate': 0.001,
    'n_train': 1000,
    'n_valid': 1000
}


class _SimulationFactoryBase(object):

    def __call__(self, seed: int, device: torch.device):
        """ Makes the simulation factory callable. It can be used by objects of
        type core.Experiment to perform runs of the same simulation with
        different seeds and on different devices. """
        raise NotImplementedError


class SimulationFactory(_SimulationFactoryBase):
    """ A factory class for core.Simulation objects. Overriding this class
    allows for designing custom experiments and prototyping/executing using
    core.Experiment class. """

    def __init__(self, simulation_class=Simulation, **simulation_kwargs):
        """
        :simulation_class: The class of simulation object to be constructed
            using the given model_factory and **simulation_kwargs.
        :simulation_kwargs: Keyword arguments to be used for setting up the
            object of type simulation_class.
        """
        self.simulation_class = simulation_class
        self.simulation_kwargs = simulation_kwargs

    def get_simulation(self, **kwargs):
        """ Returns a core.Simulation object. Override for custom behavior.
        """
        params = copy.deepcopy(_default_params_dict)
        for (key, value) in kwargs.items():
            if key == 'learning_rate':
                key = '_learning_rate'
            params[key] = value
        simulation = self.simulation_class(**params)
        return simulation

    def __call__(self, seed: int, device: torch.device):
        params = copy.deepcopy(self.simulation_kwargs)
        params['seed'] = seed
        params['device'] = device
        simulation = self.get_simulation(**params)
        return simulation
