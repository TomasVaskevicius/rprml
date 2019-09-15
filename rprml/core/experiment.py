from dataclasses import is_dataclass
from itertools import cycle
import numbers
from typing import List, Union
import torch

from .simulation_factory import SimulationFactory
from ..utils.hashable_dict import HashableDict


class Experiment(object):
    """ A class designed for running custom experiments. Provides API for
        1) prototyping -- executing the experiment using a single seed.
        2) full run -- repeating the same experiment with multiple seeds and
            collecting the results.
    """

    def __init__(self, name: str,
                 simulation_factories: List[SimulationFactory]):
        """
        :name: Name of the experiment.
        :simulation_factories: A list of core.SimulationFactory objects.
            This list defines the experiment as a collection of simulations
            to be performed.
        """
        self.name = name
        self.simulation_factories = simulation_factories
        self.log_frequency = -1
        self.print_frequency = 1

    def construct_simulation_identifier(self, simulation):
        """ Default implementation of simulation identifier. Simulation
        identifiers will be used to retrieve the results. This class is
        meant to be used by simulation objects of type core.Simulation."""
        # Construct relevant keys, using field names of dataclass and its
        # subclasses recursively.
        identifier = {}

        def append_identifiers(cls):
            if is_dataclass(cls):
                identifier.update(cls.__annotations__)
                for base_cls in cls.__bases__:
                    append_identifiers(base_cls)

        append_identifiers(simulation.__class__)

        # Update the dictionary keys to the ones taken by the current
        # simulation object.
        for key in identifier.keys():
            identifier[key] = simulation.__getattribute__(key)
            if not isinstance(identifier[key], numbers.Number):
                identifier[key] = str(identifier[key])
        if '_learning_rate' in identifier.keys():
            identifier['learning_rate'] = identifier['_learning_rate']
            del identifier['_learning_rate']

        return HashableDict(identifier)

    def handle_simulation_output(self, simulation):
        """ Implements a default simulation output handler. Override for
        custom behavior. """
        return simulation.executor.history

    def prototype_run(self, seed: int, device: torch.device,
                      epochs_per_simulation: Union[int, List[int]]):
        """ Runs the experiment -- performs all the simulations in the given
        list simulation_factories. Performs just a single run for each
        simulation.

        Returns a dictionary from simulation identifiers to simulation outputs.
        """
        if isinstance(epochs_per_simulation, int):
            epochs_per_simulation = [epochs_per_simulation]

        results = {}
        for simulation_factory, epochs in zip(self.simulation_factories,
                                              cycle(epochs_per_simulation)):
            simulation = simulation_factory(seed, device)
            simulation.executor.log_frequency = self.log_frequency
            simulation.executor.print_frequency = self.print_frequency
            simulation.run(epochs)
            simulation_identifier = self.construct_simulation_identifier(
                simulation)
            result = self.handle_simulation_output(simulation)
            results[simulation_identifier] = result

        return results

    def full_run(self, n_runs, devices_list):
        """ Runs the experiment with multiple seeds, distributing the
        simulations across different devices and saving the results to disk.
        """
        raise NotImplementedError
