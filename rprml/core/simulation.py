import torch
from dataclasses import dataclass
from typing import Callable, Union, List
from ignite.engine import Engine, create_supervised_evaluator
from ignite.metrics import Loss
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from .executor import Executor
from ..data.data import dataset_factory_methods
from ..utils.random import reset_all_seeds


def _create_supervised_trainer(simulation):
    """ Creates a ignite.engine.Engine object, which reads the optimizer,
    model and loss function information from the Simulation object. """

    def _process_function(engine, batch):
        simulation.model.train()
        simulation.optimizer.zero_grad()
        X, y = batch
        y_pred = simulation.model(X)
        loss = simulation.loss_function(y_pred, y)
        loss.backward()
        simulation.optimizer.step()
        if simulation.lr_scheduler is not None:
            simulation.lr_scheduler.step()
        return loss.item()

    return Engine(_process_function)


@dataclass
class Simulation(object):
    """ A class for storing configuration of the simulation to be performed
    with Executor. """

    seed: int  # Random seed for reproducibility.
    device: torch.device  # Device on which simulation will be performed.
    model_factory: Callable  # A function returning the model.
    data: Union[str, Callable]  # If string then must be a key to the
    # dataset_factory_methods dictionary defined in ..data.data.py.
    # Otherwhise, needs to be a Callable, returning training and
    # validation datasets of specified sizes.
    n_train: int  # Number of training data points.
    n_valid: int  # Number of validation data points.
    loss_function: Callable  # For example, torch.nn.CrossEntropyLoss().
    batch_size: int
    _learning_rate: int
    # The below lr scheduler is optional. Its step function will be called
    # after every iteration, rather than every epoch.
    lr_scheduler: _LRScheduler = None
    simulation_name: str = 'Simulation'
    custom_handlers: List[Callable] = None  # Functions of the form
    # simulation -> void
    # to be called after everyting is
    # set up.
    executor_class = Executor  # The default Executor to be used.
    # Can be changed upon instantiation of this class but not after.

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self._learning_rate = val
        if self.__initialized:
            # Set the new learning rate.
            for group in self.optimizer.param_groups:
                group['lr'] = val

    # A dictionary of extra keyword arguments to be used when creating data
    # loaders and model.
    data_factory_kwargs: dict = None
    model_factory_kwargs: dict = None

    __initialized = False

    def __post_init__(self):
        """ A method for setting up the learner object. """

        # Reset the seed before setting up the data and calling model factory.
        reset_all_seeds(self.seed)

        if self.data_factory_kwargs is None:
            self.data_factory_kwargs = {}
        if self.model_factory_kwargs is None:
            self.model_factory_kwargs = {}

        # Set up the datasets.
        dataset_factory = self.data
        if type(self.data) is str:
            dataset_factory = dataset_factory_methods[self.data]

        datasets = \
            dataset_factory(
                n_train=self.n_train, n_valid=self.n_valid, device=self.device,
                **self.data_factory_kwargs)

        if len(datasets) == 2:
            # If dataset_factory has returned a tuple of 2 objects we have
            # a train and valid datasets. Append None for test dataset.
            datasets = (*datasets, None)

        if len(datasets) != 3:
            raise ValueError('Dataset factory has to return 2 or 3 datasets.')

        self.train_dataset, self.valid_dataset, self.test_dataset = datasets

        # Create model and move to device.
        self.model = self.model_factory(**self.model_factory_kwargs)
        self.model.to(self.device)

        # Set up the optimizer.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), self.learning_rate, momentum=0,
            dampening=0, weight_decay=0, nesterov=False)

        # Set up ignite trainer and evaluator computing the loss.
        self.trainer = _create_supervised_trainer(self)
        self.evaluator = create_supervised_evaluator(
            self.model, metrics={'loss': Loss(self.loss_function)},
            device=self.device)

        # Reset the seed again once the data and the model is set up.
        reset_all_seeds(self.seed)

        self.executor = Executor(self)

        # Set up the data loaders before registering custom handlers.
        self._reset_data_loaders()

        # Call the custom handlers.
        if self.custom_handlers is not None:
            for handler in self.custom_handlers:
                handler(self)

        # Mark instance as initialized to prohibit changing certain instance
        # variables.
        self.__initialized = True


    def _reset_data_loaders(self):
        """ Resets the data loader objects. """
        self.train_dl = DataLoader(self.train_dataset,
                                   batch_size=self.batch_size, shuffle=True)
        self.valid_dl = DataLoader(self.valid_dataset,
                                   batch_size=self.n_valid, shuffle=False)

        # If test dataset exists, create a test data loader.
        self.test_dl = None
        if self.test_dataset is not None:
            # Try to use the validation dataset size for evaluating test
            # metrics.
            test_batch_size = self.n_valid
            if test_batch_size > len(self.test_dataset):
                test_batch_size = len(self.test_dataset)
            self.test_dl = DataLoader(self.test_dataset,
                                      batch_size=test_batch_size,
                                      shuffle=False)

    def run(self, epochs: int):
        """ Runs the trainer for the given number of epochs. """
        # Need to reset the data loaders, for example, if the
        # batch size was changed, or if the dataset objects were modified.
        self._reset_data_loaders()
        # Run the simulation.
        self.executor.run(epochs)
