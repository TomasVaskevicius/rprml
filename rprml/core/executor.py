import numpy as np
from tqdm.auto import tqdm
from ignite.engine import Events
from ignite.metrics import Accuracy
import numbers

# We define a custom event for ignite.Engine, which will fire
# after pre-specified number of iterations.
_iteration_level_event = 'iteration_level_event'


# Implementation of our custom event, which fires after every specified
# number of iterations.
def _iteration_level_event_handler(engine, executor):
    iteration_frequency = executor.log_frequency
    executor._complete_iterations += 1
    if executor._complete_iterations % iteration_frequency == 0:
        engine.fire_event(_iteration_level_event)


class _ExecutorBase(object):
    """ Base class for the executor. Provides interface for running the
    trainer, registering event handlers and handles displaying results. """

    def __init__(self, simulation, log_frequency=-1, print_frequency=1):
        """
        :simulation: A subclass of simulation.Simulation, with references to
                to train_dl, valid_dl, optimizer, ignite trainer, ignite
                evaluator.
        :log_frequency: After every specified number of iterations registered
            metrics will be tracked. If not specified (or -1) then uses the
            trining dataset size of the simulation object (i.e., one log per
            epoch).
        :print_frequency: Display every print_frequency-th logged metric. Value
            -1 is reserved for disabling printing.
        """
        # First save a handle to the simulation object and register our custom
        # event handler, which fires after every log_frequency number of
        # iterations.
        self.simulation = simulation
        self.simulation.trainer.register_events(_iteration_level_event)
        self._complete_iterations = 0
        @self.simulation.trainer.on(Events.ITERATION_COMPLETED)
        def call_custom_event_handler(engine):
            _iteration_level_event_handler(engine, self)

        self.log_frequency = log_frequency
        self.print_frequency = print_frequency
        self._pbar = None  # The core pbar displaying number of epochs.
        self._n_log_pbars = 0  # Number of pbars for displaying metrics.
        self._log_pbars = []  # List for storing metrics pbars.
        self._output_dict = {}  # Metrics can be added to this dict via
        # a call to register_printable_metric.
        self._printable_metric_names = []  # Registers metrics that should be
        # printed.
        self._not_printable_metric_names = []  # Registered metrics for logging
        # only.
        self.__initialized = False

    @property
    def log_frequency(self):
        return self._log_frequency

    @log_frequency.setter
    def log_frequency(self, val):
        if val == -1:
            simulation = self.simulation
            self._log_frequency = simulation.n_train // simulation.batch_size
        elif val > 0:
            self._log_frequency = val
        else:
            raise ValueError('Invalid value ' + str(val) + ' for setting' +
                             'log frequency of core.Executor.')

    def register_printable_metric(self, handler, name,
                                  event=_iteration_level_event):
        """ Registers a metric for the evaluator, which will be computed
        at the end of every epoch, stored into self._output_dict and
        displayed live during training using one of the progress bars.
        Metrics are computed in the order they are added. """

        trainer = self.simulation.trainer
        self._output_dict[name] = []
        self._n_log_pbars += 1
        self._printable_metric_names.append(name)
        @trainer.on(event)
        def compute_metric(engine):
            metric = handler(engine, self)
            self._output_dict[name].append(metric)

    def register_not_printable_metric(self, handler, name,
                                      event=_iteration_level_event):
        """ Registers a metric for the evaluator which will be logged but
        not printed. """
        trainer = self.simulation.trainer
        self._output_dict[name] = []
        self._not_printable_metric_names.append(name)
        @trainer.on(event)
        def compute_metric(engine):
            metric = handler(engine, self)
            self._output_dict[name].append(metric)

    def display_output_dict(self):
        """ A method for refreshing the output dict to be displayed on
        self._log_pbars. """
        raise NotImplementedError

    def _register_handlers(self):
        """ To be implemented by subclasses for setting up custom handlers. """
        # Show the output dict immediately as the training begins.
        @self.simulation.trainer.on(Events.STARTED)
        def display_output_dict_on_start(engine):
            if self.print_frequency != -1:
                self.display_output_dict()
        # Update the output dict after the training ends.
        @self.simulation.trainer.on(Events.COMPLETED)
        def display_output_dict_on_end(engine):
            if self.print_frequency != -1:
                self.display_output_dict()

    def run(self, epochs: int):
        """ Runs the trainer of the current simulation configuration for the
        given number of epochs. """

        if not self.__initialized:
            self._register_handlers()
            # Start logging metrics at initialization.
            self.simulation.trainer.fire_event(_iteration_level_event)
            self.__initialized = True

        simulation = self.simulation
        print('Starting simulation ' + simulation.simulation_name +
              ' with seed ' + str(simulation.seed) + ' for ' + str(epochs) +
              ' epochs.')

        # Create a progress bar displaying the number of epochs.
        self._pbar = tqdm(initial=0, total=epochs, position=0,
                          unit='epoch')
        # Create progress bars for logging.
        self._log_pbars = []
        for i in range(self._n_log_pbars):
            self._log_pbars.append(tqdm(
                total=0, position=i + 1, bar_format='{desc}'))

        if self.print_frequency != -1:
            self._register_display_handlers()

        try:
            simulation.trainer.run(simulation.train_dl, max_epochs=epochs)
        except KeyboardInterrupt:
            # Clean up if interrupted.
            self._remove_display_handlers()
            self._pbar.close()
            for pbar in self._log_pbars:
                pbar.close()
            raise

        self._remove_display_handlers()
        self._pbar.close()
        for pbar in self._log_pbars:
            pbar.close()

    def _register_display_handlers(self):
        """ Registers handlers responsible for displaying the output.
        The purpose of this method is so that the display handlers are
        registered just before the training process begins, to make sure
        they are the last ones. """

        # Define the display handlers.
        def display_output_dict_handler(engine):
            if self.print_frequency == -1:
                return
            if (len(self._output_dict['iter'])-1) % self.print_frequency == 0:
                self.display_output_dict()

        def epoch_count_handler(engine):
            self._pbar.update(1)
            self._pbar.refresh()

        # Keep the reference to handlers for removing it later.
        self._display_output_dict_handler = display_output_dict_handler
        self._epoch_count_handler = epoch_count_handler

        # Add the display handlers.
        self.simulation.trainer.add_event_handler(
            _iteration_level_event, self._display_output_dict_handler)
        self.simulation.trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                                  self._epoch_count_handler)

    def _remove_display_handlers(self):
        """ Removes the display handles. Called after training is finished.
        This allows to register custom handlers in between the calls to
        self.run(). """

        self.simulation.trainer.remove_event_handler(
            self._display_output_dict_handler, _iteration_level_event)
        self._display_output_dict_handler = None
        self.simulation.trainer.remove_event_handler(
            self._epoch_count_handler, Events.EPOCH_COMPLETED)
        self._epoch_count_handler = None


# Below we define handlers used by the Executor class below.

def _compute_iteration_handler(engine, executor):
    log_frequency = executor.log_frequency
    if executor._output_dict['iter'] == []:
        # This is the exceptional call, during the 0-th iteration (before
        # the training has strated).
        return 0
    else:
        return executor._output_dict['iter'][-1] + log_frequency


def _compute_epoch_handler(engine, executor):
    if executor._output_dict['epoch'] == []:
        # This is the exceptional call, during the 0-th iteration (before
        # the training has strated).
        return 0
    else:
        return executor._output_dict['epoch'][-1]


def _compute_training_loss(engine, executor):
    evaluator = executor.simulation.evaluator
    evaluator.run(executor.simulation.train_dl)
    return evaluator.state.metrics['loss']


def _compute_validation_loss(engine, executor):
    evaluator = executor.simulation.evaluator
    evaluator.run(executor.simulation.valid_dl)
    return evaluator.state.metrics['loss']


def _compute_generalization_error(engine, executor):
    return executor._output_dict['valid_loss'][-1] - \
        executor._output_dict['train_loss'][-1]


def _update_optimal_iteration(engine, executor):
    current_loss = executor._output_dict['valid_loss'][-1]
    if current_loss < executor._optimal_validation_loss:
        executor._optimal_validation_loss = current_loss
        return executor._output_dict['iter'][-1]
    return executor._output_dict['opt_iter'][-1]


def _compute_training_accuracy(engine, executor):
    evaluator = executor.simulation.evaluator
    evaluator.run(executor.simulation.train_dl)
    return evaluator.state.metrics['accuracy']


def _compute_validation_accuracy(engine, executor):
    evaluator = executor.simulation.evaluator
    evaluator.run(executor.simulation.valid_dl)
    return evaluator.state.metrics['accuracy']


class Executor(_ExecutorBase):
    """ Executor implementation with registered default metrics tracking.
    Includes training loss, validation loss, generalization loss. Also
    computes the same metrics at early stopped model based on validation
    loss. """

    def __init__(self, simulation, log_frequency=-1, print_frequency=1):
        super().__init__(simulation, log_frequency, print_frequency)
        self.__accuracy_handlers_registered = False

    def _register_handlers(self):
        """ Overrides base class method, registering some handlers at
        initialization of the self. """

        # First register base class handlers.
        super()._register_handlers()

        self.register_printable_metric(_compute_iteration_handler, 'iter')

        self.register_printable_metric(_compute_epoch_handler, 'epoch')
        @self.simulation.trainer.on(Events.EPOCH_COMPLETED)
        def call_custom_event_handler(engine):
            self._output_dict['epoch'][-1] += 1

        self.register_printable_metric(_compute_validation_loss, 'valid_loss')
        self.register_printable_metric(_compute_training_loss, 'train_loss')
        self.register_printable_metric(
            _compute_generalization_error, 'gen_error')

        self._optimal_validation_loss = np.finfo(np.float32).max
        self.register_not_printable_metric(
            _update_optimal_iteration, 'opt_iter')

    def register_accuracy_handlers(self):
        """ Registers accuracy metric to the evaluator. We leave this
        behavior optimal, since we may sometimes consider regression models.
        """
        # The below code can only be called once.
        if self.__accuracy_handlers_registered is False:
            evaluator = self.simulation.evaluator
            accuracy_metric = Accuracy()
            accuracy_metric.attach(evaluator, 'accuracy')
            self.register_printable_metric(_compute_training_accuracy,
                                           'train_acc')
            self.register_printable_metric(_compute_validation_accuracy,
                                           'valid_acc')
            self.__accuracy_handlers_registered = True

    def display_output_dict(self):
        """ Display the _output_dict showing each metric side by side with
        the metric at early stopped model. Each metric is displayed on its
        own pbar. """

        keys = []
        vals_current = []
        vals_early_stopped = []

        # First populate the above lists displaying row name and values.
        for key, value in self._output_dict.items():
            if key not in self._printable_metric_names:
                continue

            keys.append(str(key))

            if len(value) > 0:
                value = value[-1]
            if isinstance(value, numbers.Number):
                value = round(value, 3)
            vals_current.append(str(value))

            # Compute the early stopped value of the current metric.
            early_stopped_value = value
            if len(self._output_dict['opt_iter']) > 0:
                opt_iter = self._output_dict['opt_iter'][-1]
                opt_iter_id = opt_iter // self.log_frequency
                early_stopped_value = \
                    self._output_dict[key][opt_iter_id]
            if isinstance(early_stopped_value, numbers.Number):
                early_stopped_value = round(early_stopped_value, 3)
            vals_early_stopped.append(str(early_stopped_value))

        # Now set up description strings for each pbar.
        # First we align the strings right or left for each metric.
        def right_justify(*lists):
            right_justified_lists = []
            for l in lists:
                max_len = len(max(l, key=len))
                right_justified_lists.append(
                    [val.rjust(max_len, ' ') for val in l])

            return tuple(right_justified_lists)

        keys, vals_current, vals_early_stopped = right_justify(
            keys, vals_current, vals_early_stopped)

        descriptions = []
        separator = ' | early stopped: '
        for (key, val, val_early_stopped) in \
                zip(keys, vals_current, vals_early_stopped):
            # Create a description string for the current metric.
            descriptions.append(key + ': ' + val + separator +
                                val_early_stopped)

        # Sanity check.
        assert len(descriptions) == self._n_log_pbars

        for pbar_id in range(self._n_log_pbars):
            self._log_pbars[pbar_id].set_description_str(descriptions[pbar_id])

    @property
    def history(self):
        """ Returns the logged history of the registered metrics. """
        return self._output_dict
