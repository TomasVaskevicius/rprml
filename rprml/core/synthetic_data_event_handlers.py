from . import rprml_path  # noqa

import torch

from ..optimizers.mirror_descent_optimizer import MirrorDescentOptimizer

###############################################################################
#
# This file implements logging of certain metrics associated to well-specified
# linear regression datasets. In particular, the ones generated using the
# rprml.data.synthetic_data.generate_synthetic_regression_dataset function.
# The model class is assumed to be of type rprml.models.LinearModel.
#
###############################################################################


def log_linear_model_complexity_parameters(simulation):
    """ Registers logging various complexity parameters, such as in-sample
    prediction-error, model parameters at iteration t,
    Bregman divergence to the true minimizer (if applied to a mirror descent
    simulation), etc. """

    X = simulation.train_dataset.X
    y = simulation.train_dataset.y
    w_star = simulation.train_dataset.w_star.view(-1, 1)
    loss_w_star = torch.sum((X.matmul(w_star) - y)**2) / X.shape[0]
    covariance_matrix = None

    if hasattr(simulation.train_dataset, 'covariance_matrix'):
        covariance_matrix = simulation.train_dataset.covariance_matrix
    if covariance_matrix is None:
        # Set covariance_matrix to identity
        covariance_matrix = torch.diag(
            torch.ones(X.shape[1], device=X.device, dtype=X.dtype))
    else:
        if not isinstance(covariance_matrix, torch.Tensor):
            covariance_matrix = torch.tensor(
                covariance_matrix, device=X.device, dtype=X.dtype)

    def _log_wt(engine, executor):
        return simulation.model.get_w().detach().cpu().numpy().flatten()

    def _log_bregman_divergence(engine, executor):
        with torch.no_grad():
            if not isinstance(simulation.optimizer, MirrorDescentOptimizer):
                return -1  # This is not a mirror descent simulation.
            w_t = simulation.model.get_w()
            mirror_map = simulation.optimizer.mirror_map
            return mirror_map.compute_bregman_divergence(w_star, w_t)

    def _log_true_excess_risk(engine, executor):
        with torch.no_grad():
            w_t = simulation.model.get_w()
            delta_t = w_t - w_star
            excess_risk_t = delta_t.t() @ covariance_matrix @ delta_t
            return excess_risk_t.item()

    def _log_rt(engine, executor):
        # Logs in sample prediction.
        with torch.no_grad():
            w_t = simulation.model.get_w()
            r_t = torch.sum(X.matmul(w_t - w_star)**2) / X.shape[0]
            return r_t.item()

    def _log_delta_t(engine, executor):
        with torch.no_grad():
            w_t = simulation.model.get_w()
            loss_w_t = torch.sum((X.matmul(w_t) - y)**2) / X.shape[0]
            return (loss_w_t - loss_w_star).item()

    simulation.executor.register_not_printable_metric(
        _log_wt, 'w_t')
    # The excess risk metric should be ignored for fixed-design simulations.
    # In such a case, covariance is assumed to be identity and excess risk
    # outputs the l2 squared parameter estimation error.
    simulation.executor.register_not_printable_metric(
        _log_true_excess_risk, 'excess_risk_t')
    simulation.executor.register_not_printable_metric(
        _log_rt, 'r_t')
    simulation.executor.register_not_printable_metric(
        _log_delta_t, 'delta_t')
    simulation.executor.register_not_printable_metric(
        _log_bregman_divergence, 'bregman_t')
