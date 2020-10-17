import torch
from torch.optim import SGD


class MirrorMap(object):
    """ Provides functionality related to mirror decent. """

    def __init__(self):
        super().__init__()

    def compute_mirror_map(self, x):
        """ Computes grad psi (x). """
        raise NotImplementedError

    def compute_mirror_map_inverse(self, x):
        """ Computes (grad psi)^{-1}(x). """
        raise NotImplementedError

    def compute_bregman_divergence(self, x, y):
        """ Computes D_{psi}(x, y). """
        raise NotImplementedError


class MirrorDescentOptimizer(SGD):
    """ An implementation of mirror descent optimizer for linear model. """

    def __init__(self, mirror_map, params, **kwargs):
        """ :mirror_map: An instance of class MirrorMap defining the mirror
                map, its inverse and associated Bregman divergence.
            :params: Model parameters (will be passed to SGD base class).
            :**kwrags: Keyword arguments passed to SGD base class. """
        super().__init__(params, **kwargs)
        self.mirror_map = mirror_map

    @torch.no_grad()
    def step(self):
        # Apply the mirror map to parameters.
        for group in self.param_groups:
            for p in group['params']:
                p.data = self.mirror_map.compute_mirror_map(p.data)

        # Take a GD step.
        super().step()

        # Map parameters back to the primal space.
        for group in self.param_groups:
            for p in group['params']:
                p.data = self.mirror_map.compute_mirror_map_inverse(p.data)
