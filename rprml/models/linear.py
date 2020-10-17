import torch


class LinearModel(torch.nn.Module):
    """ An implementation of a linear model. """

    def __init__(self, d, bias=False):
        super().__init__()
        self.d = d
        self.w = torch.nn.Linear(d, 1, bias=bias)

    def forward(self, x):
        return self.w(x)

    def get_w(self):
        return self.w.weight.data.view(-1, 1).clone()

    def set_w(self, w):
        self.w.weight.data = w.clone().view(1, self.d)


class LinearModelFactory(object):
    """ A factory class for generating MLPs. """

    def __init__(self, d, init_w=None):
        self.d = d
        if init_w is None:
            init_w = torch.zeros(d, dtype=torch.float32)
        self.init_w = init_w

    def __call__(self):
        model = LinearModel(self.d)
        model.set_w(self.init_w.clone())
        return model
