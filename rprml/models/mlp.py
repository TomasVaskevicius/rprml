import torch


class Mlp(torch.nn.Module):
    """ An implementation of multi layer perceptron. """

    def __init__(self, layer_sizes, bias=True, activation=torch.nn.ReLU):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Flatten())
        for i in range(len(layer_sizes)):
            if i == 0:
                continue
            in_d = layer_sizes[i-1]
            out_d = layer_sizes[i]
            self.layers.append(torch.nn.Linear(in_d, out_d, bias=bias))
            if i < len(layer_sizes)-1:
                # If this is not the last layer, append the activation
                # function.
                self.layers.append(activation())

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def get_mlp_factory(layer_sizes, bias=True, activation=torch.nn.ReLU):
    """ A factory method implementation for the Mlp class. """
    def get_model():
        return Mlp(layer_sizes, bias, activation)
    return get_model
