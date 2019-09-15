import numpy as np
import random
import torch


def reset_all_seeds(seed: int):
    """ A method for resetting all RNG seeds to the one stored by this
    object. """

    # See https://github.com/pytorch/pytorch/issues/7068.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
