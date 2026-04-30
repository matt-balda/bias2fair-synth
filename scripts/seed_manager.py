import random
import numpy as np
import torch

def get_fixed_seeds():
    """Returns the 10 fixed seeds for the experiment."""
    return [42, 123, 999, 2024, 7, 88, 101, 13, 555, 777]

def set_seed(seed):
    """Sets the seed for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
