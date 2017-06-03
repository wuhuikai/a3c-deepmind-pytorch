import random

import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def split_weight_bias(model):
    weights, biases = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            biases += [p]
        else:
            weights += [p]
    return weights, biases