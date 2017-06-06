import random

import numpy as np

import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def split_weight_bias(model):
    weights, biases = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            biases.append(p)
        else:
            weights.append(p)
    return weights, biases

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad