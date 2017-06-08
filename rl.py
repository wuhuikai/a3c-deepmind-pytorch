import numpy as np

import torch.nn.functional as F

from cached_property import cached_property

from collections import namedtuple

class Environment(object):
    @property
    def state(self):
        pass

    @property
    def reward(self):
        pass

    @property
    def is_terminal(self):
        pass

    @property
    def exceed_max(self):
        pass

    @property
    def current_raw_screen(self):
        pass

    def receive_action(self, action):
        pass

    def reset(self):
        pass

class SoftmaxPolicy(object):
    """Abstract softmax policy class."""
    def compute_policy(self, state):
        raise NotImplementedError

    def logits2policy(self, logits):
        return SoftmaxPolicyOutput(logits)

class SoftmaxPolicyOutput(object):
    def __init__(self, logits):
        self.logits = logits

    @cached_property
    def most_probable_actions(self):
        return np.argmax(self.probs.data.numpy(), axis=1)

    @cached_property
    def probs(self):
        return F.softmax(self.logits)

    @cached_property
    def log_probs(self):
        return F.log_softmax(self.logits)

    @cached_property
    def action_indices_var(self):
        return self.probs.multinomial(1).detach()

    @cached_property
    def action_indices(self):
        return self.action_indices_var.data.numpy().squeeze(1)

    @cached_property
    def sampled_actions_log_probs(self):
        return self.log_probs.gather(1, self.action_indices_var)

    @cached_property
    def entropy(self):
        return - (self.probs*self.log_probs).sum(1)

EvalResult = namedtuple('EvalResult', ('reward', 'duration'))