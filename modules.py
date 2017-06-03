import torch.nn as nn
import torch.nn.functional as F

from rl import SoftmaxPolicy

class NIPSDQNHead(nn.Module):
    """DQN's head (NIPS workshop version)"""
    def __init__(self, n_input_channels=4, n_output_channels=256, activation=F.relu):
        super(NIPSDQNHead, self).__init__()
        self.activation = activation
        self.n_output_channels = n_output_channels

        self.conv1 = nn.Conv2d(n_input_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.linear = nn.Linear(2592, n_output_channels)

    def forward(self, state):
        h = self.activation(self.conv1(state))
        h = self.activation(self.conv2(h))
        h = h.view(-1, 2592)
        h = self.activation(self.linear(h))

        return h

class FCSoftmaxPolicy(nn.Module, SoftmaxPolicy):
    """Softmax policy that consists of FC layers and rectifiers"""
    def __init__(self, n_input_channels, n_actions):
        super(FCSoftmaxPolicy, self).__init__()

        self.linear = nn.Linear(n_input_channels, n_actions)

    def forward(self, state):
        return self.linear(state)

    def compute_policy(self, state):
        return self.logits2policy(self(state))

class FCVFunction(nn.Module):
    def __init__(self, n_input_channels):
        super(FCVFunction, self).__init__()

        self.linear = nn.Linear(n_input_channels, 1)

    def forward(self, state):
        return self.linear(state)