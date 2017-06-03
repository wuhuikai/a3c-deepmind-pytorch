import torch
import torch.nn as nn

from torch.autograd import Variable

from modules import NIPSDQNHead, FCSoftmaxPolicy, FCVFunction

class A3CModel(object):
    def pi_and_v(self, state, keep_same_state=False):
        raise NotImplementedError()

    def reset_state(self):
        pass

    def unchain_backward(self):
        pass

class A3CLSTM(nn.Module, A3CModel):
    def __init__(self, n_actions):
        super(A3CLSTM, self).__init__()

        self.head = NIPSDQNHead()
        self.pi = FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = FCVFunction(self.head.n_output_channels)
        self.lstm = nn.LSTMCell(self.head.n_output_channels, self.head.n_output_channels)
        self.reset_state()

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        h, c = self.lstm(out, (self.h, self.c))
        if not keep_same_state:
            self.h, self.c = h, c
        return self.pi.compute_policy(h), self.v(h)

    def reset_state(self):
        self.h, self.c = Variable(torch.zeros(1, self.head.n_output_channels)), Variable(torch.zeros(1, self.head.n_output_channels))

    def unchain_backward(self):
        self.h.detach_()
        self.c.detach_()