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
        if self.un_init:
            batch_size = state.size()[0]
            self.h, self.c = Variable(torch.zeros(batch_size, self.head.n_output_channels)), Variable(torch.zeros(batch_size, self.head.n_output_channels))
            self.un_init = False

        out = self.head(state)
        h, c = self.lstm(out, (self.h, self.c))
        if not keep_same_state:
            self.h, self.c = h, c
        return self.pi.compute_policy(h), self.v(h)

    def reset_state(self):
        self.un_init = True

    def unchain_backward(self):
        if self.un_init:
            return
        self.h.detach_()
        self.c.detach_()