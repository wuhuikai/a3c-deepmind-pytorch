import torch.optim as optim

class RMSpropAsync(optim.Optimizer):
    """
        Implements RMSprop algorithm with shared states.
    """
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(RMSpropAsync, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['ms'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['ms'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                ms = state['ms']
                alpha = group['alpha']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                ms.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = ms.add(group['eps']).sqrt_()
                p.data.addcdiv_(-group['lr'], grad, avg)

        return loss