import copy

import torch
from torch.autograd import Variable

import numpy as np

from utils import ensure_shared_grads

class A3C(object):
    """A3C: Asynchronous Advantage Actor-Critic.
    """
    def __init__(self, model, optimizer, env, t_max, gamma, beta=1e-2, process_idx=0,
                 clip_reward=True, phi=lambda x: x, pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False):

        # Globally shared model
        self.shared_model = model
        # Thread specific model
        self.model = copy.deepcopy(self.shared_model)
        self.optimizer = optimizer
        self.env = env

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.process_idx = process_idx
        self.clip_reward = clip_reward
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same

        self.update_state()

    def update_state(self):
        self.state_var = Variable(torch.from_numpy(self.phi(self.env.state)).unsqueeze_(0))

    def act(self):
        self.model.load_state_dict(self.shared_model.state_dict())
        self.model.train()

        log_probs, entropies, rewards, values = [], [], [], []
        for _ in range(self.t_max):
            pout, vout = self.model.pi_and_v(self.state_var)
            reward = self.env.receive_action(pout.action_indices[0])
            if self.clip_reward:
                reward = np.clip(reward, -1, 1)

            log_probs.append(pout.sampled_actions_log_probs)
            entropies.append(pout.entropy)
            values.append(vout)
            rewards.append(reward)

            if self.env.is_terminal:
                break

            self.update_state()

        R = 0
        if not self.env.is_terminal:
            _, vout = self.model.pi_and_v(self.state_var, keep_same_state=True)
            R = float(vout.data.numpy())
        else:
            self.env.reset()
            self.model.reset_state()
            self.update_state()

        t = len(rewards)
        pi_loss, v_loss = 0, 0
        for i in reversed(range(t)):
            R = self.gamma*R + rewards[i]
            v = values[i]

            advantage = R - float(v.data.numpy()[0, 0])
            # Accumulate gradients of policy
            log_prob = log_probs[i]
            entropy = entropies[i]
            # Log probability is increased proportionally to advantage
            pi_loss -= log_prob * advantage
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function
            v_loss += (v - R).pow(2).div_(2)

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and t < self.t_max:
            factor = self.t_max / t
            pi_loss *= factor
            v_loss *= factor

        total_loss = pi_loss + v_loss

        # Compute gradients using thread-specific model
        self.optimizer.zero_grad()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
        # Copy the gradients to the globally shared model
        ensure_shared_grads(self.model, self.shared_model)

        self.optimizer.step()

        self.model.unchain_backward()

        return t

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def finish(self):
        print('Trainer [{}] Finished !!!'.format(self.process_idx))