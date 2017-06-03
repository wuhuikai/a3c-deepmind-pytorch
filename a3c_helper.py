import time
import numpy as np
from collections import deque

import torch
from torch.autograd import Variable

from a3c_model import A3CLSTM
from optimizer import RMSpropAsync

from a3c import A3C

from rl import EvalResult
from rl_helper import build_env, async_train

from utils import split_weight_bias

def build_master_model(n_actions, args):
    model = A3CLSTM(n_actions)
    model.share_memory()

    weights, biases = split_weight_bias(model)
    opt = RMSpropAsync([{'params': weights}, {'params': biases, 'weight_decay': 0}], 
                            lr=args.lr, eps=1e-1, alpha=0.99, weight_decay=args.weight_decay)
    opt.share_memory()

    return model, opt

def a3c_train(args):
    n_actions = build_env(args.type, args).number_of_actions

    model, opt = build_master_model(n_actions, args)

    def creat_agent(process_idx):
        env = build_env(args.type, args)
        return A3C(model, opt, env, args.t_max, 0.99, beta=args.beta, process_idx=process_idx, phi=dqn_phi)

    def model_eval_func(model, env, stuck_prevent=True):
        return model_eval(model, env, dqn_phi, stuck_prevent=stuck_prevent)

    async_train(args, creat_agent, model, model_eval_func)

def dqn_phi(screens):
    """Phi (feature extractor) of DQN for ALE
    Args:
      screens: List of N screen objects. Each screen object must be
      numpy.ndarray whose dtype is numpy.uint8.
    Returns:
      numpy.ndarray
    """
    raw_values = np.asarray(screens, dtype=np.float32)
    # [0,255] -> [0, 1]
    raw_values /= 255.0
    return raw_values

def model_eval(model, env, phi, stuck_prevent=True):
    if stuck_prevent:
        # a quick hack to prevent the agent from stucking
        actions = deque(maxlen=100)

    reward, start_time = 0, time.time()
    while True:
        state_var = Variable(torch.from_numpy(phi(env.state)).unsqueeze_(0), volatile=True)
        pout, _ = model.pi_and_v(state_var)
        action = pout.action_indices[0]
        reward += env.receive_action(action)
        if env.is_terminal:
            break
        if stuck_prevent:
            actions.append(action)
            if actions.count(actions[0]) == actions.maxlen or time.time() - start_time > 60*5:
                break

    return EvalResult(reward, time.time()-start_time)