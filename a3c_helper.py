import time
import numpy as np

import torch
from torch.autograd import Variable

from a3c import A3C

from a3c_model import A3CLSTM
from optimizer import RMSpropAsync

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
        env = build_env(args.type, args, max_episode_length=args.max_episode_length)
        return A3C(model, opt, env, args.t_max, 0.99, beta=args.beta, process_idx=process_idx, phi=dqn_phi)

    def model_eval_func(model, env, **params):
        return model_eval(model, env, dqn_phi, **params)

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

def model_eval(model, env, phi, random=True, vis=None):
    if vis:
        vis, window_id, fps = vis
        frame_dur = 1.0 / fps
        last_time = time.time()

    reward, start_time = 0, time.time()
    while True:
        state_var = Variable(torch.from_numpy(phi(env.state)).unsqueeze_(0), volatile=True)
        pout, _ = model.pi_and_v(state_var)
        action = pout.action_indices[0] if random else pout.most_probable_actions[0]
        reward += env.receive_action(action)

        if vis and time.time() > last_time + frame_dur:
            vis.image(env.current_raw_screen.transpose((2, 0, 1)), win=window_id)
            last_time = time.time()

        if env.is_terminal:
            break

    return EvalResult(reward, time.time()-start_time)