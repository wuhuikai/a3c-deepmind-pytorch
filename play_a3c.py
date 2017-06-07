import os
import json
import argparse
import datetime
from subprocess import call

import torch

from rl_helper import build_env
from a3c_helper import model_eval, dqn_phi
from a3c_model import A3CLSTM

from utils import set_random_seed

def main():
    parser = argparse.ArgumentParser('A3C: Play')
    parser.add_argument('--name', type=str, required=True, help='Experiment name, all outputs will be saved in checkpoints/[name]/')
    parser.add_argument('--model_name', default='best_model.pth', help='Which model to play with (default: best_model.pth)')

    parser.add_argument('--no_render', action='store_true', help='Do not render to screen (default: False)')
    parser.add_argument('--duration', default=5, type=float, help='How long does the play last (default: 5 [min])')
    parser.add_argument('--fps', default=60, type=int, help='FPS for recording video (default: 60)')
    args = parser.parse_args()

    args.save_path = os.path.join('checkpoints', args.name)
    args.gif_path = os.path.join(args.save_path, 'gifs', '{}_{}'.format(args.model_name.split('.')[0], 
                            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    if not os.path.isdir(args.gif_path):
        os.makedirs(args.gif_path)

    with open(os.path.join(args.save_path, 'config')) as f:
        vargs = json.loads(''.join(f.readlines()))
    vargs.update(vars(args))
    args.__dict__ = vargs

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    set_random_seed(args.seed)

    env = build_env(args.type, args, render=not args.no_render, treat_life_lost_as_terminal=False, record_screen_dir=args.gif_path)
    model = A3CLSTM(env.number_of_actions)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name)))
    model.eval()
    
    model_eval(model, env, dqn_phi, random=False, duration=args.duration)

    call(['ffmpeg', '-r', str(args.fps), '-i', os.path.join(args.gif_path, '%06d.png'), '-f', 'mov', '-c:v', 'libx264', args.gif_path+'.mov'])
    call(['rm', '-rf', args.gif_path])
    
if __name__ == '__main__':
    main()