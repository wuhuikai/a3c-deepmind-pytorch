import os
import json
import random
import argparse

import numpy as np

from a3c_helper import a3c_train

from utils import set_random_seed

def main():
    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser('A3C: Train')
    parser.add_argument('--name', type=str, required=True, help='Experiment name, all outputs will be saved in checkpoints/[name]/')

    parser.add_argument('--type', type=str, required=True, help='Which type of game to play [ALE]')
    parser.add_argument('--rom_path', type=str, required=True, help='Path of rom, only used when [type==ALE]')
    parser.add_argument('--no_render', action='store_true', help='Do not render to screen (default: False)')

    parser.add_argument('--lr', type=float, default=7e-4, help='Learning rate (default: 7e-4)')
    parser.add_argument('--beta', type=float, default=1e-2, help='Weight for entropy (default: 1e-2)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight deacy (default: 0)')

    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--t_max', type=int, default=5, help='# of forward steps in A3C (default: 5)')        
    parser.add_argument('--n_steps', type=int, default=8*10**7, help='Max steps to run (default: 8e7)')
    parser.add_argument('--n_processes', type=int, default=8, help='# of training processes (default: 8)')
    parser.add_argument('--save_intervel', type=int, default=2000000, help='Frequency of model saving (default: 2000000)')
    
    parser.add_argument('--n_eval', type=int, default=10, help='# of evaluation runs per model (default: 10)')
    args = parser.parse_args()

    args.save_path = os.path.join('checkpoints', args.name)
    args.model_path = os.path.join(args.save_path, 'snapshots')

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    set_random_seed(args.seed)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    # Save all the arguments
    with open(os.path.join(args.save_path, 'config'), 'w') as f:
        f.write(json.dumps(vars(args)))

    a3c_train(args)

if __name__ == '__main__':
    main()