import os
import time
import copy
import numpy as np

from pycrayon import CrayonClient

import torch
import torch.multiprocessing as mp

from setproctitle import setproctitle

import visdom

from rl import EvalResult

from utils import set_random_seed

def build_env(game_type, args, **params):
    game_type = game_type.upper()
    if game_type == 'ALE':
        from ale_env import ALE
        return ALE(args.rom_path, **params)
    else:
        raise RuntimeError('Unknown game_type [{}], should be one of [ALE]'.format(game_type))

def async_train(args, create_agent, model, model_eval):
    setproctitle('{}:train[MASTER]'.format(args.name))
    counter = mp.Value('l', 0)

    def run_trainer(process_idx):
        setproctitle('{}:train[{}]'.format(args.name, process_idx))
        set_random_seed(np.random.randint(0, 2 ** 32))

        agent = create_agent(process_idx)

        train_loop(counter, args, agent)

    def run_evalator():
        setproctitle('{}:eval'.format(args.name))

        eval_loop(counter, args, model, model_eval)

    def run_player():
        setproctitle('{}:play'.format(args.name))

        play_loop(counter, args, model, model_eval)        

    processes = []
    processes.append(mp.Process(target=run_evalator))
    if not args.no_render:
        processes.append(mp.Process(target=run_player))
    for process_idx in range(args.n_processes):
        processes.append(mp.Process(target=run_trainer, args=(process_idx+1,)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

def train_loop(counter, args, agent):
    try:
        global_t = 0
        while True:
            agent.set_lr((args.n_steps - global_t - 1) / args.n_steps * args.lr)

            t = agent.act()

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += t
                global_t = counter.value

            if global_t > args.n_steps:
                break
    except KeyboardInterrupt:
        agent.finish()
        raise

    agent.finish()

def eval_loop(counter, args, shared_model, model_eval):
    try:
        SEC_PER_DAY = 24*60*60

        vis = visdom.Visdom(env='A3C:'+args.name)

        env = build_env(args.type, args, treat_life_lost_as_terminal=False)
        model = copy.deepcopy(shared_model)
        model.eval()

        set_random_seed(np.random.randint(0, 2 ** 32))
        # Create a new experiment
        cc = CrayonClient()
        names = cc.get_experiment_names()
        summaries = []
        for idx in range(args.n_eval):
            name = "{} [{}]".format(args.name, idx+1)
            if name in names:
                cc.remove_experiment(name)
            summaries.append(cc.create_experiment(name))

        max_reward = None
        save_condition = args.save_intervel
        
        rewards = []
        start_time = time.time()
        while True:
            # Sync with the shared model
            model.load_state_dict(shared_model.state_dict())

            eval_start_time, eval_start_step = time.time(), counter.value
            results = []
            for i in range(args.n_eval):
                model.reset_state()
                results.append(model_eval(model, env, vis=(vis, i+1, 12)))
                env.reset()
            eval_end_time, eval_end_step = time.time(), counter.value
            results = EvalResult(*zip(*results))
            rewards.append((counter.value, results.reward))

            local_max_reward = np.max(results.reward)
            if max_reward is None or max_reward < local_max_reward:
                max_reward = local_max_reward
                
            if local_max_reward >= max_reward:
                # Save model
                torch.save(model.state_dict(), os.path.join(args.model_path, 'best_model.pth'))

            time_since_start = time.time() - start_time
            day = time_since_start // SEC_PER_DAY
            time_since_start %= SEC_PER_DAY

            seconds_to_finish = (args.n_steps - eval_end_step)/(eval_end_step-eval_start_step)*(eval_end_time-eval_start_time)
            days_to_finish = seconds_to_finish // SEC_PER_DAY
            seconds_to_finish %= SEC_PER_DAY
            print("STEP:[{}|{}], Time: {}d {}, Finish in {}d {}".format(
                counter.value, args.n_steps, '%02d' % day, time.strftime("%Hh %Mm %Ss", time.gmtime(time_since_start)),
                '%02d' % days_to_finish, time.strftime("%Hh %Mm %Ss", time.gmtime(seconds_to_finish))))
            print('\tMax reward: {}, avg_reward: {}, std_reward: {}, min_reward: {}, max_reward: {}'.format(
                max_reward, np.mean(results.reward), np.std(results.reward), np.min(results.reward), local_max_reward))

            # Plot
            for summary, reward in zip(summaries, results.reward):
                summary.add_scalar_value('reward', reward, step=eval_start_step)

            if counter.value > save_condition:
                save_condition += args.save_intervel
                torch.save(model.state_dict(), os.path.join(args.model_path, 'model_iter_{}.pth'.format(counter.value)))

            if counter.value >= args.n_steps or len(rewards) > 10000:
                with open(os.path.join(args.save_path, 'rewards'), 'a+') as f:
                    for record in rewards:
                        f.write('{}: {}\n'.format(record[0], record[1]))
                del rewards[:]

            if counter.value >= args.n_steps:
                print('Evaluator Finished !!!')
                break
    except KeyboardInterrupt:
        torch.save(shared_model.state_dict(), os.path.join(args.model_path, 'model_iter_{}.pth'.format(counter.value)))
        raise

    torch.save(shared_model.state_dict(), os.path.join(args.model_path, 'model_final.pth'))

def play_loop(counter, args, shared_model, model_eval):
    try:
        env = build_env(args.type, args, render=not args.no_render, treat_life_lost_as_terminal=False)
        model = copy.deepcopy(shared_model)
        model.eval()

        set_random_seed(np.random.randint(0, 2 ** 32))
        
        while True:
            # Sync with the shared model
            model.load_state_dict(shared_model.state_dict())
            model.reset_state()
            model_eval(model, env)
            env.reset()
            
            if counter.value >= args.n_steps:
                print('Player Finished !!!')
                break
    except KeyboardInterrupt:
        raise