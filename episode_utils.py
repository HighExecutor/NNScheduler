import numpy as np
import env.context as ctx
import requests
from wf_gen_funcs import tree_data_wf, read_wf
from argparse import ArgumentParser
from draw_figures import write_schedule
from lstm_deq import LSTMDeque
from heft_deps.heft_settings import run_heft
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pathlib
import os
import time
import csv
import glob
import pandas as pd
from datetime import datetime

DEFAULT_CONFIG = {'task_par': 30, 'agent_task': 5, 'task_par_min': 20,
                  'nodes': np.array([4, 8, 8, 16]), 'state_size': 64,
                  'batch_size': 64, 'wfs_name': ['Montage_25'], 'seq_size': 5}


def parameter_setup(args, config):
    dict_args = vars(args)
    for key, value in dict_args.items():
        if value is not None:
            if key is 'wfs_name':
                config[key] = [value]
            else:
                config[key] = value
    return config


def wf_setup(wfs_names):
    wfs_real = [read_wf(name) for name in wfs_names]
    test_wfs = []
    test_times = dict()
    test_scores = dict()
    test_size = len(wfs_real)
    for i in range(test_size):
        wf_components = tree_data_wf(wfs_real[i])
        # tasks_n = np.random.randint(task_par_min, task_par+1)
        # wf_components = tree_data_gen(tasks_n)
        test_wfs.append(wf_components)
        test_times[i] = list()
        test_scores[i] = list()
    return test_wfs, test_times, test_scores, test_size


def episode(ei, config, test_wfs, test_size, URL):
    ttree, tdata, trun_times = test_wfs[ei % test_size]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    wfl.name = config['wfs_name'][ei % test_size]
    if config['actor_type'] == 'lstm':
        deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
    done = wfl.completed
    state = list(map(float, list(wfl.state)))
    if config['actor_type'] == 'lstm':
        deq.push(state)
        state = deq.show()
    sars_list = list()
    reward = 0
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = list(map(int, list(wfl.get_mask())))
        state = state.tolist() if type(state) != list else state
        action = requests.post(f"{URL}act", json={'state': state, 'mask': mask, 'sched': False}).json()['action']
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = list(map(float, list(wfl.state)))
        if config['actor_type'] == 'lstm':
            deq.push(next_state)
            next_state = deq.show()
            next_state = next_state.tolist()
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return reward, sars_list
    return reward, sars_list


def remember(sars_list, URL):
    for sars in sars_list:
        _ = requests.post(f'{URL}remember', json={'SARSA': sars})


def replay(batch_size, URL):
    loss = requests.post(f'{URL}replay', json={'batch_size': batch_size}).json()['loss']
    return loss


def run_episode(ei, logger, args):
    URL = f"http://{args.host}:{args.port}/"
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode(ei, config, test_wfs, test_size, URL)
    remember(sars_list, URL)
    replay(config['batch_size'], URL)
    if ei % 100 == 0:
        print("episode {} completed".format(ei))
    if logger is not None:
        logger.log_scalar('main/reward', reward, ei)
    return reward


def test(args, URL):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        if config['actor_type'] == 'lstm':
            deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
        done = wfl.completed
        state = list(map(float, list(wfl.state)))
        if config['actor_type'] == 'lstm':
            deq.push(state)
            state = deq.show()
        for time in range(wfl.n):
            mask = list(map(int, list(wfl.get_mask())))
            action = requests.post(f'{URL}test', json={'state': state.tolist(), 'mask': mask, 'sched': False}).json()[
                'action']
            act_t, act_n = wfl.actions[action]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = list(map(float, list(wfl.state)))
            if config['actor_type'] == 'lstm':
                deq.push(next_state)
                next_state = deq.show()
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
        write_schedule(args.run_name, i, wfl)


def save(URL):
    model = requests.post(f'{URL}save')


def plot_reward(args, rewards, heft_reward=None):
    cur_dir = os.getcwd()
    means = np.convolve(rewards, np.ones((500,)))[499:-499] / 500
    means = means.tolist()

    if heft_reward is not None:
        heft_rewards = [heft_reward for _ in range(args.num_episodes)]

    plt.style.use("seaborn-muted")
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, '--', label="rewards")
    plt.plot(means, '-', label="avg")

    if heft_reward is not None:
        plt.plot(heft_rewards, label='heft-rewards')

    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.legend()
    plt_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_plt.png'
    plt.savefig(plt_path)

    reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_rewards.csv'
    rewards = np.array(rewards)
    result = pd.DataFrame()
    result['reward'] = rewards
    result.to_csv(reward_path, sep=',', index=None, columns=['reward'])

    mean_reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_mean_rewards.csv'
    means = np.array(means)
    result = pd.DataFrame()
    result['reward'] = means
    result.to_csv(mean_reward_path, sep=',', index=None, columns=['reward'])

    if heft_reward is not None:
        reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_heft_reward.csv'
        rewards = np.array(rewards)
        result = pd.DataFrame()
        result['reward'] = rewards
        result.to_csv(reward_path, sep=',', index=None, columns=['reward'])


def do_heft(args, URL, logger):
    config = parameter_setup(args, DEFAULT_CONFIG)
    response = requests.post(f'{URL}heft', json={'wf_name': config['wfs_name'], 'nodes': config['nodes'].tolist()}).json()
    cur_dir = os.getcwd()
    reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_heft_reward.csv'
    rewards = response['reward']

    if args.logger:
        for i in range(args.num_episodes):
            logger.log_scalar('main/reward', rewards, i)

    makespan = response['makespan']
    rewards = np.array(rewards)
    result = pd.DataFrame()
    result['reward'] = rewards
    result.to_csv(reward_path, sep=',', index=None, columns=['reward'])
    print(f'Schedule makespan: {makespan}')
    return response
