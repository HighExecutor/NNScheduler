import os
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from datetime import datetime


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
