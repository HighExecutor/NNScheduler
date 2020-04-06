import os
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from datetime import datetime


def plot_reward(args, rewards, heft_reward=None):
    """
    Plot history of reward on one plot for reward from algorithm based on NN and heft

    :param args:
    :param rewards:
    :param heft_reward:
    :return:
    """
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
    plt_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now().strftime("%d%b%y_%I%M%p")}_plt.png'
    plt.savefig(plt_path)

    reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now().strftime("%d%b%y_%I%M%p")}_rewards.csv'
    rewards = np.array(rewards)
    result = pd.DataFrame()
    result['reward'] = rewards
    result.to_csv(reward_path, sep=',', index=None, columns=['reward'])

    mean_reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now().strftime("%d%b%y_%I%M%p")}_mean_rewards.csv'
    means = np.array(means)
    result = pd.DataFrame()
    result['reward'] = means
    result.to_csv(mean_reward_path, sep=',', index=None, columns=['reward'])

    if heft_reward is not None:
        reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now().strftime("%d%b%y_%I%M%p")}_heft_reward.csv'
        rewards = np.array(rewards)
        result = pd.DataFrame()
        result['reward'] = rewards
        result.to_csv(reward_path, sep=',', index=None, columns=['reward'])


def plot_reward_together(args, reward0, reward1):
    """
    Plot two different reward history together

    :param args:
    :param reward0:
    :param reward1:
    :return:
    """
    cur_dir = os.getcwd()
    means0 = np.convolve(reward0, np.ones((500,)))[499:-499] / 500
    means0 = means0.tolist()

    means1 = np.convolve(reward1, np.ones((500,)))[499:-499] / 500
    means1 = means1.tolist()

    plt.style.use("seaborn-muted")
    plt.figure(figsize=(10, 5))
    plt.plot(reward0, '--', label="rewards our")
    plt.plot(means0, '-', label="avg our")

    plt.plot(reward1, '-.', label="rewards dqts")
    plt.plot(means1, '-o', label="avg dqts")

    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.legend()
    plt_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now().strftime("%d%b%y_%I%M%p")}_plt.png'
    plt.savefig(plt_path)

