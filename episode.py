import numpy as np
from argparse import ArgumentParser
import pathlib
import os
from datetime import datetime
from logger import Logger
from episode_utils import plot_reward, run_episode, test, save

parser = ArgumentParser()

parser.add_argument('--alg', type=str, default='nns')

parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--task-par', type=int, default=None)
parser.add_argument('--agent-task', type=int, default=None)
parser.add_argument('--task-par-min', type=int, default=None)
parser.add_argument('--nodes', type=np.ndarray, default=None)
parser.add_argument('--state-size', type=int, default=None)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--wfs-name', type=str, default=None)
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=1)
parser.add_argument('--actor-type', type=str, default='fc')
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def main(args):
    if args.logger:
        logger = Logger(pathlib.Path(os.getcwd()) / 'train_logs' / f'RL-agent-{datetime.now()}')
    else:
        logger = None
    if not args.is_test:
        rewards = [run_episode(ei, logger, args) for ei in range(args.num_episodes)]
        plot_reward(args, rewards)
    else:
        test(args)
    if args.save:
        save()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)











