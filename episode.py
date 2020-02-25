import numpy as np
from argparse import ArgumentParser
import pathlib
import os
from datetime import datetime
from logger import setup_logger
from episode_utils import plot_reward, run_episode, test, save, do_heft

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
    URL = f"http://{args.host}:{args.port}/"
    logger_nns, logger_heft = setup_logger(args)
    if args.alg == 'nns':
        if not args.is_test:
            rewards = [run_episode(ei, logger_nns, args) for ei in range(args.num_episodes)]
            plot_reward(args, rewards)
        else:
            test(args, URL)
        if args.save:
            save(URL)
    elif args.alg == 'heft':
        do_heft(args, URL, logger_heft)
    elif args.alg == 'compare':
        response = do_heft(args, URL, logger_heft)
        rewards = [run_episode(ei, logger_nns, args) for ei in range(args.num_episodes)]
        plot_reward(args, rewards, heft_reward=response['reward'])
        test(args, URL)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)











