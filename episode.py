import numpy as np
from argparse import ArgumentParser
from logger import setup_logger
from ep_utils.draw_rewards import plot_reward
from ep_utils.do_episode import run_episode, run_dqts_episode
from ep_utils.test import test
from ep_utils.save import save
from ep_utils.heft import do_heft

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
parser.add_argument('--model-type', type=str, default='ours')
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def main(args):
    """
    Console running program.
    Using parameter args.alg you can run 3 different strategy.
    1. Run algorithm on NN if args.alg = nns
    2. Run heft algorithm if args.heft = heft
    3. Run first heft algorithm, then algorithm based on NN if args.alg = compare

    :param args:
    :return:
    """
    URL = f"http://{args.host}:{args.port}/"
    logger_nns, logger_heft = setup_logger(args)
    if args.model_type == 'ours':
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
    elif args.model_type == 'dqts':
        if not args.is_test:
            rewards = [run_dqts_episode(ei, logger_nns, args) for ei in range(args.num_episodes)]
            plot_reward(args, rewards)
        else:
            test(args, URL)
        if args.save:
            save(URL)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)











