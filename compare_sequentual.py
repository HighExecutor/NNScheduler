from argparse import ArgumentParser
import numpy as np
from sequential import get_model
from DQTS import get_model as get_dqts_model
from sequential import run_episode, test
from ep_utils.draw_rewards import plot_reward_together

parser = ArgumentParser()

parser.add_argument('--state-size', type=int, default=64)
parser.add_argument('--agent-tasks', type=int, default=5)

parser.add_argument('--actor-type', type=str, default='fc')
parser.add_argument('--first-layer', type=int, default=1024)
parser.add_argument('--second-layer', type=int, default=512)
parser.add_argument('--third-layer', type=int, default=256)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--load-path', type=str, default=None)

parser.add_argument('--n_nodes', type=int, default=4)
parser.add_argument('--nodes', type=np.ndarray, default=None)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--model-name', type=str, default='')

parser.add_argument('--task-par', type=int, default=None)
parser.add_argument('--task-par-min', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None)

parser.add_argument('--wfs-name', type=str, default='CyberShake_30')
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=20)
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def main(args):
    """
    Enter point for Sequential Compare Function. Compare DQTS and our algorithm.

    :param args:
    :return:
    """
    model = get_model(args)
    dqts_model = get_dqts_model(args)
    reward = [run_episode(model, ei, args) for ei in range(args.num_episodes)]
    dqts_reward = [run_episode(dqts_model, ei, args) for ei in range(args.num_episodes)]
    plot_reward_together(args, reward, dqts_reward)
    test(model, args)
    test(dqts_model, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
