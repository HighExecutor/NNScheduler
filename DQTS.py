from actor import DQNActor
from rnn_deq import RNNDeque
from argparse import ArgumentParser
from ep_utils.setups import parameter_setup, DEFAULT_CONFIG
import numpy as np
import env.context as ctx
from ep_utils.setups import wf_setup
from draw_figures import write_schedule
from interactive import ScheduleInterectivePlotter
from copy import deepcopy
from ep_utils.draw_rewards import plot_reward
from sequential import run_episode

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

parser.add_argument('--wfs-name', type=str, default=None)
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=20)
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def get_model(args):
    action_size = args.agent_tasks * args.n_nodes
    model_name = 'model_dqts.h5'
    if not args.load:
        return DQNActor(first=20, second=20, third=20,
                        state_size=args.state_size, action_size=action_size,
                        seq_size=args.seq_size, actor_type='fc')
    else:
        model = DQNActor(first=20, second=20, third=20,
                         state_size=args.state_size, action_size=action_size,
                         seq_size=args.seq_size, actor_type='fc')
        if args.load_path is not '':
            model.load(model_name, path=args.load_path)
        else:
            model.load(model_name)
        return model


def main(args):
    """
    Enter point for DQTS algorithm run

    :param args:
    :return:
    """
    model = get_model(args)
    reward = [run_episode(model, ei, args) for ei in range(args.num_episodes)]
    plot_reward(args, reward)
    # interective_test(model, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
