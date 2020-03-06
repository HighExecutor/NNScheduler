from actor import DQNActor
from lstm_deq import LSTMDeque
from argparse import ArgumentParser
from ep_utils.setups import parameter_setup, DEFAULT_CONFIG
import numpy as np
import env.context as ctx
from ep_utils.setups import wf_setup
from draw_figures import write_schedule
from interective import ScheduleInterectivePlotter
from copy import deepcopy

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
parser.add_argument('--num-episodes', type=int, default=1)
parser.add_argument('--logger', type=bool, default=True)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')


def get_model(args):
    action_size = args.agent_tasks * args.n_nodes
    if not args.model_name:
        model_name = 'model_fc.h5' if not args.actor_type=='fc' else 'model_lstm.h5'
    if not args.load:
        return DQNActor(first=args.first_layer, second=args.second_layer, third=args.third_layer,
                        state_size=args.state_size, action_size=action_size,
                        seq_size=args.seq_size, actor_type=args.actor_type)
    else:
        model = DQNActor(first=args.first_layer, second=args.second_layer, third=args.third_layer,
                         state_size=args.state_size, action_size=action_size,
                         seq_size=args.seq_size, actor_type=args.actor_type)
        if args.load_path is not '':
            model.load(model_name, path=args.load_path)
        else:
            model.load(model_name)
        return model


def episode(model, ei, config, test_wfs, test_size):
    ttree, tdata, trun_times = test_wfs[ei % test_size]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    wfl.name = config['wfs_name'][ei % test_size]
    if config['actor_type'] == 'lstm':
        deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
    done = wfl.completed
    state = wfl.state
    if config['actor_type'] == 'lstm':
        deq.push(state)
        state = deq.show()
    sars_list = list()
    reward = 0
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = wfl.get_mask()
        # action = requests.post(f"{URL}act", json={'state': state, 'mask': mask, 'sched': False}).json()['action']
        action = model.act(state.reshape(1, state.shape[0]), mask, False)
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = wfl.state
        if config['actor_type'] == 'lstm':
            deq.push(next_state)
            next_state = deq.show()
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return reward, sars_list
    return reward, sars_list


def remember(model, sars_list, args):
    for sarsa in sars_list:
        if args.actor_type == 'fc':
            state = sarsa[0].reshape(1, model.STATE)
            next_state = sarsa[3].reshape(1, model.STATE)
        elif args.actor_type == 'lstm':
            state = sarsa[0].reshape((1, model.seq_size, model.STATE))
            next_state = sarsa[3].reshape((1, model.seq_size, model.STATE))
        action = int(sarsa[1])
        reward = float(sarsa[2])
        done = bool(sarsa[4])
        model.remember((state, action, reward, next_state, done))


def replay(model, batch_size):
    model.replay(batch_size)


def run_episode(model, ei, args):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode(model, ei, config, test_wfs, test_size)
    remember(model, sars_list, args)
    replay(model, config['batch_size'])
    return reward


def test(model, args):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        if config['actor_type'] == 'lstm':
            deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
        done = wfl.completed
        state = wfl.state
        if config['actor_type'] == 'lstm':
            deq.push(state)
            state = deq.show()
        for time in range(wfl.n):
            mask = wfl.get_mask()
            action = model.act(state.reshape(1, state.shape[0]), mask, False)
            act_t, act_n = wfl.actions[action]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = wfl.state
            if config['actor_type'] == 'lstm':
                deq.push(next_state)
                next_state = deq.show()
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
        write_schedule(args.run_name, i, wfl)


def interective_test(model, args):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        sch = ScheduleInterectivePlotter(wfl.worst_time, wfl.m, wfl.n)
        wfl.name = config['wfs_name'][i]
        if config['actor_type'] == 'lstm':
            deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
        done = wfl.completed
        state = wfl.state
        for time in range(wfl.n):
            mask = wfl.get_mask()
            q = model.act_q(state.reshape(1, state.shape[0]), mask, False)
            q = np.squeeze(q, axis=0) if len(q.shape) > 1 else q
            action_idx = np.argmax(q)
            actions = [wfl.actions[action] for action in range(q.shape[-1])]
            best_t, best_n = actions[action_idx]
            copies_of_wfl = [deepcopy(wfl) for _ in range(len(actions))]
            reward, wf_time = wfl.make_action(best_t, best_n)
            next_state = wfl.state

            acts = []
            for idx, action in enumerate(actions):
                wfl_copy = copies_of_wfl[idx]
                t, n = action
                if q[idx] != 0 or idx == action_idx:
                    reward, wf_time, item = wfl_copy.make_action_item(t, n)
                    acts.append((item, reward, n))
            sch.draw_item(wfl.schedule, acts)
            if config['actor_type'] == 'lstm':
                deq.push(next_state)
                next_state = deq.show()
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
        write_schedule(args.run_name, i, wfl)


def main(args):
    model = get_model(args)
    reward = [run_episode(model, ei, args) for ei in range(args.num_episodes)]
    interective_test(model, args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
