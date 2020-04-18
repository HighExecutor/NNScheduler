import env.context as ctx
import env.dqts_context as dqts_ctx
import requests
from rnn_deq import RNNDeque
import numpy as np
from ep_utils.setups import parameter_setup, wf_setup, DEFAULT_CONFIG
from ep_utils.remember import remember
from ep_utils.replay import replay


def episode(ei, config, test_wfs, test_size, URL):
    tree, data, run_times = test_wfs[ei % test_size]
    wfl = ctx.Context(config['agent_task'], config['nodes'], run_times, tree, data)
    wfl.name = config['wfs_name'][ei % test_size]
    if config['actor_type'] == 'rnn':
        deq = RNNDeque(seq_size=config['seq_size'], size=config['state_size'])
    done = wfl.completed
    state = list(map(float, list(wfl.state)))
    if config['actor_type'] == 'rnn':
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
        if config['actor_type'] == 'rnn':
            deq.push(next_state)
            next_state = deq.show()
            next_state = next_state.tolist()
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            replay(config['batch_size'], URL)
            return reward, sars_list

    return reward, sars_list


def episode_dqts(ei, config, test_wfs, test_size, URL):
    tree, data, run_times = test_wfs[ei % test_size]
    wfl = dqts_ctx.Context(config['agent_task'], config['nodes'], run_times, tree, data)
    wfl.name = config['wfs_name'][ei % test_size]
    done = wfl.completed
    state = list(map(float, list(wfl.state)))
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
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
        state = next_state
        replay(config['batch_size'], URL)
        if done:
            return reward, sars_list
    return reward, sars_list


def run_episode(ei, logger, args):
    URL = f"http://{args.host}:{args.port}/"
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode(ei, config, test_wfs, test_size, URL)
    remember(sars_list, URL)
    if ei % 100 == 0:
        print("episode {} completed".format(ei))
    if logger is not None:
        logger.log_scalar('main/reward', reward, ei)
    return reward


def run_dqts_episode(ei, logger, args):
    URL = f"http://{args.host}:{args.port}/"
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode_dqts(ei, config, test_wfs, test_size, URL)
    remember(sars_list, URL)
    if logger is not None:
        logger.log_scalar('main/reward', reward, ei)
    return reward
