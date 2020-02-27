from ep_utils.setups import parameter_setup, wf_setup, DEFAULT_CONFIG
import env.context as ctx
from lstm_deq import LSTMDeque
import requests
from draw_figures import write_schedule


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
