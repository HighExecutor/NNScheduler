from ep_utils.setups import parameter_setup, DEFAULT_CONFIG
import requests
import os
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime


def do_heft(args, URL, logger):
    config = parameter_setup(args, DEFAULT_CONFIG)
    response = requests.post(f'{URL}heft', json={'wf_name': config['wfs_name'], 'nodes': config['nodes'].tolist()}).json()
    cur_dir = os.getcwd()
    reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_heft_reward.csv'
    rewards = response['reward']

    if args.logger:
        for i in range(args.num_episodes):
            logger.log_scalar('main/reward', rewards, i)

    makespan = response['makespan']
    rewards = np.array(rewards)
    result = pd.DataFrame()
    result['reward'] = rewards
    result.to_csv(reward_path, sep=',', index=None, columns=['reward'])
    print(f'Schedule makespan: {makespan}')
    return response
