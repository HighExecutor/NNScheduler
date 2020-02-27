from wf_gen_funcs import tree_data_wf, read_wf
import numpy as np

DEFAULT_CONFIG = {'task_par': 30, 'agent_task': 5, 'task_par_min': 20,
                  'nodes': np.array([4, 8, 8, 16]), 'state_size': 64,
                  'batch_size': 64, 'wfs_name': ['Montage_25'], 'seq_size': 5}


def parameter_setup(args, config):
    dict_args = vars(args)
    for key, value in dict_args.items():
        if value is not None:
            if key is 'wfs_name':
                config[key] = [value]
            else:
                config[key] = value
    return config


def wf_setup(wfs_names):
    wfs_real = [read_wf(name) for name in wfs_names]
    test_wfs = []
    test_times = dict()
    test_scores = dict()
    test_size = len(wfs_real)
    for i in range(test_size):
        wf_components = tree_data_wf(wfs_real[i])
        # tasks_n = np.random.randint(task_par_min, task_par+1)
        # wf_components = tree_data_gen(tasks_n)
        test_wfs.append(wf_components)
        test_times[i] = list()
        test_scores[i] = list()
    return test_wfs, test_times, test_scores, test_size
