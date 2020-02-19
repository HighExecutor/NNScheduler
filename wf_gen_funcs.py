from env.entities import Node, Workflow, Cluster, volume_to_transfer, estimate_comp_times
from env.context import Context
from utilities.DAXParser import read_workflow

import numpy as np
import os
import pathlib


def runtime_gen(size):
    return np.random.exponential(scale=400, size=size)


def input_gen(size):
    return np.random.weibull(a=1, size=size)*100 #used earlier
    # return np.random.weibull(a=1, size=size)*10


def tree_data_gen(tree_length):
    tree = np.zeros((tree_length, tree_length), dtype=np.int)
    data = np.zeros((tree_length, tree_length), dtype=np.float)
    for j in range(len(tree)):
        tree[j, (j + 1):tree_length] = np.random.randint(0, 2, len(tree[j, (j + 1):len(tree)]))
        # tree[j, (j + 1):tree_length] = runtime_gen(len(tree[j, (j + 1):len(tree)]))
        # data[j, (j + 1):tree_length] = tree[j, (j + 1):tree_length] * (np.random.rand(len(tree) - j - 1) * 90 + 10)
        data[j, (j + 1):tree_length] = tree[j, (j + 1):tree_length] * input_gen(len(tree) - j - 1)
    tree = tree - np.transpose(tree)
    data = data - np.transpose(data)
    run_times = runtime_gen(tree_length)
    return tree, data, run_times


def tree_data_wf(wf):
    tasks = wf.tasks
    n = len(tasks)
    tree = np.zeros((n, n), dtype=np.int)
    data = np.zeros((n, n), dtype=np.float)
    run_times = np.zeros(n, dtype=np.float)
    task_ids = np.array([t.id for t in tasks])
    for i in range(n):
        run_times[i] = tasks[i].runtime
        children = tasks[i].children
        c_ids = [c.id for c in children]
        for c in c_ids:
            c_i = np.where(task_ids == c)
            data_req = volume_to_transfer(tasks[c_i[0][0]], tasks[i])
            tree[i, c_i] = 1
            data[i, c_i] = data_req

    tree = tree - np.transpose(tree)
    data = data - np.transpose(data)
    return tree, data, run_times


def read_wf(wf_name):
    # wf_name = "CyberShake_30"
    project_path = pathlib.Path(os.getcwd())
    resources_path = os.path.join(project_path, 'resources')
    wf_path = os.path.join(resources_path,"{0}.xml".format(wf_name))
    return read_workflow(wf_path, wf_name)


if __name__ == "__main__":
    wf_name = "Montage_25"
    # wf_name = "CyberShake_30"
    project_path = pathlib.Path(os.getcwd())
    resources_path = os.path.join(project_path, 'resources')
    wf_path = os.path.join(resources_path, "{0}.xml".format(wf_name))
    wfl = read_workflow(wf_path, wf_name)

    nodes = np.array([4, 8, 16])  # железно
    tree, data, run_times = tree_data_wf(wfl)
    # tree, data, run_times = tree_data_gen(20)

    wf = Context(nodes, run_times, tree, data)
    for i in range(20):
        mask = wf.get_mask()
        wf.make_action(i, np.random.randint(3))
        print(wf.wf_end_time())
    import pprint
    pprint.pprint(wf.get_state_map())