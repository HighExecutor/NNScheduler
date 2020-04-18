import numpy as np


def volume_to_transfer(task1, task2):
    input = task1.input_files
    output = task2.output_files
    req_files = input.keys() & output.keys()
    volume = sum([input[f].size for f in list(req_files)])
    return volume / 1024 / 1024 # from B to MB


# составление матрицы оценочного времени выполнения задач на нодах в зависимости от количества ядер
def estimate_comp_times(run_times, nodes):
    n = len(run_times)
    m = len(nodes)
    comp_times = np.empty((m, n))
    for i in range(m):
        comp_times[i] = run_times / np.sqrt(nodes[i] / 8)
    return comp_times.T


class Node:
    def __init__(self, id, cores):
        self.id = id
        self.cores = cores


class Cluster:
    def __init__(self):
        self.nodes = list()
        self.id_to_node = dict()

    def add_node(self, node):
        self.nodes.append(node)
        self.id_to_node[node.id] = node


class Workflow:
    def __init__(self, id, head_task):
        self.id = id
        self.head_task = head_task
        self.tasks = self.get_all_tasks()
        self.id_to_task = self.get_id_to_task_map()

    def get_all_tasks(self):
        """
        Get all unique tasks in sorted order
        """

        def add_tasks(tasks, task):
            tasks.update(task.children)
            for child in task.children:
                add_tasks(tasks, child)

        tasks_set = set()
        if self.head_task is None:
            return tasks_set
        else:
            add_tasks(tasks_set, self.head_task)
        return sorted(list(tasks_set), key=lambda x: x.id)

    def get_id_to_task_map(self):
        id_map = dict()
        for t in self.tasks:
            if not t.is_head:
                id_map[t.id] = t
        return id_map


class Task:
    def __init__(self, id, is_head=False):
        self.id = id
        self.parents = set()  # set of parents tasks
        self.children = set()  # set of children tasks
        self.runtime = None  # flops for calculating
        self.input_files = None
        self.output_files = None
        self.is_head = is_head


class File:
    def __init__(self, name, size):
        self.name = name
        self.size = size
