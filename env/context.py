import numpy as np
from env.entities import volume_to_transfer, estimate_comp_times


class Context:
    def __init__(self, state_n, nodes, run_times, tree, data):
        self.n = len(run_times)  # tasks amount
        self.m = len(nodes)  # nodes amount
        self.state_n = state_n
        self.actions = [[i, j] for i in range(self.state_n) for j in range(self.m)]

        # требуемые для внутренний логики
        self.nodes = nodes  # массив с колличеством ядер на узлах
        # self.bandwidth = 0.1  # скорость передачи данных в MB = 50MB
        self.bandwidth = 10.0  # скорость передачи данных в MB = 50MB
        self.run_times = run_times  # время выполнения задач на узлах
        self.comp_times = estimate_comp_times(self.run_times, self.nodes)
        self.tree = tree  # матрица инцидентности
        self.data_map = data  # матрица зависимостей по данным (как инцидентности, только с объёмом данных в значениях)
        self.endtime_map = np.ones((self.n,
                                    self.m)) * -1  # матрица с кортежами (время старта; время финиша) для каждой задачи на нодах -1, если нет
        self.scheduled = []  # индексы уже запланированных задач
        self.schedule = self.init_schedule()

        # доп статы и метрики
        # static
        self.worst_time = self.wf_worst_schedule_time()
        self.lvls, self.task_lvls, self.height, self.widths = self.levels()
        self.max_width = max(self.widths)
        self.avg_width = sum(self.widths) / len(self.widths)
        self.width_eq_one = len(np.where(np.array(self.widths) == 1))
        self.tasks_on_lvl = np.array([self.widths[self.task_lvls[t]] for t in range(self.n)])
        self.lvl_runtime = self.levels_runtimes()
        self.t_parents = [self.parents(t) for t in range(self.n)]
        self.t_children = [self.children(t) for t in range(self.n)]
        self.t_parents_avg = np.mean([len(par) for par in self.t_parents])
        self.t_children_avg = np.mean([len(ch) for ch in self.t_children])
        self.t_parents_max = np.max([len(par) for par in self.t_parents])
        self.t_children_max = np.max([len(ch) for ch in self.t_children])
        self.runtime_sum = self.run_times.sum()
        self.input_sum = self.data_map[np.where(self.data_map > 0)].sum()
        self.runtime_med = np.median(self.run_times)
        self.input_avg = 0.0
        if self.data_map.max() > 0:
            self.input_avg = np.mean(self.data_map[np.where(self.data_map > 0)])
        self.t_input = self.tasks_input()
        self.lvl_input = self.levels_input()

        # dynamic
        self.completed = False
        self.load = np.zeros(self.m)
        self.tasks_ready = np.zeros(self.n, dtype=np.int)
        self.scheduled_tasks = np.zeros(self.n, dtype=np.int)
        self.p_on_node = np.zeros((self.n, self.m), dtype=np.int)
        self.input_on_node = np.zeros((self.n, self.m))

        self.state = []
        self.update_state()

    def init_schedule(self):
        sched = dict()
        for n in range(len(self.nodes)):
            sched[n] = []
        return sched

    def make_action(self, task_act, node):
        candidates_len = len(self.candidates)
        if task_act >= candidates_len:
            raise Exception("chosen action is not valid")
        task = self.candidates[task_act]
        placed_item = self.find_time_slot(task, node)

        # look for the worst time slot
        # time_slots = []
        # for n in range(self.m):
        #     time_slots.append(self.find_time_slot(task, n).end_time)
        # worst_time_slot = max(time_slots)

        self.scheduled.append(task)
        self.endtime_map[task, node] = placed_item.end_time
        self.schedule[node].append(placed_item)
        self.update_state()
        end_time = max(self.endtime_map.max(), 0.00001)

        reward = 0.0
        if self.completed:
            reward = self.worst_time / end_time
        else:
            reward = 0.3 * (self.worst_time / placed_item.end_time) * len(self.scheduled) / self.n
        return reward, end_time

    def is_ready_task(self, task):
        for p in self.parents(task):
            if p not in self.scheduled:
                return False
        return True

    def is_not_scheduled(self, task):
        return task not in self.scheduled

    def is_valid_action(self, task_id):
        return self.is_ready_task(task_id) and self.is_not_scheduled(task_id)

    def find_time_slot(self, task, node):
        if not self.is_valid_action(task):
            raise Exception("Not valid action")
        run_time = self.comp_times[task, node]

        node_last_time = 0.0
        node_sched = self.schedule[node]
        if len(node_sched) > 0:
            node_last_time = node_sched[-1].end_time

        parent_times = [0.0]
        for p in self.parents(task):
            p_node, = np.where(self.endtime_map[p] > -1)[0]
            p_end_time = self.endtime_map[p, p_node]
            transfer_time = 0.0
            if node != p_node:
                transfer_time = self.data_map[p, task] / self.bandwidth
            parent_time = max(node_last_time, p_end_time) + transfer_time
            parent_times.append(parent_time)

        min_start_time = max(parent_times)
        start_time = max(min_start_time, node_last_time)
        end_time = start_time + run_time
        return ScheduleItem(task, node, start_time, end_time)

    def find_task_item(self, task_id):
        for n in self.schedule.keys():
            for it in self.schedule[n]:
                if it.task == task_id:
                    return it
        return None

    # выводит непосредственных предшественников i-той задачи
    def parents(self, task):
        parents, = np.where(self.tree[task] == -1)
        return parents

    # список дочерних задач
    def children(self, task):
        childs, = np.where(self.tree[task] == 1)
        return childs

    def wf_end_time(self):
        return self.endtime_map.max()

    def wf_worst_schedule_time(self):
        worst_time = 0.0
        for i in range(self.n):
            worst_time += self.comp_times[i].max()
        worst_time += self.data_map[np.where(self.data_map > 0)].sum() / self.bandwidth
        return worst_time

    def actions_arr(self, n, m):
        actions = []
        for i in range(n):
            for j in range(m):
                actions.append([i, j])

    def levels(self):
        lvl = []
        lvl_map = np.ones(self.n, dtype=np.int) * -1

        cur_tasks = []
        for t in range(self.n):
            if self.parents(t).size == 0:
                cur_tasks.append(t)

        def tasks_levels(cur_level, cur_tasks, lvl_map):
            for cur in cur_tasks:
                if lvl_map[cur] == -1:
                    lvl_map[cur] = cur_level
                else:
                    lvl_map[cur] = max(cur_level, lvl_map[cur])
                children = self.children(cur)
                if children.size > 0:
                    tasks_levels(cur_level + 1, children, lvl_map)

        tasks_levels(0, cur_tasks, lvl_map)
        max_level = int(lvl_map.max()) + 1
        for l in range(max_level):
            lvl.append(list(np.where(lvl_map == l)[0]))
        widths = [len(arr) for arr in lvl]
        return lvl, lvl_map, max_level, widths

    def tasks_input(self):
        return np.array([self.task_input(t) for t in range(self.n)])

    def task_input(self, t):
        return -self.data_map[t, np.where(self.data_map[t] < 0)].sum()

    def levels_runtimes(self):
        levels = np.zeros(self.height)
        for l in range(self.height):
            cur = self.lvls[l]
            levels[l] = self.run_times[cur].sum()
        return levels

    def levels_input(self):
        levels = np.zeros(self.height)
        for l in range(self.height):
            cur = self.lvls[l]
            levels[l] = self.t_input[cur].sum()
        return levels

    def update_state(self):
        self.completed = len(self.scheduled) == self.n
        # nodes load (last time)
        for node in range(self.m):
            if len(self.schedule[node]) > 0:
                self.load[node] = self.schedule[node][-1].end_time
        # tasks ready to run
        self.tasks_ready = np.vectorize(self.is_valid_action)(np.arange(self.n)).astype(int)
        # tasks scheduled
        for t in range(self.n):
            if t in self.scheduled:
                self.scheduled_tasks[t] = 1
        # parents on each node and part of input volume
        for t in range(self.n):
            parents = self.parents(t)
            for node in range(self.m):
                # parents
                parents_on_node = np.intersect1d(np.where(self.endtime_map[:, node] > 0), parents)
                self.p_on_node[t, node] = len(parents_on_node)
                # volume
                if len(parents_on_node) > 0:
                    volume_on_node = self.data_map[parents_on_node, t].sum()
                    self.input_on_node[t, node] = volume_on_node / self.t_input[t]
                else:
                    self.input_on_node[t, node] = 0.0
        self.state = self.construct_state()

    def construct_state(self):
        ready_tasks = np.arange(0, len(self.tasks_ready))[np.where(self.tasks_ready == 1)]
        np.random.shuffle(ready_tasks)
        ready_tasks = ready_tasks[:self.state_n]
        ready_tasks = np.stack((self.run_times[ready_tasks], ready_tasks), axis=-1)
        ready_tasks = np.flip(np.sort(ready_tasks, axis=0).T[1].astype(int), axis=0)
        ready_len = len(ready_tasks)
        self.candidates = ready_tasks

        state = list()
        # general wfl state
        # state.append(self.n)
        state.append(len(self.scheduled) / self.n)
        # state.append(self.m)
        # state.append(self.height)
        # state.append(self.max_width)
        # state.append(self.avg_width)
        # state.append(self.width_eq_one)
        # state.append(self.worst_time)
        # state.append(self.runtime_sum)
        state.append(self.runtime_sum / self.worst_time)
        # state.append(self.runtime_med)
        state.append(self.runtime_med / self.worst_time)
        # state.append(self.input_sum)
        state.append(self.input_sum / self.bandwidth / self.worst_time)
        # state.append(self.input_avg)
        state.append(self.input_avg / self.bandwidth / self.worst_time)
        # state.append(self.t_parents_max)
        # state.append(self.t_parents_avg)
        # state.append(self.t_children_max)
        # state.append(self.t_children_avg)
        # nodes state
        for node in range(self.m):
        #     state.append(self.nodes[node])
        #     state.append(self.load[node])
            state.append(self.load[node] / self.worst_time)
        # tasks state
        for rt in range(self.state_n):
            # only task
            if rt < ready_len:
                t = ready_tasks[rt]
                # state.append(self.tasks_ready[t])
                # state.append(self.scheduled_tasks[t])
                state.append(self.run_times[t] / self.runtime_sum)
                t_input = 0.0
                if self.input_sum > 0:
                    t_input = self.t_input[t] / self.input_sum
                state.append(t_input)
                # lvl = self.task_lvls[t]
                # state.append(lvl)
                # state.append(self.tasks_on_lvl[lvl])
                # state.append(self.run_times[t] / self.lvl_runtime[lvl])
                # if self.t_input[t] > 0:
                #     state.append(self.t_input[t] / self.lvl_input[lvl])
                # else:
                #     state.append(0.0)
                # state.append(len(self.t_parents[t]))
                state.append(len(self.t_children[t]))
            else:
                # state.append(0)
                # state.append(0)
                state.append(0.0)
                state.append(0.0)
                # state.append(0)
                # state.append(0)
                # state.append(0.0)
                # state.append(0.0)
                # state.append(0)
                state.append(0)
            # task on each node
            for node in range(self.m):
                if rt < ready_len:
                    t = ready_tasks[rt]
                    state.append(self.comp_times[t, node] / self.worst_time)
                    # state.append(self.comp_times[t, node] / self.runtime_sum)
                    # state.append(self.input_on_node[t, node])
                    state.append(self.input_on_node[t, node] * self.t_input[t] / self.bandwidth / self.worst_time)
                else:
                    state.append(0.0)
                    state.append(0.0)
        return np.array(state,dtype=np.float32)

    def get_state_map(self):
        result = dict()
        idx = 0
        result['tasks'] = self.state[idx]
        idx+=1
        result['scheduled_tasks'] = self.state[idx]
        idx+=1
        result['nodes'] = self.state[idx]
        idx+=1
        result['height'] = self.state[idx]
        idx+=1
        result['max_width'] = self.state[idx]
        idx+=1
        result['avg_width'] = self.state[idx]
        idx+=1
        result['width_eq_one'] = self.state[idx]
        idx+=1
        result['worst_time'] = self.state[idx]
        idx+=1
        result['runtime_sum'] = self.state[idx]
        idx+=1
        result['runtime_median'] = self.state[idx]
        idx+=1
        result['input_sum'] = self.state[idx]
        idx+=1
        result['input_avg'] = self.state[idx]
        idx+=1
        result['max_parents'] = self.state[idx]
        idx+=1
        result['avg_parents'] = self.state[idx]
        idx+=1
        result['max_parents'] = self.state[idx]
        idx+=1
        result['avg_children'] = self.state[idx]
        idx+=1
        for node in range(self.m):
            result['n_{0}_cores'.format(node)] = self.state[idx]
            idx+=1
            result['n_{0}_load'.format(node)] = self.state[idx]
            idx+=1
        for t in range(self.state_n):
            #state.append(self.tasks_ready[t])
            result['t_{0}_valid'.format(t)] = self.state[idx]
            idx+=1
            #state.append(self.scheduled_tasks[t])
            result['t_{0}_scheduled'.format(t)] = self.state[idx]
            idx+=1
            #state.append(self.run_times[t] / self.runtime_sum)
            result['t_{0}_runtime'.format(t)] = self.state[idx]
            idx+=1
            #state.append(self.t_input[t] / self.input_sum)
            result['t_{0}_input'.format(t)] = self.state[idx]
            idx+=1
            #state.append(lvl)
            result['t_{0}_level'.format(t)] = self.state[idx]
            idx+=1
            #state.append(self.tasks_on_lvl[lvl])
            result['t_{0}_tasks_on_lvl'.format(t)] = self.state[idx]
            idx+=1
            #state.append(self.run_times[t] / self.lvl_runtime[lvl])
            result['t_{0}_runtime_to_lvl'.format(t)] = self.state[idx]
            idx+=1
            #    state.append(self.t_input[t] / self.lvl_input[lvl])
            result['t_{0}_input_to_lvl'.format(t)] = self.state[idx]
            idx+=1
            #state.append(len(self.t_parents[t]))
            result['t_{0}_parents'.format(t)] = self.state[idx]
            idx+=1
            #state.append(len(self.t_children[t]))
            result['t_{0}_children'.format(t)] = self.state[idx]
            idx+=1
            for node in range(self.m):
                result['t_{0}_n_{1}_comp_time'.format(t, node)] = self.state[idx]
                idx += 1
                result['t_{0}_n_{1}_input_on_node'.format(t, node)] = self.state[idx]
                idx += 1

        return result

    #маска валидных задач
    def get_mask(self):
        mask = np.zeros(self.state_n * self.m, dtype=np.int)
        for t in range(self.state_n):
            valid = 0
            if t < len(self.candidates):
                valid = self.is_valid_action(self.candidates[t])
            for node in range(self.m):
                mask[t * self.m + node] = valid
        return mask


class ScheduleItem:
    def __init__(self, task, node, st_time, end_time):
        self.task = task
        self.node = node
        self.st_time = st_time
        self.end_time = end_time
