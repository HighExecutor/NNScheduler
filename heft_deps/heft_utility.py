import cProfile
import functools
import json
import operator
import os
from pprint import pprint
import pstats
import io
import time
from random import Random
from heft_deps.aggregate_utilities import interval_statistics

from heft_deps.settings import __root_path__
from heft_deps.resource_manager import ScheduleItem, Schedule
from heft_deps.DAXExtendParser import DAXParser

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def f_eq(a, b):
    """
    check equality for two float numbers
    """
    return abs(a - b) < 0.00000001


def wf(wf_name, task_postfix_id="00", deadline=1000, is_head=True):
    # dax_filepath = "../../resources/{0}.xml".format(wf_name)
    dax_filepath = "{0}/resources/{1}.xml".format(__root_path__, wf_name)
    _wf = Utility.readWorkflow(dax_filepath, wf_name, task_postfix_id, deadline=deadline, is_head=is_head)
    return _wf


def draw_heft_schedule(schedule, worst_time, n, run_name, test_i):
    m = len(schedule.keys())
    colors = sns.color_palette("Set2", n)
    fig, ax = plt.subplots(1)
    keys = list(schedule.keys())
    used_colors = 0
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            print("Task {}, st {} end {}".format(it.job, it.start_time, it.end_time))
            coords = (it.start_time, k)
            rect = patches.Rectangle(coords, it.end_time - it.start_time, 1, fill=True, facecolor=colors[used_colors],
                                     label=it.job, alpha=0.5, edgecolor="black")
            used_colors += 1
            ax.add_patch(rect)
            ax.text(coords[0] + (it.end_time - it.start_time) / 3, coords[1] + 0.5, str(it.job))

    plt.legend()
    plt.ylim(0, m)
    plt.xlim(0, worst_time)
    plt.title("{}_{}".format(run_name, test_i))
    # plt.show()
    path = os.path.join(os.getcwd(), 'results')
    plt.savefig(os.path.join(path, "schedule_{}_{}.png".format(run_name, test_i)))
    plt.close()


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        # print('{0} function took {1:0.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret

    return wrap


class RepeatableTiming:
    def __init__(self, repeat_count):
        self._repeat_count = repeat_count

    def __call__(self, func):
        def wrap(*args, **kwargs):
            def measure():
                time1 = time.time()
                x = func(*args, **kwargs)
                time2 = time.time()
                return (time2 - time1) * 1000.0

            measures = [measure() for _ in range(self._repeat_count)]
            mean, mn, mx, std, left, right = interval_statistics(measures)
            print("Statistics - mean: {0}, min: {1}, max: {2}, std: {3}, left: {4}, right: {5} by {6} runs".format(mean,
                                                                                                                   mn,
                                                                                                                   mx,
                                                                                                                   std,
                                                                                                                   left,
                                                                                                                   right,
                                                                                                                   self._repeat_count))

            return func(*args, **kwargs)

        return wrap


def profile_decorator(func):
    def wrap_func(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        # =============
        result = func(*args, **kwargs)
        # =============
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result

    return wrap_func


def reverse_dict(d):
    """ Reverses direction of dependence dict
    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
    """
    result = {}
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key,)
    return result


class GraphVisualizationUtility:
    @staticmethod
    def visualize_task_node_mapping(wf, schedule):
        import matplotlib.pyplot as plt
        import networkx

        def extract_edges_and_vertex(parent, edge_set, vertex_set):
            for child in parent.children:
                vertex_set.add(child.id)
                edge_set.add((parent.id, child.id))
                extract_edges_and_vertex(child, edge_set, vertex_set)
                pass
            pass

        def get_task_node_mapping(schedule):
            result = {i.job.id: node.name for node, items in schedule.mapping.items() for i in items}
            return result

        def draw_graph():
            graph = networkx.DiGraph()
            edge_set = set()
            vertex_set = set()
            extract_edges_and_vertex(wf.head_task, edge_set, vertex_set)
            edge_set = filter(lambda x: False if x[0] == wf.head_task.id else True, edge_set)
            vertex_set = filter(lambda x: x == wf.head_task.id, vertex_set)
            tnmap = get_task_node_mapping(schedule)
            for v in vertex_set:
                graph.add_node(v)
            for v1, v2 in edge_set:
                graph.add_edge(v1, v2)
            labels = dict((t, str(t) + "/" + str(n)) for t, n in tnmap.items())
            # networkx.draw(graph)
            networkx.draw(graph, labels=labels)
            plt.show()
            pass

        draw_graph()
        pass


def tracing(func):
    def wrap(*args, **kwargs):
        print("function {0} started".format(func.__name__))
        result = func(*args, **kwargs)
        print("function {0} finished".format(func.__name__))
        return result

    return wrap


def signal_if_true(func):
    def wrap(*args, **kwargs):
        x = func(*args, **kwargs)
        if isinstance(x, bool):
            if x is True:
                print("Event {0} appeared".format(func.__name__))
            return x
        else:
            raise ValueError("result of function {0} is not of a boolean type".format(func.__name__))

    return wrap


class Utility:
    MIN_PIPELINE_SIZE = 10
    MAX_PIPELINE_SIZE = 40

    def __init__(self):
        pass

    @staticmethod
    def is_enough_to_be_executed(wf, t_id, completed_tasks_ids):
        pids = [p.id for p in wf.byId(t_id).parents if p != wf.head_task]
        return all(pid in completed_tasks_ids for pid in pids)

    @staticmethod
    def get_default_bundle():
        ## dedicated resource are the same for all bundles
        _wf = wf('CyberShake_30')
        path = '{0}/resources/saved_schedules/CyberShake_30_bundle_backup.json'.format(__root_path__)
        bundle = Utility.load_schedule(path, _wf)
        return bundle

    @staticmethod
    def generateUrgentPipeline(dax_filepath, wf_name, wf_start_id, task_postfix_id, deadline):
        parser = DAXParser()
        random = Random()
        pipelineSize = 1  ##random.randint(Utility.MIN_PIPELINE_SIZE,Utility.MAX_PIPELINE_SIZE)
        wfs = [parser.parseXml(dax_filepath, wf_start_id + str(i), task_postfix_id + str(i), wf_name) for i in
               range(0, pipelineSize)]
        for wf in wfs:
            wf.deadline = deadline
        return wfs

    @staticmethod
    def readWorkflow(dax_filepath, wf_name, wf_start_id="00", task_postfix_id="00", deadline=1000, is_head=True):
        parser = DAXParser()
        wf = parser.parseXml(dax_filepath, wf_start_id + "0", task_postfix_id + "0", wf_name, is_head=is_head)
        wf.deadline = deadline
        return wf

    @staticmethod
    def validate_time_seq(items):
        time = -1
        for item in items:
            if time > item.start_time:
                return False
                # raise Exception("Node: " + str(node) + " all time: " + str(time) + " st_time: " + str(item.start_time))
            else:
                time = item.start_time
            if time > item.end_time:
                return False
            else:
                time = item.end_time
        return True

    @staticmethod
    def validateNodesSeq(schedule):
        for (node, items) in schedule.mapping.items():
            result = Utility.validate_time_seq(items)
            if result is False:
                return False
        return True

    ## TODO: under development now
    @staticmethod
    def validateParentsAndChildren(schedule, workflow, AllUnstartedMode=False, RaiseException=False):
        INCORRECT_SCHEDULE = "Incorrect schedule"
        # {
        #   task: (node,start_time,end_time),
        #   ...
        # }
        task_to_node = dict()
        for (node, items) in schedule.mapping.items():
            for item in items:
                seq = task_to_node.get(item.job.id, [])
                seq.append(item)
                ##seq.append(node, item.start_time, item.end_time, item.state)
                task_to_node[item.job.id] = seq

        def check_failed(seq):
            ## in schedule items sequence, only one finished element must be
            ## resulted schedule can contain only failed and finished elements
            states = [item.state for item in seq]
            if AllUnstartedMode:
                if len(states) > 1 or states[0] != ScheduleItem.UNSTARTED:
                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
            else:
                if states[-1] != ScheduleItem.FINISHED:
                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
                finished = [state for state in states if state == ScheduleItem.FINISHED]
                if len(finished) != 1:
                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
                failed = [state for state in states if state == ScheduleItem.FAILED]
                if len(states) - len(finished) != len(failed):
                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
            return True

        task_to_node = {job_id: sorted(seq, key=lambda x: x.start_time) for (job_id, seq) in task_to_node.items()}
        for (job_id, seq) in task_to_node.items():
            result = Utility.validate_time_seq(seq)
            if result is False:
                if RaiseException:
                    raise Exception(INCORRECT_SCHEDULE)
                else:
                    return False
            if check_failed(seq) is False:
                if RaiseException:
                    raise Exception(INCORRECT_SCHEDULE)
                else:
                    return False

        def check(task):
            for child in task.children:
                p_end_time = task_to_node[task.id][-1].end_time
                c_start_time = task_to_node[child.id][-1].start_time
                if c_start_time < p_end_time:

                    # TODO: debug
                    print("Parent task: ", task.id)
                    print("Child task: ", child.id)

                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
                res = check(child)
                if res is False:
                    if RaiseException:
                        raise Exception(INCORRECT_SCHEDULE)
                    else:
                        return False
            return True

        for task in workflow.head_task.children:
            res = check(task)
            if res is False:
                if RaiseException:
                    raise Exception(INCORRECT_SCHEDULE)
                else:
                    return False
        return True

    @staticmethod
    def is_static_schedule_valid(_wf, schedule):
        try:
            Utility.validate_static_schedule(_wf, schedule)
        except:
            return False
        return True

    @staticmethod
    def makespan(schedule):
        def get_last_time(node_items):
            return 0 if len(node_items) == 0 else node_items[-1].end_time

        last_time = max([get_last_time(node_items) for (node, node_items) in schedule.mapping.items()])
        return last_time

    @staticmethod
    def overall_transfer_time(schedule, wf, estimator):
        """
        This method extracts OVERALL transfer time during execution.
        It should be noted that common_transfer_time + common_execution_time != makespan,
        due to as transfer as execution can be executed in parallel, so overlap may occur.
        """
        t_n = schedule.task_to_node()
        # t_n = {task.id: node for task, node in t_n.items()}
        tasks = wf.get_all_unique_tasks()

        def calc(p, child):
            return estimator.estimate_transfer_time(t_n[p], t_n[child], p, child)

        relations_iter = (calc(p, child) for p in tasks if p != wf.head_task.id for child in p.children)
        transfer_time = functools.reduce(operator.add, relations_iter)
        return transfer_time

    @staticmethod
    def overall_execution_time(schedule):
        """
        This method extracts OVERALL execution time during execution.
        It should be noted that common_transfer_time + common_execution_time != makespan,
        due to as transfer as execution can be executed in parallel, so overlap may occur.
        """
        execution_iters = (item.end_time - item.start_time for node, items in schedule.mapping.items() for item in
                           items)
        execution_time = functools.reduce(operator.add, execution_iters)
        return execution_time

    @staticmethod
    def load_schedule(path, wf):
        decoder = Utility.build_bundle_decoder(wf.head_task)
        f = open(path, 'r')
        bundle = json.load(f, object_hook=decoder)
        f.close()
        return bundle

    @staticmethod
    def check_and_raise_for_fixed_part(resulted_schedule, fixed_schedule_part, current_time):

        # TODO: Urgent! make a check for consistency with fixed schedule

        fpart_check = Utility.check_fixed_part(resulted_schedule, fixed_schedule_part, current_time)

        # TODO: Urgent! make a check for abandoning of presence of duplicated tasks with state Finished, unstarted,
        #  executing

        duplicated_check = Utility.check_duplicated_tasks(resulted_schedule)

        if fpart_check is False:
            raise Exception("check for consistency with fixed schedule didn't pass")
        else:
            print("Time: " + str(current_time) + " fpart_check passed")
        if duplicated_check is False:
            raise Exception("check for duplicated tasks didn't pass")
        else:
            print("Time: " + str(current_time) + " duplicated_check passed")
        pass

    @staticmethod
    def check_fixed_part(schedule, fixed_part, current_time):
        def item_equality(item1, fix_item):

            is_equal = item1.state == fix_item.state
            not_finished = (fix_item.state == ScheduleItem.UNSTARTED or fix_item.state == ScheduleItem.EXECUTING)
            is_finished_now = (
                    not_finished and item1.state == ScheduleItem.FINISHED and fix_item.end_time <= current_time)
            is_executing_now = (
                    not_finished and item1.state == ScheduleItem.EXECUTING and fix_item.start_time <= current_time <= fix_item.end_time)
            is_state_correct = is_equal or is_finished_now or is_executing_now

            return item1.job.id == fix_item.job.id and is_state_correct and item1.start_time == fix_item.start_time and item1.end_time == fix_item.end_time

        for (node, items) in fixed_part.mapping.items():

            # TODO: need to make here search by node.name

            itms = schedule.mapping[node]
            for i in range(len(items)):
                if not item_equality(itms[i], items[i]):
                    return False
        return True

    @staticmethod
    def check_duplicated_tasks(schedule):
        task_instances = dict()
        for (node, items) in schedule.mapping.items():
            for item in items:
                instances = task_instances.get(item.job.id, [])
                instances.append((node, item))
                task_instances[item.job.id] = instances

        for (id, items) in task_instances.items():
            sts = [item.state for (node, item) in items]
            inter_excluded_states = list(filter(
                lambda x: x == ScheduleItem.FINISHED or x == ScheduleItem.EXECUTING or x == ScheduleItem.UNSTARTED,
                sts))
            if len(inter_excluded_states) > 1:
                return False
            pass
        return True
