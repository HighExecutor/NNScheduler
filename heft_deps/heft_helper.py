from functools import partial
from heft_deps.resource_manager import Scheduler, ScheduleItem, Schedule


class HeftHelper(Scheduler):

    @staticmethod
    def heft_rank(wf, rm, estimator):
        nodes = rm.get_nodes()
        ranking = HeftHelper.build_ranking_func(nodes,
                                                lambda job, agent: estimator.estimate_runtime(job, agent),
                                                lambda ni, nj, A, B: estimator.estimate_transfer_time(A, B, ni, nj))
        sorted_tasks = [t.id for t in ranking(wf)]
        return sorted_tasks

    @staticmethod
    def to_nodes(resources):
        result = set()
        for resource in resources:
            result.update(resource.nodes)
        result = list(sorted(result, key=lambda x: x.name))
        return result

    @staticmethod
    def build_ranking_func(nodes, compcost, commcost):
        task_rank_cache = dict()

        def ranking_func(wf):
            wf_dag = HeftHelper.convert_to_parent_children_map(wf)
            rank = partial(HeftHelper.ranking, nodes=nodes, succ=wf_dag,
                           compcost=compcost, commcost=commcost,
                           task_rank_cache=task_rank_cache)
            jobs = set(wf_dag.keys()) | set(x for xx in wf_dag.values() for x in xx)

            # TODO: sometimes sort gives different results
            # TODO: it's normal because of only elements with the same rank change their place
            # TODO: relatively each other with the same rank
            # TODO: need to get deeper understanding of this situation

            jobs = sorted(jobs, key=rank)

            return list(reversed(jobs))

        return ranking_func

    @staticmethod
    def ranking(ni, nodes, succ, compcost, commcost, task_rank_cache):

        w = partial(HeftHelper.avr_compcost, compcost=compcost, nodes=nodes)
        c = partial(HeftHelper.avr_commcost, nodes=nodes, commcost=commcost)

        def estimate(ni):
            result = task_rank_cache.get(ni, None)
            if result is not None:
                return result
            if ni in succ and succ[ni]:

                # the last component cnt(ni)/nodes.len is needed to account
                # software restrictions of particular task
                # and

                # TODO: include the last component later

                result = w(ni) + max(c(ni, nj) + estimate(nj) for nj in
                                     succ[ni])  ##+ math.pow((nodes.len - cnt(ni)),2)/nodes.len - include it later.
            else:
                result = w(ni)
            task_rank_cache[ni] = result
            return result

        """print( "%s %s" % (ni, result))"""
        result = estimate(ni)
        if hasattr(ni, 'priority'):
            if ni.priority > 0:
                result += pow(120, ni.priority)
            result = float(round(result, 5)) + HeftHelper.get_seq_number(ni)
        else:
            result = int(round(result, 5) * 1000000) + HeftHelper.get_seq_number(ni)
        return result

    @staticmethod
    def get_seq_number(task):
        # It is assumed that task.id have only one format ID000[2 digits number]_000
        id = task.id
        number = id[5:7]
        return int(number)

    @staticmethod
    def avr_compcost(ni, nodes, compcost):
        """ Average computation cost """
        return sum(compcost(ni, node) for node in nodes) / len(nodes)

    @staticmethod
    def avr_commcost(ni, nj, nodes, commcost):
        # TODO: remake it later.
        # return 10
        """ Average communication cost """
        n = len(nodes)
        if n == 1:
            return 0
        npairs = n * (n - 1)
        return 1. * sum(commcost(ni, nj, a1, a2) for a1 in nodes for a2 in nodes
                        if a1 != a2) / npairs

    @staticmethod
    def convert_to_parent_children_map(wf):
        head = wf.head_task
        map = dict()

        def mapp(parents, map):
            for parent in parents:
                st = map.get(parent, set())
                st.update(parent.children)
                map[parent] = st
                mapp(parent.children, map)

        mapp(head.children, map)
        return map

    @staticmethod
    def get_all_tasks(wf):
        map = HeftHelper.convert_to_parent_children_map(wf)
        tasks = [task for task in map.keys()]
        return tasks

    @staticmethod
    def clean_unfinished(schedule):
        def clean(items):
            return [item for item in items if
                    item.state == ScheduleItem.FINISHED or item.state == ScheduleItem.EXECUTING]

        new_mapping = {node: clean(items) for (node, items) in schedule.mapping.items()}
        return Schedule(new_mapping)

    @staticmethod
    def get_tasks_for_planning(wf, schedule):
        # TODO: remove duplicate code later
        def clean(items):
            return [item.job for item in items if
                    item.state == ScheduleItem.FINISHED or item.state == ScheduleItem.EXECUTING]

        def get_not_for_planning_tasks(schedule):
            result = set()
            for (node, items) in schedule.mapping.items():
                unfin = clean(items)
                result.update(unfin)
            return result

        all_tasks = HeftHelper.get_all_tasks(wf)
        not_for_planning = get_not_for_planning_tasks(schedule)
        # def check_in_not_for_planning(tsk):
        #     for t in not_for_planning:
        #         if t.id == tsk.id:
        #             return True
        #     return False
        # for_planning = [tsk for tsk in all_tasks if not(check_in_not_for_planning(tsk))]
        for_planning = set(all_tasks) - set(not_for_planning)
        return for_planning

    pass
