from functools import partial
from pprint import pprint

from heft_deps.ScheduleBuilder import FreeSlotIterator
from heft_deps.heft_helper import HeftHelper
from heft_deps.resource_manager import Scheduler
from heft_deps.resource_manager import Node
from heft_deps.resource_manager import SoftItem
from heft_deps.resource_manager import ScheduleItem
from heft_deps.resource_manager import Schedule
from heft_deps.heft_utility import reverse_dict


# TODO: obsolete remove this test later

class StaticHeftPlanner(Scheduler):
    global_count = 0

    def __init__(self):
        super().__init__()
        self.task_rank_cache = dict()
        self.current_time = 0

    def compcost(self, job, agent):
        return self.estimator.estimate_runtime(job, agent)

    def commcost(self, ni, nj, A, B):
        return self.estimator.estimate_transfer_time(A, B, ni, nj)

    def make_ranking(self, wf, nodes):
        ##resources = self.resource_manager.get_resources()
        ##print("common nodes count:" + str(len(toNodes(resources))))
        ##nodes = HeftHelper.to_nodes(resources)
        ranking_func = HeftHelper.build_ranking_func(nodes, self.compcost, self.commcost)
        wf_jobs = ranking_func(wf)
        return wf_jobs

    def schedule(self):
        """
         create inter-priority
        """
        def byPriority(wf):
            return 0 if wf.priority is None else wf.priority

        ##simple inter priority sorting
        sorted_wfs = sorted(self.workflows, key=byPriority)
        wf_jobs = {wf: [] for wf in sorted_wfs}
        resources = self.resource_manager.get_resources()
        ##print("common nodes count:" + str(len(toNodes(resources))))
        nodes = HeftHelper.to_nodes(resources)

        wf_jobs = {wf: self.make_ranking(wf, nodes) for wf in sorted_wfs}

        ##new_schedule = self.get_unchanged_schedule(self.old_schedule, time)
        new_schedule = Schedule({node: [] for node in nodes})
        new_plan = new_schedule.mapping

        for (wf, jobs) in wf_jobs.items():
            new_schedule = self.mapping([(wf, jobs)],
                                        new_plan,
                                        nodes,
                                        self.commcost,
                                        self.compcost)
            new_plan = new_schedule.mapping

        return new_schedule

    def mapping(self, sorted_jobs, existing_plan, live_nodes, commcost, compcost):
        """def allocate(job, orders, jobson, prec, compcost, commcost):"""
        """ Allocate job to the machine with earliest finish time

        Operates in place
        """
        # TODO: add finished tasks
        jobson = dict()
        for (node, items) in existing_plan.items():
            for item in items:
                if item.state == ScheduleItem.FINISHED or item.state == ScheduleItem.EXECUTING:
                    jobson[item.job] = node

        new_plan = existing_plan

        def ft(machine):
            # cost = st(machine)
            runtime = compcost(task, machine)
            cost = st(machine, runtime) + runtime
            ##print("machine: %s job:%s cost: %s" % (machine.name, task.id, cost))
            ##print("machine: " + str(machine.name) + " cost: " + str(cost))

            return cost

        if len(live_nodes) != 0:
            ## in case if there is not any live nodes we just return the same cleaned schedule
            for wf, tasks in sorted_jobs:
                ##wf_dag = self.convert_to_parent_children_map(wf)
                wf_dag = HeftHelper.convert_to_parent_children_map(wf)
                prec = reverse_dict(wf_dag)
                for task in tasks:
                    st = partial(self.start_time, wf, task, new_plan, jobson, prec, commcost)

                    # ress = [(key, ft(key)) for key in new_plan.keys()]
                    # agent_pair = min(ress, key=lambda x: x[1][0])
                    # agent = agent_pair[0]
                    # start = agent_pair[1][0]
                    # end = agent_pair[1][1]

                    # agent = min(new_plan.keys(), key=ft)
                    agent = min(live_nodes, key=ft)
                    runtime = compcost(task, agent)
                    start = st(agent, runtime)
                    end = ft(agent)

                    # new_plan[agent].append(ScheduleItem(task, start, end))
                    Schedule.insert_item(new_plan, agent, ScheduleItem(task, start, end))

                    jobson[task] = agent

        new_sched = Schedule(new_plan)
        return new_sched

    def start_time(self, wf, task, orders, jobson, prec, commcost, node, runtime):

        ## check if soft satisfy requirements
        if self.can_be_executed(node, task):
            ## static or running virtual machine
            ## or failed it works here too
            if node.state is not Node.Down:

                if len(task.parents) == 1 and wf.head_task.id == list(task.parents)[0].id:
                    comm_ready = 0
                else:
                    parent_tasks = set()
                    for p in task.parents:
                        val = self.endtime(p, orders[jobson[p]]) + commcost(p, task, node, jobson[p])
                        parent_tasks.add(val)
                    comm_ready = max(parent_tasks)

                (st, end) = next(FreeSlotIterator(self.current_time, comm_ready, runtime, orders[node]))
                return st

                # agent_ready = orders[node][-1].end_time if orders[node] else 0
                # return max(agent_ready, comm_ready, self.current_time)
            else:
                return 1000000
        else:
            return 1000000

    def can_be_executed(self, node, job):
        ## check it
        return (job.soft_reqs in node.soft) or (SoftItem.ANY_SOFT in node.soft)

    def endtime(self, job, events):
        """ Endtime of job in list of events """
        # for e in reverse(events):
        #     if e.job.id == job.id:
        #         return e.end_time

        for e in events:
            if e.job == job:
                return e.end_time
