from heft_deps.heft_helper import HeftHelper
from heft_deps.resource_manager import Node
from heft_deps.resource_manager import Schedule, ScheduleItem
from heft_deps.heft_utility import timing
from heft_deps.simple_heft import StaticHeftPlanner

class DynamicHeft(StaticHeftPlanner):
    executed_tasks = set()

    def get_nodes(self):
        resources = self.resource_manager.get_resources()
        nodes = HeftHelper.to_nodes(resources)
        return nodes
        # return self.resource_manager.get_nodes()

    def __init__(self, workflow, resource_manager, estimator, ranking=None):
        self.current_schedule = Schedule(dict())
        self.workflow = workflow
        self.resource_manager = resource_manager
        self.estimator = estimator
        self.ranking = ranking

        self.current_time = 0

        nodes = self.get_nodes()



        # print("A: " + str(self.wf_jobs))

        #TODO: remove it later
        # to_print = ''
        # for job in self.wf_jobs:
        #     to_print = to_print + str(job.id) + " "
        # print(to_print)
        pass

    @timing
    def run(self, current_cleaned_schedule):
        ## current_cleaned_schedule - this schedule contains only
        ## finished and executed tasks, all unfinished and failed have been removed already
        ## current_cleaned_schedule also have down nodes and new added
        ## ALGORITHM DOESN'T CHECK ADDING OF NEW NODES BY ITSELF
        ## nodes contain only available now

        ## 1. get all unscheduled tasks
        ## 2. sort them by rank
        ## 3. map on the existed nodes according to current_cleaned_schedule

        nodes = self.get_nodes()
        live_nodes = [node for node in nodes if node.state != Node.Down]

        for_planning = HeftHelper.get_tasks_for_planning(self.workflow, current_cleaned_schedule)
        ## TODO: check if it sorted properly
        for_planning = set([task.id for task in for_planning])

        self.wf_jobs = self.make_ranking(self.workflow, live_nodes) if self.ranking is None else self.ranking

        sorted_tasks = [task for task in self.wf_jobs if task.id in for_planning]

        # print("P: " + str(sorted_tasks))

        new_sched = self.mapping([(self.workflow, sorted_tasks)], current_cleaned_schedule.mapping, nodes, self.commcost, self.compcost)
        return new_sched

    def endtime(self, job, events):
        """ Endtime of job in list of events """
        for e in events:
            if e.job == job and (e.state == ScheduleItem.FINISHED or e.state == ScheduleItem.EXECUTING or e.state == ScheduleItem.UNSTARTED):
                return e.end_time


def run_heft(workflow, resource_manager, estimator):
    """
    It simply runs src with empty initial schedule
    and returns complete schedule
    """
    heft = DynamicHeft(workflow, resource_manager, estimator)
    nodes = resource_manager.get_nodes()
    init_schedule = Schedule({node: [] for node in nodes})
    return heft.run(init_schedule)