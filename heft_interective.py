from heft_deps.ExperimentalManager import ExperimentResourceManager, ModelTimeEstimator
from heft_deps.resource_generator import ResourceGenerator as rg
from heft_deps.heft_utility import wf, Utility, draw_heft_schedule
from heft_deps.heft_settings import run_heft
import env.context as ctx
from ep_utils.setups import wf_setup


def heft(wf_name, nodes):
    """
    Heft algorithm

    :return:
    """
    rm = ExperimentResourceManager(rg.r(nodes))
    estimator = ModelTimeEstimator(bandwidth=10)
    _wf = wf(wf_name[0])
    heft_schedule = run_heft(_wf, rm, estimator)
    actions = [(proc.start_time, int(proc.job.global_id), node.name_id)
               for node in heft_schedule.mapping
               for proc in heft_schedule.mapping[node]]
    actions = sorted(actions, key=lambda x: x[0])
    actions = [(action[1], action[2]) for action in actions]

    test_wfs, test_times, test_scores, test_size = wf_setup(wf_name)
    ttree, tdata, trun_times = test_wfs[0]
    wfl = ctx.Context(len(_wf.get_all_unique_tasks()), nodes, trun_times, ttree, tdata)
    reward = 0
    end_time = 0
    for task, node in actions:
        task_id = wfl.candidates.tolist().index(task)
        reward, end_time = wfl.make_action(task_id, node)

    draw_heft_schedule(heft_schedule.mapping, wfl.worst_time, len(actions), 'h', '1')
    response = {'reward': reward, 'makespan': end_time}
    return reward, end_time