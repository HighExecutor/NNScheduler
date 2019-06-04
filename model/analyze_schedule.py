import numpy as np
import env.context as ctx
from wf_gen_funcs import tree_data_wf, read_workflow
import actor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_schedule(wfl):
    schedule = wfl.schedule
    worst = wfl.worst_time
    n = wfl.n
    colors = ["r", "g", "b", "yellow", "orange", "purple", "brown", "violet"]
    fig, ax = plt.subplots(1)
    m = len(schedule.keys())
    keys = list(schedule.keys())
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            print("Task {}, st {} end {}".format(it.task, it.st_time, it.end_time))
            coords = (it.st_time, k)
            rect = patches.Rectangle(coords, it.end_time - it.st_time, 1, fill=True, facecolor="r", label=it.task, alpha=0.5, edgecolor="black")
            ax.add_patch(rect)
            ax.text(coords[0] + (it.end_time-it.st_time)/3, coords[1]+0.5, str(it.task))

    plt.legend()
    plt.ylim(0,m)
    plt.xlim(0, worst)
    plt.title(wfl.wf_name)
    plt.show()

if __name__ == "__main__":

    task_par = 10
    task_par_min = 8
    proc_par = 2
    state_size = 87
    action_size = task_par * proc_par
    agent = actor.DQNAgent(state_size, action_size)
    # agent.load("model_10000.h5")
    # agent.load("model_10000.h5")
    agent.load("model_50000.h5")
    # agent.load("model_0.h5")

    nodes = np.array([4, 8])

    wf_name = "floodplain"
    wf_path = "..\\resources\\{0}.xml".format(wf_name)
    wfl = read_workflow(wf_path, wf_name)

    tree, data, run_times = tree_data_wf(wfl)
    wfl = ctx.Context(nodes, run_times, tree, data)

    done = wfl.completed
    state = wfl.state
    state = np.reshape(state, [1, state_size])
    for i in range(wfl.n):
        mask = wfl.get_mask()
        action = agent.act(state, mask, True)
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = wfl.state
        done = wfl.completed
        next_state = np.reshape(next_state, [1, state_size])
        if done:
            print("Completed")
            wfl.wf_name=wf_name
            draw_schedule(wfl)
            break