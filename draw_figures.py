import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nkx
import os


def draw_schedule_file(input_path, worst, name):
    schedule = dict()
    file = open(input_path, 'r')
    cur_r = 0
    n = 0
    for row in file.readlines():
        if 'res' in row:
            cur_r = int(row[-2])
            schedule[cur_r] = list()
        else:
            arr = row.split("_")
            task_id = int(arr[0])
            start_t = float(arr[1])
            fin_t = float(arr[2])
            schedule[cur_r].append((task_id, start_t, fin_t))
            n += 1

    file.close()
    colors = sns.color_palette("Set2", n)
    fig, ax = plt.subplots(1, figsize=(6,3))
    m = len(schedule.keys())
    keys = list(schedule.keys())
    used_colors = 0
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            print("Task {}, st {} end {}".format(it[0], it[1], it[2]))
            coords = (it[1], k)
            rect = patches.Rectangle(coords, it[2] - it[1], 1, fill=True, facecolor=colors[used_colors],
                                     label=str(it[0]), alpha=0.5, edgecolor="black")
            used_colors += 1
            ax.add_patch(rect)
            ax.text(coords[0] + (it[2] - it[1]) / 3, coords[1] + 0.5, str(it[0]), fontsize=13)

    # plt.legend()
    plt.ylim(0, m)
    plt.xlim(0, worst)
    plt.xlabel("Time, s", fontsize=16)
    plt.yticks([0.5, 1.5, 2.5, 3.5], ["R1", "R2", "R3", "R4"], fontsize=18)
    plt.title(name, fontsize=16)

    plt.tight_layout()
    plt.show()

def scores_draw_file(path):
    data = np.load(path)
    s = len(data)
    scores_avg = list()
    scores_last_avg = list()
    cur_scores = list()
    core_sum = 0.0
    avg_amount = 500
    for i in range(s):
        if i % 1000 == 0:
            print(i)
        score = data[i]
        core_sum += score
        cur_scores.append(score)
        scores_avg.append(core_sum / (i+1))
        scores_last_avg.append(np.mean(cur_scores[-avg_amount:]))

    plt.style.use("seaborn-muted")
    plt.figure(figsize=(10, 5))
    # plt.plot(scores[100:], '-*', label="scores")
    plt.plot(scores_last_avg[1000:], '--', label="avg over last {}".format(avg_amount))
    plt.plot(scores_avg[1000:], '-', label="avg")
    # plt.axhline(y=776, color='b', linestyle='-')
    plt.ylabel('reward', fontsize=18)
    plt.xlabel('episodes', fontsize=18)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    pass


def draw_schedule(run_name, test_i, wfl):
    schedule = wfl.schedule
    worst = wfl.worst_time
    n = wfl.n
    colors = sns.color_palette("Set2", n)
    fig, ax = plt.subplots(1)
    m = len(schedule.keys())
    keys = list(schedule.keys())
    used_colors = 0
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            print("Task {}, st {} end {}".format(it.task, it.st_time, it.end_time))
            coords = (it.st_time, k)
            rect = patches.Rectangle(coords, it.end_time - it.st_time, 1, fill=True, facecolor=colors[used_colors],
                                     label=it.task, alpha=0.5, edgecolor="black")
            used_colors += 1
            ax.add_patch(rect)
            ax.text(coords[0] + (it.end_time - it.st_time) / 3, coords[1] + 0.5, str(it.task))

    plt.legend()
    plt.ylim(0, m)
    plt.xlim(0, worst)
    plt.title("{}_{}".format(run_name, test_i))
    # plt.show()
    path = os.path.join(os.getcwd(), 'results')
    plt.savefig(os.path.join(path, "schedule_{}_{}.png".format(run_name, test_i)))
    plt.close()


def draw_graph(components, idx, wf_name=""):
    tree = components[0]
    g = nkx.DiGraph()
    g.add_nodes_from(range(len(tree)))
    for n in range(len(tree)):
        edges = tree[n]
        edges = [(n, v) for v in range(len(edges)) if edges[v] == 1]

        g.add_edges_from(edges)

    nkx.draw_networkx(g, with_labels=True, arrows=True)
    plt.title("WF {} {}".format(idx, wf_name))
    path = os.path.join(os.getcwd(), 'results')
    plt.savefig(os.path.join(path,"testwf_{}_graph.png".format(idx)))
    plt.close()


def write_schedule(run_name, test_i, wfl):
    path = os.path.join(os.getcwd(), 'results')
    file = open(os.path.join(path, "schedule_{}_{}".format(run_name, test_i)), 'w')
    for r in list(wfl.schedule.keys()):
        file.write("res_{}\n".format(r))
        sched = wfl.schedule[r]
        for item in sched:
            file.write("{}_{}_{}\n".format(item.task, item.st_time, item.end_time))
    file.close()
    draw_schedule(run_name, test_i, wfl)


if __name__ == '__main__':
    worst = 1400.0
    name = "Gene2life"
    draw_schedule_file(path, worst, name)

    # path = "C:\wspace\papers\ysc2019\\nns\exps\chosen\\small\\scores.npy"
    # scores_draw_file(path)