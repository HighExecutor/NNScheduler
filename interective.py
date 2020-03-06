import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt
from ep_utils.setups import parameter_setup, DEFAULT_CONFIG, wf_setup
import env.context as ctx
from lstm_deq import LSTMDeque
import requests

import numpy as np
import threading

matplotlib.use("TkAgg")


class ScheduleInterectivePlotter(object):
    def __init__(self, worst_time, k, n):

        self.worst_time = worst_time
        self.k = k
        self.n = n

        self.schedule = {}
        self.act_rew = [((1, 1), 2), ((2, 0), 0)]

    def draw_item(self, schedule, actions=None):
        self.act_rew = actions
        self.schedule = schedule
        self.put_item()

    def put_item(self):

        colors = sns.color_palette("Set2", self.n)
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
        m = len(self.schedule.keys())
        keys = list(self.schedule.keys())
        used_colors = 0
        for k in range(m):
            items = self.schedule[keys[k]]
            for it in items:
                coords = (it.st_time, k)
                rect = patches.Rectangle(coords, it.end_time - it.st_time, 1,
                                         fill=True,
                                         facecolor=colors[used_colors],
                                         label=it.task, alpha=0.5,
                                         edgecolor="black")
                used_colors += 1
                ax[0].add_patch(rect)
                ax[0].text(coords[0] + (it.end_time - it.st_time) / 3,
                           coords[1] + 0.5, str(it.task))
        ax[0].legend()
        ax[0].axis(xmin=0, xmax=self.worst_time, ymin=0, ymax=self.k)
        texts = []
        for item, reward, proc in self.act_rew:
            texts.append(f'Item {item.task} on proc {proc} with reward {round(reward, 3)}.')

        for idx, text in enumerate(texts):
            ax[1].text(0.8, 0.95 - idx / 20, text,
                       verticalalignment='center', horizontalalignment='right',
                       transform=ax[1].transAxes, fontsize=10, wrap=True)

        plt.show()


def interective_test(args, URL):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        if config['actor_type'] == 'lstm':
            deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
        done = wfl.completed
        state = list(map(float, list(wfl.state)))
        if config['actor_type'] == 'lstm':
            deq.push(state)
            state = deq.show()
        for time in range(wfl.n):
            mask = list(map(int, list(wfl.get_mask())))
            action = requests.post(f'{URL}test', json={'state': state.tolist(), 'mask': mask, 'sched': False}).json()[
                'action']
            act_t, act_n = wfl.actions[action]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = list(map(float, list(wfl.state)))
            if config['actor_type'] == 'lstm':
                deq.push(next_state)
                next_state = deq.show()
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
        write_schedule(args.run_name, i, wfl)
    pass


def draw_schedule(test_i, episode, wfl):
    schedule = wfl.schedule
    worst = wfl.worst_time
    n = wfl.n
    colors = sns.color_palette("Set2", n)
    fig, ax = plt.subplots(2, 1)
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
    plt.title("test {} {}".format(test_i, episode))
    # plt.show()
    path = "C:\\wspace\\papers\\ysc2019\\nns\exps\\last_exp\\"
    plt.savefig(path + "testwf_{}_episode_{}.png".format(test_i, episode))
    plt.close()


def put_item(root, schedule, worst_time, k, n):
    colors = sns.color_palette("Set2", n)
    fig, ax = plt.subplots(1)
    m = len(schedule.keys())
    keys = list(schedule.keys())
    used_colors = 0
    for k in range(m):
        items = schedule[keys[k]]
        for it in items:
            coords = (it.st_time, k)
            rect = patches.Rectangle(coords, it.end_time - it.st_time, 1,
                                     fill=True,
                                     facecolor=colors[used_colors],
                                     label=it.task, alpha=0.5,
                                     edgecolor="black")
            used_colors += 1
            ax.add_patch(rect)
            ax.text(coords[0] + (it.end_time - it.st_time) / 3,
                    coords[1] + 0.5, str(it.task))
    plt.legend()
    plt.ylim(0, k)
    plt.xlim(0, worst_time)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


class IT(object):
    def __init__(self, start, end, task):
        self.st_time = start
        self.end_time = end
        self.task = task


if __name__ == '__main__':
    # sch = SchedulePlotter(worst_time=10, k=3, n=5)
    it = IT(0, 5, '0')
    it2 = IT(1, 2, '1')
    it3 = IT(2, 5, '2')
    a = ScheduleInterectivePlotter(25, 4, 3)
    a.draw_item(it, 0)
    a.draw_item(it2, 1)
    a.draw_item(it3, 1)
