# -*- coding: utf-8 -*-
"""
все используемые пакеты

import random
from collections import deque
import numpy as np
import json
#from keras import initializations
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from collections import Iterable
from itertools import chain
import numpy as np
import tensorflow as tf
#модуль с правилами для составления расписания
import workflowenv as wf
#модуль с агентом и сетью
import actor


"""

import numpy as np
import tensorflow as tf
import env.context as ctx
from wf_gen_funcs import tree_data_gen, tree_data_wf, read_wf
# модуль с агентом и сетью
import actor
import time as timer
import random

from multiprocessing import Pool
from draw_figures import draw_schedule, draw_graph, write_schedule


import os
import glob

files = glob.glob("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\*")
for f in files:
    os.remove(f)
# инициализируем Tensorflow
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
sess = tf.Session()
# инициализируем Keras
from keras import backend as K

K.set_session(sess)
learning61 = []
scores_learn = []
scores = []
time61 = []
# число задач
task_par = 30
agent_tasks = 5
# минимальное число задач
task_par_min = 20
# число процессоров
nodes = 4
# длинна вектора состояния (вылезает ошибка с числом которое надо подставить при первом запуске с новыми параметрами)
state_size = 64
# state_size = 2514
# число действий
action_size = agent_tasks * nodes
# инициализируем агента
process_id = random.randint(0, 1000)
print("process {}".format(process_id))
agent = actor.DQNAgent(state_size, action_size, name="actor_{}".format(process_id))
# функция загрузки весов (сохраняются agent.save(name))
# agent.load("model61.h5")
done = False
# размер выборки из памяти для обучения
batch_size = 16
# метрики
scoreavg = 0
EPISODES = 50001
loss = 0
# nodes = np.array([4, 8, 8, 16])
nodes = np.array([4, 8, 8, 16])
# nodes = np.array([4, 4, 8, 8, 16, 16])
# wfs_names = ["gene2life", "floodplain", "scoop_small", "leadadas", "leadmm", "molsci"]
wfs_names = ["Montage_25"]
# wfs_names = ["Montage_25", "CyberShake_30", "Inspiral_30", "Epigenomics_24"]
# wfs_names = ["Montage_25", "CyberShake_30"]
wfs_real = [read_wf(name) for name in wfs_names]
test_wfs = []
test_times = dict()
test_scores = dict()
test_size = len(wfs_real)
for i in range(test_size):
    wf_components = tree_data_wf(wfs_real[i])
    # tasks_n = np.random.randint(task_par_min, task_par+1)
    # wf_components = tree_data_gen(tasks_n)
    test_wfs.append(wf_components)
    test_times[i] = list()
    test_scores[i] = list()
    draw_graph(wf_components, i, wfs_names[i])
last_mean_size = 500


def episode(ei):

    # генерируем wf
    # random
    # tasks_n = np.random.randint(task_par_min, task_par+1)
    # tree, data, run_times = tree_data_gen(tasks_n)
    # test sample
    ttree, tdata, trun_times = test_wfs[ei % test_size]
    wfl = ctx.Context(agent_tasks, nodes, trun_times, ttree, tdata)
    wfl.name = wfs_names[ei % test_size]
    # real
    # wf_real = random.choice(wfs_real)
    # tree, data, run_times = tree_data_wf(wf_real)
    # real or random
    # wfl = ctx.Context(agent_tasks, nodes, run_times, tree, data)
    # wfl.name = "random"
    done = wfl.completed
    state = wfl.state
    state = np.reshape(state, [1, state_size])
    # ep_memory = []
    # import pprint
    # pprint.pprint(wfl.get_state_map())
    # state_memory = np.zeros(shape=(3, 64))
    # state_memory[0] = state
    sars_list = list()
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = wfl.get_mask()
        action = agent.act(state, mask)
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = wfl.state
        done = wfl.completed
        next_state = np.reshape(next_state, [1, state_size])
        # запоминаем цепочку действий
        #agent.remember((state, action, reward, next_state, done))
        sars_list.append((state, action, reward, next_state, done))
        # ep_memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return reward, sars_list
            # # for ep_idx in range(len(ep_memory)):
            # #     agent.remember((ep_memory[ep_idx][0], ep_memory[ep_idx][1], reward * (ep_idx+1) / wfl.n, ep_memory[ep_idx][3],ep_memory[ep_idx][4]))
            # # выводим метрики, если расписание составлено
            # scoreavg += reward
            # scores.append(reward)
            # neps = e + 1
            # learning61.append([scoreavg / neps])
            # last_mean_scores = np.mean(scores[len(scores) - last_mean_size:len(scores)])
            # scores_learn.append(last_mean_scores)
            # if e % 100 == 0:
            #     print(
            #         "episode: {}/{},ntask: {}, wf_time: {},  score: {}, normalized score avg: {:.4}, last scores {}".format(
            #             e,
            #             EPISODES,
            #             wfl.n,
            #             wf_time,
            #             reward,
            #             scoreavg / neps, last_mean_scores))
            # break

from functools import partial

if __name__ == '__main__':
    parallel = 1
    # pool = Pool(parallel)
    start = timer.time()
    for e in range(0, EPISODES, parallel):
        # rewards = list(pool.map(episode, range(e, e + parallel, 1)))
        rewards = episode(e)
        rewards = [rewards]

        for ei in range(parallel):
            reward = rewards[ei][0]
            sars_list = rewards[ei][1]
            for sars in sars_list:
                agent.remember(sars)
            scoreavg += reward
            scores.append(reward)
            neps = e + ei + 1
            mean_scores = np.mean(scores)
            learning61.append(mean_scores)
            last_mean_scores = np.mean(scores[len(scores) - last_mean_size:len(scores)])
            scores_learn.append(last_mean_scores)
            if (e + ei) % 10 == 0:
                print(
                    "episode: {}/{},  score: {}, normalized score avg: {:.4}, last scores {}".format(
                        e,
                        EPISODES,
                        reward,
                        mean_scores, last_mean_scores))

        # обучаем нейронку
        if e % 1 == 0:
            if len(agent.D) > batch_size:
                loss += agent.replay(batch_size, e)
        if e % (5000) == 0:
            # test wfs
            eps = agent.epsilon
            agent.epsilon = 0.0
            for i in range(test_size):
                best_saves = list()
                for k in range(5):
                    ttree, tdata, trun_times = test_wfs[i]
                    wfl = ctx.Context(agent_tasks, nodes, trun_times, ttree, tdata)
                    wfl.name = wfs_names[i]
                    done = wfl.completed
                    state = wfl.state
                    state = np.reshape(state, [1, state_size])
                    for time in range(wfl.n):
                        mask = wfl.get_mask()
                        action = agent.act(state, mask)
                        act_t, act_n = wfl.actions[action]
                        reward, wf_time = wfl.make_action(act_t, act_n)
                        next_state = wfl.state
                        done = wfl.completed
                        next_state = np.reshape(next_state, [1, state_size])
                        state = next_state
                        if done:
                            test_scores[i].append(reward)
                            test_times[i].append(wf_time)
                            best_saves.append((wfl, wf_time))
                            break
                best_saves = sorted(best_saves, key=lambda x: x[1])
                write_schedule(i, e, best_saves[0][0])

            agent.epsilon = eps
    end_time = timer.time()
    print("Time spent = {}".format(end_time - start))
    # print test data
    for i in range(test_size):
        print("--test wf {}--".format(i))
        print("scores = {}".format(test_scores[i]))
        print("times = {}".format(test_times[i]))
    # строим кривую обучения
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-muted")
    plt.figure(figsize=(10, 5))
    # plt.plot(scores[100:], '-*', label="scores")
    plt.plot(scores_learn[500:], '--', label="last {} avg".format(last_mean_size))
    plt.plot(learning61[500:], '-', label="avg")
    # plt.axhline(y=776, color='b', linestyle='-')
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.legend()
    plt.savefig("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\exp_plot.png")
    np.save("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\scores.npy", np.array(scores))

    # from playsound import playsound
    # playsound("airhorn-short.wav")

    plt.show()
