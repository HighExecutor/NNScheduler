import numpy as np

import env.context as ctx
from wf_gen_funcs import tree_data_gen, tree_data_wf, read_wf
# модуль с агентом и сетью
import actor
import time as timer
import random
import uuid
from multiprocessing import Pool, Lock
from draw_figures import draw_schedule, draw_graph, write_schedule
import os
import glob
from deap import tools
from scoop import shared, futures




avg_scores = []
last_avg_scores = []
loss = []
last_avg_size = 500
scores = []
agent_tasks = 5
nodes = 4
state_size = 64
action_size = agent_tasks * nodes

process_id = uuid.uuid4()

batch_size = 64
EPISODES = 20001
nodes = np.array([4, 8, 8, 16])
# wfs_names = ["gene2life", "floodplain", "scoop_small", "leadadas", "leadmm", "molsci"]
wfs_names = ["gene2life"]
# wfs_names = ["Montage_25", "CyberShake_30", "Inspiral_30", "Epigenomics_24"]
# wfs_names = ["Montage_25", "CyberShake_30"]
wfs_real = [read_wf(name) for name in wfs_names]
# for tests
test_wfs = []
test_times = dict()
test_scores = dict()
test_size = len(wfs_real)
for i in range(test_size):
    wf_components = tree_data_wf(wfs_real[i])
    test_wfs.append(wf_components)
    test_times[i] = list()
    test_scores[i] = list()
    draw_graph(wf_components, i, wfs_names[i])


def episode(ei):
    shared_agent = shared.getConst('agent{}'.format(ei%10))
    test_size = len(test_wfs)
    tree, data, run_times = test_wfs[ei % test_size]
    wfl = ctx.Context(agent_tasks, nodes, run_times, tree, data)
    wfl.name = wfs_names[ei % test_size]
    done = wfl.completed
    state = wfl.state
    state = np.reshape(state, [1, state_size])
    sars_list = list()
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = wfl.get_mask()
        # lock.acquire()
        action = shared_agent(state, mask)
        # lock.release()
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = wfl.state
        done = wfl.completed
        next_state = np.reshape(next_state, [1, state_size])
        # запоминаем цепочку действий
        sars_list.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return reward, sars_list


if __name__ == "__main__":
    files = glob.glob("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\*")
    for f in files:
        os.remove(f)

    import tensorflow as tf
    from keras import backend as K
    sess = tf.Session()
    # инициализируем Keras
    K.set_session(sess)

    shared_agent = actor.DQNAgent(state_size, action_size, name=process_id)
    shared.setConst(agent=shared_agent)
    agent = shared.getConst("agent")
    parallel = 10
    # Learning
    start = timer.time()
    for e in range(0, EPISODES, parallel):
        rewards = list(futures.map(episode, list(range(e, e+parallel, 1))))
        # rewards = list(map(episode, range(e, e+parallel, 1)))

        for ei in range(parallel):
            reward = rewards[ei][0]
            sars_list = rewards[ei][1]
            for sars in sars_list:
                agent.remember(sars)

            scores.append(reward)
            mean_scores = np.mean(scores)
            avg_scores.append(mean_scores)
            last_mean_scores = np.mean(scores[len(scores) - last_avg_size:len(scores)])
            last_avg_scores.append(last_mean_scores)
            if (e + ei) % 10 == 0:
                print(
                    "episode: {}/{},  score: {}, normalized score avg: {:.4}, last scores {}".format(
                        e,
                        EPISODES,
                        reward,
                        mean_scores, last_mean_scores))

            # обучаем нейронку
            if (e + ei) % 10 == 0:
                if len(agent.D) > batch_size:
                    curloss = agent.replay(batch_size, e)
                    loss.append(curloss)
        if e % (10000) == 0:
            print("Testing")
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
            print("Test finished")
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
    plt.plot(last_avg_scores[500:], '--', label="last {} avg".format(last_avg_size))
    plt.plot(avg_scores[500:], '-', label="avg")
    # plt.axhline(y=776, color='b', linestyle='-')
    plt.ylabel('reward')
    plt.xlabel('episodes')
    plt.legend()
    plt.savefig("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\exp_plot.png")
    np.save("C:\\wspace\\papers\\ysc2019\\nns\\exps\\last_exp\\scores.npy", np.array(scores))
    print(loss)

    # from playsound import playsound
    # playsound("airhorn-short.wav")

    plt.show()




