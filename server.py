import flask
from flask import request, send_from_directory
from actor import DQNActor
from heft_deps.ExperimentalManager import ExperimentResourceManager, ModelTimeEstimator
from heft_deps.resource_generator import ResourceGenerator as rg
from heft_deps.heft_utility import wf, Utility, draw_heft_schedule
from flask import jsonify
from argparse import ArgumentParser
from heft_deps.heft_settings import run_heft
from heft_deps.heft_utility import Utility
import numpy as np
import tensorflow as tf
import os
import pathlib
import datetime

parser = ArgumentParser()

parser.add_argument('--alg', type=str, default='nns')


# HEFT parameters


# NNs paramters

parser.add_argument('--state-size', type=int, default=64)
parser.add_argument('--agent-tasks', type=int, default=5)
parser.add_argument('--actor-type', type=str, default='fc')
parser.add_argument('--first-layer', type=int, default=1024)
parser.add_argument('--second-layer', type=int, default=512)
parser.add_argument('--third-layer', type=int, default=256)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--load-path', type=str, default=None)

parser.add_argument('--nodes', type=int, default=4)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--model-name', type=str, default='')

args = parser.parse_args()
app = flask.Flask(__name__)

app.config['alg'] = args.alg

app.config['state_size'] = args.state_size
app.config['action_size'] = args.agent_tasks * args.nodes
app.config['actor_type'] = args.actor_type
app.config['first_layer'] = args.first_layer
app.config['second_layer'] = args.second_layer
app.config['third_layer'] = args.third_layer
app.config['seq_size'] = args.seq_size
app.config['load'] = args.load
app.config['load_path'] = args.load_path
app.config['model_name'] = args.model_name


@app.route('/')
def get_model():
    state_size = app.config.get('state_size')
    action_size = app.config.get('action_size')
    actor_type = app.config.get('actor_type')
    seq_size = app.config.get('seq_size')
    load = app.config.get('load')
    first_layer = app.config.get('first_layer')
    second_layer = app.config.get('second_layer')
    third_layer = app.config.get('third_layer')
    load_path = app.config.get('load_path')
    model_name = app.config.get('model_name')

    if not model_name:
        model_name = 'model_fc.h5' if not actor_type=='fc' else 'model_lstm.h5'
    if not load:
        return DQNActor(first=first_layer, second=second_layer, third=third_layer, state_size=state_size, action_size=action_size, seq_size=seq_size, actor_type=actor_type)
    else:
        model = DQNActor(first=first_layer, second=second_layer, third=third_layer, state_size=state_size, action_size=action_size, seq_size=seq_size, actor_type=actor_type)
        if load_path is not '':
            model.load(model_name, path=load_path)
        else:
            model.load(model_name)
        return model


@app.route('/act', methods=['POST'])
def act():
    global graph
    data = request.get_json(force=True)
    if args.actor_type == 'fc':
        state = np.array(data['state']).reshape(1, model.STATE)
        mask = np.array(data['mask'])
        sched = data['sched']
    elif args.actor_type == 'lstm':
        state = np.asarray(data['state']).reshape((1, model.seq_size, model.STATE))
        mask = np.array(data['mask'])
        sched = data['sched']
    with graph.as_default():
        action = model.act(state, mask, sched)
    return jsonify(action=int(action))


@app.route('/test', methods=['POST'])
def test():
    global graph
    data = request.get_json(force=True)
    if args.actor_type == 'fc':
        state = np.asarray(data['state']).reshape(1, model.STATE)
    elif args.actor_type == 'lstm':
        state = np.asarray(data['state']).reshape((1, model.seq_size, model.STATE))
    mask = np.array(data['mask'])
    sched = data['sched']
    with graph.as_default():
        eps = model.epsilon
        model.epsilon = 0.0
        action = model.act(state, mask, sched)
        model.epsilon = eps
    return jsonify(action=int(action))


@app.route('/replay', methods=['POST'])
def replay():
    global graph
    data = request.get_json()
    batch_size = data['batch_size']
    with graph.as_default():
        response = model.replay(batch_size)
    return response


@app.route('/remember', methods=['POST'])
def remember():
    data = request.get_json()
    SARSA = data['SARSA']
    if args.actor_type == 'fc':
        state = np.asarray(SARSA[0]).reshape(1, model.STATE)
        next_state = np.asarray(SARSA[3]).reshape(1, model.STATE)
    elif args.actor_type == 'lstm':
        state = np.asarray(SARSA[0]).reshape((1, model.seq_size, model.STATE))
        next_state = np.asarray(SARSA[3]).reshape((1, model.seq_size, model.STATE))
    action = int(SARSA[1])
    reward = float(SARSA[2])
    done = bool(SARSA[4])
    response = {'is_remembered': model.remember((state, action, reward, next_state, done))}
    return response


@app.route('/save', methods=['POST'])
def save():
    json_model = model.save('model')
    return json_model


@app.route('/heft', methods=['POST'])
def heft():
    data = request.get_json()
    wf_name = data['wf_name']
    rm = ExperimentResourceManager(rg.r(data['nodes']))
    estimator = ModelTimeEstimator(bandwidth=10)
    _wf = wf(wf_name)
    heft_schedule = run_heft(_wf, rm, estimator)
    print(heft_schedule)
    actions = [(proc.start_time, int(proc.job.global_id), node.name_id) for node in heft_schedule.mapping for proc in heft_schedule.mapping[node]]
    print(actions)
    actions = sorted(actions, key=lambda x: x[0])
    actions = [(action[1], action[2]) for action in actions]
    print(actions)
    makespan = Utility.makespan(heft_schedule)
    # reward = worst_time / makespan
    draw_heft_schedule(heft_schedule.mapping, data['worst_time'], len(actions), 'h', '1')
    response = {'makespan': makespan, 'actions': actions}
    return response


if __name__ == '__main__':
    graph = tf.get_default_graph()
    model = get_model()
    URL = f'http://{args.host}:{args.port}/'
    app.run(host=args.host, port=args.port, debug=True)




