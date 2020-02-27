import requests


def replay(batch_size, URL):
    loss = requests.post(f'{URL}replay', json={'batch_size': batch_size}).json()['loss']
    return loss
