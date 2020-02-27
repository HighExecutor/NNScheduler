import requests


def remember(sars_list, URL):
    for sars in sars_list:
        _ = requests.post(f'{URL}remember', json={'SARSA': sars})
