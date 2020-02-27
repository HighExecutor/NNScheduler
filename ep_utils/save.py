import requests


def save(URL):
    model = requests.post(f'{URL}save')