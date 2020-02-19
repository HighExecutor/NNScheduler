import numpy as np


class LSTMDeque(object):
    def __init__(self, seq_size, size):
        self.seq_size = seq_size
        self.size = size
        self.deque = np.zeros((seq_size, size))

    def push(self, x):
        slc = self.deque[1:]
        self.deque = np.zeros((self.seq_size, self.size))
        self.deque[:-1] = slc
        self.deque[-1] = x

    def show(self):
        return self.deque
