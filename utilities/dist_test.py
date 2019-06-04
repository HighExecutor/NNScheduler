import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def runtim_choice():

    # data = rnd.weibull(a=1, size=1000)*1000
    # plt.hist(data, label="weibull", alpha=0.4)
    # data = rnd.exponential(scale=400, size=1000)
    # plt.hist(data, label="exp", alpha=0.4)
    data = rnd.gamma(1, size=1000) * 1000 # norm
    plt.hist(data, label="gamma", alpha=0.4)
    # data = rnd.exponential(scale=500, size=1000)
    plt.legend()
    plt.show()

def input_choice():

    data = rnd.weibull(a=1, size=1000)*100
    plt.hist(data, label="weibull", alpha=0.4)
    # data = rnd.exponential(scale=50, size=1000)
    # plt.hist(data, label="exp", alpha=0.4)
    # data = rnd.gamma(1, size=1000) * 100
    # plt.hist(data, label="gamma", alpha=0.4)
    # data = rnd.exponential(scale=500, size=1000)
    plt.legend()
    plt.show()

if __name__ == "__main__":
   runtim_choice()