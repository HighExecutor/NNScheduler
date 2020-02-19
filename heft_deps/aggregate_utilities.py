import json
import os
import math
from pprint import pprint
import matplotlib.pyplot as plt
from heft_deps.settings import TEMP_PATH
from scipy import stats
import scipy
import numpy

# HACK
BASE_PARAMS = {
    "alg_name": "cga",
    "executor_params": {
        "base_fail_duration": 40,
        "base_fail_dispersion": 1,
        "fixed_interval_for_ga": 15,
        "fail_count_upper_limit": 15,
        "replace_anyway": True
    },
    "resource_set": {
        "nodes_conf": [(10, 15, 25, 30)],
        "rules_list": [(80, 30)]
    },
    "estimator_settings": {
        "ideal_flops": 20,
        "transfer_nodes": 100,
        "reliability": 1.0,
        "transfer_blades": 100
    }
}


WFS_COLORS_30 = {
    # 30 - series
    "Montage_25": "-gD",
    "CyberShake_30": "-rD",
    "Inspiral_30": "-bD",
    "Sipht_30": "-yD",
    "Epigenomics_24": "-mD",
}

WFS_COLORS_50 = {
    # 50 - series
    "Montage_50": "-gD",
    "CyberShake_50": "-rD",
    "Inspiral_50": "-bD",
    "Sipht_60": "-yD",
    "Epigenomics_46": "-mD",
}


WFS_COLORS_75 = {
    # 75 - series
    "Montage_75": "-gD",
    "CyberShake_75": "-rD",
    "Inspiral_72": "-bD",
    "Sipht_73": "-yD",
    "Epigenomics_72": "-mD",
}


WFS_COLORS_100 = {
    # 100 - series
    "Montage_100": "-gD",
    "CyberShake_100": "-rD",
    "Inspiral_100": "-bD",
    "Sipht_100": "-yD",
    "Epigenomics_100": "-mD",
}

WFS_COLORS = dict()
WFS_COLORS.update(WFS_COLORS_30)
WFS_COLORS.update(WFS_COLORS_50)
WFS_COLORS.update(WFS_COLORS_75)
WFS_COLORS.update(WFS_COLORS_100)


def visualize(data, functions, path_to_save=None):


    for i in range(len(functions)):
        #plt.subplot(len(functions), 1, i + 1)
        plt.clf()
        functions[i](data)

    plt.tight_layout()

    if path_to_save is None:
        plt.show()
    else:
        directory = os.path.dirname(path_to_save)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path_to_save, dpi=96.0, format="png")
        plt.clf()
    pass


def aggregate(pathes,  picture_path="gh.png", extract_and_add=None, functions=None):
    files = [os.path.join(path, p) for path in pathes for p in os.listdir(path) if p.endswith(".json")]
    data = {}
    for p in files:
        with open(p, "r") as f:
            d = json.load(f)
            extract_and_add(data, d)

    path = os.path.join(TEMP_PATH, picture_path) if not os.path.isabs(picture_path) else picture_path
    visualize(data, functions, path)



def interval_statistics(points, confidence_level=0.95):
    s = numpy.array(points)
    n, min_max, mean, var, skew, kurt = stats.describe(s)
    std = math.sqrt(var)
    left, right = stats.norm.interval(confidence_level, loc=mean, scale=std)
    mn, mx = min_max
    return mean, mn, mx, std, left, right


class InMemoryDataAggregator:

    def __init__(self, pathes):
        files = [os.path.join(path, p) for path in pathes for p in os.listdir(path) if p.endswith(".json")]
        self._data_array = []
        for p in files:
            with open(p, "r") as f:
                d = json.load(f)
                self._data_array.append(d)
        pass

    def __call__(self, picture_path="gh.png", extract_and_add=None, functions=None):
        data = {}
        for d in self._data_array:
            extract_and_add(data, d)

        path = os.path.join(TEMP_PATH, picture_path) if not os.path.isabs(picture_path) else picture_path
        visualize(data, functions, path)
        pass
    pass


def interval_stat_string(stat_result):
    mean, mn, mx, std, left, right = stat_result
    st = "Mean: {0:.0f}, Min: {1:.0f}, Max: {2:.0f}, Std: {3:.0f}, Left: {4:.0f}, Right: {5:.0f}"\
        .format(mean, mn, mx, std, left, right)
    return st