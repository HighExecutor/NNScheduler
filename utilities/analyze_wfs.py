import numpy as np
import matplotlib.pyplot as plt
from utilities.DAXParser import read_workflow
from wf_gen_funcs import tree_data_wf
from env.context import Context

if __name__ == "__main__":
    # wf_names = ["Montage_25", "Montage_50", "Montage_100", "CyberShake_30", "CyberShake_50", "CyberShake_100",
    #             "Epigenomics_24", "Epigenomics_46", "Epigenomics_100", "Sipht_30", "Sipht_60", "Sipht_100",
    #             "Inspiral_30", "Inspiral_50", "Inspiral_100"]
    wf_names = ["Montage_25", "Montage_50", "CyberShake_30", "CyberShake_50",
                "Epigenomics_24", "Epigenomics_46", "Sipht_30",
                "Inspiral_30", "Inspiral_50"]
    wf_names2 = ["cadsr", "gene2life", "glimmer", "leadadas", "leaddm", "mememast", "molsci",
                "motif_small", "psload_small", "scoop_small"]
    # wf_names.extend(wf_names2)
    # wf_names = ["Montage_25"]

    nodes = [4, 8 ,16]

    wfs = []

    for wf_name in wf_names:
        wf_path = ".\\resources\\{0}.xml".format(wf_name)
        wfl = read_workflow(wf_path, wf_name)
        tree, data, run_times = tree_data_wf(wfl)
        wf = Context(len(run_times), nodes, run_times, tree, data)
        wfs.append((wf, wf_name))

        print(wf_name)
        print(wf.n)
        # print("Sum runtime = {}".format(wf.runtime_sum))
        # print("Sum input = {}".format(wf.input_sum))
        # print("Runtime / input = {}".format(wf.runtime_sum / wf.input_sum))

        run_times = wf.run_times
        inputs = np.round(wf.t_input, 1)
        # print("Run_times = {}".format(run_times))
        # print("Inputs = {}".format(inputs))
        # print("Avg runtimes = {}".format(np.mean(run_times)))
        # print("Avg input = {}".format(np.mean(inputs)))
        # print("Std runtimes = {}".format(np.std(run_times)))
        # print("Std input = {}".format(np.std(inputs)))


    plt.figure()
    plt.subplot(211)
    for wf, wf_name in wfs:
        run_times = wf.run_times
        plt.hist(run_times, alpha=0.4, label=wf_name)
    plt.legend()
    plt.subplot(212)
    for wf, wf_name in wfs:
        inputs = np.round(wf.t_input, 1)
        plt.hist(inputs, alpha=0.4, label=wf_name)
    plt.legend()
    plt.show()

