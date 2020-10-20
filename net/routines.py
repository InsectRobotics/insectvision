from plot3mbon import plot_hist, plot_hist_old
from mb import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def hist2dataset(hist, version=2):
    dataset = []
    ts = 3 + int(version > 1)
    columns = ["type", "name", "trial", "condition", "time"]
    keys = ["m1s", "m2s", "m3s", "d1s", "d2s", "d3s"]
    types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu", r"PPL1", r"PAM", r"PAM"]
    names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a", u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    trials = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    conditions = ["CS-", "CS+"]
    for nn, tp, nm in zip(keys, types, names):
        vls = hist[nn].reshape((-1, ts, 2))[np.arange(17), :, [0, 1] * 8 + [0]].flatten()
        data_dict = {
            "type": np.array([tp] * len(vls)),
            "name": np.array([nm] * len(vls)),
            "trial": np.array([[tl] * 2 * ts for tl in trials]).flatten()[:-ts],
            "condition": np.array(conditions)[([0] * ts + [1] * ts) * 8 + [0] * ts],
            "time": np.array(["pre-odour", "odour", "shock", "post-odour"] * 17),
            0: vls
        }

        df = pd.DataFrame(data_dict)
        df.set_index(columns, inplace=True)
        dataset.append(df)

    return pd.concat(dataset, axis=0)[0]


def get_mse(df_pred, df_target):
    return np.square(df_target - df_pred).mean(axis=0)


def get_mae(df_pred, df_target):
    return np.absolute(df_target.mean(axis=0) - df_pred)


def run_net(net, version=2, verbose=2, show=True, interactive=True, filename=None):
    hist = {
        "names": {},
        "m1": [], "m2": [], "m3": [],
        "d1": [], "d2": [], "d3": [],
        "m1s": [], "m2s": [], "m3s": [],
        "d1s": [], "d2s": [], "d3s": [],
        "m1c": [], "m2c": [], "m3c": [],
    }

    if verbose > 0:
        print unicode(net)

    for n, nid in zip(net.neurons, ["d1", "d2", "d3", "m1", "m2", "m3"]):
        hist["names"][nid] = n.name

    for i, s in enumerate([0., 0., 0., 1., 0., 1., 0., 1., 0., 1.,
                           0., 1., 0., 0., 1., 0., 1.]):
        if verbose > 1:
            print ["%d CS-", "%d CS+"][i % 2] % (i // 2 + 1), ["", "shock"][int(s)]
        odours = [np.array([1., 0.]), np.array([0., 1.])]
        net(odour=odours[i % 2], shock=s)

        hist["m1"].append(net.short_hist['m1'][-1 - int(version > 1)].copy())
        hist["m1s"].append(net.short_hist['m1'])
        hist["m1c"].append(net.short_hist['cm1'])
        hist["m2"].append(net.short_hist['m2'][-1 - int(version > 1)].copy())
        hist["m2s"].append(net.short_hist['m2'])
        hist["m2c"].append(net.short_hist['cm2'])
        hist["m3"].append(net.short_hist['m3'][-1 - int(version > 1)].copy())
        hist["m3s"].append(net.short_hist['m3'])
        hist["m3c"].append(net.short_hist['cm3'])

        hist["d1"].append(net.short_hist['d1'][-1 - int(version > 1)].copy())
        hist["d1s"].append(net.short_hist['d1'])
        hist["d2"].append(net.short_hist['d2'][-1 - int(version > 1)].copy())
        hist["d2s"].append(net.short_hist['d2'])
        hist["d3"].append(net.short_hist['d3'][-1 - int(version > 1)].copy())
        hist["d3s"].append(net.short_hist['d3'])


    hist["m1"] = np.array(hist["m1"])
    hist["m1s"] = np.array(hist["m1s"]).reshape((-1, 2))
    hist["m1c"] = np.array(hist["m1c"]).reshape((-1, 2))
    hist["m2"] = np.array(hist["m2"])
    hist["m2s"] = np.array(hist["m2s"]).reshape((-1, 2))
    hist["m2c"] = np.array(hist["m2c"]).reshape((-1, 2))
    hist["m3"] = np.array(hist["m3"])
    hist["m3s"] = np.array(hist["m3s"]).reshape((-1, 2))
    hist["m3c"] = np.array(hist["m3c"]).reshape((-1, 2))
    hist["d1"] = np.array(hist["d1"])
    hist["d1s"] = np.array(hist["d1s"]).reshape((-1, 2))
    hist["d2"] = np.array(hist["d2"])
    hist["d2s"] = np.array(hist["d2s"]).reshape((-1, 2))
    hist["d3"] = np.array(hist["d3"])
    hist["d3s"] = np.array(hist["d3s"]).reshape((-1, 2))

    for n in ["m1", "m2", "m3", "d1", "d2", "d3"]:
        hist[n][1::2, 0] = (hist[n][0:-1:2, 0] + hist[n][2::2, 0]) / 2.
        hist[n][2:-2:2, 1] = (hist[n][1:-2:2, 1] + hist[n][3::2, 1]) / 2.

    if show:
        if version < 2:
            plot_hist_old(**hist)
        else:
            plot_hist(**hist)
        if interactive:
            plt.pause(0.05)

    if filename is not None:
        network_to_file(net, filename=filename)

    return hist2dataset(hist, version=version)
