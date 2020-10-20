from data.fruitfly import DataFrame
from mb import *
from net.routines import run_net, get_mse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def cost_func(x, x_default, df_data, samples=[], mse=[], verbose=0, show=True):
    x_default[0:12] = x[0:12]
    x_default[36:48] = x[12:24]
    x_default[81] = x[24]
    x_default[90] = x[25]
    net = network_from_features_old(x_default)
    df_model = run_net(net, verbose=verbose, show=show)  # , filename="opt-current.yaml")
    if verbose > 2:
        print df_model[["MBON-GABA", "PPL1"]]
    mse.append(get_mse(df_model[["MBON-GABA", "PPL1"]], df_data[["MBON-GABA", "PPL1"]]).mean())
    if verbose > 0:
        print "Iter: % 2d;\tMSE: %.2f" % (len(mse), mse[-1])

    if show:
        samples.append(np.array(df_model))
        if len(samples) > 1:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(np.array(samples))

            plt.figure("cost-distribution", figsize=(6, 5))
            plt.clf()
            plt.scatter(pcs[:, 0], pcs[:, 1], c=mse, vmin=0, vmax=15, cmap='viridis', marker='.')
            plt.colorbar()
            plt.scatter(pcs[-1, 0], pcs[-1, 1], c='red', marker='o')
            plt.xlabel("principal component 1")
            plt.ylabel("principal component 2")
            # plt.xlim([-10, 10])
            # plt.ylim([-10, 10])
            plt.tight_layout()
            plt.pause(0.05)

    return mse[-1]



if __name__ == '__main__':
    from learn.optimisation import minimize

    plt.ion()
    verbose = 1

    df_data = DataFrame().dataset6neuron_old
    if verbose > 1:
        print df_data

    net = network_from_file_old('opt-current.yaml', verbose=verbose)
    x0 = network_to_features_old(net)

    x_default = x0.copy()
    x0 = np.concatenate([x0[0:12], x0[36:48], [x0[81]], [x0[90]]])

    # df_model = run_net(net, verbose=verbose, show=False, filename="new-network.yaml")
    # mse = get_mse(df_model, df_data)
    # if verbose > 1:
    #     print mse.groupby(["type", "name"]).mean()

    mse, samples = [], []
    res = minimize(cost_func, x0, args=(x_default, df_data, samples, mse, verbose),
                   method="Nelder-Mead",
                   # method="powell",
                   options={'maxiter': 2000000, 'xtol': 1e-01, 'disp': True, 'adaptive': True})

    nnet = network_from_features_old(res.x)
    print unicode(nnet)
    network_to_file_old(nnet, "opt-new.yaml")

    plt.ioff()
    plt.show()
