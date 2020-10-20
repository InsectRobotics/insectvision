from data.fruitfly import DataFrame
from mb import *
from net.routines import get_mse, run_net

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def cost_func(x, df_data, samples=[], mse=[], sparse=False, verbose=0, show=True):
    net = network_from_features(x, sparse=sparse)
    df_model = run_net(net, verbose=verbose, show=show)  # , filename="opt-current.yaml")
    if verbose > 2:
        print df_model
    mse.append(get_mse(df_model, df_data).mean())
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
            plt.scatter(pcs[0, 0], pcs[0, 1], c='blue', marker='o')
            plt.xlabel("principal component 1")
            plt.ylabel("principal component 2")
            plt.tight_layout()
            plt.pause(0.05)

    return mse[-1]


if __name__ == '__main__':
    from learn.optimisation import minimize
    import matplotlib.pyplot as plt

    plt.ion()
    verbose = 1
    sparse = False

    df_data = DataFrame().dataset6neuron
    if verbose > 1:
        print df_data

    net = network_from_file('opt-new.yaml', verbose=verbose)
    x0 = network_to_features(net, sparse=sparse)

    mse, samples = [], []
    res = minimize(cost_func, x0, args=(df_data, samples, mse, sparse, verbose),
                   method="Nelder-Mead",
                   options={'maxiter': 10000, 'xtol': 1e-01, 'disp': True, 'adaptive': True})

    nnet = network_from_features(res.x, sparse=sparse)
    print unicode(nnet)
    network_to_file(nnet, "opt-new.yaml")

    plt.ioff()
    plt.show()
