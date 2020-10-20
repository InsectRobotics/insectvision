from data.fruitfly import DataFrame
from mb import *
from net.routines import get_mse, run_net

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def connected_neurons_network():
    mm1 = MBNeuron(a=np.array([1., 1.]), c_odour=np.array([3., 3.]), b=4.,
                   u=np.array([.5, 0.]), c_shock=np.array([0., 0.]), utype="mc")
    mm2 = MBNeuron(a=np.array([1., 1.]), c_odour=np.array([-4., -4.]), b=4.,
                   u=np.array([.7, 0.]), c_shock=np.array([0., 0.]), utype="mc")
    mm3 = MBNeuron(a=np.array([1., 1.]), c_odour=np.array([.5, .5]), b=0.,
                   u=np.array([.2, .2]), c_shock=np.array([0., 0.]), utype="mc")
    dd1 = MBNeuron(a=np.array([.5, .5]), c_odour=np.array([0., 0.]), b=0.,
                   u=np.array([0., .0]), c_shock=np.array([10., 10.]), utype="mc")
    dd2 = MBNeuron(a=np.array([.5, .5]), c_odour=np.array([0., 0.]), b=2.,
                   u=np.array([.9, .0]), c_shock=np.array([0., 0.]), utype="mc")
    dd3 = MBNeuron(a=np.array([.5, .5]), c_odour=np.array([0., 0.]), b=7.,
                   u=np.array([0., 0.]), c_shock=np.array([3., 3.]), utype="mc")

    # initial values
    mm1.v = np.array([0., 0.])
    mm2.v = np.array([0., 0.])
    mm3.v = np.array([0., 0.])
    dd1.v = np.array([0., 0.])
    dd2.v = np.array([5., -5.])
    dd3.v = np.array([10., 10.])

    # the network
    nn = MANetwork(dan=[dd1, dd2, dd3], mbon=[mm1, mm2, mm3])
    nn.w_mbon2mbon = np.array([
        # [0., -.3, -.3],
        # [0., 0., .1],
        # [0., 0., 0.],
        [0., -.2, -.3],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    nn.w_dan2kc = np.array([
        # [-.3, 0., 0.],
        # [0., -.3, 0.],
        # [0., 0., .02],
        [-.3, 0., 0.],
        [0., -.3, 0.],
        [0., 0., .0],
    ])
    nn.w_mbon2dan = np.array([
        # [.1, 0., 0.],
        # [0., 2., -.3],
        # [0., -.2, -.5],
        [.2, 0., 0.],
        [0., -1., 0.],
        [0., -2.5, 0.],
    ])

    return nn


def cost_func(x, df_data, samples=[], mse=[], verbose=0, show=True):
    net = network_from_features_old(x)
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
            plt.xlabel("principal component 1")
            plt.ylabel("principal component 2")
            # plt.xlim([-20, 20])
            # plt.ylim([-20, 20])
            plt.tight_layout()
            plt.pause(0.05)

    return mse[-1]



if __name__ == '__main__':
    from learn.optimisation import minimize
    import matplotlib.pyplot as plt

    plt.ion()
    verbose = 1

    df_data = DataFrame().dataset6neuron_old
    if verbose > 1:
        print df_data

    # net = connected_neurons_network()
    net = network_from_file_old('opt-current.yaml', verbose=verbose)
    x0 = network_to_features_old(net)

    # x_default = x0.copy()
    # x0 = np.concatenate([x0[0:12], x0[36:48], x0[81], x0[90]])

    # df_model = run_net(net, verbose=verbose, show=False, filename="new-network.yaml")
    # mse = get_mse(df_model, df_data)
    # if verbose > 1:
    #     print mse.groupby(["type", "name"]).mean()

    mse, samples = [], []
    res = minimize(cost_func, x0, args=(df_data, samples, mse, verbose),
    # res = minimize(cost_func_2, x0, args=(x_default, df_data, samples, mse, verbose),
                   method="Nelder-Mead",
                   # method="powell",
                   options={'maxiter': 200000, 'xtol': 1e-01, 'disp': True, 'adaptive': True})

    nnet = network_from_features_old(res.x)
    print unicode(nnet)
    network_to_file_old(nnet, "opt-new.yaml")

    print

    plt.ioff()
    plt.show()
