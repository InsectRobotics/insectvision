from compoundeye.geometry import angles_distribution
from comp_model_plots import evaluate

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    tilt = False
    samples = 1000
    nb_noise = 1
    max_noise = 1

    theta, phi, fit = angles_distribution(60, 60)
    alpha = (phi - np.pi / 2) % (2 * np.pi) - np.pi
    noises = np.linspace(0, max_noise, int(max_noise * 2) + 1, endpoint=True)

    costs, d_effs, eles = [], [], []
    x = []
    for noise in noises:
        costs.append([])
        d_effs.append([])
        eles.append([])
        x.append([])
        for _ in xrange(nb_noise):
            n = np.absolute(np.random.randn(*theta.shape)) < noise
            cloud = 100. * n.sum() / np.float32(n.size)
            x[-1].append(cloud)
            d_err, d_eff, tau = evaluate(noise=noise, tilting=False)
            costs[-1].append(d_err)
            d_effs[-1].append(d_eff)
            eles[-1].append(tau.flatten())

            print "Mean cost: %.2f" % d_err.mean(),
            print "| Noise level: %.4f (%.2f %%)" % (noise, cloud)

    costs = np.array(costs)[:, 0, :, 0]
    d_effs = np.array(d_effs)[:, 0, :, 0]
    eles = np.rad2deg(np.array(eles)[:, 0, :])
    x = np.array([np.array(x).T] * costs.shape[-1])[:, 0, :].T
    print x.shape
    print costs.shape
    print d_effs.shape
    print eles.shape
    # np.savez_compressed("%s-layer.npz" % activation, x=x, y=costs)

    max_ele = 90
    sky0 = np.all([x < 5, eles < max_ele], axis=0)
    sky1 = np.all([x >= 5, x <= 50, eles < max_ele], axis=0)
    sky2 = np.all([x > 50, x <= 80, eles < max_ele], axis=0)
    sky3 = np.all([x > 80, eles < max_ele], axis=0)
    print "Sky: 0 - %d, 1 - %d, 2 - %d, 3 - %d" % (sky0.sum(), sky1.sum(), sky2.sum(), sky3.sum())

    plt.figure("cost-function-clouds", figsize=(20, 15))
    plt.subplot(223)
    # plt.scatter(costs[sky3].flatten(), d_effs[sky3].flatten(), s=10,
    #             marker="o", edgecolors="black", color="black", label=r"$> 80\%$ clouds")
    plt.scatter(costs[sky2].flatten(), d_effs[sky2].flatten(), s=10,
                marker="o", edgecolors="black", color="white", label=r"$\leq 80\%$ clouds")
    plt.scatter(costs[sky1].flatten(), d_effs[sky1].flatten(), s=10,
                marker="^", edgecolors="black", color="black", label=r"$\leq 50\%$ clouds")
    plt.scatter(costs[sky0].flatten(), d_effs[sky0].flatten(), s=10,
                marker="^", edgecolors="black", color="white", label=r"$\leq 5\%$ clouds")

    # plt.legend()  # ncol=2)
    plt.grid()
    plt.ylim([0, 1])
    plt.xlim([90, -1])
    plt.ylabel(r"Polarisation contrast, $d_{eff}$")
    plt.xlabel(r"Error ($^\circ$)")

    plt.subplot(222)
    # plt.scatter(eles[sky3].flatten(), costs[sky3].flatten(), s=10,
    #             marker="o", edgecolors="black", color="black", label=r"$> 80\%$ clouds")
    plt.scatter(eles[sky2].flatten(), costs[sky2].flatten(), s=10,
                marker="o", edgecolors="black", color="white", label=r"$\leq 80\%$ clouds")
    plt.scatter(eles[sky1].flatten(), costs[sky1].flatten(), s=10,
                marker="^", edgecolors="black", color="black", label=r"$\leq 50\%$ clouds")
    plt.scatter(eles[sky0].flatten(), costs[sky0].flatten(), s=10,
                marker="^", edgecolors="black", color="white", label=r"$\leq 5\%$ clouds")

    # plt.legend()  # ncol=2)
    plt.grid()
    plt.xlim([-1, 90])
    plt.ylim([-1, 90])
    plt.xlabel(r"Solar elevation ($^\circ$)")
    plt.ylabel(r"Error ($^\circ$)")

    plt.subplot(224)
    # plt.scatter(eles[sky3].flatten(), d_effs[sky3].flatten(), s=10,
    #             marker="o", edgecolors="black", color="black", label=r"$> 80\%$ clouds")
    plt.scatter(eles[sky2].flatten(), d_effs[sky2].flatten(), s=10,
                marker="o", edgecolors="black", color="white", label=r"$\leq 80\%$ clouds")
    plt.scatter(eles[sky1].flatten(), d_effs[sky1].flatten(), s=10,
                marker="^", edgecolors="black", color="black", label=r"$\leq 50\%$ clouds")
    plt.scatter(eles[sky0].flatten(), d_effs[sky0].flatten(), s=10,
                marker="^", edgecolors="black", color="white", label=r"$\leq 5\%$ clouds")

    plt.legend()  # ncol=2)
    plt.grid()
    plt.xlim([-1, 90])
    plt.ylim([0, 1])
    plt.xlabel(r"Solar elevation ($^\circ$)")
    plt.ylabel(r"Polarisation contrast, $d_{eff}$")

    plt.subplot(221)
    plt.hist([costs[sky3].flatten(), costs[sky2].flatten(), costs[sky1].flatten(), costs[sky0].flatten()],
             bins=45, range=(0, 90), histtype="step", stacked=True,
             color=["C1", "C3", "C2", "C0"])  # , edgecolor="black")
    # plt.hist(costs[np.any([sky0, sky1, sky2], axis=0)].flatten(), bins=90, range=(0, 90),
    #          edgecolor="black", facecolor="white")
    # plt.legend()  # ncol=2)
    plt.grid()
    plt.xlim([90, -1])
    plt.ylabel(r"Number of occurences")
    plt.xlabel(r"Error ($^\circ$)")
    plt.show()
