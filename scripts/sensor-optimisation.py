#!/usr/bin/env python
# Optimise the sensor design using global optimisation algorithms.
# There are also different modes, to test the weights and visualise the results of optimisation.
#

from compoundeye import CompassSensor
from compoundeye.geometry import fibonacci_sphere, angles_distribution
from learn import SensorObjective, optimise
from learn.optimisation import __datadir__, get_log
from sphere import azidist

from datetime import datetime
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import re
import os

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2018, The Invisible Cues Project"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

# mode = "init"
mode = "noise"
# mode = "pol-contrast"
# mode = "single"
# mode = "archipelago"
# mode = "plot-overall"
# mode = "plot-params"


if __name__ == "__main__":
    if mode == "init":
        tilt = False
        samples = 1000
        theta, phi, fit = angles_distribution(60, 60)
        # theta, phi = fibonacci_sphere(samples=60, fov=60)
        alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi
        phi_tb1 = np.linspace(0., 2 * np.pi, 8, endpoint=False)  # TB1 preference angles
        w = -8 / (2. * 60) * np.cos(phi_tb1[np.newaxis] - phi[:, np.newaxis]) * np.sin(theta[:, np.newaxis]) ** 4

        # w = np.maximum(w, 0)
        # cost = SensorObjective._fitness(theta, phi, alpha, w=w, tilt=tilt, error=azidist)
        cost = SensorObjective._fitness(theta, phi, alpha, tilt=tilt, error=azidist, noise=.0)

        print "Mean cost: %.2f" % cost

        # s = CompassSensor(thetas=theta, phis=phi, alphas=alpha)
        # CompassSensor.visualise(s, sL=100.*np.sqrt(np.square(w).sum(axis=1)), colormap="Reds")

        # if tilt:
        #     angles = np.array([
        #         [0, 0],
        #         [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
        #         [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4],
        #         [np.pi / 6, 7 * np.pi / 4],
        #         [np.pi / 3, 0], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
        #         [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4],
        #         [np.pi / 3, 7 * np.pi / 4]
        #     ])  # 17
        # else:
        #     angles = np.array([[0., 0.]])  # 1
        #
        # theta_s, phi_s = fibonacci_sphere(samples=samples, fov=180)
        # samples = angles.shape[0] * samples
        # d = np.zeros(samples)
        #
        # for theta_t, phi_t in angles:
        #     v_t = sph2vec(theta_t, phi_t, zenith=True)
        #     v_s = sph2vec(theta_s, phi_s, zenith=True)
        #     v = sph2vec(theta, phi, zenith=True)
        #     v_a = sph2vec(np.full(alpha.shape[0], np.pi/2), alpha, zenith=True)
        #     R = point2rotmat(v_t)
        #     v_s_ = R.dot(v_s)
        #     theta_s_, phi_s_, _ = vec2sph(v_s_, zenith=True)
        #     # theta_s_, phi_s_, _ = vec2sph(v_s, zenith=True)
        #     print theta_s_[0], phi_s_[0]
        #     theta_, phi_, _ = vec2sph(R.T.dot(v), zenith=True)
        #     _, alpha_, _ = vec2sph(R.T.dot(v_a), zenith=True)
        #     # theta_, phi_, _ = vec2sph(v, zenith=True)
        #     s = CompassSensor(thetas=theta_, phis=phi_, alphas=alpha_)
        #     ax = s.visualise_structure(s)
        #     ax.plot(-s.R_c * v_s_[0, 0], s.R_c * v_s_[1, 0], marker="o", color="yellow", markeredgecolor="black", markersize=5)
        #     plt.show()
    elif mode == "noise":
        tilt = True
        samples = 1000
        nb_noise = 10
        max_noise = 2
        xaxis = "cloud"

        theta, phi, fit = angles_distribution(60, 60)
        alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi
        noises = np.linspace(0, max_noise, int(max_noise * 5) + 1, endpoint=True)

        plt.figure("cost-function-noise")
        for activation, ls in [("linear", "-"), ("relu", "-."), ("sigmoid", ":"), ("tanh", "--")]:
        # for order, ls in [(1, "-"), (5, "-."), (10, "--"), (15, ":")]:
            costs, c_000, c_030, c_060 = [], [], [], []
            x = []
            print activation.capitalize()
            # print "Order:", order
            for noise in noises:
                costs.append([])
                c_000.append([])
                c_030.append([])
                c_060.append([])
                x.append([])
                for _ in xrange(nb_noise):
                    n = np.absolute(np.random.randn(*theta.shape)) < noise
                    cloud = 100. * n.sum() / np.float32(n.size)
                    if "noise" in xaxis:
                        x[-1].append(noise)
                    elif "cloud" in xaxis:
                        x[-1].append(cloud)
                    cost, _, _ = SensorObjective._fitness(
                        theta, phi, alpha, tilt=tilt, error=azidist, noise=noise,
                        activation=activation,
                        # gate_order=order,
                        return_mean=False)
                    costs[-1].append(cost.mean())
                    c_000[-1].append(cost[:, 0].mean())
                    c_030[-1].append(cost[:, 1:9].mean())  # axis=1))
                    c_060[-1].append(cost[:, 9:].mean())  # axis=1))

                    print "Mean cost: %.2f, 00: %.2f, 30: %.2f, 60: %.2f" % (
                        cost.mean(), c_000[-1][-1], c_030[-1][-1], c_060[-1][-1]),
                    print "| Noise level: %.4f (%.2f %%)" % (noise, cloud)

            np.savez_compressed("%s-layer.npz" % activation, x=x, y=costs, y00=c_000, y30=c_030, y60=c_060)
            plt.plot(np.mean(x, axis=1), np.mean(costs, axis=1),
                     c="C0", ls=ls, label=r"overall" if ls is "-" else None)
            plt.plot(np.mean(x, axis=1), np.mean(c_000, axis=1),
                     c="C1", ls=ls, label=r"$0^\circ$" if ls is "-" else None)
            plt.plot(np.mean(x, axis=1), np.mean(c_030, axis=1),
                     c="C2", ls=ls, label=r"$30^\circ$" if ls is "-" else None)
            plt.plot(np.mean(x, axis=1), np.mean(c_060, axis=1),
                     c="C3", ls=ls, label=r"$60^\circ$" if ls is "-" else None)
            plt.plot([], [], c='black', ls=ls, label="%s layer" % activation)
            # np.savez_compressed("%02d-order.npz" % order, x=x, y=costs, y00=c_000, y30=c_030, y60=c_060)
            # plt.plot(np.mean(x, axis=1), np.mean(costs, axis=1),
            #          c="C0", ls=ls, label=r"overall" if ls is "-" else None)
            # plt.plot(np.mean(x, axis=1), np.mean(c_000, axis=1),
            #          c="C1", ls=ls, label=r"$0^\circ$" if ls is "-" else None)
            # plt.plot(np.mean(x, axis=1), np.mean(c_030, axis=1),
            #          c="C2", ls=ls, label=r"$30^\circ$" if ls is "-" else None)
            # plt.plot(np.mean(x, axis=1), np.mean(c_060, axis=1),
            #          c="C3", ls=ls, label=r"$60^\circ$" if ls is "-" else None)
            # plt.plot([], [], c='black', ls=ls, label="order %02d" % order)

        plt.legend(ncol=2)
        plt.grid()
        plt.ylim([0, 90])
        if "noise" in xaxis:
            plt.xlim([0, max_noise])
            plt.xlabel(r"noise ($\sigma$)")
        elif "cloud" in xaxis:
            plt.xlim([0, 100])
            plt.xlabel(r"noise ($\%$)")
        plt.ylabel(r"cost ($^\circ$)")
        plt.show()
    elif mode == "pol-contrast":
        tilt = False
        samples = 1000
        nb_noise = 1
        max_noise = 3

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
                cost, d_eff, theta_s = SensorObjective._fitness(theta, phi, alpha, samples=samples,
                                                                tilt=tilt, error=azidist, noise=noise,
                                                                gate_shift=np.deg2rad(54),
                                                                gate_order=2,
                                                                return_mean=False)
                costs[-1].append(cost)
                d_effs[-1].append(d_eff)
                eles[-1].append(theta_s)

                print "Mean cost: %.2f" % cost.mean(),
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
    elif mode == "single":

        algo_name = "sea"
        samples = 130
        fov = 150
        tilt = False
        seed = 1
        thetas = True
        phis = False
        alphas = False
        ws = False

        name = "%s-%s-%03d-%03d%s" % (
            datetime.now().strftime("%Y%m%d"),
            algo_name,
            samples,
            fov,
            "-tilt" if tilt else ""
        )
        if thetas and phis and alphas and ws:
            name += "-%04d" % seed
        else:
            name += "-"
            name += "t" if thetas else "f"
            name += "t" if phis else "f"
            name += "t" if alphas else "f"
            name += "t" if ws else "f"

        print name
        so = SensorObjective(nb_lenses=samples, fov=fov, consider_tilting=tilt,
                             b_thetas=thetas, b_phis=phis, b_alphas=alphas, b_ws=ws)
        x, f, log = optimise(so, algo_name, name=name, verbosity=100, gen=500)
        # x = so.x_init
        # f = 0.
        # log = np.array([])

        x = so.correct_vector(x)
        print "CHAMP x:", x
        print "CHAMP f:", f

        thetas, phis, alphas, w = SensorObjective.devectorise(x)

        s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
        s.visualise_structure(s, title="%s-struct" % name, show=True)
    elif mode == "archipelago":
        import pygmo as pg

        # Initialise the problem
        sf = SensorObjective()
        prob = pg.problem(sf)

        # Initialise the random seed
        pg.set_global_rng_seed(2018)

        # Initialise the algorithms
        sa = pg.algorithm(pg.simulated_annealing(Ts=1., Tf=.01, n_T_adj=1000))

        de = pg.algorithm(pg.de(gen=10000, F=.8, CR=.9))

        # local = pg.algorithm(pg.nlopt("cobyla"))

        # Initialise archipelago
        archi = pg.archipelago(n=10, udi=pg.thread_island(), algo=sa, prob=prob, pop_size=100)
        archi.push_back(algo=de, prob=prob, size=100, udi=pg.thread_island())
        archi.push_back(algo=sa, prob=prob, size=100, udi=pg.thread_island())
        archi.push_back(algo=de, prob=prob, size=100, udi=pg.thread_island())
        archi.push_back(algo=sa, prob=prob, size=100, udi=pg.thread_island())

        archi.evolve(100)

        print archi

        archi.wait_check()

        x = archi.get_champions_x()
        f = archi.get_champions_f()
        print "CHAMP X:", x[np.argmin(f)]
        print "CHAMP F:", f.min()

        thetas, phis, alphas, w = SensorObjective.devectorise(x[np.argmin(f)])

        from compoundeye.sensor import CompassSensor

        s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
        s.visualise_structure(s)
    elif mode == "plot-overall":

        algo_name = "pso"
        names = [
            # "sea-060-060",
            # "pso-060-060",
            "%s-060-060" % algo_name,
            "%s-130-150" % algo_name,
            "%s-060-060-tilt" % algo_name,
            "%s-130-150-tilt" % algo_name
        ]
        labels = [
            # "SEA",
            # "PSO",
            "normal - non-tilting",
            "wide - non-tilting",
            "normal - tilting",
            "wide - tilting",
        ]

        log_names = get_log(algo_name)
        gens = 1000
        for c, (name, label) in enumerate(zip(names, labels)):
            x = np.empty((0, gens/100), dtype=np.float32)
            y = np.empty((0, gens/100), dtype=np.float32)
            for seed in xrange(1, 10):
                for f in os.listdir(__datadir__):
                    if not f.endswith("%s-%04d.npz" % (name, seed)):
                        continue
                    print f
                    data = np.load(__datadir__ + f)
                    # print log_names
                    # x_new = data["log"][:, log_names.index("gen")]
                    x_new = np.arange(1, gens+1, 100)
                    y_new = data["log"][:, log_names.index("gbest")]
                    missing = gens/100 - y_new.shape[0]
                    if missing > 0:
                        # x_new = np.append(x_new, np.full(missing, np.nan))
                        y_new = np.append(y_new, np.full(missing, np.nan))
                    x = np.vstack([x, x_new[:gens/100]])
                    y = np.vstack([y, y_new[:gens/100]])

            x_mean = np.nanmean(x, axis=0)
            y_mean = np.nanmean(y, axis=0)
            y_std = np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))

            plt.figure(algo_name)
            plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, facecolor="C%d" % c, alpha=.5)
            plt.plot(x_mean, y_mean, color="C%d" % c, label=label)

        plt.figure(algo_name)
        plt.legend()
        plt.xlabel("generations")
        plt.ylabel("objective function (degrees)")
        plt.ylim([0, 90])
        plt.xlim([0, gens])
        plt.show()
    elif mode == "plot-params":
        nb_lenses = 60
        fov = 60
        thetas = True
        phis = True
        alphas = True
        ws = True
        label = "pso-%03d-%03d-tilt" % (nb_lenses, fov)
        style = "img"

        so = SensorObjective(nb_lenses, fov,
                             b_thetas=thetas, b_phis=phis, b_alphas=alphas, b_ws=ws)

        if thetas and phis and alphas and ws:
            p = re.compile(r"[0-9]{8}-%s-[0-9]{4}.npz" % label)
        else:
            tag = ""
            tag += "t" if thetas else "f"
            tag += "t" if phis else "f"
            tag += "t" if alphas else "f"
            tag += "t" if ws else "f"
            p = re.compile(r"[0-9]{8}-%s-%s.npz" % (label, tag))
        x_champ = None
        f_champ = None
        file_champ = None
        for f in os.listdir(__datadir__):
            if p.match(f):
                data = np.load(__datadir__ + f)
                of = data["f"]
                if f_champ is None or of < f_champ:
                    f_champ = of
                    x_champ = so.correct_vector(data["x"])
                    file_champ = f

        if not (thetas or phis or alphas or ws):
            x_champ = so.x_init
            f_champ = 0.

        if f_champ is not None:
            print file_champ

            so = SensorObjective()
            thetas, phis, alphas, w = SensorObjective.devectorise(x_champ)
            # thetas, phis, alphas, w = SensorObjective.devectorise(so.x_init)

            s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
            s.visualise_structure(s, title="%s-struct" % label, show=False)

            if style == "plot":
                phi_tb1 = np.linspace(0., 2 * np.pi, 9, endpoint=True)  # TB1 preference angles
                cmap = cm.get_cmap("hsv")
                plt.figure("weights", figsize=(7, 5))
                ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=5, polar=True)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                for phi, wi in zip(alphas, w):
                    c = (phi % (2 * np.pi)) / (2 * np.pi)
                    wii = np.append(wi, wi[0])
                    plt.plot(phi_tb1, wii, color=cmap(c))
                plt.ylim([-.1, .1])
                plt.xticks((phi_tb1[:-1] + np.pi) % (2*np.pi) - np.pi)
                plt.subplot2grid((1, 6), (0, 5), rowspan=1, colspan=1)
                l = len(w)
                plt.imshow(np.linspace(0, 2*np.pi, l+1)[:, np.newaxis],
                           vmin=0, vmax=2*np.pi, cmap="hsv", origin="lower")
                plt.tick_params(axis='y', labelleft='off', labelright='on')
                plt.yticks([0, l/4., l/2., 3.*l/4., l],
                           [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$3\frac{\pi}{2}$", r"$2\pi$"])
                plt.xticks([])
            elif style == "img":
                plt.figure("weights-img", figsize=(10, 5))
                plt.imshow(w.T, vmin=-1., vmax=1., cmap="coolwarm")
                plt.yticks([0, 7], ["1", "8"])
                ticks = np.linspace(0, w.shape[0]-1, 7)
                plt.xticks(ticks, ["%d" % tick for tick in (ticks+1)])
            plt.show()
