import numpy as np
from learn import SensorObjective, optimise
from sensor import CompassSensor
from datetime import datetime


if __name__ == "__main_2__":
    from compoundeye.geometry import fibonacci_sphere
    from learn import SensorObjective
    from sphere import sph2vec, vec2sph, azidist
    from sphere.transform import point2rotmat
    import matplotlib.pyplot as plt

    tilt = True
    samples = 1000
    theta, phi = fibonacci_sphere(samples=60, fov=60)
    alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi

    cost = SensorObjective._fitness(theta, phi, alpha, tilt=False, error=azidist)

    print cost

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


# single
if __name__ == "__main__":

    algo_name = "pso"
    samples = 130
    fov = 150
    tilt = False
    seed = 1

    name = "%s-%s-%03d-%03d%s-%04d" % (
        datetime.now().strftime("%Y%m%d"),
        algo_name,
        samples,
        fov,
        "-tilt" if tilt else "",
        seed
    )

    print name
    so = SensorObjective(nb_lenses=samples, fov=fov, consider_tilting=tilt)
    x, f, log = optimise(so, algo_name, name=name, gen=1000, seed=seed,
                         save_log_x=False, plot=False)
    # x = so.x_init
    # f = 0.
    # log = np.array([])

    # print "CHAMP x:", x
    # print "CHAMP f:", f

    # thetas, phis, alphas, w = SensorObjective.devectorise(x)

    # s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
    # s.visualise_structure(s, title="%s-struct" % name, show=True)


# archipelago
if __name__ == "__main_2__":
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

    from sensor import CompassSensor

    s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
    s.visualise_structure(s)


if __name__ == "__main_2__":
    from learn.optimisation import __datadir__, get_log
    import matplotlib.pyplot as plt
    import os

    names = [
        "sea-060-060",
        "sea-130-150",
        "sea-060-060-tilt",
        "sea-130-150-tilt"
    ]
    labels = [
        "normal - non-tilting",
        "wide - non-tilting",
        "normal - tilting",
        "wide - tilting",
    ]

    log_names = get_log("sea")
    gens = 20000
    for c, (name, label) in enumerate(zip(names, labels)):
        x = np.empty((0, gens/100), dtype=np.float32)
        y = np.empty((0, gens/100), dtype=np.float32)
        for seed in xrange(1, 10):
            for f in os.listdir(__datadir__):
                if not f.endswith("%s-%04d.npz" % (name, seed)):
                    continue
                print f
                data = np.load(__datadir__ + f)
                # x_new = data["log"][:, log_names.index("gen")]
                x_new = np.arange(1, gens+1, 100)
                y_new = data["log"][:, log_names.index("best")]
                missing = gens/100 - y_new.shape[0]
                if missing > 0:
                    # x_new = np.append(x_new, np.full(missing, np.nan))
                    y_new = np.append(y_new, np.full(missing, np.nan))
                x = np.vstack([x, x_new[:gens/100]])
                y = np.vstack([y, y_new[:gens/100]])

        x_mean = np.nanmean(x, axis=0)
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0) / np.sqrt(np.sum(~np.isnan(y), axis=0))

        plt.figure("SEA")
        plt.fill_between(x_mean, y_mean - y_std, y_mean + y_std, facecolor="C%d" % c, alpha=.5)
        plt.plot(x_mean, y_mean, color="C%d" % c, label=label)

    plt.figure("SEA")
    plt.legend()
    plt.xlabel("generations")
    plt.ylabel("objective function (degrees)")
    plt.ylim([0, 90])
    plt.xlim([0, gens])
    plt.show()
    #


if __name__ == "__main_2__":
    from learn.optimisation import __datadir__
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import os
    import re

    label = "sea-060-060-tilt"

    p = re.compile(r"[0-9]{8}-%s-[0-9]{4}.npz" % label)
    x_champ = None
    f_champ = None
    file_champ = None
    for f in os.listdir(__datadir__):
        if p.match(f):
            data = np.load(__datadir__ + f)
            of = data["f"]
            if f_champ is None or of < f_champ:
                f_champ = of
                x_champ = data["x"]
                file_champ = f

    if f_champ is not None:
        print file_champ

        so = SensorObjective()
        thetas, phis, alphas, w = SensorObjective.devectorise(x_champ)
        # thetas, phis, alphas, w = SensorObjective.devectorise(so.x_init)

        s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
        s.visualise_structure(s, title="%s-struct" % label, show=False)

        cmap = cm.get_cmap("hsv")
        plt.figure("weights", figsize=(10, 5))
        for phi, wi in zip(alphas, w):
            c = (phi % (2 * np.pi)) / (2 * np.pi)
            plt.plot(wi, color=cmap(c))
        plt.xlim([0, 7])
        plt.ylim([-.5, .5])
        plt.show()
