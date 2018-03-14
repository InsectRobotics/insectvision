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

    cost = SensorObjective._fitness(theta, phi, alpha, tilt=True, error=azidist)

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

    algo_name = "sea"
    samples = 130
    fov = 150
    tilt = False

    name = "%s-%s-%03d-%03d%s" % (
        datetime.now().strftime("%Y%m%d"),
        algo_name,
        samples,
        fov,
        "-tilt" if tilt else ""
    )
    so = SensorObjective(nb_lenses=samples, fov=fov, consider_tilting=tilt)
    x, f, log = optimise(so, algo_name, name=name, gen=1000000)
    # x = so.x_init
    # f = 0.
    # log = np.array([])

    print "CHAMP x:", x
    print "CHAMP f:", f

    thetas, phis, alphas, w = SensorObjective.devectorise(x)

    s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
    s.visualise_structure(s, title="%s-struct" % name, show=True)


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
    from learn.optimisation import __datadir__, plot_log

    name = "20180313-sea-130-150-tilt"

    data = np.load(__datadir__ + "%s.npz" % name)
    plot_log(data["log"], algo_name="sea", title=name)

    thetas, phis, alphas, w = SensorObjective.devectorise(data["x"])

    s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
    s.visualise_structure(s, title="%s-struct" % name, show=True)
