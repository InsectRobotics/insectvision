import numpy as np
from geometry import fibonacci_sphere
from sphere import angdist, eledist, azidist
from datetime import datetime
import os

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/opt/"


# Transformation matrix of turbidity to luminance coefficients
T_L = np.array([[0.1787, -1.4630],
                [-0.3554, 0.4275],
                [-0.0227, 5.3251],
                [0.1206, -2.5771],
                [-0.0670, 0.3703]])


class SensorFunction(object):

    def __init__(self, nb_lenses=60, fov=60, nb_tl2=16, nb_tb1=8):
        # Initialise the position and orientation of the lenses with fibonacci distribution
        thetas, phis = fibonacci_sphere(samples=nb_lenses, fov=fov)
        thetas = (thetas + np.pi) % (2 * np.pi) - np.pi
        phis = (phis + np.pi) % (2 * np.pi) - np.pi
        alphas = (phis + 3 * np.pi / 2) % (2 * np.pi) - np.pi

        # initialise weights of the computational model
        phi_tl2 = np.linspace(0., 4 * np.pi, nb_tl2, endpoint=False)  # TL2 preference angles
        phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles
        w_tl2 = 1. / nb_lenses * np.sin(phi_tl2[np.newaxis] - alphas[:, np.newaxis])
        w_tb1 = float(nb_tb1) / float(nb_tl2) * np.sin(phi_tb1[np.newaxis] - phi_tl2[:, np.newaxis] + np.pi / 2)
        w = w_tl2.dot(w_tb1)

        # create initial feature-vector
        self.x_init = self.vectorise(thetas, phis, alphas, w)
        self.ndim = (3 + nb_tb1) * nb_lenses  # number of features
        self.lb = np.hstack((  # lower bound of parameters
            np.zeros(nb_lenses),
            np.full(nb_lenses, -np.pi),
            np.full(nb_lenses, -np.pi),
            -np.ones(nb_tb1 * nb_lenses)
        ))
        self.ub = np.hstack((  # upper bound of parameters
            np.full(nb_lenses, np.deg2rad(fov) / 2),
            np.full(nb_lenses, np.pi),
            np.full(nb_lenses, np.pi),
            np.ones(nb_tb1 * nb_lenses)
        ))

    def fitness(self, x):
        theta, phi, alpha, w = self.devectorise(x)
        return [self._fitness(theta, phi, alpha, w=w)]

    @staticmethod
    def _fitness(theta, phi, alpha, w=None, samples=1000, error=angdist):
        theta_s, phi_s = fibonacci_sphere(samples=samples, fov=180)
        d = np.zeros_like(theta_s)
        for i, (e, a) in enumerate(zip(theta_s, phi_s)):
            _, dop, aop = SensorFunction.encode(e, a, theta, phi)
            ele, azi = SensorFunction.decode(dop, aop, alpha, w=w)
            d[i] = np.absolute(error(np.array([e, a]), np.array([ele, azi])))
        return np.rad2deg(d.mean())

    def get_bounds(self):
        return list(self.lb), list(self.ub)

    def get_name(self):
        return "Sensor Function"

    def get_extra_info(self):
        return "\tDimensions: % 4d" % self.ndim

    @staticmethod
    def encode(theta_s, phi_s, theta, phi, tau=2., c1=.6, c2=4.):
        """

        :param theta: element elevation (from Z point)
        :param phi: element azimuth (from N point)
        :param theta_s: sun elevation (from Z point)
        :param phi_s: sun azimuth (from N point)
        :param tau: turbidity
        :param c1: maximum polarisation parameter 1
        :param c2: maximum polarisation parameter 2
        :return:
        """

        eps = np.finfo(float).eps
        A, B, C, D, E = T_L.dot(np.array([tau, 1.]))  # sky parameters
        T_T = np.linalg.pinv(T_L)
        tau, c = T_T.dot(np.array([A, B, C, D, E]))
        tau /= c  # turbidity correction

        gamma = np.arccos(
            np.sin(np.pi / 2 - theta) * np.sin(np.pi / 2 - theta_s) +
            np.cos(np.pi / 2 - theta) * np.cos(np.pi / 2 - theta_s) * np.cos(np.absolute(phi - phi_s))
        )

        def L(chi, z):  # Prez. et. al. Luminance function
            return (1 + A * np.exp(B / (np.cos(z) + eps))) * (1 + C * np.exp(D * chi) + E * np.square(np.cos(chi)))

        # Intensity
        I_prez, I_00, I_90 = L(gamma, theta), L(0., theta_s), L(np.pi / 2, np.absolute(theta_s - np.pi / 2))
        # influence of sky intensity
        I = (1. / (I_prez + eps) - 1. / (I_00 + eps)) * I_00 * I_90 / (I_00 - I_90 + eps)
        chi = (4. / 9. - tau / 120.) * (np.pi - 2 * theta_s)
        Y_z = (4.0453 * tau - 4.9710) * np.tan(chi) - 0.2155 * tau + 2.4192
        L = np.clip(Y_z * I_prez / (I_00 + np.finfo(float).eps) / 15., 0., 1.)

        # Degree of Polarisation
        M_p = np.exp(-(tau - c1) / (c2 + eps))
        LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
        # P = np.clip(2. / np.pi * M_p * LP * (theta * np.cos(theta) + (np.pi/2 - theta) * I), 0., 1.)
        P = LP

        # Angle of polarisation
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).T
        v_s = np.array([np.sin(theta_s) * np.cos(phi_s), np.sin(theta_s) * np.sin(phi_s), np.cos(theta_s)])

        R = point2rotmat(v_s)
        v_rot = v.dot(R)
        A = np.arctan2(v_rot[:, 1], v_rot[:, 0]) % (2 * np.pi)

        return L, P, A

    @staticmethod
    def decode(P, A, alpha, nb_tl2=16, nb_tb1=8, w_tl2=None, w_tb1=None, w=None):
        """

        :param P: degree of polarisation
        :param A: angle of polarisation (from N point)
        :param alpha: element orientation (from N point)
        :param nb_tl2: number of Tl2 neurons
        :param nb_tb1: number of TB1 neurons
        :param w_tl2: TL2 layer weights
        :param w_tb1: TB1 layer weights
        :param w: total weights
        :return:
        """
        phi_tl2 = np.linspace(0., 4 * np.pi, nb_tl2, endpoint=False)  # TL2 preference angles
        phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles

        # polarisation filter
        r = (np.sqrt(np.square(np.sin(A - alpha)) + np.square(np.cos(A - alpha)) * np.square(1. - P)) -
             np.sqrt(np.square(np.cos(A - alpha)) + np.square(np.sin(A - alpha)) * np.square(1. - P))) / \
            (np.sqrt(np.square(np.sin(A - alpha)) + np.square(np.cos(A - alpha)) * np.square(1. - P)) +
             np.sqrt(np.square(np.cos(A - alpha)) + np.square(np.sin(A - alpha)) * np.square(1. - P)))

        # import matplotlib.pyplot as plt
        #
        # plt.figure("sensor", figsize=(5, 5))
        #
        # ax = plt.subplot(111, polar=True)
        # ax.scatter(phis, thetas, s=50 * fov / nb_lenses, c=r*10., marker='o', cmap="coolwarm", vmin=-1, vmax=1)
        # ax.scatter(azi_sun, ele_sun, s=100, marker='o', edgecolor='black', facecolor='yellow')
        # ax.set_theta_zero_location("N")
        # ax.set_ylim([0, np.pi/2])
        # ax.set_yticks([])
        # ax.set_title("Sensor Response")
        #
        # plt.show()

        # computational model
        S = P.shape[-1]
        if w is None and w_tl2 is None:
            w_tl2 = 1. / S * np.sin(phi_tl2[np.newaxis] - alpha[:, np.newaxis])
        if w is None and w_tb1 is None:
            w_tb1 = float(nb_tb1) / float(nb_tl2) * np.sin(phi_tb1[np.newaxis] - phi_tl2[:, np.newaxis] + np.pi / 2)
        if w is None:
            w = w_tl2.dot(w_tb1)
        r_tb1 = r.dot(w)

        # decode signal - FFT
        R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * np.pi / 4.))
        theta_s_ = np.absolute(R) % (2 * np.pi)  # sun elevation (prediction)
        phi_s_ = (np.pi - np.arctan2(R.imag, R.real)) % (2 * np.pi) - np.pi  # sun azimuth (prediction)

        return theta_s_, phi_s_

    @staticmethod
    def vectorise(theta, phi, alpha, w):
        return np.hstack((theta, phi, alpha, w.flatten()))

    @staticmethod
    def devectorise(v):
        nb_lenses = v.shape[0] / 11
        theta = v[:nb_lenses]
        phi = v[nb_lenses:(2 * nb_lenses)]
        alpha = v[(2 * nb_lenses):(3 * nb_lenses)]
        w = v[(3 * nb_lenses):].reshape((nb_lenses, 8))
        return theta, phi, alpha, w


def point2rotmat(p):
    z = np.array([0., 0., 1.])
    v = np.cross(z, p)
    c = np.dot(z, p)
    v_x = np.array([[0., -v[2], v[1]],
                    [v[2], 0., -v[0]],
                    [-v[1], v[0], 0.]])
    return np.eye(3) + v_x + np.matmul(v_x, v_x) / (1 + c)


if __name__ == "__main__":
    import pygmo as pg

    sf = SensorFunction()

    pg.set_global_rng_seed(2018)

    algo = pg.algorithm(pg.sea(1000))
    algo.set_verbosity(50)

    prob = pg.problem(sf)
    pop = pg.population(prob, size=20)
    # pop.push_back(sf.x_init)

    print "POP (len):", len(pop.get_x())
    pop = algo.evolve(pop)
    uda = algo.extract(pg.sea)
    print "UDA", uda.get_log()

    x = np.array(pop.champion_x)
    f = np.array(pop.champion_f)
    # print "CHAMP X:", pop.champion_x
    # print "CHAMP F:", pop.champion_f

    name = "%s-%s-%.2f" % (datetime.now().strftime("%Y%m%d"), algo.get_name(), f)
    np.savez_compressed(__datadir__ + "%s.npz" % name, x=x, f=f)

    thetas, phis, alphas, w = SensorFunction.devectorise(pop.champion_x)

    from sensor import CompassSensor

    s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
    s.visualise_structure(s)


if __name__ == "__main_2__":
    from geometry import angles_distribution

    nb_lenses = 60
    fov = 60
    nb_tl2 = 16
    nb_tb1 = 8

    try:
        thetas, phis, fit = angles_distribution(nb_lenses=nb_lenses, fov=fov)
    except ValueError:
        thetas = np.empty(0, dtype=np.float32)
        phis = np.empty(0, dtype=np.float32)
        fit = False

    if not fit:
        thetas, phis = fibonacci_sphere(samples=nb_lenses, fov=fov)
    thetas = (thetas + np.pi) % (2 * np.pi) - np.pi
    # phis = (phis + np.pi) % (2 * np.pi) - np.pi
    alphas = (phis + 3*np.pi/2) % (2 * np.pi) - np.pi
    print np.rad2deg(alphas.min()), np.rad2deg(alphas.max())
    # alphas = phis + np.pi / 2

    phi_tl2 = np.linspace(0., 4 * np.pi, nb_tl2, endpoint=False)  # TL2 preference angles
    phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles
    w_tl2 = 1. / nb_lenses * np.sin(phi_tl2[np.newaxis] - alphas[:, np.newaxis])
    w_tb1 = float(nb_tb1) / float(nb_tl2) * np.sin(phi_tb1[np.newaxis] - phi_tl2[:, np.newaxis] + np.pi / 2)
    w = w_tl2.dot(w_tb1)

    d = SensorFunction._fitness(thetas, phis, alphas, w)
    print np.rad2deg(d.mean())


if __name__ == "__main_2__":
    from geometry import angles_distribution

    nb_lenses = 60
    fov = 60
    # ele_sun = 0.
    ele_sun = np.pi/6
    # azi_sun = 0.
    azi_sun = np.pi/2

    try:
        thetas, phis, fit = angles_distribution(nb_lenses=nb_lenses, fov=fov)
    except ValueError:
        thetas = np.empty(0, dtype=np.float32)
        phis = np.empty(0, dtype=np.float32)
        fit = False

    if not fit:
        thetas, phis = fibonacci_sphere(samples=nb_lenses, fov=fov)

    i, p, a = SensorFunction.encode(ele_sun, azi_sun, thetas, phis)

    # import matplotlib.pyplot as plt
    #
    # plt.figure("sky-features", figsize=(15, 5))
    #
    # ax = plt.subplot(131, polar=True)
    # ax.scatter(phis, thetas, s=50 * fov / nb_lenses, c=i, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
    # ax.scatter(azi_sun, ele_sun, s=100, marker='o', edgecolor='black', facecolor='yellow')
    # ax.set_theta_zero_location("N")
    # ax.set_ylim([0, np.pi/2])
    # ax.set_yticks([])
    # ax.set_title("Intensity")
    #
    # ax = plt.subplot(132, polar=True)
    # ax.scatter(phis, thetas, s=50 * fov / nb_lenses, c=p, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
    # ax.scatter(azi_sun, ele_sun, s=100, marker='o', edgecolor='black', facecolor='yellow')
    # ax.set_theta_zero_location("N")
    # ax.set_ylim([0, np.pi/2])
    # ax.set_yticks([])
    # ax.set_title("Degree of Polarisation")
    #
    # ax = plt.subplot(133, polar=True)
    # ax.scatter(phis, thetas, s=50 * fov / nb_lenses, c=a, marker='o', cmap="hsv", vmin=0, vmax=2*np.pi)
    # ax.scatter(azi_sun, ele_sun, s=100, marker='o', edgecolor='black', facecolor='yellow')
    # ax.set_theta_zero_location("N")
    # ax.set_ylim([0, np.pi/2])
    # ax.set_yticks([])
    # ax.set_title("Angle of Polarisation")
    #
    # plt.show()

    # above 8 works fine
    ele, azi = SensorFunction.decode(p, a, phis + np.pi/2, nb_tl2=16, nb_tb1=8)
    ele = ele % (2 * np.pi)
    azi = azi % (2 * np.pi)

    print "E: % 3d --> % 3d,  A: % 3d --> % 3d" % (np.rad2deg(ele_sun), np.rad2deg(ele), np.rad2deg(azi_sun), np.rad2deg(azi))

    dist = angdist(np.array([ele_sun, azi_sun]), np.array([ele, azi]))
    edist = eledist(np.array([ele_sun, azi_sun]), np.array([ele, azi]))
    adist = azidist(np.array([ele_sun, azi_sun]), np.array([ele, azi]))
    print "D: % 3d, E: % 3d, A: % 3d" % (np.rad2deg(dist), np.rad2deg(edist), np.rad2deg(adist))
