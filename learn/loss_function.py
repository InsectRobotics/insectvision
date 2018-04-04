#!/usr/bin/env python
# Loss functions for data in spherical representations
#

from code import decode_sph
from compoundeye.geometry import fibonacci_sphere
from sphere import angle_between, sph2vec, vec2sph, angdist
from sphere.transform import point2rotmat, tilt as transtilt

import numpy as np
import pygmo as pg

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2018, The Invisible Cues Project"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


class SensorObjective(object):
    # Transformation matrix of turbidity to luminance coefficients
    T_L = np.array([[0.1787, -1.4630],
                    [-0.3554, 0.4275],
                    [-0.0227, 5.3251],
                    [0.1206, -2.5771],
                    [-0.0670, 0.3703]])

    def __init__(self, nb_lenses=60, fov=60, nb_tb1=8, consider_tilting=False,
                 b_thetas=True, b_phis=True, b_alphas=True, b_ws=True):
        # Initialise the position and orientation of the lenses with fibonacci distribution
        thetas, phis = fibonacci_sphere(samples=nb_lenses, fov=fov)
        thetas = (thetas + np.pi) % (2 * np.pi) - np.pi
        phis = (phis + np.pi) % (2 * np.pi) - np.pi
        alphas = (phis + 3 * np.pi / 2) % (2 * np.pi) - np.pi

        # initialise weights of the computational model
        phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles
        w = nb_tb1 / (2. * nb_lenses) * np.sin(phi_tb1[np.newaxis] - alphas[:, np.newaxis]) * np.sin(thetas)

        # create initial feature-vector
        self.x_init = self.vectorise(thetas, phis, alphas, w)
        self.theta_init = thetas
        self.phi_init = phis
        self.w_f = lambda a: nb_tb1 / (2. * nb_lenses) * np.maximum(np.sin(phi_tb1[np.newaxis] - a[:, np.newaxis]), 0.)
        self.ndim = (3 + nb_tb1) * nb_lenses  # number of features
        self.lb = np.hstack((  # lower bound of parameters
            np.zeros(nb_lenses),          # theta
            np.full(nb_lenses, -np.pi),   # phi
            np.full(nb_lenses, -np.pi),   # alpha
            -np.ones(nb_tb1 * nb_lenses)  # weights
        ))
        self.ub = np.hstack((  # upper bound of parameters
            np.full(nb_lenses, np.deg2rad(fov) / 2),  # theta
            np.full(nb_lenses, np.pi),                # phi
            np.full(nb_lenses, np.pi),                # alpha
            np.ones(nb_tb1 * nb_lenses)               # weights
        ))
        self._thetas = b_thetas
        self._phis = b_phis
        self._alphas = b_alphas
        self._ws = b_ws
        self.__consider_tilting = consider_tilting

    def correct_vector(self, v):
        l = v.shape[0] / 11
        if not self._thetas:
            v[:l] = self.theta_init
        if not self._phis:
            v[l:2*l] = self.phi_init
        if not self._alphas:
            v[2*l:3*l] = (v[l:2*l] + 3 * np.pi / 2) % (2 * np.pi) - np.pi
        if not self._ws:
            v[3*l:] = self.w_f(v[2*l:3*l]).flatten()
        return v

    def fitness(self, x):
        theta, phi, alpha, w = self.devectorise(self.correct_vector(x))
        return [self._fitness(theta, phi, alpha, w=w, tilt=self.__consider_tilting)]

    def gradient(self, x):
        return pg.estimate_gradient_h(self.fitness, x)

    def get_bounds(self):
        return list(self.lb), list(self.ub)

    def get_name(self):
        return "Sensor Objective Function"

    def get_extra_info(self):
        return "\tDimensions: % 4d" % self.ndim

    @staticmethod
    def _fitness(theta, phi, alpha, w=None, samples=1000, tilt=False, error=angdist):
        if tilt:
            angles = np.array([
                [0., 0.],
                [np.pi/6, 0.], [np.pi/6, np.pi/4], [np.pi/6, 2*np.pi/4], [np.pi/6, 3*np.pi/4],
                [np.pi/6, 4*np.pi/4], [np.pi/6, 5*np.pi/4], [np.pi/6, 6*np.pi/4], [np.pi/6, 7*np.pi/4],
                [np.pi/3, 0.], [np.pi/3, np.pi/4], [np.pi/3, 2*np.pi/4], [np.pi/3, 3*np.pi/4],
                [np.pi/3, 4*np.pi/4], [np.pi/3, 5*np.pi/4], [np.pi/3, 6*np.pi/4], [np.pi/3, 7*np.pi/4]
            ])  # 17
            if samples == 1000:
                samples /= 2
        else:
            angles = np.array([[0., 0.]])  # 1

        theta_s, phi_s = fibonacci_sphere(samples=samples, fov=180)
        d = np.zeros((samples, angles.shape[0]))
        phi_tb1 = np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi/2  # TB1 preference angles
        # shift = 0.
        shift = np.pi/6

        for j, (theta_t, phi_t) in enumerate(angles):
            v_t = sph2vec(theta_t, phi_t, zenith=True)
            v_s = sph2vec(theta_s, phi_s, zenith=True)
            v = sph2vec(theta, phi, zenith=True)
            v_a = sph2vec(np.full(alpha.shape[0], np.pi/2), alpha, zenith=True)
            R = point2rotmat(v_t)
            theta_s_, phi_s_, _ = vec2sph(R.dot(v_s), zenith=True)
            # theta_s_, phi_s_ = transtilt(-theta_t, -phi_t, theta=theta_s, phi=phi_s)

            theta_, phi_, _ = vec2sph(R.T.dot(v), zenith=True)
            # theta_, phi_ = transtilt(theta_t, phi_t, theta=theta, phi=phi)
            _, alpha_, _ = vec2sph(R.T.dot(v_a), zenith=True)
            # _, alpha_ = transtilt(theta_t, phi_t, theta=np.pi/2, phi=alpha)


            if w is None:
                # Gate
                g = np.sqrt(1 - np.square(np.cos(theta + shift) * np.cos(theta_t) +
                                          np.sin(theta + shift) * np.sin(theta_t) * np.cos(np.absolute(phi_t - phi))))
                # alternative form
                # g = np.sin(theta_)  # dynamic gating

                # other versions
                # g = np.sin(theta)  # static gating
                # g = np.ones_like(theta)  # uniform - no gating

                # from compoundeye import CompassSensor
                # s = CompassSensor(thetas=theta, phis=phi, alphas=alpha)
                # CompassSensor.visualise(s, sL=g, colormap="Reds", sides=False, scale=None,
                #                         title="E%03d-A%03d" % (np.rad2deg(theta_t), np.rad2deg(phi_t)))

                w = -8. / (2. * 60.) * np.cos(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * g[:, np.newaxis]
                # w = np.maximum(w, 0)

            for i, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):
                _, dop, aop = SensorObjective.encode(e_org, a_org, theta_, phi_)
                ele, azi = SensorObjective.decode(dop, aop, alpha_, w=w)
                d[i, j] = np.absolute(error(np.array([e, a]), np.array([ele, azi])))

        d_deg = np.rad2deg(d)

        import matplotlib.pyplot as plt

        # plt.figure("Tilts", figsize=(10, 15))
        # for i, ang, dd in zip(range(angles.shape[0]), angles, d.T):
        #     ax = plt.subplot2grid((5, 4), ((i-1) // 4, (i-1) % 4), polar=True)
        #     ax.set_theta_zero_location("N")
        #     ax.set_theta_direction(-1)
        #     plt.scatter(phi_s, np.rad2deg(theta_s), marker=".", c=dd, cmap="Reds", vmin=0, vmax=np.pi/2)
        #     plt.scatter(ang[1]+np.pi, np.rad2deg(ang[0]), marker="o", c="yellow", edgecolors="black")
        #     plt.title(r"$\epsilon_t=%03d, \alpha_t=%03d$" % tuple(np.rad2deg(ang)))
        #     plt.axis("off")
        # plt.subplot2grid((5, 4), (4, 0), colspan=3)
        # plt.imshow([np.arange(0, np.pi/2, np.pi/180)] * 3, cmap="Reds")
        # plt.yticks([])
        # plt.xticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])
        # plt.show()

        plt.figure("Tilts", figsize=(10, 3))
        for i, ang, dd in zip(range(3), angles[[0, 1, 9]], d.T[[0, 1, 9]]):
            ax = plt.subplot2grid((1, 10), (0, i * 3), colspan=3, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            plt.scatter(phi_s, np.rad2deg(theta_s), marker="o", c=dd, cmap="Reds", vmin=0, vmax=np.pi/2)
            plt.scatter(ang[1]+np.pi, np.rad2deg(ang[0]), marker="o", c="yellowgreen", edgecolors="black")
            plt.text(-np.deg2rad(50), 145, ["A.", "B.", "C."][i], fontsize=12)
            plt.axis("off")
        plt.subplot2grid((3, 10), (1, 9))
        plt.imshow(np.array([np.arange(0, np.pi/2, np.pi/180)] * 10).T, cmap="Reds")
        plt.xticks([])
        plt.yticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])
        plt.show()

        # plt.figure("cost-function")
        # w = np.bartlett(10)
        # w /= w.sum()
        # d_000 = np.convolve(w, np.rad2deg(d[:, 0]), mode="same")
        # plt.plot(np.rad2deg(theta_s), d_000, label=r"$0^\circ$")
        # if angles.shape[0] > 1:
        #     d_030 = np.convolve(w, np.rad2deg(d[:, 1]), mode="same")
        #     d_060 = np.convolve(w, np.rad2deg(d[:, 9]), mode="same")
        #     plt.plot(np.rad2deg(theta_s), d_030, label=r"$30^\circ$")
        #     plt.plot(np.rad2deg(theta_s), d_060, label=r"$60^\circ$")
        #     plt.legend()
        # plt.xlim([0, 90])
        # plt.ylim([0, 180])
        # plt.xlabel(r"sun elevation ($^\circ$)")
        # plt.ylabel(r"cost ($^\circ$)")
        # plt.show()

        return d_deg.mean()

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
        A, B, C, D, E = SensorObjective.T_L.dot(np.array([tau, 1.]))  # sky parameters
        T_T = np.linalg.pinv(SensorObjective.T_L)
        tau, c = T_T.dot(np.array([A, B, C, D, E]))
        tau /= c  # turbidity correction

        gamma = np.arccos(
            np.sin(np.pi / 2 - theta) * np.sin(np.pi / 2 - theta_s) +
            np.cos(np.pi / 2 - theta) * np.cos(np.pi / 2 - theta_s) * np.cos(np.absolute(phi - phi_s))
        )

        def L(chi, z):  # Prez. et. al. Luminance function
            i = z < np.pi/2
            f_ = np.zeros_like(z)
            if z.ndim > 0:
                f_[i] = (1 + A * np.exp(B / (np.cos(z[i]) + eps)))
            elif i:
                f_ = (1 + A * np.exp(B / (np.cos(z) + eps)))
            phi_ = (1 + C * np.exp(D * chi) + E * np.square(np.cos(chi)))
            return f_ * phi_

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
            w = nb_tb1 / (2. * S) * np.sin(phi_tb1[np.newaxis] - alpha[:, np.newaxis])
            # w = w_tl2.dot(w_tb1)
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


def angular_distance_rad(y_target, y_predict):
    return angle_between(decode_sph(y_target), decode_sph(y_predict))


def angular_distance_deg(y_target, y_predict):
    return 180 * angular_distance_rad(y_target, y_predict) / np.pi


def angular_distance_per(y_target, y_predict):
    return angle_between(decode_sph(y_target), decode_sph(y_predict)) / np.pi


def angular_distance_3d(y_predict, y_target, theta=True, phi=True):
    if theta:
        thy = y_predict[:, 1]
        tht = y_target[:, 1]
    else:
        thy = np.zeros_like(y_predict[:, 1])
        tht = np.zeros_like(y_target[:, 1])
    if phi:
        phy = y_predict[:, 0]
        pht = y_target[:, 0]
    else:
        phy = np.zeros_like(y_predict[:, 0])
        pht = np.zeros_like(y_target[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).mean())


def angular_deviation_3d(y_predict, y_target, theta=True, phi=True):
    if theta:
        thy = y_predict[:, 1]
        tht = y_target[:, 1]
    else:
        thy = np.zeros_like(y_predict[:, 1])
        tht = np.zeros_like(y_target[:, 1])
    if phi:
        phy = y_predict[:, 0]
        pht = y_target[:, 0]
    else:
        phy = np.zeros_like(y_predict[:, 0])
        pht = np.zeros_like(y_target[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).std())


losses = {
    "adr": angular_distance_rad,
    "angular distance rad": angular_distance_rad,
    "add": angular_distance_deg,
    "angular distance degrees": angular_distance_deg,
    "adp": angular_distance_per,
    "angular distance percentage": angular_distance_per,
    "ad3": angular_distance_3d,
    "angular distance 3D": angular_distance_3d,
    "astd3": angular_deviation_3d,
    "angular deviation 3D": angular_deviation_3d
}


def get_loss(name):
    assert name in losses.keys(), "Name of loss function does not exist."
    return losses[name]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    angles = np.array([
        [0., 0.],
        [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
        [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4], [np.pi / 6, 7 * np.pi / 4],
        [np.pi / 3, 0.], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
        [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4], [np.pi / 3, 7 * np.pi / 4]
    ])  # 17

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    plt.scatter(angles[:, 1], np.rad2deg(angles[:, 0]), c="r", marker="o")
    plt.ylim([0, 90])
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4])
    plt.yticks([0, 30, 60, 90])
    plt.show()
