from geometry import angles_distribution, fibonacci_sphere
from environment import Environment, spectrum, spectrum_influence, eps
from sphere.transform import tilt
from sphere import angdist

import numpy as np


class CompoundEye(object):

    def __init__(self, n, omega, rho, nb_pr=8, theta_c=0., phi_c=0., name="compound-eye"):
        """

        :param n: number of ommatidia
        :type n: int
        :param omega: receiptive field (degrees)
        :type omega: float
        :param nb_pr: number of photo-receptors per ommatidium
        :type nb_pr: int
        :param rho: acceptance angle (degrees)
        :type rho: float, np.ndarray
        """
        try:
            self.theta, self.phi, fit = angles_distribution(n, float(omega))
        except ValueError:
            self.theta = np.empty(0, dtype=np.float32)
            self.phi = np.empty(0, dtype=np.float32)
            fit = False

        if not fit or n > 100:
            self.theta, self.phi = fibonacci_sphere(n, float(omega))

        # create transformation matrix of the perceived light with respect to the optical properties of the eye
        rho = np.deg2rad(rho)
        self.rho = rho if rho.size == n else np.full(n, rho)
        sph = np.array([self.theta, self.phi])
        i1, i2 = np.meshgrid(np.arange(n), np.arange(n))
        i1, i2 = i1.flatten(), i2.flatten()
        sph1, sph2 = sph[:, i1], sph[:, i2]
        d = np.square(angdist(sph1, sph2).reshape((n, n)))
        sigma = np.square([self.rho] * n) + np.square([self.rho] * n).T
        self._rho_gaussian = np.exp(-d/sigma)
        self._rho_gaussian /= np.sum(self._rho_gaussian, axis=1)

        # by default the phoro-receptors are white light sensitive and no POL sensitive
        self.rhabdom = np.array([[spectrum["w"]] * n] * nb_pr)  # spectrum sensitivity of each rhabdom
        self.mic_l = np.zeros((nb_pr, n), dtype=float)  # local (in the ommatidium) angle of microvilli
        self.mic_a = (self.phi + np.pi/2) % (2 * np.pi) - np.pi  # global (in the compound eye) angle of mictovilli
        self.mic_p = np.zeros((nb_pr, n), dtype=float)  # polarisation sensitivity of microvilli
        self._theta_c = theta_c
        self._phi_c = phi_c
        self._theta_t = 0.
        self._phi_t = 0.
        self.__r = np.full((n), np.nan)

        self._is_called = False
        self.name = name

    def __call__(self, env, *args, **kwargs):
        """

        :param env: the environment where the photorectors can percieve light
        :type env: Environment
        :param args: unlabeled arguments
        :type args: list
        :param kwargs: labeled arguments
        :type kwargs: dict
        :return:
        """
        env.theta_t = self.theta_t
        env.phi_t = self.phi_t

        _, alpha = tilt(self.theta_t, self.phi_t + np.pi, theta=np.pi / 2, phi=self.mic_a)
        y, p, a = env(self.theta, self.phi, *args, **kwargs)

        # influence of the acceptance angle on the luminance and DOP
        # y = y.dot(self._rho_gaussian)
        # p = p.dot(self._rho_gaussian)

        # influence of the wavelength on the perceived light
        ry = spectrum_influence(y, self.rhabdom)

        s = ry * ((np.square(np.sin(a - alpha + self.mic_l)) +
                   np.square(np.cos(a - alpha + self.mic_l)) * np.square(1. - p)) * self.mic_p + (1. - self.mic_p))
        self.__r = np.sqrt(s)

        self._is_called = True

        return self.__r

    @property
    def theta_t(self):
        return self._theta_t

    @theta_t.setter
    def theta_t(self, value):
        theta_t, phi_t = tilt(self._theta_c, self._phi_c - np.pi, theta=self._theta_t, phi=self._phi_t)
        self._theta_t, phi_t = tilt(self._theta_c, self._phi_c, theta=value, phi=self.phi_t)

    @property
    def phi_t(self):
        return self._phi_t

    @phi_t.setter
    def phi_t(self, value):
        theta_t, phi_t = tilt(self._theta_c, self._phi_c - np.pi, theta=self._theta_t, phi=self._phi_t)
        theta_t, self._phi_t = tilt(self._theta_c, self._phi_c, theta=self.theta_t, phi=value)

    @property
    def r(self):
        assert self._is_called, "No light has passed through the sensors yet."
        return self._r

    def __repr__(self):
        return "%s(name=%s, n=%d, omega=%f, rho=%f)" % (
            self.__class__.__name__, self.name, self.theta.size, np.max(self.theta) * 2, self.rho[0])


class DRA(CompoundEye):

    def __init__(self, n=60, omega=56, rho=5.4, nb_pr=8, theta_c=0., phi_c=0., name="dra"):
        """

        :param n: number of ommatidia
        :type n: int
        :param omega: receiptive field (degrees)
        :type omega: float
        :param rho: acceptance angle (degrees)
        :type rho: float, np.ndarray
        """
        super(DRA, self).__init__(n=n, omega=omega, rho=rho, nb_pr=nb_pr, theta_c=theta_c, phi_c=phi_c, name=name)

        # set as default the desert ants' ommatidia set-up
        self.rhabdom = np.array([[spectrum["uv"], spectrum["uv"], spectrum["w"], spectrum["uv"],
                                  spectrum["uv"], spectrum["uv"], spectrum["w"], spectrum["uv"]]] * n).T
        self.mic_l = np.array([[0., np.pi/2, np.pi/2, np.pi/2, 0., np.pi/2, np.pi/2, np.pi/2]] * n).T
        self.mic_p[:] = 1.
        self.__r_op = np.full(n, np.nan)
        self.__r_po = np.full(n, np.nan)
        self.__r_pol = np.full(n, np.nan)

        self.__is_called = False

    def __call__(self, my_sky, *args, **kwargs):
        r = super(DRA, self).__call__(my_sky, *args, **kwargs)

        self.__r_op = np.sum(np.cos(2 * self.mic_l) * r, axis=0)
        self.__r_po = np.sum(r, axis=0)
        self.__r_pol = self.__r_op / (self.__r_po + eps)
        self.__r_po = 2. * self.__r_po / np.max(self.__r_po) - 1.

        self.__is_called = True

        return self.r_pol

    @property
    def r_pol(self):
        assert self._is_called, "No light has passed through the sensors yet."
        return self.__r_pol

    @property
    def r_po(self):
        assert self._is_called, "No light has passed through the sensors yet."
        return self.__r_po

    @property
    def r_op(self):
        assert self._is_called, "No light has passed through the sensors yet."
        return self.__r_op


def visualise(my_sky, y):
    import matplotlib.pyplot as plt

    plt.figure("Luminance", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(my_sky.theta_t, my_sky.phi_t, theta=my_sky.theta_s, phi=my_sky.phi_s)
    ax.scatter(my_sky.phi, my_sky.theta, s=100, c=y, marker='.', cmap='coolwarm', vmin=-1, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    ax.scatter(my_sky.phi_t, my_sky.theta_t, s=50, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


if __name__ == "__main_2__":
    from datetime import datetime
    from ephem import city
    from beeeye import load_both_eyes
    import matplotlib.pyplot as plt

    angle = [np.pi/2, 0., np.pi/3]

    # initialise sky
    obs = city("Edinburgh")
    obs.date = datetime.now()

    # initialise ommatidia features
    ommatidia_left, ommatidia_right = load_both_eyes()
    l_eye = CompoundEye(ommatidia_left)
    l_eye.rotate(*angle)
    r_eye = CompoundEye(ommatidia_right)
    r_eye.rotate(*angle)
    l_eye.sky.obs = obs
    r_eye.sky.obs = obs

    # plot result
    s, p = 20, 4
    # plot eye's structure
    norm = lambda x: x / 25. / np.sqrt(2)

    r_theta, r_phi = r_eye.theta_local, r_eye.phi_local
    l_theta, l_phi = l_eye.theta_local, l_eye.phi_local
    if True:
        plt.figure("Compound eyes - Structure", figsize=(8, 21))

        lum_r = norm(l_eye.sky.L) + (1. - norm(l_eye.sky.L)) * .05
        lum_g = norm(l_eye.sky.L) + (1. - norm(l_eye.sky.L)) * .53
        lum_b = norm(l_eye.sky.L) + (1. - norm(l_eye.sky.L)) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(321)
        plt.title("Left")
        print np.rad2deg(l_eye.yaw_pitch_roll)
        plt.scatter(l_phi, l_theta, c=L, marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        lum_r = norm(r_eye.sky.L) + (1. - norm(r_eye.sky.L)) * .05
        lum_g = norm(r_eye.sky.L) + (1. - norm(r_eye.sky.L)) * .53
        lum_b = norm(r_eye.sky.L) + (1. - norm(r_eye.sky.L)) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_phi, r_theta, c=L, marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_phi, l_theta, c=l_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_phi, r_theta, c=r_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_phi, l_theta, c=l_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_phi, r_theta, c=r_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])
    # plot bee's view
    if True:
        plt.figure("Compound eyes - Bee's view", figsize=(8, 21))

        plt.subplot(321)
        plt.title("Left")
        plt.scatter(l_phi, l_theta, c=l_eye.L, marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_phi, r_theta, c=r_eye.L, marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_phi, l_theta, c=l_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_phi, r_theta, c=r_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_phi, l_theta, c=l_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_phi, r_theta, c=r_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(r_theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])
    # plot filters
    if False:
        plt.figure("Compound eyes - Filters", figsize=(15, 21))

        plt.subplot(3, 4, 1)
        plt.title("microvilli")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 2)
        plt.title("aop")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 3)
        plt.title("microvilli")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 4)
        plt.title("aop")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        l_aop_r = l_eye.filter(l_eye.AOP, colour="r")
        l_aop_g = l_eye.filter(l_eye.AOP, colour="g")
        l_aop_b = l_eye.filter(l_eye.AOP, colour="b")

        plt.subplot(3, 4, 5)
        plt.title("green")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_g, cmap="Greens", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 6)
        plt.title("blue")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_b, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        r_aop_r = r_eye.filter(r_eye.AOP, colour="r")
        r_aop_g = r_eye.filter(r_eye.AOP, colour="g")
        r_aop_b = r_eye.filter(r_eye.AOP, colour="b")

        plt.subplot(3, 4, 7)
        plt.title("green")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_g, cmap="Greens", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 8)
        plt.title("blue")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_b, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        l_aop = np.clip(np.array([l_aop_r * .05, l_aop_g * .53, l_aop_b * .79]).T, 0, 1)
        l_aop_avg = l_aop.mean(axis=1)

        plt.subplot(3, 4, 9)
        plt.title("avg")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop_avg, cmap="Greys", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(3, 4, 10)
        plt.title("sin")
        plt.scatter(l_eye.phi, l_eye.theta, c=l_aop,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        r_aop = np.array([r_aop_r * .05, r_aop_g * .53, r_aop_b * .79]).T
        r_aop_avg = r_aop.mean(axis=1)

        plt.subplot(3, 4, 11)
        plt.title("avg")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop_avg, cmap="Greys", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(3, 4, 12)
        plt.title("sin")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_aop,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

    # from sky import cubebox, skydome
    #
    #
    # def plot_luminance(**kwargs):
    #     plt.figure("Luminance", figsize=(6, 9))
    #     ax = plt.subplot(2, 1, 1)
    #     plt.imshow(kwargs["skydome"])
    #     ax.set_anchor('W')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 17)
    #     plt.imshow(kwargs["left"])
    #     plt.text(32, 40, "left", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 18)
    #     plt.imshow(kwargs["front"])
    #     plt.text(32, 40, "front", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 19)
    #     plt.imshow(kwargs["right"])
    #     plt.text(32, 40, "right", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 20)
    #     plt.imshow(kwargs["back"])
    #     plt.text(32, 40, "back", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 14)
    #     plt.imshow(kwargs["top"])
    #     plt.text(32, 32, "top", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.subplot(6, 4, 22)
    #     plt.imshow(kwargs["bottom"])
    #     plt.text(32, 32, "bottom", fontsize="16", fontweight="bold", color="white",
    #              horizontalalignment="center", verticalalignment="center")
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    #
    #
    # # create cubebox parts
    # L_left, DOP_left, AOP_left = cubebox(sky, "left")
    # L_front, DOP_front, AOP_front = cubebox(sky, "front")
    # L_right, DOP_right, AOP_right = cubebox(sky, "right")
    # L_back, DOP_back, AOP_back = cubebox(sky, "back")
    # L_top, DOP_top, AOP_top = cubebox(sky, "top")
    # L_bottom, DOP_bottom, AOP_bottom = cubebox(sky, "bottom")
    #
    # # create skydome
    # L, DOP, AOP = skydome(sky)
    #
    # # plot cubeboxes
    # plot_luminance(skydome=L,
    #                left=L_left, front=L_front, right=L_right, back=L_back, top=L_top, bottom=L_bottom)

    plt.show()
