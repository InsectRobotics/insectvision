from geometry import angles_distribution, fibonacci_sphere
from sphere.transform import tilt
from sky.model import eps

import numpy as np

spectrum = {
    "uv": 350,  # WL: 350 nm
    "v": 400,  # WL: 400 nm
    "b": 475,  # WL: 475 nm
    "g": 535,  # WL: 535 nm
    "y": 580,  # WL: 580 nm
    "o": 605,  # WL: 605 nm
    "r": 715,  # WL: 715 nm
    "ir": 1200,  # WL: 1200 nm
    "w": -1
}


class DRA(object):

    def __init__(self, n=60, omega=56, rho=np.deg2rad(5.4)):
        try:
            self.theta, self.phi, fit = angles_distribution(n, float(omega))
        except ValueError:
            self.theta = np.empty(0, dtype=np.float32)
            self.phi = np.empty(0, dtype=np.float32)
            fit = False

        if not fit or n > 100:
            self.theta, self.phi = fibonacci_sphere(n, float(omega))
        self.alpha = (self.phi + np.pi/2) % (2 * np.pi) - np.pi

        self.rho = rho if rho.size == n else np.full(n, rho)
        self.rhabdom = np.array([[spectrum["w"], spectrum["w"]]] * n).T
        # self.rhabdom = np.array([[spectrum["uv"], spectrum["uv"], spectrum["w"], spectrum["uv"],
        #                           spectrum["uv"], spectrum["uv"], spectrum["w"], spectrum["uv"]]] * n)
        self.mic = np.array([[0., np.pi/2]] * n).T
        # self.mic = np.array([[0., np.pi/2, np.pi/2, np.pi/2, 0., np.pi/2, np.pi/2, np.pi/2]] * n)
        self.theta_t = 0.
        self.phi_t = 0.
        self.__r_op = np.full(n, np.nan)
        self.__r_po = np.full(n, np.nan)
        self.__r_pol = np.full(n, np.nan)

        self.__is_called = False

    def __call__(self, sky, *args, **kwargs):
        sky.theta_t = self.theta_t
        sky.phi_t = self.phi_t

        _, alpha = tilt(self.theta_t, self.phi_t + np.pi, theta=np.pi/2, phi=self.alpha)
        y, p, a = sky(self.theta, self.phi, *args, **kwargs)

        ry = spectrum_influence(y, self.rhabdom)
        s = ry * (np.square(np.sin(a - alpha + self.mic)) + np.square(np.cos(a - alpha + self.mic)) * np.square(1. - p))
        r = np.sqrt(s)
        self.__r_op = np.sum(np.cos(2 * self.mic) * r, axis=0)
        self.__r_po = np.sum(r, axis=0)
        self.__r_pol = self.__r_op / (self.__r_po + eps)
        self.__r_po = 2. * self.__r_po / np.max(self.__r_po) - 1.

        self.__is_called = True

        return self.r_pol

    @property
    def r_pol(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_pol

    @property
    def r_po(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_po

    @property
    def r_op(self):
        assert self.__is_called, "No light has passed through the sensors yet."
        return self.__r_op


def spectrum_influence(v, wl):
    l1 = 10.0 * np.power(wl/1000., 8) * np.square(v) / float(v.size)
    l2 = 0.001 * np.power(1000./wl, 8) * np.square(v).sum() / float(v.size)
    v_max = v.max()
    w_max = (v + l1 + l2).max()
    w = v_max * (v + l1 + l2) / w_max
    if type(wl) is np.ndarray:
        w[wl < 0] = np.array([v] * wl.shape[0])[wl < 0]
    elif wl < 0:
        w = v
    return w


def visualise(sky, y):
    import matplotlib.pyplot as plt

    plt.figure("Luminance", figsize=(4.5, 4.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta_s, phi_s = tilt(sky.theta_t, sky.phi_t, theta=sky.theta_s, phi=sky.phi_s)
    ax.scatter(sky.phi, sky.theta, s=100, c=y, marker='.', cmap='coolwarm', vmin=-1, vmax=1)
    ax.scatter(phi_s, theta_s, s=100, edgecolor='black', facecolor='yellow')
    ax.scatter(sky.phi_t, sky.theta_t, s=50, edgecolor='black', facecolor='greenyellow')
    ax.set_ylim([0, np.pi/2])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$ (N)', r'$45^\circ$ (NE)', r'$90^\circ$ (E)', r'$135^\circ$ (SE)',
                        r'$180^\circ$ (S)', r'$-135^\circ$ (SW)', r'$-90^\circ$ (W)', r'$-45^\circ$ (NW)'])

    plt.show()


if __name__ == "__main__":
    from sky import Sky

    sky = Sky(theta_s=np.pi/3)
    dra = DRA()
    dra.theta_t = np.pi/6
    dra.phi_t = np.pi/2
    # s = dra(sky)
    r_pol = dra(sky)
    r_po = dra.r_po
    # print s.shape

    visualise(sky, r_po)
