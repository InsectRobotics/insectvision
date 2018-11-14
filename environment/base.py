import numpy as np


class Environment(object):

    def __init__(self, name="environment"):
        self.__name = name

        # Tilting of the environment with respect to the sensor
        self.theta_t = 0.
        self.phi_t = 0.

    def __call__(self, theta=None, phi=None, *args, **kwargs):
        return None

    @property
    def name(self):
        return self.__name


# spectrum colours to wavelength
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
