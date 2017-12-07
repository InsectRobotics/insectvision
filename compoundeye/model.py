import numpy as np
from utils import get_microvilli_angle, load_beeeye
from sky import SkyModel, ChromaticitySkyModel

SkyBlue = np.array([.05, .53, .79])[..., np.newaxis]
# SkyBlue = np.array([1.0, 1.0, 1.0])[..., np.newaxis]


class CompoundEye(object):

    def __init__(self, ommatidia, central_microvili=(np.pi/6, np.pi/18), noise_factor=.1,
                 activate_dop_sensitivity=False):

        # eye specifications (ommatidia topography)
        self.theta = ommatidia[:, 0]
        self.phi = ommatidia[:, 1]
        if ommatidia.shape[1] > 3:
            self._dop_filter = ommatidia[:, 3]
            self._aop_filter = ommatidia[:, 2]
        elif ommatidia.shape[1] > 2:
            self._aop_filter = ommatidia[:, 2]
            _, self._dop_filter = get_microvilli_angle(
                self.theta, self.phi, theta=central_microvili[0], phi=central_microvili[1], n=noise_factor
            )
        else:
            self._aop_filter, self._dop_filter = get_microvilli_angle(
                self.theta, self.phi, theta=central_microvili[0], phi=central_microvili[1], n=noise_factor
            )
        if not activate_dop_sensitivity:
            self._dop_filter[:] = 1.
        self._active_pol_filters = True

        self._channel_filters = {}
        self._update_filters()

        # the raw receptive information
        self._lum = np.zeros_like(self.theta)
        self._dop = np.ones_like(self._dop_filter)
        self._aop = np.zeros_like(self._aop_filter)

        self.facing_direction = 0

    def activate_pol_filters(self, value):
        """

        :param value:
        :type value: bool
        :return:
        """
        old_value = self._active_pol_filters
        self._active_pol_filters = value
        if value != old_value:
            self._update_filters()

    @property
    def dop_filter(self):
        if self._active_pol_filters:
            return self._dop_filter
        else:
            return np.zeros_like(self._dop_filter)

    @property
    def L(self):
        ks = self._channel_filters.keys()
        k = []
        for kk in ['r', 'g', 'b', 'uv']:
            if kk in ks:
                k.append(kk)
        lum_channels = np.array([
            pf(cf(self._lum), self._aop, self._dop) for cf, pf in [self._channel_filters[c] for c in k]
        ]).T
        return np.clip(lum_channels, 0, 1)

    @property
    def DOP(self):
        return self._dop

    @property
    def AOP(self):
        return self._aop

    def set_sky(self, sky):
        self._lum, self._dop, self._aop = sky.get_features(np.pi/2-self.theta, np.pi-self.phi+self.facing_direction)
        self._lum /= np.sqrt(2)
        self._dop[np.isnan(self._dop)] = 1.

    def rotate(self, angle):
        self.facing_direction -= angle

    def _update_filters(self):

        self._channel_filters = {
            "g": [
                WLFilter(WLFilter.RGB_WL[0], name="GreenWLFilter"),
                POLFilter(self._aop_filter + np.pi / 4, self.dop_filter, name="GreenPOLFilter")
            ],
            "b": [
                WLFilter(WLFilter.RGB_WL[1], name="BlueWLFilter"),
                POLFilter(self._aop_filter + np.pi / 2, self.dop_filter, name="BluePOLFilter")
            ],
            "uv": [
                WLFilter(WLFilter.RGB_WL[2], name="UVWLFilter"),
                POLFilter(self._aop_filter, self.dop_filter, name="UVPOLFilter")
            ]
        }


class Filter(object):
    __counter = 0

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        else:
            Filter.__counter += 1
            self.name = "Filter-%d" % Filter.__counter

    def __call__(self, *args, **kwargs):
        assert len(args) > 0, "Not enough arguments."
        return args[0]


class WLFilter(Filter):
    Red_WL = 685.
    Green_WL = 532.5
    Blue_WL = 472.5
    UV_WL = 300.
    RGB_WL = np.array([Red_WL, Green_WL, Blue_WL])
    GBUV_WL = np.array([Green_WL, Blue_WL, UV_WL])

    def __init__(self, wl_max, wl_min=None, name=None):
        super(WLFilter, self).__init__(name)
        self.alpha, self.beta = self.__build_map()
        if wl_min is not None:
            wl_max = (wl_max + wl_min) / 2.
        self.weight = self.wl2int(wl_max)

    def __call__(self, *args, **kwargs):
        i0 = super(WLFilter, self).__call__(*args, **kwargs)
        i0 += self.weight * (1. - i0)
        return np.clip(i0, 0., 1.)

    def wl2int(self, wl):
        return self.alpha / np.power(wl, 4) + self.beta

    @classmethod
    def __build_map(cls):
        A = np.array([1 / np.power(cls.RGB_WL, 4), np.ones(3)]).T
        W = np.linalg.pinv(A.T).T.dot(SkyBlue)
        return W


class POLFilter(Filter):

    def __init__(self, angle, degree=1., name=None):
        super(POLFilter, self).__init__(name)
        self.angle = angle
        self.degree = degree

    def __call__(self, *args, **kwargs):  # TODO: not working properly
        assert len(args) > 2, "Not enough arguments."
        lum, aop, dop = args[:3]
        if 'lum' in kwargs.keys():
            lum = kwargs['lum']
        if 'aop' in kwargs.keys():
            aop = kwargs['aop']
        if 'dop' in kwargs.keys():
            dop = kwargs['dop']
        # print self.name,

        # create the light coordinates
        d = aop - self.angle
        E1 = np.array([
            np.cos(d),
            np.sin(d)
        ]) * lum
        E2 = np.array([
            np.cos(d + np.pi/2),
            np.sin(d + np.pi/2)
        ]) * lum * (1. - dop)

        # filter the separate coordinates
        E1[1] *= (1. - self.degree)
        E2[1] *= (1. - self.degree)

        # the total intensity is the integration of the ellipse
        E = np.array([np.sqrt(np.square(E1).sum(axis=0)), np.sqrt(np.square(E2).sum(axis=0))])
        # print E[0].max(), E[1].max()
        return np.sqrt(np.square(E).sum(axis=0))


def sph2vec(theta, phi, rho=1.):
    """
    Transforms the spherical coordinates to a cartesian 3D vector.
    :param theta: elevation
    :param phi:   azimuth
    :param rho:   radius length
    :return vec:    the cartessian vector
    """

    y = rho * (np.sin(phi) * np.cos(theta))
    x = rho * (np.cos(phi) * np.cos(theta))
    z = rho * np.sin(theta)

    return np.asarray([x, -y, z])


if __name__ == "__main__":
    from datetime import datetime
    from ephem import city
    import matplotlib.pyplot as plt

    angle = 0

    # initialise sky
    obs = city("Edinburgh")
    obs.date = datetime.now()
    sky = ChromaticitySkyModel(observer=obs, nside=1)
    sky.generate()

    # initialise ommatidia features
    ommatidia_left, ommatidia_right = load_beeeye()
    l_eye = CompoundEye(ommatidia_left)
    l_eye.rotate(angle)
    r_eye = CompoundEye(ommatidia_right)
    r_eye.rotate(angle)
    l_eye.set_sky(sky)
    r_eye.set_sky(sky)

    # plot result
    s, p = 20, 4
    # plot eye's structure
    if True:
        plt.figure("Compound eyes - Structure", figsize=(8, 21))

        lum_r = l_eye._lum + (1. - l_eye._lum) * .05
        lum_g = l_eye._lum + (1. - l_eye._lum) * .53
        lum_b = l_eye._lum + (1. - l_eye._lum) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(321)
        plt.title("Left")
        plt.scatter(l_eye.phi, l_eye.theta, c=L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        lum_r = r_eye._lum + (1. - r_eye._lum) * .05
        lum_g = r_eye._lum + (1. - r_eye._lum) * .53
        lum_b = r_eye._lum + (1. - r_eye._lum) * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_eye.phi, r_eye.theta, c=L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._dop_filter, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye._aop_filter % np.pi, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
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
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(322)
        plt.title("Right")
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.L, marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi/2, np.pi/2])
        plt.xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], [])
        plt.yticks([-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2], [])

        plt.subplot(323)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(324)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.DOP, cmap="Blues", vmin=0, vmax=1,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2], [])

        plt.subplot(325)
        plt.scatter(l_eye.phi, l_eye.theta, c=l_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
        plt.xlabel("alpha (azimuth)")
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   ["-180", "-135", "-90", "-45", "0", "45", "90", "135", "180"])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.subplot(326)
        plt.scatter(r_eye.phi, r_eye.theta, c=r_eye.AOP, cmap="hsv", vmin=0, vmax=np.pi,
                    marker=".", s=np.power(s, p * np.absolute(l_eye.theta) / np.pi))
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
