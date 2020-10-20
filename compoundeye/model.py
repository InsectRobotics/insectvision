import numpy as np
from copy import copy
from .utils import get_microvilli_angle
from sky import SkyModel

SkyBlue = np.array([.05, .53, .79])[..., np.newaxis]
# SkyBlue = np.array([1.0, 1.0, 1.0])[..., np.newaxis]


class CompoundEye(object):

    def __init__(self, ommatidia, central_microvili=(np.pi/6, np.pi/18), noise_factor=.1,
                 activate_dop_sensitivity=False):

        # the eye facing direction
        self.yaw_pitch_roll = np.zeros(3)

        # eye specifications (ommatidia topography)
        self._sky = SkyModel()
        self.theta_global = ommatidia[:, 0]
        self.phi_global = ommatidia[:, 1]
        if ommatidia.shape[1] > 3:
            self._dop_filter = ommatidia[:, 3]
            self._aop_filter = ommatidia[:, 2]
        elif ommatidia.shape[1] > 2:
            self._aop_filter = ommatidia[:, 2]
            _, self._dop_filter = get_microvilli_angle(
                self.theta_global, self.phi_global, theta=central_microvili[0], phi=central_microvili[1], n=noise_factor
            )
        else:
            self._aop_filter, self._dop_filter = get_microvilli_angle(
                self.theta_global, self.phi_global, theta=central_microvili[0], phi=central_microvili[1], n=noise_factor
            )
        if not activate_dop_sensitivity:
            self._dop_filter[:] = 1.
        self._active_pol_filters = True

        self._channel_filters = {}
        self._update_filters()

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
    def sky(self):
        return self._sky

    @sky.setter
    def sky(self, value):
        value = value.copy()
        value.theta_z = self.sky.theta_z
        value.phi_z = self.sky.phi_z
        self._sky = value

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
        i, d, a = self.sky.L / np.sqrt(2), self.DOP, self.AOP
        lum_channels = np.array([
            pf(cf(i), a, d) for cf, pf in [self._channel_filters[c] for c in k]
        ]).T
        lum_channels[np.isclose(i, 0.)] *= 0.
        return np.clip(lum_channels, 0, 1)

    @property
    def DOP(self):
        return self.sky.DOP

    @property
    def AOP(self):
        return self.sky.AOP % (2 * np.pi)

    @property
    def theta_global(self):
        return (self.sky.theta_z + np.pi) % (2 * np.pi) - np.pi  # type: np.ndarray

    @theta_global.setter
    def theta_global(self, value):
        """
        :param value: the ommatidia elevation
        :type value: np.ndarray
        :return:
        """
        self.sky.theta_z = (value + np.pi) % (2 * np.pi) - np.pi

    @property
    def phi_global(self):
        return (self.sky.phi_z + np.pi) % (2 * np.pi) - np.pi

    @phi_global.setter
    def phi_global(self, value):
        self.sky.phi_z = (value + np.pi) % (2 * np.pi) - np.pi

    @property
    def theta_local(self):
        theta_z, phi_z = SkyModel.rotate(self.theta_global, self.phi_global, yaw=-self.yaw)
        theta_z, phi_z = SkyModel.rotate(theta_z, phi_z, pitch=-self.pitch)
        theta_z, phi_z = SkyModel.rotate(theta_z, phi_z, roll=-self.roll)
        return (theta_z + np.pi) % (2 * np.pi) - np.pi

    @property
    def phi_local(self):
        theta_z, phi_z = SkyModel.rotate(self.theta_global, self.phi_global, yaw=-self.yaw)
        theta_z, phi_z = SkyModel.rotate(theta_z, phi_z, pitch=-self.pitch)
        theta_z, phi_z = SkyModel.rotate(theta_z, phi_z, roll=-self.roll)
        return (phi_z + np.pi) % (2 * np.pi) - np.pi

    @property
    def yaw(self):
        return self.yaw_pitch_roll[0]

    @yaw.setter
    def yaw(self, value):
        self.yaw_pitch_roll[0] = value

    @property
    def pitch(self):
        return self.yaw_pitch_roll[1]

    @pitch.setter
    def pitch(self, value):
        self.yaw_pitch_roll[1] = value

    @property
    def roll(self):
        return self.yaw_pitch_roll[2]

    @roll.setter
    def roll(self, value):
        self.yaw_pitch_roll[2] = value

    def rotate(self, yaw=0., pitch=0., roll=0.):
        sky = self.sky

        # rotate back to the default orientation
        sky = SkyModel.rotate_sky(sky, yaw=-self.yaw)
        sky = SkyModel.rotate_sky(sky, pitch=-self.pitch)
        sky = SkyModel.rotate_sky(sky, roll=-self.roll)

        # update the facing direction of the eye
        self.yaw_pitch_roll = self.rotate_centre(
            self.yaw_pitch_roll, yaw=yaw, pitch=-pitch, roll=roll
        )

        # rotate the sky according to the new facing direction
        sky = SkyModel.rotate_sky(
            sky, yaw=self.yaw, pitch=self.pitch, roll=self.roll
        )

        self.sky.theta_z = sky.theta_z
        self.sky.phi_z = sky.phi_z

        self._update_filters()

    def _update_filters(self):

        self._channel_filters = {
            "g": [
                WLFilter(WLFilter.RGB_WL[0], name="GreenWLFilter"),
                POLFilter(self._aop_filter - self.yaw + np.pi / 4, self.dop_filter, name="GreenPOLFilter")
            ],
            "b": [
                WLFilter(WLFilter.RGB_WL[1], name="BlueWLFilter"),
                POLFilter(self._aop_filter - self.yaw + np.pi / 2, self.dop_filter, name="BluePOLFilter")
            ],
            "uv": [
                WLFilter(WLFilter.RGB_WL[2], name="UVWLFilter"),
                POLFilter(self._aop_filter - self.yaw, self.dop_filter, name="UVPOLFilter")
            ]
        }

    @staticmethod
    def rotate_centre(centre, yaw=0., pitch=0., roll=0.):
        # centre[[1, 0]] = SkyModel.rotate(centre[1], centre[0], yaw=yaw, pitch=pitch, roll=roll)
        centre[[1, 0]] = SkyModel.rotate(np.pi / 2 - centre[1], np.pi - centre[0], yaw=yaw, pitch=-pitch, roll=-roll)

        centre[0] = (2 * np.pi - centre[0]) % (2 * np.pi) - np.pi
        centre[1] = (3 * np.pi/2 - centre[1]) % (2 * np.pi) - np.pi
        centre[2] = (centre[2] + roll + np.pi) % (2 * np.pi) - np.pi

        return centre


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
        i0 = copy(super(WLFilter, self).__call__(*args, **kwargs))
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

        i0 = lum / (1. - dop)
        # create the light coordinates
        d = (aop - self.angle + np.pi) % (2 * np.pi) - np.pi
        E1 = np.array([
            np.cos(d),
            np.sin(d)
        ]) * np.sqrt(i0)
        E2 = np.array([
            np.cos(d + np.pi/2),
            np.sin(d + np.pi/2)
        ]) * np.sqrt(i0) * (1. - dop)

        # filter the separate coordinates
        E1[1] *= (1. - self.degree)
        E2[1] *= (1. - self.degree)

        # the total intensity is the integration of the ellipse
        E = np.array([np.sqrt(np.square(E1).sum(axis=0)), np.sqrt(np.square(E2).sum(axis=0))])
        return np.sqrt(np.square(E).sum(axis=0))
        # A = np.sqrt(np.square(E1).sum(axis=0)) * np.sqrt(np.square(E2).sum(axis=0))
        # return A


if __name__ == "__main__":
    from datetime import datetime
    from ephem import city
    from .beeeye import load_both_eyes
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
        print(np.rad2deg(l_eye.yaw_pitch_roll))
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