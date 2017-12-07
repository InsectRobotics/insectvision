import numpy as np
import numpy.linalg as la
import healpy as hp

from model import CompoundEye, WLFilter
from utils import pca_kernel
from sky import ChromaticitySkyModel, sph2vec
from geometry import fibonacci_sphere, angles_distribution
import os


__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/sensor/"

LENS_RADIUS = 1  # mm
A_lens = np.pi * np.square(LENS_RADIUS)  # mm ** 2
NB_EN = 8
DEBUG = False


class CompassSensor(CompoundEye):

    def __init__(self, nb_lenses=20, fov=np.deg2rad(60), mode="normal"):

        self.nb_lenses = nb_lenses
        self.fov = fov

        # we assume that the lens is a hexagon
        self.r_l = LENS_RADIUS
        self.R_l = LENS_RADIUS * 2 / np.sqrt(3)
        self.S_l = 3 * self.r_l * self.R_l
        # an estimation of the dome area
        self.S_a = nb_lenses * self.S_l  # the surface area of the sensor
        self.R_c = np.sqrt(self.S_a / (2 * np.pi * (1. - np.cos(fov / 2))))  # the radius of the curvature
        self.alpha = self.R_c * np.sin(fov / 2)
        self.height = self.R_c * (1. - np.cos(fov / 2))
        self.learning_rate = 0.1

        assert mode in ["normal", "cross", "event"],\
            "Mode has to be one of 'normal', 'cross' or 'event'."
        self.mode = mode

        if DEBUG:
            print "Number of lenses:              %d" % nb_lenses
            print "Field of view:                %.2f degrees" % np.rad2deg(fov)
            print "Lens radius (r):               %.2f mm" % LENS_RADIUS
            print "Lens surface area (A):         %.2f mm" % A_lens
            print "Sensor area (S_a):            %.2f mm^2" % self.S_a
            print "Radius of the curvature (R_c): %.2f mm" % self.R_c
            print "Sphere area (S):             %.2f mm^2" % (4 * np.pi * np.square(self.R_c))
            print "Sensor height (h):             %.2f mm" % self.height
            print "Surface coverage:              %.2f" % self.coverage

        try:
            thetas, phis, fit = angles_distribution(nb_lenses, np.rad2deg(fov))
        except ValueError:
            thetas = np.empty(0, dtype=np.float32)
            phis = np.empty(0, dtype=np.float32)
            fit = False
        if not fit or nb_lenses > 100:
            thetas, phis = fibonacci_sphere(nb_lenses, np.rad2deg(fov))

        super(CompassSensor, self).__init__(
            ommatidia=np.array([thetas.flatten(), phis.flatten()]).T,
            central_microvili=(0, 0),
            noise_factor=.0,
            activate_dop_sensitivity=False)

        self._channel_filters.pop("g")
        if self.mode != "cross":
            self._channel_filters.pop("b")
        self.__x_new = np.zeros(nb_lenses, dtype=np.float32)
        self.__x_last = np.zeros(nb_lenses, dtype=np.float32) * np.nan

        self.w = np.random.randn(thetas.size, NB_EN)
        self.w_whitening = np.eye(thetas.size)
        self.m = np.zeros(thetas.size)

    @property
    def coverage(self):
        """

        :return: the percentage of the sphere's surface that is covered from lenses
        """
        # the surface area of the complete sphere
        S = 4 * np.pi * np.square(self.R_c)
        return self.S_a / S

    @property
    def L(self):
        if self.mode == "cross":
            b_max = self._channel_filters["b"][0](1.)
            uv_max = self._channel_filters["uv"][0](1.)
            x = self.__x_new.reshape((-1, 2))
            uv = x[:, 1] / uv_max
            b = x[:, 0] / b_max
            x = uv / (uv + b)
        else:
            x = self.__x_new

        x_max = x.max()
        x_min = x.min()
        x = (x - x_min) / (x_max - x_min)
        if self.mode == "event":
            if not np.all(np.isnan(self.__x_last)):
                x_last_max = self.__x_last.max()
                x_last_min = self.__x_last.min()
                x_last = (self.__x_last - x_last_min) / (x_last_max - x_last_min)
                x = x - x_last
            else:
                x = self.__x_last
        return x

    def set_sky(self, sky):
        super(CompassSensor, self).set_sky(sky)
        if not np.all(self.__x_new == 0.):
            self.__x_last = self.__x_new
        self.__x_new = super(CompassSensor, self).L.flatten()

    def update_parameters(self, x, t=None):
        """

        :param x:
        :type x: np.ndarray, ChromaticitySkyModel
        :param t:
        :type t: np.ndarray
        :return:
        """
        if isinstance(x, ChromaticitySkyModel):
            sky_model = x  # type: ChromaticitySkyModel
            x = np.empty((0, self.L.size), dtype=np.float32)
            t = np.empty((0, NB_EN), dtype=np.float32)
            r = self.facing_direction
            for j in xrange(180):
                self.rotate(np.deg2rad(2))
                self.set_sky(sky_model)
                lon = (sky_model.lon + self.facing_direction) % (2 * np.pi)
                lat = sky_model.lat
                x = np.vstack([x, self.L.flatten()])
                t = np.vstack([t, encode_sun(lon, lat)])
            self.facing_direction = r
            self.set_sky(sky_model)

        # self.w = (1. - self.learning_rate) * self.w + self.learning_rate * la.pinv(x).dot(t)
        self.w_whitening = pca_kernel(x)
        self.m = x.mean(axis=0)
        if t is not None:
            self.w = la.pinv(self._pprop(x), 1e-01).dot(t)

        return self._fprop(x)

    def __call__(self, *args, **kwargs):
        if isinstance(args[0], np.ndarray):
            self._lum = args[0]  # type: np.ndarray
        elif isinstance(args[0], ChromaticitySkyModel):
            self.set_sky(args[0])
        else:
            raise AttributeError("Unknown attribute type: %s" % type(args[0]))
        decode = False
        if 'decode' in kwargs.keys():
            decode = kwargs['decode']
        elif len(args) > 1:
            decode = args[1]  # type: bool
        return self._fprop(self.L, decode=decode).flatten()

    def _pprop(self, x):
        return (x.reshape((-1, self.nb_lenses)) - self.m).dot(self.w_whitening)

    def _fprop(self, x, decode=True):
        x = self._pprop(x)
        if decode:
            y = []
            for x0 in x.dot(self.w):
                y.append(decode_sun(x0))
            return np.array(y)
        else:
            return x.dot(self.w)

    def save_weights(self):
        name = "sensor-L%03d-V%03d.npz" % (self.nb_lenses, np.rad2deg(self.fov))
        np.savez_compressed(__datadir__ + name, w=self.w, w_whitening=self.w_whitening)

    def load_weights(self, filename=None):
        if filename is None:
            name = "sensor-L%03d-V%03d.npz" % (self.nb_lenses, np.rad2deg(self.fov))
            filename = __datadir__ + name
        weights = np.load(filename)
        self.w = weights["w"]
        self.w_whitening = weights["w_whitening"]

    @classmethod
    def visualise(cls, sensor):
        """

        :param sensor:
        :type sensor: CompassSensor
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle
        from sky.utils import sph2vec

        xyz = sph2vec(np.pi/2 - sensor.theta, sensor.phi, sensor.R_c)

        plt.figure("Sensor Design", figsize=(10, 10))

        # top view
        ax_t = plt.subplot2grid((4, 4), (1, 1), colspan=2, rowspan=2, aspect="equal", adjustable='box-forced')
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        sensor_outline = Ellipse(xy=np.zeros(2),
                                 width=2 * sensor.alpha + 2 * sensor.R_l,
                                 height=2 * sensor.alpha + 2 * sensor.R_l)
        ax_t.add_artist(outline)
        outline.set_clip_box(ax_t.bbox)
        outline.set_alpha(.2)
        outline.set_facecolor("grey")
        ax_t.add_artist(sensor_outline)
        sensor_outline.set_clip_box(ax_t.bbox)
        sensor_outline.set_alpha(.5)
        sensor_outline.set_facecolor("grey")

        stheta, sphi, sL = sensor.theta, sensor.phi, sensor.L
        if sensor.mode == "event":
            sL = np.clip(1. * sL + .5, 0., 1.)
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):

            lens = Ellipse(xy=[x, y], width=1.5 * sensor.r_l, height=1.5 * np.cos(th) * sensor.r_l,
                           angle=np.rad2deg(-ph))
            ax_t.add_artist(lens)
            lens.set_clip_box(ax_t.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax_t.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_xticklabels([])
        ax_t.set_yticklabels([])

        # side view #1 (x, z)
        ax = plt.subplot2grid((4, 4), (0, 1), colspan=2, aspect="equal", adjustable='box-forced', sharex=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c],
                             width=2 * sensor.R_c,
                             height=2 * sensor.R_c - sensor.height - sensor.R_l)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):
            if y > 0:
                continue
            lens = Ellipse(xy=[x, z], width=1.5 * sensor.r_l, height=np.sin(-y / sensor.R_c) * 1.5 * sensor.r_l,
                           angle=np.rad2deg(np.arcsin(-x / sensor.R_c)))
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_ylim(0, sensor.R_c + 2)
        ax.set_xticks([])
        ax.set_yticks([])

        # side view #2 (-x, z)
        ax = plt.subplot2grid((4, 4), (3, 1), colspan=2, aspect="equal", adjustable='box-forced', sharex=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c + sensor.height + sensor.R_l],
                             width=2 * sensor.R_c,
                             height=2 * sensor.R_c - sensor.height - sensor.R_l)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):
            if y < 0:
                continue
            lens = Ellipse(xy=[x, -z], width=1.5 * sensor.r_l, height=np.sin(-y / sensor.R_c) * 1.5 * sensor.r_l,
                           angle=np.rad2deg(np.arcsin(x / sensor.R_c)))
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_ylim(-sensor.R_c - 2, 0)
        ax.set_yticks([])

        # side view #3 (y, z)
        ax = plt.subplot2grid((4, 4), (1, 3), rowspan=2, aspect="equal", adjustable='box-forced', sharey=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c, -sensor.R_c],
                             width=2 * sensor.R_c - sensor.height - sensor.R_l,
                             height=2 * sensor.R_c)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):
            if x > 0:
                continue
            lens = Ellipse(xy=[z, y], width=1.5 * sensor.r_l, height=np.sin(-x / sensor.R_c) * 1.5 * sensor.r_l,
                           angle=np.rad2deg(np.arcsin(y / sensor.R_c)) + 90)
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_xlim(0, sensor.R_c + 2)
        ax.set_yticks([])
        ax.set_xticks([])

        # side view #4 (-y, z)
        ax = plt.subplot2grid((4, 4), (1, 0), rowspan=2, aspect="equal", adjustable='box-forced', sharey=ax_t)
        outline = Ellipse(xy=np.zeros(2),
                          width=2 * sensor.R_c,
                          height=2 * sensor.R_c)
        fade_one = Rectangle(xy=[-sensor.R_c + sensor.height + sensor.R_l, -sensor.R_c],
                             width=2 * sensor.R_c - sensor.height - sensor.R_l,
                             height=2 * sensor.R_c)
        ax.add_artist(outline)
        outline.set_clip_box(ax.bbox)
        outline.set_alpha(.5)
        outline.set_facecolor("grey")
        ax.add_artist(fade_one)
        fade_one.set_clip_box(ax.bbox)
        fade_one.set_alpha(.6)
        fade_one.set_facecolor("white")
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):
            if x < 0:
                continue
            lens = Ellipse(xy=[-z, y], width=1.5 * sensor.r_l, height=np.sin(-x / sensor.R_c) * 1.5 * sensor.r_l,
                           angle=np.rad2deg(np.arcsin(-y / sensor.R_c)) - 90)
            ax.add_artist(lens)
            lens.set_clip_box(ax.bbox)
            lens.set_facecolor(np.array([0., 0., 1.]) * L)

        ax.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax.set_xlim(-sensor.R_c - 2, 0)
        ax.set_xticks([])

        plt.tight_layout(pad=0.)

        plt.show()


def encode_sun(lon, lat):
    return np.sin(np.linspace(0, 2 * np.pi, NB_EN, endpoint=False) + lon + np.pi / 2) * lat / (NB_EN / 2.)


def decode_sun(x):
    fund_freq = np.fft.fft(x)[1]
    lon = -np.angle(np.conj(fund_freq))
    lat = np.absolute(fund_freq)
    return lon, lat


def mse(y, t, theta=True, phi=True):
    if theta:
        thy = y[:, 1]
        tht = t[:, 1]
    else:
        thy = np.zeros_like(y[:, 1])
        tht = np.zeros_like(t[:, 1])
    if phi:
        phy = y[:, 0]
        pht = t[:, 0]
    else:
        phy = np.zeros_like(y[:, 0])
        pht = np.zeros_like(t[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).mean())


if __name__ == "__main__":
    from sky import get_seville_observer
    from datetime import datetime

    # modes: "normal", "cross", "event"
    s = CompassSensor(nb_lenses=100, fov=np.deg2rad(60), mode="normal")
    # s.load_weights()

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime.now()

    # create and generate a sky instance
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    # lon, lat = sky.lon, sky.lat
    # print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    # lon, lat = s.update_parameters(sky)
    # print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    s.set_sky(sky)
    # s.rotate(np.deg2rad(90))
    # s.set_sky(sky)
    CompassSensor.visualise(s)


if __name__ == "__main__2__":
    import matplotlib.pyplot as plt
    from sky import ChromaticitySkyModel, get_seville_observer
    from datetime import datetime

    s = CompassSensor(nb_lenses=12, fov=np.pi/6)
    p = np.zeros(hp.nside2npix(s.nside))
    i = hp.ang2pix(s.nside, s.theta, s.phi)

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime.now()

    # create and generate a sky instance
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    s.set_sky(sky)
    p[i] = s.L
    # p[i] = s.DOP
    # p[i] = np.rad2deg(s.AOP)
    p_i_max = p[i].max()
    p_i_min = p[i].min()
    p[i] = (p[i] - p_i_min) / (p_i_max - p_i_min)
    hp.orthview(p, rot=(0, 90, 0))
    plt.show()
