import os
import healpy as hp
import numpy as np

from .geometry import fibonacci_sphere, angles_distribution, LENS_RADIUS, A_lens
from learn.whitening import zca
from .model import CompoundEye
from sky import SkyModel
from code.compass import encode_sph, decode_sph
from sphere import sph2vec

__dir__ = os.path.dirname(os.path.realpath(__file__))
__datadir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "sensor"))

np.random.seed(2018)
NB_EN = 8
DEBUG = False


class CompassSensor(CompoundEye):

    def __init__(self, nb_lenses=60, fov=np.deg2rad(60), kernel=None, mode="cross", fibonacci=False):

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
        self.kernel = kernel

        assert mode in ["normal", "cross", "event"],\
            "Mode has to be one of 'normal', 'cross' or 'event'."
        self.mode = mode

        if DEBUG:
            print("Number of lenses:              %d" % nb_lenses)
            print("Field of view:                %.2f degrees" % np.rad2deg(fov))
            print("Lens radius (r):               %.2f mm" % LENS_RADIUS)
            print("Lens surface area (A):         %.2f mm" % A_lens)
            print("Sensor area (S_a):            %.2f mm^2" % self.S_a)
            print("Radius of the curvature (R_c): %.2f mm" % self.R_c)
            print("Sphere area (S):             %.2f mm^2" % (4 * np.pi * np.square(self.R_c)))
            print("Sensor height (h):             %.2f mm" % self.height)
            print("Surface coverage:              %.2f" % self.coverage)

        try:
            thetas, phis, fit = angles_distribution(nb_lenses, np.rad2deg(fov))
        except ValueError:
            thetas = np.empty(0, dtype=np.float32)
            phis = np.empty(0, dtype=np.float32)
            fit = False
        self.fibonacci = False
        if fibonacci or not fit or nb_lenses > 100:
            thetas, phis = fibonacci_sphere(nb_lenses, np.rad2deg(fov))
            self.fibonacci = True
        thetas = (thetas - np.pi) % (2 * np.pi) - np.pi
        phis = (phis + np.pi) % (2 * np.pi) - np.pi

        super(CompassSensor, self).__init__(
            ommatidia=np.array([thetas.flatten(), phis.flatten()]).T,
            central_microvili=(0, 0),
            noise_factor=.0,
            activate_dop_sensitivity=False)

        # self._channel_filters.pop("g")
        # if self.mode != "cross":
        #     self._channel_filters.pop("b")
        self.__x_new = np.zeros(nb_lenses, dtype=np.float32)
        self.__x_last = np.zeros(nb_lenses, dtype=np.float32) * np.nan

        # computational parameters
        self.nb_tl2 = 16
        self.nb_cl1 = 16
        self.nb_tb1 = 8
        self.w_whitening = np.eye(thetas.size)
        self.w_tl2, self.w_cl1, self.w_tb1 = self.__init_weights()
        self.w = [self.w_tl2, self.w_cl1, self.w_tb1]
        self.b_tl2 = np.zeros(self.nb_tl2)
        self.b_cl1 = np.zeros(self.nb_cl1)
        self.b_tb1 = np.zeros(self.nb_tb1)
        self.b = [self.b_tl2, self.b_cl1, self.b_tb1]
        self.m = .5 * np.ones(thetas.size)
        self.tl2 = np.zeros(self.nb_tl2)
        self.cl1 = np.zeros(self.nb_cl1)
        self.tb1 = np.zeros(self.nb_tb1)

    def _update_filters(self):
        super(CompassSensor, self)._update_filters()
        self._channel_filters.pop("g")
        if self.mode != "cross":
            self._channel_filters.pop("b")

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
            x = np.zeros_like(uv)
            z = uv + b + np.finfo(float).eps
            # s = uv * np.sin(2 * self._aop_filter) + b * np.sin(2 * (self._aop_filter + np.pi/2))
            # c = uv * np.cos(2 * self._aop_filter) + b * np.cos(2 * (self._aop_filter + np.pi/2))
            # d = np.sqrt(np.square(s) + np.square(c))
            m = np.any([np.isnan(z), np.isclose(z, 0.)], axis=0)
            # x[~m] = 2 * d[~m] / z + .5
            x[~m] = (uv / z)[~m]
            # x[np.isnan(x)] = 0.
            return x
            # return np.clip((x - x[x > 0].min()) / (x.max() - x[x > 0].min()), 0, 1)
        else:
            x = self.__x_new

        x[np.isnan(x)] = 0.
        x_max = x.max()
        x_min = x.min()
        if (x_max - x_min) > 0:
            x = np.clip((x - x[x > 0].min()) / (x.max() - x[x > 0].min()), 0, 1)
        else:
            x -= x_min
        if self.mode == "event":
            if not np.all(np.isnan(self.__x_last)):
                x_last_max = self.__x_last.max()
                x_last_min = self.__x_last.min()
                x_last = (self.__x_last - x_last_min) / (x_last_max - x_last_min)
                x = x - x_last
            else:
                x = self.__x_last
        return x

    def refresh(self):
        if not np.all(self.__x_new == 0.):
            self.__x_last = self.__x_new
        self.__x_new = super(CompassSensor, self).L.flatten()

    def update_parameters(self, x, t=None, **kwargs):
        """
        :param x:
        :type x: np.ndarray, ChromaticitySkyModel
        :param t:
        :type t: np.ndarray
        :return:
        """

        kwargs["activation"] = kwargs.get("activation", "relu")
        kwargs["optimizer"] = kwargs.get("optimizer", "rmsprop")
        kwargs["loss"] = kwargs.get("loss", "mse")
        kwargs["nb_epoch"] = kwargs.get("nb_epoch", 100)
        kwargs["batch_size"] = kwargs.get("batch_size", 1000)
        kwargs["random_state"] = kwargs.get("random_state", 2018)
        kwargs["l2"] = kwargs.get("l2", 1e-05)
        kwargs["constraint"] = kwargs.get("constraint", "unitnorm")
        kwargs["verbose"] = kwargs.get("verbose", True)

        if isinstance(x, SkyModel):
            sky_model = x  # type: SkyModel
            x = np.empty((0, self.L.size), dtype=np.float32)
            t = np.empty((0, NB_EN), dtype=np.float32)
            r = self.yaw
            for j in range(180):
                self.rotate(np.deg2rad(2))
                self.sky = sky_model
                lon = (sky_model.lon + self.yaw) % (2 * np.pi)
                lat = sky_model.lat
                x = np.vstack([x, self.L.flatten()])
                t = np.vstack([t, encode_sun(lon, lat)])
            self.yaw = r
            self.sky = sky_model

        x[np.isnan(x)] = 0.
        if self.m is None:
            self.m = x.mean(axis=0)
        if self.kernel is not None:
            self.w_whitening = self.kernel(x, m=self.m)
        # if t is not None:
        #     np.random.seed(kwargs["random_state"])
        #     import tensorflow as tf
        #     tf.set_random_seed(kwargs["random_state"])
        #     from keras.models import Model
        #     from keras.layers import Dense, Input
        #     from keras.regularizers import l2
        #     from keras.constraints import get as get_constraint
        #
        #     hargs = {
        #         "activation": kwargs["activation"],
        #         "kernel_constraint": get_constraint(kwargs["constraint"]),
        #         "bias_constraint": get_constraint(kwargs["constraint"])
        #     }
        #     oargs = {
        #         "activation": "linear",
        #         "kernel_regularizer": l2(kwargs["l2"]),
        #         "bias_regularizer": l2(kwargs["l2"])
        #     }
        #
        #     # create layers
        #     inp = Input(shape=(self.nb_lenses,), name="DRA")
        #     tl2 = Dense(self.nb_tl2, name="TL2", weights=[self.w_tl2, self.b_tl2], **hargs)
        #     # tl2.trainable = False
        #     cl1 = Dense(self.nb_cl1, name="CL1", weights=[self.w_cl1, self.b_cl1], **hargs)
        #     # cl1.trainable = False
        #     tb1 = Dense(self.nb_tb1, name="TB1", weights=[self.w_tb1, self.b_tb1], **oargs)
        #     # tb1.trainable = False
        #
        #     # create model
        #     model = Model(inputs=inp, outputs=tb1(cl1(tl2(inp))))
        #     model.compile(optimizer=kwargs["optimizer"], loss=kwargs["loss"], metrics=[])
        #
        #     # fit data in the model
        #     x_white = self._pprop(x)
        #     model.fit(x_white, t, nb_epoch=kwargs["nb_epoch"], batch_size=kwargs["batch_size"])
        #
        #     self.w[0][:] = self.w_tl2[:] = tl2.get_weights()[0][:]
        #     self.w[1][:] = self.w_cl1[:] = cl1.get_weights()[0][:]
        #     self.w[2][:] = self.w_tb1[:] = tb1.get_weights()[0][:]
        #     self.b[0][:] = self.b_tl2[:] = tl2.get_weights()[1][:]
        #     self.b[1][:] = self.b_cl1[:] = cl1.get_weights()[1][:]
        #     self.b[2][:] = self.b_tb1[:] = tb1.get_weights()[1][:]

        return self._fprop(x)

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            x = self.L
        elif isinstance(args[0], np.ndarray):
            x = args[0]  # type: np.ndarray
        elif isinstance(args[0], SkyModel):
            self.sky = args[0]
            self.refresh()
            x = self.L
        else:
            raise AttributeError("Unknown attribute type: %s" % type(args[0]))
        decode = kwargs.get('decode', False)
        if len(args) > 1:
            decode = args[1]  # type: bool
        return self._fprop(x, decode=decode)

    def _pprop(self, x):
        # return x
        return (x.reshape((-1, self.nb_lenses)) - self.m).dot(self.w_whitening)

    def _fprop(self, x, decode=True):
        h = self._pprop(x)
        for j, (w, b, v) in enumerate(zip(self.w, self.b, [self.tl2, self.cl1, self.tb1])):
            h = h.dot(w) + b
            if j < len(self.w) - 1:
                h = np.clip(h, 0, np.finfo(h.dtype).max)
            if h.size == v.size:
                v[:] = h[:]
        if decode:
            return np.array([decode_sun(h0) for h0 in h])
        else:
            return h

    def save_weights(self, filename=None, name=None):
        if filename is None:
            mode = '' if self.mode == 'normal' else self.mode + '-'
            if name is None:
                name = "%ssensor-L%03d-V%03d" % (mode, self.nb_lenses, np.rad2deg(self.fov))
                if self.fibonacci:
                    name += "-fibonacci"
            if ".npz" not in name:
                name += ".npz"
            filename = __datadir__ + name
        np.savez_compressed(filename, w=self.w, b=self.b, w_whitening=self.w_whitening, m=self.m)

    def load_weights(self, filename=None, name=None):
        if filename is None:
            if name is None:
                mode = '' if self.mode == 'normal' else self.mode + '-'
                name = "%ssensor-L%03d-V%03d" % (mode, self.nb_lenses, np.rad2deg(self.fov))
                if self.fibonacci:
                    name += "-fibonacci"
            if ".npz" not in name:
                name += ".npz"
            filename = __datadir__ + name
        try:
            weights = np.load(filename)
            self.w_whitening = weights["w_whitening"]
            self.w[0][:] = self.w_tl2[:] = weights["w"][0][:]
            self.w[1][:] = self.w_cl1[:] = weights["w"][1][:]
            self.w[2][:] = self.w_tb1[:] = weights["w"][2][:]
            self.b[0][:] = self.b_tl2[:] = weights["b"][0][:]
            self.b[1][:] = self.b_cl1[:] = weights["b"][1][:]
            self.b[2][:] = self.b_tb1[:] = weights["b"][2][:]
            self.m = weights["m"]
        except ValueError:
            print("Weights do not fit the current structure.")

    def __init_weights(self, layers=True):
        w_tl2 = .5 * np.sin(-self.phi_local[..., np.newaxis] + np.linspace(0, 4 * np.pi, self.nb_tl2, endpoint=False))

        w_cl1 = []
        for j, b in enumerate(np.linspace(0, 4 * np.pi, self.nb_cl1, endpoint=False)):
            w_cl1.append(.5 * np.sin(-np.linspace(0, 4 * np.pi, self.nb_tl2, endpoint=False) + b))
        w_cl1 = np.array(w_cl1)

        if not layers:
            w_tl2 = np.pi * (2. * w_tl2).dot(2. * w_cl1) / 60.
            w_cl1 = -np.eye(self.nb_tl2, self.nb_cl1)

            w_tb1_1 = np.eye(self.nb_cl1, self.nb_tb1)
            w_tb1_2 = np.roll(np.roll(np.eye(self.nb_cl1, self.nb_tb1), self.nb_tb1, axis=0), self.nb_tb1/2, axis=1)
            w_tb1 = -(.5 * w_tb1_1 - .5 * w_tb1_2)
        else:
            w_tl2 *= np.absolute(self.theta_local[..., np.newaxis]) / 2.
            w_tb1 = np.tile(np.eye(self.nb_tb1), 2).T
            w_tb1_1 = np.eye(self.nb_cl1, self.nb_tb1)
            w_tb1_2 = np.roll(np.roll(np.eye(self.nb_cl1, self.nb_tb1), self.nb_tb1, axis=0),
                              int(self.nb_tb1 / 2), axis=1)
            w_tb1 = w_tb1_1 - w_tb1_2

        return w_tl2, w_cl1, w_tb1

    @classmethod
    def visualise(cls, sensor, sL=None, show_sun=False, sides=True, interactive=False, colormap=None, scale=[0, 1],
                  title=None):
        """
        :param sensor:
        :type sensor: CompassSensor
        :return:
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle
        from matplotlib.cm import get_cmap
        import matplotlib as mpl

        if interactive:
            plt.ion()

        if colormap is not None:
            get_colour = lambda x: get_cmap(colormap)(x)
        else:
            get_colour = lambda x: x * np.array([.5, .5, 1.])
        xyz = sph2vec(np.pi/2 - sensor.theta_local, np.pi + sensor.phi_local, sensor.R_c)
        xyz[0] *= -1

        lat, lon = hp.Rotator(rot=(
            np.rad2deg(-sensor.yaw), np.rad2deg(-sensor.pitch), np.rad2deg(-sensor.roll)
        ))(sensor.sky.lat, np.pi-sensor.sky.lon)
        xyz_sun = sph2vec(np.pi/2 - lat, np.pi + lon, sensor.R_c)
        xyz_sun[0] *= -1

        if sides:
            figsize = (10, 10)
        else:
            if colormap is None or scale is None:
                figsize = (5, 5)
            else:
                figsize = (5.05, 5.05)
        plt.figure("Sensor Design" if title is None else title, figsize=figsize)

        # top view
        if sides:
            ax_t = plt.subplot2grid((4, 4), (1, 1), colspan=2, rowspan=2, aspect="equal", adjustable='box-forced')
        else:
            if colormap is not None and scale is not None:
                ax_t = plt.subplot2grid((10, 10), (0, 0), colspan=9, rowspan=9, aspect="equal", adjustable='box-forced')
            else:
                ax_t = plt.subplot(111)
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

        stheta, sphi = sensor.theta_local, np.pi + sensor.phi_local
        sL = sL if sL is not None else sensor.L
        if sensor.mode == "event":
            sL = np.clip(1. * sL + .5, 0., 1.)
        for (x, y, z), th, ph, L in zip(xyz.T, stheta, sphi, sL):

            lens = Ellipse(xy=[x, y], width=1.5 * sensor.r_l, height=1.5 * np.cos(th) * sensor.r_l,
                           angle=np.rad2deg(ph))
            ax_t.add_artist(lens)
            lens.set_clip_box(ax_t.bbox)
            lens.set_facecolor(get_colour(np.asscalar(L)))
        if show_sun:
            ax_t.plot(xyz_sun[0], xyz_sun[1], 'ro', markersize=22)

        ax_t.set_xlim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
        ax_t.set_xticklabels([])
        ax_t.set_yticklabels([])

        if sides:
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
                lens.set_facecolor(get_colour(L))

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
                lens.set_facecolor(get_colour(L))

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
                lens.set_facecolor(get_colour(L))

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
                lens.set_facecolor(get_colour(L))

            ax.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
            ax.set_xlim(-sensor.R_c - 2, 0)
            ax.set_xticks([])
        else:
            plt.axis('off')

        if colormap is not None and not sides and scale is not None:
            ax = plt.subplot2grid((10, 10), (9, 9), aspect="equal", adjustable='box-forced', projection='polar')
            ax._direction = 2*np.pi
            ax.set_theta_zero_location("N")

            # quant_steps = 360
            cb = mpl.colorbar.ColorbarBase(
                ax,
                cmap=get_cmap(colormap),
                norm=mpl.colors.Normalize(0.0, 2*np.pi),
                orientation='horizontal',
                ticks=np.linspace(0, 2*np.pi, 4, endpoint=False)
            )

            cb.outline.set_visible(False)
            # cb.ax.set_xticks([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi])
            cb.ax.set_xticklabels(np.linspace(scale[0], scale[1], 4, endpoint=False))
            cb.ax.tick_params(pad=1, labelsize=8)

        plt.tight_layout(pad=0.)

        if interactive:
            plt.draw()
            plt.pause(.1)
        else:
            plt.show()


def encode_sun(lon, lat):
    return encode_sph(lat, lon, length=NB_EN)


def decode_sun(x):
    lat, lon = decode_sph(x)
    return lon, lat


if __name__ == "__main__1__":
    import matplotlib.pyplot as plt
    from sky import get_seville_observer
    from datetime import datetime

    s, p = 20, 4
    # modes: "normal", "cross", "event"
    sensor = CompassSensor(nb_lenses=1000, fov=np.deg2rad(90), mode="normal")
    # s.load_weights()

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime(2018, 6, 21, 10, 0, 0)

    sensor.sky.obs = observer
    for angle in np.linspace(0, np.pi/2, 9):
        plt.figure("Sensor - values", figsize=(8, 21))
        sensor.rotate(pitch=np.pi / 18)
        sensor.refresh()

        lum_r = sensor.L * .05
        lum_g = sensor.L * .53
        lum_b = sensor.L * .79
        L = np.clip(np.concatenate((
            lum_r[..., np.newaxis],
            lum_g[..., np.newaxis],
            lum_b[..., np.newaxis]
        ), axis=-1), 0, 1)
        print(np.rad2deg(sensor.yaw_pitch_roll))
        plt.scatter(sensor.phi_local, sensor.theta_local,
                    c=L, marker=".", s=np.power(s, p * np.absolute(sensor.theta_local) / np.pi))
        plt.ylabel("epsilon (elevation)")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi / 2, np.pi / 2])
        plt.xticks([-np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], [])
        plt.yticks([-np.pi / 2, -np.pi / 3, -np.pi / 6, 0, np.pi / 6, np.pi / 3, np.pi / 2],
                   ["-90", "-60", "-30", "0", "30", "60", "90"])

        plt.show()


if __name__ == "__main__":
    from sky import get_seville_observer
    from datetime import datetime

    tilting = False
    step = 45
    # modes: "normal", "cross", "event"
    s = CompassSensor(fov=np.deg2rad(60), nb_lenses=3000, mode="cross", fibonacci=False)
    # s.activate_pol_filters(False)
    # s.load_weights()

    # default observer is in Seville (where the data come from)
    observer = get_seville_observer()
    observer.date = datetime(2018, 6, 21, 12, 0, 0)

    s.sky.obs = observer
    # s.rotate(pitch=np.pi / 2)

    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 13, endpoint=True)):
        s.refresh()
        # lon, lat = sky.lon, sky.lat
        # print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
        # lon, lat = s.update_parameters(sky)
        # print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
        # sL = (s.L - .5).dot(s.w_whitening) * 3 + .5
        sL = s.L
        # print sL.min(), sL.max()
        # sL = (sL - sL.min()) / (sL.max() - sL.min())
        # sL = (sL - .5) / (sL.max() - sL.min()) + .5
        # sL = ((s._aop_filter + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) + .5
        # sL = ((s.AOP + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) + .5
        # d = (s.AOP - s._aop_filter + np.pi) % (2 * np.pi) - np.pi
        # sL = .5 * np.cos(d) + .5
        # sL = np.sqrt((np.square(np.cos(d) * s.sky.L / 14) + np.square(np.sin(d + np.pi/2) * s.sky.L / 14)))
        # sL = s.sky.L / 14
        print(sL.min(), sL.max())
        CompassSensor.visualise(s, sL=sL, colormap="coolwarm",
                                title="%s-sensor_design_%03d_%03d" % (s.mode, np.rad2deg(s.fov), s.nb_lenses),
                                show_sun=True, sides=False, interactive=False, scale=[0, 1])
        s.rotate(yaw=np.pi/6)
        # observer.date = datetime(2018, 6, 21, 8 + i, 0, 0)
        # s.sky.obs = observer
        # break


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
    sky = SkyModel(observer=observer, nside=1)
    sky.generate()

    s.sky = sky
    p[i] = s.L
    # p[i] = s.DOP
    # p[i] = np.rad2deg(s.AOP)
    p_i_max = p[i].max()
    p_i_min = p[i].min()
    p[i] = (p[i] - p_i_min) / (p_i_max - p_i_min)
    hp.orthview(p, rot=(0, 90, 0))
    plt.show()