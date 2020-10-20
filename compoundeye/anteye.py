import numpy as np
from .model import CompoundEye


class AntEye(CompoundEye):

    def __init__(self, ommatidia=None, height=10, width=36):

        if ommatidia is None:
            fov = (-np.pi/6, np.pi/3)

            ground = np.abs(fov[0]) / (np.pi / 2)
            sky = np.abs(fov[1]) / (np.pi / 2)

            Z = (sky + ground) / 2

            thetas = np.linspace(fov[1], fov[0], height, endpoint=True)
            phis = np.linspace(np.pi, -np.pi, width, endpoint=False)
            thetas, phis = np.meshgrid(thetas, phis)
            ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        super(AntEye, self).__init__(
            ommatidia=ommatidia,
            central_microvili=(0., 0.),
            noise_factor=.1,
            activate_dop_sensitivity=False
        )

        self._channel_filters.pop("uv")
        self._channel_filters.pop("b")
        self.activate_pol_filters(False)
