import numpy as np
from .model import CompoundEye
from .beeeye import load_both_eyes


class BumbleBeeEye(CompoundEye):

    def __init__(self, left=True, right=False):
        ommatidia_left, ommatidia_right = load_both_eyes()

        ommatidia = np.empty((0, 2))
        if left:
            ommatidia = np.append(ommatidia, ommatidia_left, axis=0)
        if right:
            ommatidia = np.append(ommatidia, ommatidia_right, axis=0)

        self.__left = np.zeros(ommatidia.shape[0], dtype=bool)
        self.__right = np.zeros(ommatidia.shape[0], dtype=bool)
        if left:
            self.__left[:ommatidia_left.shape[0]] = True
        if right:
            self.__right[-ommatidia_right.shape[0]:] = True
        super(BumbleBeeEye, self).__init__(
            ommatidia=ommatidia,
            central_microvili=(np.pi/6, np.pi/18),
            noise_factor=.1,
            activate_dop_sensitivity=True
        )

        self._channel_filters.pop("g")
        self.activate_pol_filters(True)
