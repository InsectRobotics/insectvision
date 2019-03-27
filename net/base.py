import numpy as np
import yaml
import os


# get path of the script
__dir__ = os.path.dirname(os.path.abspath(__file__))
# load parameters
with open(os.path.join(__dir__, 'params.yaml'), 'rb') as f:
    params = yaml.safe_load(f)

GAIN = params['gain']
RNG = np.random.RandomState(2018)


class Network(object):

    def __init__(self, gain=GAIN, rng=RNG, dtype=np.float32):
        """

        :param gain:
        :type gain: float, int
        :param rng: the random state generator
        :type rng: np.random.RandomState
        :param dtype: the type of the values in the network
        :type dtype: Type[np.dtype]
        """
        self.gain = gain
        self.dtype = dtype
        self.rng = rng

        self.params = []

        self.__update = False

    @property
    def update(self):
        return self.__update

    @update.setter
    def update(self, value):
        self.__update = value

    def reset(self):
        self.__update = False
