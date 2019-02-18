import numpy as np


class Hybrid(object):
    """
    Condition based on the distance and rotation thresholds.
    """

    def __init__(self, tau_x=100., tau_phi=np.pi):
        """

        :param tau_x: the distance threshold in meters
        :type tau_x: float
        :param tau_phi: the rotation threshold in rads
        :type tau_phi: float
        """
        super(Hybrid, self).__init__()
        self.__step = np.abs(tau_x)  # type: float
        self.__phi = np.abs(tau_phi)  # type: float

    @property
    def step(self):
        """
        The distance threshold.
        """
        return self.__step  # type: float

    @property
    def phi(self):
        """
        The rotation threshold.
        """
        return self.__phi  # type: float

    def __call__(self, *args, **kwargs):
        """
        Indicates whether the distance and rotation exceed the thresholds
        :param d_x: the input distance
        :type d_x: float
        :param d_phi: the input rotation
        :type d_phi: float
        :return: whether the condition is valid or not
        """
        d_x, d_phi = -1., np.pi
        if len(args) > 1:
            d_x, d_phi = np.abs(args[:2])
        elif len(args) > 0:
            d_x = np.abs(args[0])
        if "d_x" in kwargs.keys():
            d_x = np.abs(kwargs["d_x"])
        if "d_phi" in kwargs.keys():
            d_phi = np.abs(kwargs["d_phi"])

        return d_x >= self.__step or d_phi >= self.__phi  # type: bool

    def to_array(self):
        """
        Transforms the parameters to an np.ndarray
        :return:
        """
        return np.array([self.step, self.phi])  # type: np.ndarray

    @classmethod
    def from_array(cls, array):
        """
        :param array: an array with the parameters
        :type array: np.ndarray, list, tuple
        :return: A Hybrid condition
        """
        d_x, d_phi = array
        return Hybrid(d_x, d_phi)

    def __str__(self):
        return "Hybrid: step=%.2f, __phi_z=%.2f" % (self.step, self.phi)


class Stepper(Hybrid):
    """
    Condition based on a distance threshold.
    """

    def __init__(self, tau_x=0.):
        super(Stepper, self).__init__(tau_x=tau_x)


class Turner(Hybrid):
    """
    Condition based on a rotation threshold.
    """

    def __init__(self, tau_phi=0.):
        super(Turner, self).__init__(tau_phi=tau_phi)


class NoneCondition(Hybrid):
    """
    Condition that is always True (independent of the distance and rotation)
    """

    def __init__(self):
        super(NoneCondition, self).__init__()
