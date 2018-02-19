import numpy as np
RNG = np.random.RandomState(2018)


def get_microvilli_angle(epsilon, alpha, theta=np.pi/6, phi=np.pi/18, n=.1):
    """
    Returns the orientation of the microvilli associated to each ommatidium.

    :param epsilon: elevation of ommatidia
    :type epsilon: np.ndarray
    :param alpha: azimuth of ommatidia
    :type alpha: np.ndarray
    :param theta: elevation shift of the centre
    :type theta: float
    :param phi: azimuth shift of the centre
    :type phi: float:
    :param n: noise of polarisation sensitivity
    :type n: float
    :return: the microvilli orientation and polarisation sensitivity
    """
    s = 1.  # sign
    if (alpha > 0).sum() < (alpha < 0).sum():
        s *= (-1)

    # code spherical coordinates of the ommatidia to 3D vectors
    xyz = np.array([
        np.cos(epsilon) * np.cos(s * alpha),
        np.cos(epsilon) * np.sin(s * alpha),
        np.sin(epsilon)
    ])

    # shift the orientation of the vectors according to the parameters
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    xyz = R_y.dot(R_z.dot(xyz))

    # calculate the orientation of each of the microvilli
    angle = (s * np.arctan2(xyz[1], xyz[0]) + 3 * np.pi/2) % (2 * np.pi) - np.pi  # type: np.ndarray
    noise = RNG.randn(angle.size) * np.square(xyz[2]) * np.maximum(n, 0.)
    dop = np.power(xyz[2], 4) + noise
    dop[xyz[2] < 0] = 0
    dop = dop > .5
    return angle + noise, np.clip(dop, 0, 1)
