import numpy as np
from .utils import pix2sph
from sphere import vec2sph


def cubebox_angles(side, width=64, height=64):
    """
    Generates the respective points of interest for a specific side of the cubebox.

    :param side: description of the side; one of: 'left', 'right', 'front', 'back', 'top' and 'bottom'
    :type side: basestring
    :param width: the width of the box
    :type width: int
    :param height: the height of the box
    :type height: int
    :return: theta, phi
    """
    if side == "left":
        y = np.linspace(1, -1, width, endpoint=False)
        z = np.linspace(1, -1, height, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = -np.ones(width * height)
    elif side == "front":
        x = np.linspace(-1, 1, width, endpoint=False)
        z = np.linspace(1, -1, height, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = -np.ones(width * height)
    elif side == "right":
        y = np.linspace(-1, 1, width, endpoint=False)
        z = np.linspace(1, -1, height, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = np.ones(width * height)
    elif side == "back":
        x = np.linspace(1, -1, width, endpoint=False)
        z = np.linspace(1, -1, height, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = np.ones(width * height)
    elif side == "top":
        x = np.linspace(-1, 1, width, endpoint=False)
        y = np.linspace(1, -1, width, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = np.ones(width * width)
    elif side == "bottom":
        x = np.linspace(-1, 1, width, endpoint=False)
        y = np.linspace(-1, 1, width, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = -np.ones(width * width)
    else:
        x, y, z = np.zeros((3, height * width))
    vec = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T
    theta, phi, _ = vec2sph(vec)
    return theta, phi


def cubebox(sky, side, width=64, height=64):
    """
    Computes the luminance, degree and angle of polarisation for the respective points of interest
    for a specific side of the cubebox.

    :param sky: the sky model
    :type sky: Sky
    :param side: description of the side; one of: 'left', 'right', 'front', 'back', 'top' and 'bottom'
    :type side: basestring
    :param width: the width of the box
    :type width: int
    :param height: the height of the box
    :type height: int
    :return: Y_cube, P_cube, A_cube
    """
    theta, phi = cubebox_angles(side, width, height)
    Y, P, A = sky(theta, phi)

    Y /= 6.  # set it in range [0, 1]
    Y_cube = ((1. - Y[..., np.newaxis]).dot(np.array([[.05, .53, .79]])).T + Y).T
    Y_cube = Y_cube.reshape((width, height, 3))
    Y_cube = np.clip(Y_cube, 0, 1)

    P[np.isnan(P)] = -1
    P = P.reshape((width, height))
    P_cube = np.zeros((width, height, 3))
    P_cube[..., 0] = P * .53 + (1. - P)
    P_cube[..., 1] = P * .81 + (1. - P)
    P_cube[..., 2] = P * 1.0 + (1. - P)
    P_cube = np.clip(P_cube, 0, 1)

    A = A.reshape((width, height))
    A_cube = A % np.pi
    A_cube = np.clip(A_cube, 0, np.pi)

    return Y_cube, P_cube, A_cube


def skydome(sky, width=64, height=64):
    """
    Computes the luminance, degree and angle of polarisation for the respective points of interest
    of a skydome.

    :param sky: the sky model
    :type sky: Sky
    :param width: the width of the box
    :type width: int
    :param height: the height of the box
    :type height: int
    :return: Y_cube, P_cube, A_cube
    """
    x, y = np.arange(width), np.arange(height)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    theta, phi = pix2sph(np.array([x, y]), height, width)

    sky_Y, sky_P, sky_A = sky(theta, phi)
    sky_Y /= 6.  # set it in range [0, 1]
    sky_Y = ((1. - sky_Y[..., np.newaxis]).dot(np.array([[.05, .53, .79]])).T + sky_Y).T
    sky_P = np.clip(sky_P, 0, 1)

    Y = np.zeros((width, height, 3))
    Y[x, y] = np.clip(sky_Y, 0., 1)

    P = np.zeros((width, height, 3))
    P[x, y, 0] = sky_P * .53 + (1. - sky_P)
    P[x, y, 1] = sky_P * .81 + (1. - sky_P)
    P[x, y, 2] = sky_P * 1.0 + (1. - sky_P)

    A = (sky_A + np.pi) % (2 * np.pi) - np.pi
    A = A.reshape((width, height))

    return Y, P, A

