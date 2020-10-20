import numpy as np


def point2rotmat(p):
    z = np.array([0., 0., 1.])
    v = np.cross(z, p)
    c = np.dot(z, p)
    v_x = np.array([[0., -v[2], v[1]],
                    [v[2], 0., -v[0]],
                    [-v[1], v[0], 0.]])
    return np.eye(3) + v_x + np.matmul(v_x, v_x) / (1 + c)


# obsolete
def sph2rotmat(theta, phi):
    r00 = 1. - 2. * np.square(np.sin(theta / 2.) * np.sin(phi))
    r01 = -np.square(np.sin(theta / 2.)) * np.sin(2 * phi)
    r02 = np.sin(theta) * np.sin(phi)
    r10 = -np.square(np.sin(theta / 2.)) * np.sin(2 * phi)
    r11 = np.cos(theta) * np.square(np.cos(phi)) + np.square(np.sin(phi))
    r12 = np.sin(theta) * np.cos(phi)
    r20 = -np.sin(theta) * np.sin(phi)
    r21 = -np.sin(theta) * np.cos(phi)
    r22 = np.cos(theta)
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])


def tilt(theta_t, phi_t, theta, phi):

    x = np.sin(theta) * (np.sin(phi) - 2 * np.square(np.sin(theta_t / 2)) * np.sin(phi_t) * np.cos(phi_t - phi)) +\
        np.sin(theta_t) * np.cos(theta) * np.sin(phi_t)
    y = -np.sin(theta) * np.sin(phi) * np.square(np.sin(theta_t / 2.)) * np.sin(2 * phi_t) +\
        np.sin(theta) * np.cos(phi) * (np.cos(theta_t) * np.square(np.cos(phi_t)) + np.square(np.sin(phi_t))) +\
        np.cos(theta) * np.sin(theta_t) * np.cos(phi_t)
    z = np.cos(theta_t) * np.cos(theta) - np.sin(theta_t) * np.sin(theta) * np.cos(phi_t - phi)

    e = np.arccos(z)
    a = np.arctan2(x, y)
    return e, a


def sph2vec(theta, phi=None, rho=1., zenith=False):
    """
    Transforms the spherical coordinates to a cartesian 3D vector.
    :param theta:  elevation
    :param phi:    azimuth
    :param rho:    radius length
    :param zenith: whether zenith is the 0 elevation point other than 90 deg elevation point
    :return:       the cartesian vector
    """
    if phi is None:
        phi = theta[1]
        if theta.shape[0] > 2:
            rho = theta[2]
        theta = theta[0]

    if not zenith:
        theta = np.pi/2 - theta

    x = rho * np.sin(theta) * np.sin(phi)
    y = rho * np.sin(theta) * np.cos(phi)
    z = rho * np.cos(theta)

    return np.asarray([x, y, z])


def vec2sph(vec, y=None, z=None, zenith=False):
    """
    Transforms a cartesian vector to spherical coordinates.
    :param vec:    the cartesian vector
    :param y:      the y component of the cartesian vector (in case vec is the x component)
    :param z:      the z component of the cartesian vector (in case vec is the x component)
    :param zenith: whether zenith is the 0 elevation point other than 90 deg elevation point
    :return:       the spherical coordinates
    """

    if y is None or z is None:
        assert vec.shape[0] == 3
        x, y, z = vec
    else:
        x = vec
    vec = np.array([x, y, z])

    rho = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    if vec.ndim == 1 and rho == 0:
        rho = 1.
    elif vec.ndim > 1:
        rho[rho == 0] = 1.
    x, y, z = vec / rho  # normalised vector

    theta = np.arccos(z)
    phi = np.arctan2(x, y)

    if vec.ndim == 1 and np.isclose(theta, 0):
        phi = 0.
    elif vec.ndim > 1:
        phi[np.isclose(theta, 0)] = 0.

    if not zenith:
        theta = np.pi/2 - theta
    return np.asarray([theta, phi, rho])


# conditions to restrict the angles to correct quadrants
def eleadj(theta):
    """
    Adjusts the elevation in [-pi, pi]
    :param theta:   the elevation
    """
    theta, _ = sphadj(theta=theta)
    return theta


def aziadj(phi):
    """
    Adjusts the azimuth in [-pi, pi].
    :param phi: the azimuth
    """
    _, phi = sphadj(phi=phi)
    return phi


def sphadj(theta=None, phi=None,
           theta_min=-np.pi / 2, theta_max=np.pi / 2,  # constrains
           phi_min=-np.pi, phi_max=np.pi):
    """
    Adjusts the spherical coordinates using the given bounds.
    :param theta:       the elevation
    :param phi:         the azimuth
    :param theta_min:   the elevation lower bound (default -pi/2)
    :param theta_max:   the elevation upper bound (default pi/2)
    :param phi_min:     the azimuth lower bound (default -pi)
    :param phi_max:     the azimuth upper bound (default pi)
    """

    # change = np.any([theta_z < -np.pi / 2, theta_z > np.pi / 2], axis=0)
    if theta is not None:
        if (theta >= theta_max).all():
            theta = np.pi - theta
            if np.all(phi):
                phi += np.pi
        elif (theta < theta_min).all():
            theta = -np.pi - theta
            if np.all(phi):
                phi += np.pi
        elif (theta >= theta_max).any():
            theta[theta >= theta_max] = np.pi - theta[theta >= theta_max]
            if np.all(phi):
                phi[theta >= theta_max] += np.pi
        elif (theta < theta_min).any():
            theta[theta < theta_min] = -np.pi - theta[theta < theta_min]
            if np.all(phi):
                phi[theta < theta_min] += np.pi

    if phi is not None:
        while (phi < phi_min).all():
            phi += 2 * np.pi
        while (phi >= phi_max).all():
            phi -= 2 * np.pi
        while (phi < phi_min).any():
            phi[phi < phi_min] += 2 * np.pi
        while (phi >= phi_max).any():
            phi[phi >= phi_max] -= 2 * np.pi

    return theta, phi


def vec2pol(vec, y=None):
    """
    Converts a vector to polar coordinates.
    """
    if y is None:
        rho = np.sqrt(np.square(vec[..., 0:1]) + np.square(vec[..., 1:2]))
        phi = np.arctan2(vec[..., 1:2], vec[..., 0:1])

        return np.append(rho, phi, axis=-1)
    else:
        rho = np.sqrt(np.square(vec) + np.square(y))
        phi = np.arctan2(vec, y)

        return rho, phi


def pol2vec(pol, phi=None):
    """
    Convert polar coordinates to vector.
    """
    if phi is None:
        rho = pol[..., 0:1]
        phi = pol[..., 1:2]

        return rho * np.append(np.cos(phi), np.sin(phi), axis=-1)
    else:
        return pol * np.cos(phi), pol * np.sin(phi)


def azirot(vec, phi):
    """
    Rotate a vector horizontally and clockwise.
    :param vec: the 3D vector
    :param phi: the azimuth of the rotation
    """
    Rz = np.asarray([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]]
    )

    return Rz.dot(vec)


if __name__ == "__main__":
    from compoundeye.geometry import fibonacci_sphere
    # v = np.array([[1], [0], [0]], dtype=float)
    # v = np.array([0, 1, 0], dtype=float)
    theta_s, phi_s = fibonacci_sphere(samples=1000, fov=161)
    phi_s = phi_s[theta_s <= np.pi/2]
    theta_s = theta_s[theta_s <= np.pi/2]
    samples = theta_s.size

    theta_t, phi_t = np.pi/4, np.pi/4

    v_s = sph2vec(theta_s, phi_s, zenith=True)
    R = sph2rotmat(theta_t, phi_t)
    theta_s_1, phi_s_1, _ = vec2sph(R.dot(v_s), zenith=True)
    theta_s_2, phi_s_2 = tilt(theta_t, phi_t, theta_s, phi_s)

    print("Elevation:",  np.all(np.isclose(theta_s_1, theta_s_2)), end="")
    print("--- Azimuth:", np.all(np.isclose(phi_s_1, phi_s_2)))


