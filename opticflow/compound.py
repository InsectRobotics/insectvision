import numpy as np
from sphere import sph2vec
from sphere import angle_between


def build_a(values, rdir, c, kernel_size):
    """
    Build a kernel containing pixel intensities
    :param values:
    :type values: np.ndarray
    :param rdir:
    :type rdir: np.ndarray
    :param c:
    :type c: np.ndarray
    :param kernel_size:
    :type kernel_size: float
    """

    window = kernel_size / 2.
    vdir = sph2vec(rdir.T)
    vc = sph2vec(c)
    d = np.abs(np.arccos(vdir.T.dot(vc)))

    home_i = np.argmin(d)
    home = values[home_i]
    j = d <= window

    a = np.zeros((j.sum(), 2), dtype=float)
    dj = angle_between(rdir[j], rdir[home_i])
    di = np.array([home - values[j]] * 2).T
    zeros = dj == 0
    a[~zeros] = (di[~zeros] / dj[~zeros].T).T

    return a[:, ::-1]


def build_b(n_val, o_val, rdir, c, kernel_size):
    window = kernel_size / 2.
    vdir = sph2vec(rdir.T)
    vc = sph2vec(c)
    d = np.arccos(vdir.T.dot(vc))

    j = d <= window
    return n_val[j] - o_val[j]


def gaussian_weight(size, rdir, c, even=False):

    window = size / 2.
    vdir = sph2vec(rdir.T)
    vc = sph2vec(c)
    d = np.arccos(vdir.T.dot(vc))
    j = d <= window
    d = d[j]

    if even:
        return np.diag(np.ones(j.sum()))

    sigma = 2 * np.pi / rdir.shape[1]  # the standard deviation of your normal curve
    correlation = 0.  # see wiki for multivariate normal distributions
    z = (2 * np.pi * sigma * sigma) * np.sqrt(1 - correlation * correlation)
    exp = -1 / (2 * (1 - correlation * correlation)) * (d * d) / (sigma * sigma)

    return np.diag(1. / z * np.exp(exp))


if __name__ == "__main__":
    vals1 = np.arange(65 * 65, dtype=float).reshape((65, 65)) / (65 * 65)
    vals2 = vals1.T
    thetas = np.linspace(-np.pi/2, np.pi/2, 65, endpoint=False)
    phis = np.linspace(-np.pi, np.pi, 65, endpoint=False)
    thetas, phis = np.meshgrid(thetas, phis)
    dirs = np.array([thetas.reshape((-1,)), phis.reshape((-1,))]).T
    centers = np.array([[0, np.pi/4], [0, -np.pi/4]])
    ksize = np.pi/1

    vals1 = vals1.reshape((-1,))
    vals2 = vals2.reshape((-1,))
    print(vals1.shape, vals2.shape, dirs.shape)
    A = build_a(vals2, dirs, centers[0], ksize)
    B = build_b(vals2, vals1, dirs, centers[0], ksize)
    W = gaussian_weight(ksize, dirs, centers[0], even=False)
    print(A.shape, B.shape, W.shape)
    vpt = np.linalg.inv(A.T.dot(W ** 2).dot(A)).dot(A.T).dot(W ** 2).dot(B)
    print(vpt)
