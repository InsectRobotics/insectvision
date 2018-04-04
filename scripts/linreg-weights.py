#!/usr/bin/env python
# Compute weights for the sensor using Linear regression.
#

from compoundeye.geometry import fibonacci_sphere, angles_distribution
from compoundeye import CompassSensor, encode_sun
from learn.loss_function import SensorObjective
from sphere import sph2vec, vec2sph, azidist
from sphere.transform import point2rotmat

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2018, The Invisible Cues Project"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Development"


if __name__ == "__main__":

    S = 1000
    D = 60
    N = 8
    tilt = True

    if tilt:
        angles = np.array([
            [0., 0.],
            [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
            [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4], [np.pi / 6, 7 * np.pi / 4],
            [np.pi / 3, 0.], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
            [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4], [np.pi / 3, 7 * np.pi / 4]
        ])  # 17
        if S == 1000:
            S /= 2
    else:
        angles = np.array([[0., 0.]])  # 1

    theta, phi, fit = angles_distribution(D, 60)
    alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi
    phi_tb1 = np.linspace(0., 2 * np.pi, N, endpoint=False)  # TB1 preference angles

    theta_s, phi_s = fibonacci_sphere(samples=S, fov=180)
    R = np.empty((0, D))
    Y = np.empty((0, N))
    for j, (theta_t, phi_t) in enumerate(angles):

        v_t = sph2vec(theta_t, phi_t, zenith=True)
        v_s = sph2vec(theta_s, phi_s, zenith=True)
        v = sph2vec(theta, phi, zenith=True)
        v_a = sph2vec(np.full(alpha.shape[0], np.pi / 2), alpha, zenith=True)
        M = point2rotmat(v_t)
        theta_s_, phi_s_, _ = vec2sph(M.dot(v_s), zenith=True)
        # theta_s_, phi_s_ = transtilt(-theta_t, -phi_t, theta=theta_s, phi=phi_s)

        theta_, phi_, _ = vec2sph(M.T.dot(v), zenith=True)
        # theta_, phi_ = transtilt(theta_t, phi_t, theta=theta, phi=phi)
        _, alpha_, _ = vec2sph(M.T.dot(v_a), zenith=True)
        # _, alpha_ = transtilt(theta_t, phi_t, theta=np.pi/2, phi=alpha)

        for i, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):
            L, P, A = SensorObjective.encode(e_org, a_org, theta_, phi_)

            # polarisation filter
            r = (np.sqrt(np.square(np.sin(A - alpha_)) + np.square(np.cos(A - alpha_)) * np.square(1. - P)) -
                 np.sqrt(np.square(np.cos(A - alpha_)) + np.square(np.sin(A - alpha_)) * np.square(1. - P))) / \
                (np.sqrt(np.square(np.sin(A - alpha_)) + np.square(np.cos(A - alpha_)) * np.square(1. - P)) +
                 np.sqrt(np.square(np.cos(A - alpha_)) + np.square(np.sin(A - alpha_)) * np.square(1. - P)))

            R = np.vstack([R, r])
            Y = np.vstack([Y, encode_sun(a, e)])
    print R.shape, Y.shape

    W = np.linalg.pinv(R).dot(Y)
    print W.shape, W.max(), W.min()

    cost = SensorObjective._fitness(theta, phi, alpha, w=W, tilt=tilt,
                                    error=azidist)
    print cost

    s = CompassSensor(thetas=theta, phis=phi, alphas=alpha)
    CompassSensor.visualise(s, sL=1. * np.sqrt(np.square(W).sum(axis=1)) + .5, colormap="coolwarm", sides=False, scale=None)
    # for i in xrange(8):
    #     CompassSensor.visualise(s, sL=10.*W[:, i]+.5, colormap="coolwarm", sides=False, scale=None)

    plt.figure()
    plt.imshow(W, cmap="coolwarm")
    plt.show()
