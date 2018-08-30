from compoundeye.geometry import angles_distribution, fibonacci_sphere
from sphere import azidist
from sphere.transform import tilt
from learn.loss_function import SensorObjective

import matplotlib.pyplot as plt
import numpy as np


def skyfeatures(noise=0., simple_pol=False, samples=1000, verbose=False):

    # default parameters
    tau_L = 2.
    c1 = .6
    c2 = 4.
    eps = np.finfo(float).eps
    AA, BB, CC, DD, EE = SensorObjective.T_L.dot(np.array([tau_L, 1.]))  # sky parameters
    T_T = np.linalg.pinv(SensorObjective.T_L)
    tau_L, c = T_T.dot(np.array([AA, BB, CC, DD, EE]))
    tau_L /= c  # turbidity correction

    # Prez. et. al. Luminance function
    def L(cchi, zz):
        ii = zz < (np.pi/2)
        ff = np.zeros_like(zz)
        if zz.ndim > 0:
            ff[ii] = (1. + AA * np.exp(BB / (np.cos(zz[ii]) + eps)))
        elif ii:
            ff = (1. + AA * np.exp(BB / (np.cos(zz) + eps)))
        pphi = (1. + CC * np.exp(DD * cchi) + EE * np.square(np.cos(cchi)))
        return ff * pphi

    theta_s, phi_s = np.array([np.pi/6]), np.array([np.pi])

    theta, phi = fibonacci_sphere(samples, 180)
    phi = phi[theta <= np.pi / 2]
    theta = theta[theta <= np.pi / 2]
    samples = theta.size

    theta = (theta - np.pi) % (2 * np.pi) - np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    alpha = (phi + np.pi / 2) % (2 * np.pi) - np.pi

    # SKY INTEGRATION
    gamma = np.arccos(np.cos(theta) * np.cos(theta_s) + np.sin(theta) * np.sin(theta_s) * np.cos(phi - phi_s))
    # Intensity
    I_prez, I_00, I_90 = L(gamma, theta), L(0., theta_s), L(np.pi / 2, np.absolute(theta_s - np.pi / 2))
    # influence of sky intensity
    I = (1. / (I_prez + eps) - 1. / (I_00 + eps)) * I_00 * I_90 / (I_00 - I_90 + eps)
    chi = (4. / 9. - tau_L / 120.) * (np.pi - 2 * theta_s)
    Y_z = (4.0453 * tau_L - 4.9710) * np.tan(chi) - 0.2155 * tau_L + 2.4192
    Y = np.maximum(Y_z * I_prez / (I_00 + eps), 0.)  # Illumination

    # Degree of Polarisation
    M_p = np.exp(-(tau_L - c1) / (c2 + eps))
    LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
    if simple_pol:
        P = np.clip(2. / np.pi * M_p * LP, 0., 1.)
    else:
        P = np.clip(2. / np.pi * M_p * LP * (theta * np.cos(theta) + (np.pi/2 - theta) * I), 0., 1.)

    # Angle of polarisation
    _, A = tilt(theta_s, phi_s + np.pi, theta, phi)

    # create cloud disturbance
    if noise > 0:
        eta = np.absolute(np.random.randn(*P.shape)) < noise
        if verbose:
            print "Noise level: %.4f (%.2f %%)" % (noise, 100. * eta.sum() / float(eta.size))
        Y[eta] = 10.
        P[eta] = 0.  # destroy the polarisation pattern
    else:
        eta = np.zeros(1)

    return Y, P, A, theta, phi


if __name__ == "__main__":
    noise = 1.33

    y, p, a, theta, phi = skyfeatures(noise=noise, samples=3000)

    print y.max()

    plt.figure("sky-lum-%02d" % (10 * noise), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    ax.scatter(phi, theta, s=10, marker='.', c=y, cmap="Blues_r", vmin=0, vmax=7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([])
    plt.xticks([])

    print p.max()

    plt.figure("sky-dop-%02d" % (10 * noise), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    ax.scatter(phi, theta, s=10, marker='.', c=p, cmap="Greys", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([])
    plt.xticks([])

    plt.show()
