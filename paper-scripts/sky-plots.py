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
    if type(noise) is np.ndarray:
        if noise.size == P.size:
            # print "yeah!"
            eta = np.array(noise, dtype=bool)
        else:
            eta = np.zeros_like(theta, dtype=bool)
            eta[:noise.size] = noise
    elif noise > 0:
        eta = np.argsort(np.absolute(np.random.randn(*P.shape)))[:int(noise * P.shape[0])]
        # eta = np.array(np.absolute(np.random.randn(*P.shape)) < noise, dtype=bool)
        if verbose:
            print "Noise level: %.4f (%.2f %%)" % (noise, 100. * eta.sum() / float(eta.size))
    else:
        eta = np.zeros_like(theta, dtype=bool)
    P[eta] = 0.  # destroy the polarisation pattern
    Y[eta] = 10.

    return Y, P, A, theta, phi


if __name__ == "__main__":
    from compoundeye import POLCompassDRA
    from environment import Sky

    samples = 1000
    noise = .0
    # noise = .99

    theta, phi = fibonacci_sphere(samples, 180)
    sky = Sky(np.pi/6, np.pi)
    y, p, a = sky(theta, phi)
    # y, p, a, theta, phi = skyfeatures(noise, samples=50000)
    # dra = POLCompassDRA()
    # r = dra(sky, noise=noise)
    #
    # print r.shape
    # print r
    #
    # plt.figure("pol-%02d" % (10 * noise), figsize=(3, 3))
    # ax = plt.subplot(111, polar=True)
    # ax.scatter(dra.phi, dra.theta, s=100, marker='.', c=r, cmap="coolwarm", vmin=-.1, vmax=.1)
    # ax.set_theta_zero_location("N")
    # ax.set_theta_direction(-1)
    # ax.set_ylim([0, np.deg2rad(30)])
    # plt.yticks([])
    # plt.xticks([])

    print y[y > 0].min()

    plt.figure("sky-lum-%02d" % (10 * noise), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    es = ax.scatter(phi, theta, s=10, marker='.', c=y, cmap="Blues_r", vmin=-0., vmax=7.)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.colorbar(es)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('sky-lum-%02d.svg' % (10 * noise))

    print p.max()

    plt.figure("sky-dop-%02d" % (10 * noise), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    es = ax.scatter(phi, theta, s=10, marker='.', c=p, cmap="Greys", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.colorbar(es)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('sky-dop-%02d.svg' % (10 * noise))

    plt.figure("sky-aop-%02d" % (10 * noise), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    es = ax.scatter(phi, theta, s=10, marker='.', c=a, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.colorbar(es)
    plt.yticks([])
    plt.xticks([])
    plt.savefig('sky-aop-%02d.svg' % (10 * noise))

    plt.show()
