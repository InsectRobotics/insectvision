from sphere.transform import tilt

import numpy as np

eps = np.finfo(float).eps

# Transformation matrix of turbidity to luminance coefficients
T_L = np.array([[0.1787, -1.4630],
                [-0.3554, 0.4275],
                [-0.0227, 5.3251],
                [0.1206, -2.5771],
                [-0.0670, 0.3703]])


def get_sky_cues(theta, phi, theta_s, phi_s, tau_L=2., c1=.6, c2=4., noise=.0):
    A, B, C, D, E = T_L.dot(np.array([tau_L, 1.]))
    T_T = np.linalg.pinv(T_L)
    tau_L, c = T_T.dot(np.array([A, B, C, D, E]))
    tau_L /= c

    gamma = np.arccos(np.cos(theta) * np.cos(theta_s) + np.sin(theta) * np.sin(theta_s) * np.cos(phi - phi_s))

    # Intensity
    I_prez = get_luminance(gamma, theta, A, B, C, D, E)
    I_00 = get_luminance(0., theta_s, A, B, C, D, E)
    I_90 = get_luminance(np.pi / 2, np.absolute(theta_s - np.pi / 2), A, B, C, D, E)

    # influence of sky intensity
    I = (1. / (I_prez + eps) - 1. / (I_00 + eps)) * I_00 * I_90 / (I_00 - I_90 + eps)
    chi = (4. / 9. - tau_L / 120.) * (np.pi - 2 * theta_s)
    Y_z = (4.0453 * tau_L - 4.9710) * np.tan(chi) - 0.2155 * tau_L + 2.4192
    Y = np.maximum(Y_z * I_prez / (I_00 + eps), 0.)  # Illumination

    # Degree of Polarisation
    M_p = np.exp(-(tau_L - c1) / (c2 + eps))
    LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
    P = np.clip(2. / np.pi * M_p * LP * (theta * np.cos(theta) + (np.pi / 2 - theta) * I), 0., 1.)
    # P = LP

    # Angle of polarisation
    _, A = tilt(theta_s, phi_s + np.pi, theta, phi)

    # create cloud disturbance
    if noise > 0:
        eta = np.absolute(np.random.randn(*P.shape)) < noise
        P[eta] = 0.  # destroy the polarisation pattern
    else:
        eta = np.zeros(1)

    return Y, P, A


def get_luminance(chi, z, A, B, C, D, E):
    """
    Prez et al. luminance function
    :param chi:
    :param z:
    :param A:
    :param B:
    :param C:
    :param D:
    :param E:
    :return:
    """
    i = z < (np.pi/2)
    f = np.zeros_like(z)
    if z.ndim > 0:
        f[i] = (1. + A * np.exp(B / (np.cos(z[i]) + eps)))
    elif i:
        f = (1. + A * np.exp(B / (np.cos(z) + eps)))
    phi = (1. + C * np.exp(D * chi) + E * np.square(np.cos(chi)))
    return f * phi


if __name__ == "__main__":
    from compoundeye.geometry import angles_distribution
    import matplotlib.pyplot as plt

    n = 60
    omega = 52
    theta, phi, fit = angles_distribution(n, float(omega))

    theta_s, phi_s = np.array([np.pi/4]), np.array([np.pi])

    Y, P, A = get_sky_cues(theta, phi, theta_s, phi_s)

    plt.figure("", figsize=(10, 3))
    plt.subplot(131, polar=True)
    plt.scatter(phi, theta, s=20, c=Y, cmap="Reds", vmin=0, vmax=2)
    plt.subplot(132, polar=True)
    plt.scatter(phi, theta, s=20, c=P, cmap="Reds", vmin=0, vmax=1)
    plt.subplot(133, polar=True)
    plt.scatter(phi, theta, s=20, c=A, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    plt.show()
