import numpy as np
from numbers import Number


def encode_sph(theta, phi=None, length=8):
    if phi is None:
        if not isinstance(theta, Number) and theta.shape[0] > 1:
            phi = theta[1]
            theta = theta[0]
        else:
            phi = theta
            theta = np.pi / 2
    theta = np.absolute(theta)
    alpha = np.linspace(0, 2 * np.pi, length, endpoint=False)
    return np.sin(alpha + phi + np.pi / 2) * theta / (length / 2.)


def decode_sph(I):
    fund_freq = np.fft.fft(I)[1]
    phi = (np.pi - np.angle(np.conj(fund_freq))) % (2 * np.pi) - np.pi
    theta = np.absolute(fund_freq)
    return np.array([theta, phi])


def decode_xy(I):
    length = I.shape[-1]
    alpha = np.linspace(0, 2 * np.pi, length, endpoint=False)
    x = np.sum(I * np.cos(alpha), axis=-1)[..., np.newaxis]
    y = np.sum(I * np.sin(alpha), axis=-1)[..., np.newaxis]
    return np.concatenate((x, y), axis=-1)
