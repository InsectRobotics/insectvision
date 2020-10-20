from datetime import datetime, timedelta
from sphere.transform import eleadj

import numpy as np

eps = np.finfo(float).eps  # type: float


def sky_clearness(Z, Dh, I, kapa=1.041):
    """

    :param Z: solar zenith angle
    :param Dh: horizontal diffuse irradiance
    :param I: normal incidence direct irradiance
    :param kapa: a constant equal to 1.041 for Z in radians
    :return: the sky's clearness noted as epsilon
    """
    return ((Dh + I) / Dh + kapa * np.power(Z, 3)) / (1 + kapa * np.power(Z, 3))


def sky_brightness(Dh, m, I_0):
    """

    :param Dh: the horizontal diffuse irradiance
    :param m: the relative optical airmass
    :param I_0: the extraterrestrial irradiance
    :return: the sky's brightness noted as Delta
    """
    return Dh * m / I_0


def water_content(Td):
    """

    :param Td: (C) the three-hourly surface dew point temperature
    :return: the atmospheric precipitable water content, denoted W (cm)
    """
    return np.exp(.07 * Td - .075)


def spectral_power(lam):
    """
    Relative spectral power distribution.
    Note: The wavelengths are to be taken as being in standard air (dry air at 15C and 101325 Pa, containing 0.03% by
     volume of carbon dioxide.
    :param lam: the wavelength in nanometres [300nm, 830nm]
    :return: the spectral power of the correspondent wavelength
    """
    a = 100. * np.power(560. / lam, 5)
    b = np.exp((1.435 * np.power(10, 7)) / (2848 * 560)) - 1
    c = np.exp((1.435 * np.power(10, 7)) / (2848 * lam)) - 1
    return a * b / c


def get_seville_observer():
    import ephem

    seville = ephem.Observer()
    seville.lat = '37.392509'
    seville.lon = '-5.983877'
    return seville


def sph2pix(sph, height=64, width=64):
    """
    Transforms the spherical coordinates to vertical and horizontal pixel
    indecies.
    :param sph:     the spherical coordinates (elevation, azimuth, radius)
    :param height:  the height of the image
    :param width:   the width of the image
    """
    theta, phi, rho = sph

    theta = theta % np.pi
    phi = phi % (2 * np.pi)

    x = np.int32((height - 1) / 2. + np.cos(phi) * np.sin(theta) * (height - 1) / 2.)
    y = np.int32((width - 1) / 2. + np.sin(phi) * np.sin(theta) * (width - 1) / 2.)

    return np.array([x, y])


def pix2sph(pix, height=64, width=64):
    """
    Transforms vertical and horizontal pixel indecies to spherical coordinates
    (elevation, azimuth).
    :param pix:     the vertical and horizontal pixel indecies
    :param height:  the height of the image
    :param width:   the width of the image
    """
    x, y = np.float32(pix)
    h, w = np.float(height - 1), np.float(width - 1)
    v = 2. * np.array([x / w, y / h]) - 1.
    l = np.sqrt(np.square(v).sum(axis=0))
    theta = np.nan * np.ones_like(l)
    theta[l <= 1] = np.arcsin(l[l <= 1])
    phi = (np.arctan2(v[1], v[0]) + np.pi) % (2 * np.pi) - np.pi
    phi[l > 1] = np.nan
    return np.array([theta, phi])


def azi2pix(phi, width=64):
    """
    Transforms an azimuth values to a horizontal pixel index.
    :param phi:     azimuth
    :param width:   the width of the image
    """
    return (width * ((np.pi - phi) % (2 * np.pi)) / (2 * np.pi)).astype(int)


def pix2azi(pix, width=64):
    """
    Transforms a horizontal pixel index to azimuth values.
    :param pix:     the horizontal pixel index
    :param width:   the width of the image
    """
    return (np.pi - 2 * np.pi * pix.astype(float) / width) % (2 * np.pi)


def ele2pix(theta, height=64):
    """
    Transforms an elevation values to vertical pixel index.
    :param theta:   the elevation
    :param height:  the height of the image
    """
    return height - (height * (eleadj(theta) + np.pi / 2) / np.pi).astype(int)


def pix2ele(pix, height=64):
    """
    Transforms a vertical pixel index to elevation value.
    :param pix:     the vertical pixel index
    :param height:  the height of the image
    """
    return (np.pi * (height - pix - 1).astype(float) / height) % (np.pi / 2)


def ang2pix(angle, num_of_pixels=640):
    """
    Transforms an angle to a pixel index.
    :param angle:           the input angle
    :param num_of_pixels:   the maximum number of pixels
    """
    return (num_of_pixels * angle / np.pi)


def pix2ang(pix, num_of_pixels=640):
    """
    Transforms a pixel index to an angle.
    :param pix:             the pixel index
    :param num_of_pixels:   the maximum number of pixels
    """
    return np.pi * pix.astype(float) / num_of_pixels


def shifted_datetime(roll_back_days=153, lower_limit=7.5, upper_limit=19.5):
    date_time = datetime.now() - timedelta(days=roll_back_days)
    if lower_limit is not None and upper_limit is not None:
        uhours = int(upper_limit // 1)
        uminutes = timedelta(minutes=(upper_limit % 1) * 60)
        lhours = int(lower_limit // 1)
        lminutes = timedelta(minutes=(lower_limit % 1) * 60)
        if (date_time - uminutes).hour > uhours or (date_time - lminutes).hour < lhours:
            date_time = date_time + timedelta(hours=12)
    return date_time
