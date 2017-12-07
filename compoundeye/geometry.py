import numpy as np
import healpy as hp


LENS_RADIUS = 1  # mm
A_lens = np.pi * np.square(LENS_RADIUS)


# def angles_distribution(nb_lenses, fov):
#     theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith
#
#     r_l = LENS_RADIUS  # the small radius of a hexagon (mm)
#     R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
#     S_l = 3 * r_l * R_l  # area of a lens (mm^2)
#
#     S_a = nb_lenses * S_l  # area of the dome surface (mm^2)
#     R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta))))  # radius of the curvature (mm)
#     S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)
#     R_a = R_c * np.sin(theta)  # radius of the dome's disk
#     C_a = 2 * np.pi * R_a  # perimeter of the dome's disk
#     C_c = 2 * np.pi * R_c  # perimeter of the sphere
#
#     d_theta = 2 * np.pi / (C_c / (2 * r_l))  # the angular distance of each row of ommatidia in the equator
#     print "d_theta:", np.rad2deg(d_theta)
#     d = 2 * r_l
#     s_default = 0
#     thetas = np.empty(0, dtype=np.float32)
#     phis = np.empty(0, dtype=np.float32)
#     i = 0
#     while thetas.size != nb_lenses:
#         C = C_a
#         theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith
#         if thetas.size > nb_lenses:
#             s_default = np.maximum(0, s_default + (1. / (i + 1.)) * r_l)
#         else:
#             s_default = np.maximum(0, s_default - (1. / (i + 1.)) * r_l)
#         print i, theta, thetas.size, s_default
#         thetas = np.empty(0, dtype=np.float32)
#         phis = np.empty(0, dtype=np.float32)
#         s = s_default
#         while theta >= 0:
#             n = int(np.floor(C / (d + s)))
#             if n <= 0:
#                 break
#             s = C / n - d
#             print i,
#             print "n:", n, "s:", s, "d:", d,
#             d_phi = 2 * np.pi / n
#             print "d_phi:", np.rad2deg(d_phi),
#             thetas = np.append(thetas, np.ones(n) * theta)
#             phis = np.append(phis, (np.linspace(0, 2 * np.pi, n, endpoint=False) + (d_phi / 2)) % (2 * np.pi))
#             theta -= d_theta  # * (1 - s / d)
#             print "theta:", np.rad2deg(theta)
#             R = R_c * np.sin(theta)
#             C = 2 * np.pi * R
#             s = 0.
#         i += 1
#         if i > 100:
#             break
#         print thetas.shape, phis.shape
#     print thetas.shape, phis.shape
#
#     return thetas, phis


def angles_distribution(nb_lenses, fov):
    theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith

    r_l = LENS_RADIUS  # the small radius of a hexagon (mm)
    R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
    S_l = 3 * r_l * R_l  # area of a lens (mm^2)

    S_a = nb_lenses * S_l  # area of the dome surface (mm^2)
    R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta))))  # radius of the curvature (mm)
    S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

    coverage = S_a / S_c
    # compute the parameters of the sphere
    nside = 0
    npix = hp.nside2npix(2 ** nside)
    nb_slots_available = int(np.ceil(npix * coverage))
    while nb_lenses > nb_slots_available:
        nside += 1
        npix = hp.nside2npix(2 ** nside)
        thetas, _ = hp.pix2ang(2 ** nside, np.arange(npix))
        nb_slots_available = (thetas < theta).sum()
    nside = 2 ** nside

    def complete_circles():
        iii = np.arange(nb_slots_available)
        thetas, phis = hp.pix2ang(nside, iii)
        u_theta = np.sort(np.unique(thetas))
        nb_slots = np.zeros_like(u_theta, dtype=int)
        for j, uth in enumerate(u_theta):
            nb_slots[j] = (thetas == uth).sum()

        j = np.zeros(nb_lenses, dtype=int)
        k = 0
        if nb_slots.sum() == nb_lenses:
            return iii
        else:
            x = []
            start = 0
            for jj, shift in enumerate(nb_slots / 2):
                shifts = np.append(np.arange(jj % 2, shift, 2), np.arange(jj % 2 + 1, shift, 2))
                for jjj in shifts:
                    x.append([])
                    x[-1].append(start + jjj)
                    x[-1].append(start + shift + jjj)
                start += 2 * shift
            x = np.append(np.array(x[0::2]), np.array(x[1::2])).flatten()
            j[:] = x[:nb_lenses]
        return j

    # calculate the pixel indices
    ii = np.sort(complete_circles())
    # get the longitude and co-latitude with respect to the zenith
    thetas, phis = hp.pix2ang(nside, ii)  # return longitude and co-latitude in radians

    return thetas, phis, nb_lenses == nb_slots_available


def fibonacci_sphere(samples, fov):

    theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith
    phi = (1. + np.sqrt(5)) * np.pi

    r_l = LENS_RADIUS  # the small radius of a hexagon (mm)
    R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
    S_l = 3 * r_l * R_l  # area of a lens (mm^2)

    S_a = samples * S_l  # area of the dome surface (mm^2)
    R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta))))  # radius of the curvature (mm)
    S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

    total_samples = int(samples * S_c / (1.2 * S_a))

    indices = np.arange(0, total_samples, dtype=float)

    thetas = np.arccos(2 * indices / (total_samples - .5) - 1)
    phis = (phi * indices) % (2 * np.pi)

    return thetas[-samples:], phis[-samples:]

if __name__ == "__main__":

    sph = angles_distribution(27, 180)
    print sph[0].shape, sph[1].shape
