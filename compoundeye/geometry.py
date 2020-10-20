import numpy as np


LENS_RADIUS = 1.  # mm
A_lens = np.pi * np.square(LENS_RADIUS)


def angles_distribution(nb_lenses, fov, verbose=False):
    import os
    __root__ = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    filename = os.path.join(__root__, "data", "compoundeye", "sensor-%d-%d.yaml" % (nb_lenses, fov))

    if os.path.isfile(filename):
        import yaml

        r_l = LENS_RADIUS  # the small radius of a hexagon (mm)
        R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
        S_l = 3 * r_l * R_l  # area of a lens (mm^2)

        S_a = nb_lenses * S_l  # area of the dome surface (mm^2)
        R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(np.deg2rad(fov / 2)))))  # radius of the curvature (mm)
        S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

        if verbose:
            print("Hexagon radius (r):", r_l)
            print("Lens radius (R):", R_l)
            print("Lens area (S):", S_l)
            print("Curvature radius (R_c):", R_c)
            print("Dome area (S_a):", S_a)

        with open(filename, "r") as f:
            params = yaml.load(f)
            thetas = np.array(params['theta'])
            phis = np.array(params['phi'])
            rhos = np.array(params['rho'])
            nb_slots_available = nb_lenses
    else:
        import healpy as hp
        # __dir__ = os.path
        theta = np.deg2rad(fov / 2)  # angular distance of the outline from the zenith

        r_l = LENS_RADIUS  # the small radius of a hexagon (mm)
        R_l = r_l * 2 / np.sqrt(3)  # the big radius of a lens (mm)
        S_l = 3 * r_l * R_l  # area of a lens (mm^2)

        S_a = nb_lenses * S_l  # area of the dome surface (mm^2)
        R_c = np.sqrt(S_a / (2 * np.pi * (1. - np.cos(theta))))  # radius of the curvature (mm)
        S_c = 4 * np.pi * np.square(R_c)  # area of the whole sphere (mm^2)

        if verbose:
            print("Hexagon radius (r):", r_l)
            print("Lens radius (R):", R_l)
            print("Lens area (S):", S_l)
            print("Curvature radius (R_c):", R_c)
            print("Dome area (S_a):", S_a)

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

        # def complete_circles():
        #     iii = np.arange(nb_slots_available)
        #     thetas, phis = hp.pix2ang(nside, iii)
        #     u_theta = np.sort(np.unique(thetas))
        #     nb_slots = np.zeros_like(u_theta, dtype=int)
        #     for j, uth in enumerate(u_theta):
        #         nb_slots[j] = (thetas == uth).sum()
        #
        #     j = np.zeros(nb_lenses, dtype=int)
        #     k = 0
        #     if nb_slots.sum() == nb_lenses:
        #         return iii
        #     else:
        #         x = []
        #         start = 0
        #         for jj, shift in enumerate(nb_slots / 2):
        #             shifts = np.append(np.arange(jj % 2, shift, 2), np.arange(jj % 2 + 1, shift, 2))
        #             for jjj in shifts:
        #                 x.append([])
        #                 x[-1].append(start + jjj)
        #                 x[-1].append(start + shift + jjj)
        #             start += 2 * shift
        #         x = np.append(np.array(x[0::2]), np.array(x[1::2])).flatten()
        #         j[:] = x[:nb_lenses]
        #     return j

        # calculate the pixel indices
        # ii = np.sort(complete_circles())
        # get the longitude and co-latitude with respect to the zenith
        # thetas, phis = hp.pix2ang(nside, ii)  # return longitude and co-latitude in radians

        thetas, phis = hp.pix2ang(2 ** nside, np.arange(nb_slots_available))

        while nb_slots_available - (thetas == thetas.max()).sum() >= nb_lenses:
            phis = phis[thetas < thetas.max()]
            thetas = thetas[thetas < thetas.max()]
            nb_slots_available = (thetas < theta).sum()

        if nb_slots_available > nb_lenses:
            c = int(nb_lenses-nb_slots_available)
            phis = phis[:c]
            thetas = thetas[:c]

        while thetas.max() < theta:
            thetas *= 1.01
        thetas /= 1.1

    return thetas, phis, nb_lenses == nb_slots_available


def fibonacci_sphere(samples, fov):

    samples = int(samples)
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

    sph = angles_distribution(60, 56)
    print(sph[0].shape, sph[1].shape, sph[2])
