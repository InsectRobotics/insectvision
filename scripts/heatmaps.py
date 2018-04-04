#!/usr/bin/env python
# Plots heat-maps of where the components of the sensor converge for different set-ups.
#

from learn import SensorObjective

import numpy as np

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2018, The Invisible Cues Project"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


if __name__ == "__main__":
    from learn.optimisation import __datadir__
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sphere.transform import sph2vec
    import os
    import re

    threshold = np.pi/9
    nb_lensess = [60, 60, 130, 130]
    fovs = [60, 60, 150, 150]
    tilts = [False, True, False, True]
    b_thetas = True
    b_phis = True
    b_alphas = True
    b_ws = True
    algo_name = "pso"

    fig = plt.figure("heatmap-%s" % algo_name)
    ax = Axes3D(fig)
    for i, nb_lenses, fov, tilt in zip(range(4), nb_lensess, fovs, tilts):
        label = "%s-%03d-%03d%s" % (algo_name, nb_lenses, fov, "-tilt" if tilt else "")

        so = SensorObjective(nb_lenses, fov,
                             b_thetas=b_thetas, b_phis=b_phis, b_alphas=b_alphas, b_ws=b_ws)

        if b_thetas and b_phis and b_alphas and b_ws:
            p = re.compile(r"[0-9]{8}-%s-[0-9]{4}.npz" % label)
        else:
            tag = ""
            tag += "t" if b_thetas else "f"
            tag += "t" if b_phis else "f"
            tag += "t" if b_alphas else "f"
            tag += "t" if b_ws else "f"
            p = re.compile(r"[0-9]{8}-%s-%s.npz" % (label, tag))

        thetas = np.empty((0, nb_lenses), dtype=np.float32)
        phis = np.empty((0, nb_lenses), dtype=np.float32)
        for f in os.listdir(__datadir__):
            if p.match(f):
                print f
                data = np.load(__datadir__ + f)
                th, ph, _, _ = so.devectorise(so.correct_vector(data["x"]))
                thetas = np.vstack([thetas, th])
                phis = np.vstack([phis, ph])

        theta = np.linspace(0, np.pi/2, 100)
        phi = np.linspace(0, 2 * np.pi, 360)
        th, ph = np.meshgrid(theta, phi)
        thetas = thetas.flatten()
        phis = phis.flatten()
        z = np.zeros_like(th)

        mes = np.array([th, ph])
        mes = sph2vec(mes, zenith=True)
        pos = np.array([thetas, phis])
        pos = sph2vec(pos, zenith=True)
        d = np.absolute(np.arccos(np.einsum("ijk,il->jkl", mes, pos)))
        z += np.mean(d < threshold, axis=-1)
        # z += np.mean(angle_between(ph[..., np.newaxis], phis, sign=False) < threshold, axis=-1)
        # z = np.absolute((z + np.pi) % (2 * np.pi) - np.pi)

        with plt.rc_context({'ytick.color': 'white'}):
            ax = plt.subplot2grid((9, 9), ((i // 2) * 5, (i % 2) * 4), rowspan=4, colspan=4, projection="polar")
            # ax = plt.subplot(221 + i, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            plt.pcolormesh(ph, np.rad2deg(th), z, cmap="hot", vmin=0, vmax=1)
            plt.yticks([0, 30, 60, 90], [r"$0$", r"$\frac{\pi}{6}$", r"$\frac{\pi}{3}$", r"$\frac{\pi}{2}$"])
            plt.xticks((np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi) % (2 * np.pi) - np.pi)
            plt.ylim([0, 90])
            plt.text(np.deg2rad(-50), 140, ["A.", "B.", "C.", "D."][i], fontsize=16)
            ax.grid(alpha=0.2)
    ax = plt.subplot2grid((4, 36), (1, 34), rowspan=2)
    plt.colorbar(cax=ax)
    plt.show()
