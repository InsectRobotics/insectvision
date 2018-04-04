#!/usr/bin/env python
# Plots the evolution of the sensor's design, extracted for the corresponding log file.
#

from learn import SensorObjective
from compoundeye.sensor import CompassSensor

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
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import os
    import re

    algo_name = "sea"
    nb_lenses = 130
    fov = 150
    seed = 5
    thetas = True
    phis = True
    alphas = True
    ws = True
    label = "%s-%03d-%03d" % (algo_name, nb_lenses, fov)
    style = "plot"

    so = SensorObjective(nb_lenses, fov,
                         b_thetas=thetas, b_phis=phis, b_alphas=alphas, b_ws=ws)

    if thetas and phis and alphas and ws:
        p = re.compile(r"[0-9]{8}-%s-%04d.npz" % (label, seed))
    else:
        tag = ""
        tag += "t" if thetas else "f"
        tag += "t" if phis else "f"
        tag += "t" if alphas else "f"
        tag += "t" if ws else "f"
        p = re.compile(r"[0-9]{8}-%s-%s.npz" % (label, tag))
    xs = None
    file_champ = None
    for f in os.listdir(__datadir__):
        if p.match(f):
            data = np.load(__datadir__ + f)
            xs = so.correct_vector(data["log_x"])
            file_champ = f

    if not (thetas or phis or alphas or ws):
        xs = so.x_init[np.newaxis]

    print xs.shape
    if xs is not None:
        print file_champ

        for i, x in enumerate(xs):
            thetas, phis, alphas, w = SensorObjective.devectorise(x)
            # thetas, phis, alphas, w = SensorObjective.devectorise(so.x_init)

            s = CompassSensor(thetas=thetas, phis=phis, alphas=alphas)
            ax = s.visualise_structure(s, title="%s-struct-%03d" % (label, i), show=False)
            ax.text(1, -1, "Gen: % 3d" % (i+1))
            plt.savefig(__datadir__ + "../../../../%s-struct-%03d.png" % (label, i))
            plt.close()

            # if style == "plot":
            #     phi_tb1 = np.linspace(0., 2 * np.pi, 9, endpoint=True)  # TB1 preference angles
            #     cmap = cm.get_cmap("hsv")
            #     plt.figure("weights", figsize=(7, 5))
            #     ax = plt.subplot2grid((1, 6), (0, 0), rowspan=1, colspan=5, polar=True)
            #     ax.set_theta_zero_location("N")
            #     ax.set_theta_direction(-1)
            #     for phi, wi in zip(alphas, w):
            #         c = (phi % (2 * np.pi)) / (2 * np.pi)
            #         wii = np.append(wi, wi[0])
            #         plt.plot(phi_tb1, wii, color=cmap(c))
            #     plt.ylim([-.1, .1])
            #     plt.xticks((phi_tb1[:-1] + np.pi) % (2*np.pi) - np.pi)
            #     plt.subplot2grid((1, 6), (0, 5), rowspan=1, colspan=1)
            #     l = len(w)
            #     plt.imshow(np.linspace(0, 2*np.pi, l+1)[:, np.newaxis],
            #                vmin=0, vmax=2*np.pi, cmap="hsv", origin="lower")
            #     plt.tick_params(axis='y', labelleft='off', labelright='on')
            #     plt.yticks([0, l/4., l/2., 3.*l/4., l],
            #                [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$3\frac{\pi}{2}$", r"$2\pi$"])
            #     plt.xticks([])
            # elif style == "img":
            #     plt.figure("weights-img", figsize=(10, 5))
            #     plt.imshow(w.T, vmin=-1., vmax=1., cmap="coolwarm")
            #     plt.yticks([0, 7], ["1", "8"])
            #     ticks = np.linspace(0, w.shape[0]-1, 7)
            #     plt.xticks(ticks, ["%d" % tick for tick in (ticks+1)])
            # plt.show()
