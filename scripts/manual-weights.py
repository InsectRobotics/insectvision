#!/usr/bin/env python
# Test thew manually set weights.
# Use of different noise levels, designs, etc.
#

from compoundeye.sensor import CompassSensor, NB_EN, decode_sun
from sky import get_seville_observer
from learn.whitening import zca, pca
from learn import get_loss

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2018, The Invisible Cues Project"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"
__sensdir__ = __dir__ + "../data/sensor/"
mse = get_loss("ad3")
std = get_loss("astd3")


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


if __name__ == "__main__":
    # parameters
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "cross"
    noise = 0.
    fibonacci = True
    tilted = False
    fov_deg = 60
    nb_lens = 60
    fov = np.deg2rad(fov_deg)
    override = True
    show_weights = True
    show_outputs = False

    name_suf = "-tilt" if tilted else ""
    date_suf = "-tilted" if tilted else ""

    md = '' if mode == 'normal' else mode + '-'
    name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d%s" % (md, fov_deg, nb_lens, NB_EN, nb_months, delta.seconds, name_suf)
    wname = "%ssensor-L%03d-V%03d" % (md, nb_lens, np.rad2deg(fov))
    if fibonacci or nb_lens >= 100:
        wname += "-fibonacci"
    if tilted:
        wname += "-tilt"
    data = np.load(__datadir__ + "%s.npz" % name)
    dates = np.load(__datadir__ + "%sM%02d-D%04d%s.npz" % (md, nb_months, delta.seconds, date_suf))['m']

    s = CompassSensor(nb_lenses=nb_lens, fov=np.deg2rad(fov_deg), mode=mode)  # , kernel=zca)
    observer.date = datetime(2018, 6, 21, 8, 0, 0)
    s.sky.obs = observer
    s.sky.generate()

    x, t = data['x'], data['t']
    n = np.absolute(np.random.randn(*x.shape)) < noise
    # xrg = (x - .5).max() - (x - .5).min()
    # nrg = n.max() - n.min()
    nrg = n.sum() / np.float32(n.size)
    print "Noise level: %.4f (%.2f %%)" % (noise, 100. * nrg)
    # x_noise = np.clip(x + n, 0, 1)
    x_noise = x.copy()
    x_noise[n] = .5

    print wname
    if override:
        y = s.update_parameters(x_noise)  # , t)
        # y = s(x, decode=True)
        s.save_weights(name=wname)
    else:
        try:
            s.load_weights(name=wname)
            # y = s.update_parameters(x, t)
            # s.save_weights()
            y = s(x_noise, decode=True)
        except IOError:
            y = s.update_parameters(x_noise, t)
            s.save_weights(name=wname)

    w_zca = s.w_whitening
    w_tl2, w_cl1, w_tb1 = s.w
    b_tl2, b_cl1, b_tb1 = s.b

    h0 = s._pprop(x_noise)
    print "zca", h0.min(), h0.max()

    h1 = np.maximum(h0.dot(w_tl2) + b_tl2, 0)
    print "tl1", h1.min(), h1.max()

    h2 = h1.dot(w_cl1) + b_cl1
    # h2 = np.maximum(h1.dot(w_cl1) + b_cl1, 0)
    print "cl1", np.rad2deg(h2.min()), np.rad2deg(h2.max())

    h3 = h2.dot(w_tb1) + b_tb1
    print "tb1", np.rad2deg(h3.min()), np.rad2deg(h3.max())

    print "_y_", np.rad2deg(t.min()), np.rad2deg(t.max())

    t = np.array([decode_sun(t0) for t0 in t])

    print ""
    print "MSE:", mse(y, t)
    print "MSE-longitude:", mse(y, t, theta=False)
    print "MSE-latitude:", mse(y, t, phi=False)

    print "SE:", std(y, t) / np.sqrt(y.shape[0])
    print "SE-longitude:", std(y, t, theta=False) / np.sqrt(y.shape[0])
    print "SE-latitude:", std(y, t, phi=False) / np.sqrt(y.shape[0])

    if show_weights:
        plt.figure("TL2 weights", figsize=(5.5, 1))
        plt.imshow(w_tl2.T, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks([0, 60], ["1", "60"])
        plt.yticks([0, 15], ["1", "16"])

        plt.figure("CL1 weights", figsize=(1, 1))
        plt.imshow(w_cl1, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks([0, 15], ["1", "16"])
        plt.yticks([0, 15], ["1", "16"])

        plt.figure("TB1 weights", figsize=(.5, 1))
        plt.imshow(w_tb1, cmap='coolwarm', vmin=-1, vmax=1)
        plt.xticks([0, 7], ["1", "8"])
        plt.yticks([0, 15], ["1", "16"])

        plt.show()

    # w = w_tl2.dot(w_cl1).dot(w_tb1)
    # ww = np.array([decode_sun(w0) for w0 in w])
    # s.visualise(s, sL=(ww[:, 0] % (2 * np.pi)) / (2 * np.pi), colormap="hsv", sides=False, scale=None)

    if show_outputs:
        cols = [13, 13, 12, 11, 10, 9, 8]
        skip = 0
        rows = len(cols)

        print "x ", x.min(), x.max()
        plt.figure("Input", figsize=(15, 30))
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(x[(i + skip)*360:(i + 1 + skip)*360:30, ::4].T,
                           cmap='coolwarm', vmin=.4, vmax=.6)
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]

        skip = 0
        plt.figure("ZCA out", figsize=(15, 30))
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(h0[(i + skip)*360:(i + 1 + skip)*360:30, ::4].T,
                           cmap='coolwarm', vmin=-.2, vmax=.2)
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]

        skip = 0
        plt.figure("TL1 out", figsize=(15, 30))
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(h1[(i + skip)*360:(i + 1 + skip)*360:30].T,
                           cmap='coolwarm', vmin=-.3, vmax=.3)
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]

        skip = 0
        plt.figure("CL1 out", figsize=(15, 30))
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(h2[(i + skip)*360:(i + 1 + skip)*360:30].T,
                           cmap='coolwarm', vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]

        plt.figure("TB1 out - azimuth", figsize=(15, 30))
        skip = 0
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(h3[(i + skip)*360:(i + 1 + skip)*360:30].T,
                           cmap='coolwarm', vmin=-1, vmax=1)
                y0 = np.rad2deg(decode_sun(h3[(i + skip)*360])[0])
                t0 = np.rad2deg(t[(i + skip)*360, 0])
                plt.title("%.2f" % ((y0 - t0 + 180.) % 360. - 180.))
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]

        plt.figure("real-output - elevation", figsize=(15, 30))
        skip = 0
        for j in xrange(rows):
            for i in xrange(cols[j]):
                plt.subplot(rows, cols[0], (cols[0] * j + i) + 1)
                plt.imshow(data["t"][(i + skip)*360:(i + 1 + skip)*360:30].T,
                           cmap='coolwarm', vmin=-1, vmax=1)
                y0 = np.rad2deg(decode_sun(h3[(i + skip)*360])[1])
                t0 = np.rad2deg(t[(i + skip)*360, 1])
                plt.title("%.2f" % ((y0 - t0 + 180.) % 360. - 180.))
                plt.xticks([])
                plt.yticks([])
            skip += cols[j]
        plt.show()
