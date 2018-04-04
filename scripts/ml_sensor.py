#!/usr/bin/env python
# Script that trains and visualises the progress and results of training
# of the sensor weights using back-propagation.
#

from compoundeye.sensor import CompassSensor, NB_EN, decode_sun
from sky import get_seville_observer
from learn.whitening import zca, pca
from learn import get_loss
from code.compass import decode_sph

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os

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
MSE = get_loss("ad3")
STD = get_loss("astd3")


if __name__ == "__main__":
    observer = get_seville_observer()

    # parameters
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "cross"
    fibonacci = False
    tilted = False
    fov_deg = 60
    nb_lens = 60
    fov = np.deg2rad(fov_deg)

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

    s = CompassSensor(nb_lenses=nb_lens, fov=np.deg2rad(fov_deg), mode=mode)
    observer.date = datetime(2018, 6, 21, 8, 0, 0)
    s.sky.obs = observer
    s.sky.generate()

    x, t = data['x'], data['t']
    w_pca = pca(x, m=.5)
    w_zca = zca(x, m=.5)

    plt.figure("whitening", figsize=(8, 4))

    ticks = [0, 4, 12, 24, 40, 60]
    plt.subplot(121)
    plt.imshow(w_pca, cmap="coolwarm", vmin=-20, vmax=20)
    plt.plot([3.5, 3.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [3.5, 3.5], 'k:')
    plt.plot([11.5, 11.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [11.5, 11.5], 'k:')
    plt.plot([23.5, 23.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [23.5, 23.5], 'k:')
    plt.plot([39.5, 39.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [39.5, 39.5], 'k:')
    plt.xlim([-.5, 59.5])
    plt.ylim([-.5, 59.5])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title("PCA")

    plt.subplot(122)
    plt.imshow(w_zca, cmap="coolwarm", vmin=-20, vmax=20)
    plt.plot([3.5, 3.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [3.5, 3.5], 'k:')
    plt.plot([11.5, 11.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [11.5, 11.5], 'k:')
    plt.plot([23.5, 23.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [23.5, 23.5], 'k:')
    plt.plot([39.5, 39.5], [-1, 60], 'k:')
    plt.plot([-1, 60], [39.5, 39.5], 'k:')
    plt.xlim([-.5, 59.5])
    plt.ylim([-.5, 59.5])
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title("ZCA")
    plt.show()

    try:
        print wname
        s.load_weights(name=wname)
        # y = s.update_parameters(x, t)
        # s.save_weights()
        y = s(x, decode=True)
    except IOError:
        y = s.update_parameters(x, t)
        s.save_weights(wname)

    plt.figure("weights", figsize=(5.5, 1))

    plt.subplot2grid((1, 11), (0, 0), colspan=8)
    # plt.figure("tl1_weights", figsize=(4, 1))
    plt.imshow(s.w_tl2.T, cmap='coolwarm', vmin=-1, vmax=1)
    plt.ylim([-.5, 15.5])
    plt.xlim([-.5, 59.5])
    plt.yticks([0, 15], ["1", "16"])
    plt.xticks([0, 29, 59], ["1", "30", "60"])
    plt.savefig("tl1_weights.png")

    plt.subplot2grid((1, 11), (0, 8), colspan=2)
    # plt.figure("cl1_weights", figsize=(1, 1))
    plt.imshow(s.w_cl1, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlim([-.5, 15.5])
    plt.ylim([-.5, 15.5])
    plt.xticks([0, 15], ["1", "16"])
    # plt.yticks([0, 15], ["1", "16"])
    plt.yticks([0, 15], ["", ""])

    plt.subplot2grid((1, 11), (0, 10))
    # plt.figure("tb1_weights", figsize=(.5, 1))
    plt.imshow(s.w_tb1, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xlim([-.5, 7.5])
    plt.ylim([-.5, 15.5])
    plt.xticks([0, 7], ["1", "8"])
    # plt.yticks([0, 15], ["1", "16"])
    plt.yticks([0, 15], ["", ""])
    plt.tight_layout()
    plt.show()

    # w = s.w_whitening
    #
    # for l, w0 in enumerate(s.w):
    #     w = w.dot(w0)
    #     for i, w1 in enumerate(w.T):
    #         sL = 1. / (1. + np.exp(-w1 / 10.))
    #         s.visualise(s, sL=sL, sides=False, colormap="coolwarm",
    #                     title="weights-%03d-L%02d-I%02d" % (nb_lens, l+1, i+1))
    #
    #     mag, phi = [], []
    #     for w1 in w:
    #         lat, lon = decode_sph(w1)
    #         phi.append(lon)
    #         mag.append(lat)
    #     phi = (np.array(phi) % (2 * np.pi)) / (2 * np.pi)
    #     mag = (np.array(mag) * np.log(len(mag) - 2)) / 5.
    #     mag_max = np.absolute(mag).max()
    #
    #     s.visualise(s, sL=np.clip(mag / mag_max, 0, 1), title="weigths-magnitude-%03d-L%02d" % (nb_lens, l+1), sides=False, colormap="Reds")
    #     s.visualise(s, sL=phi, title="weights-phase-%03d-L%02d" % (nb_lens, l+1), sides=False, colormap="hsv")
    #
    # w = s.w_whitening.dot(s.w_tl2).dot(s.w_cl1).dot(s.w_tb1) / 27.
    # # w = np.maximum(s.w_tl1, 0.).dot(np.maximum(s.w_cl1, 0.)).dot(s.w_tb1)
    # plt.figure("weights-total", figsize=(4, .5))
    # plt.imshow(w.T, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.xlim([-.5, 59.5])
    # plt.ylim([-.5, 7.5])
    # plt.xticks([0, 29, 59], ["1", "30", "60"])
    # plt.yticks([0, 7], ["1", "8"])
    #
    # plt.show()
    #
    # mag, phi = [], []
    # for w0 in w:
    #     lat, lon = decode_sph(w0)
    #     phi.append(lon)
    #     mag.append(lat)
    # phi = (np.array(phi) % (2 * np.pi)) / (2 * np.pi)
    # mag = (np.array(mag) * np.log(len(mag) - 2)) / 5.
    # mag_max = np.absolute(mag).max()
    # print mag_max
    #
    # for i, w0 in enumerate(w.T):
    #     sL = np.sqrt((w0 / mag_max + 1.) / 2)
    #     sL = 1. / (1. + np.exp(-w0 * 10.))
    #     s.visualise(s, sL=sL, sides=False, colormap="coolwarm",
    #                 title="weights-%02d-%03d" % (i+1, nb_lens))
    #
    # s.visualise(s, sL=np.clip(mag / mag_max, 0, 1), title="weigths-magnitude-%03d" % nb_lens, sides=False, colormap="Reds")
    # s.visualise(s, sL=phi, title="weights-phase-%03d" % nb_lens, sides=False, colormap="hsv")
    #
    # t_dec = np.array([decode_sun(t0) for t0 in t])
    # print ""
    # print "MSE:", MSE(y, t_dec)
    # print "MSE-longitude:", MSE(y, t_dec, theta=False)
    # print "MSE-latitude:", MSE(y, t_dec, phi=False)
    #
    # print "SE:", STD(y, t_dec) / np.sqrt(y.shape[0])
    # print "SE-longitude:", STD(y, t_dec, theta=False) / np.sqrt(y.shape[0])
    # print "SE-latitude:", STD(y, t_dec, phi=False) / np.sqrt(y.shape[0])

    # for i in xrange(13):
    #     s.refresh()
    #     sL = np.clip((s.L - s.L[s.L > 0].min()) / (s.L.max() - s.L[s.L > 0].min()), 0, 1)
    #     s.visualise(s, sL=sL, sides=False,
    #                 title="%s_%03d_%03d_%03d" % (mode, fov_deg, nb_lens, i * 30),)
    #     s.visualise(s, sL=(s.L - s.m).dot(s.w_whitening) + .5, colormap="coolwarm", sides=False,
    #                 title="%s_%03d_%03d_%03d_zca" % (mode, fov_deg, nb_lens, i * 30))
    #     s.rotate(yaw=np.pi/6)
