import numpy as np
import matplotlib.pyplot as plt
from sensor import CompassSensor, NB_EN, decode_sun
from sky import get_seville_observer
from datetime import datetime, timedelta
from learn.whitening import zca, pca
from learn import get_loss
import os

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"
__sensdir__ = __dir__ + "../data/sensor/"
MSE = get_loss("ad3")


if __name__ == "__main__":
    observer = get_seville_observer()

    # parameters
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "cross"
    tilted = True
    fov_deg = 60
    nb_lens = 60
    fov = np.deg2rad(fov_deg)

    name_suf = "-tilt" if tilted else ""
    date_suf = "-tilted" if tilted else ""

    md = '' if mode == 'normal' else mode + '-'
    name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d%s" % (md, fov_deg, nb_lens, NB_EN, nb_months, delta.seconds, name_suf)
    data = np.load(__datadir__ + "%s.npz" % name)
    dates = np.load(__datadir__ + "%sM%02d-D%04d%s.npz" % (md, nb_months, delta.seconds, date_suf))['m']

    s = CompassSensor(nb_lenses=nb_lens, fov=np.deg2rad(fov_deg), mode=mode)
    observer.date = datetime(2018, 6, 21, 8, 0, 0)
    s.sky.obs = observer
    s.sky.generate()

    x, t = data['x'], data['t']
    # t = np.array([decode_sun(t0) for t0 in t])
    w_pca = pca(x)
    w_zca = zca(x)

    # plt.figure("whitening", figsize=(20, 10))
    #
    # plt.subplot(121)
    # plt.imshow(w_pca, cmap="coolwarm", vmin=-20, vmax=20)
    # plt.plot([3.5, 3.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [3.5, 3.5], 'k:')
    # plt.plot([11.5, 11.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [11.5, 11.5], 'k:')
    # plt.plot([23.5, 23.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [23.5, 23.5], 'k:')
    # plt.plot([39.5, 39.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [39.5, 39.5], 'k:')
    # plt.xlim([-.5, 59.5])
    # plt.ylim([-.5, 59.5])
    # plt.title("PCA")
    #
    # plt.subplot(122)
    # plt.imshow(w_zca, cmap="coolwarm", vmin=-20, vmax=20)
    # plt.plot([3.5, 3.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [3.5, 3.5], 'k:')
    # plt.plot([11.5, 11.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [11.5, 11.5], 'k:')
    # plt.plot([23.5, 23.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [23.5, 23.5], 'k:')
    # plt.plot([39.5, 39.5], [-1, 60], 'k:')
    # plt.plot([-1, 60], [39.5, 39.5], 'k:')
    # plt.xlim([-.5, 59.5])
    # plt.ylim([-.5, 59.5])
    # plt.title("ZCA")
    # plt.show()

    try:
        s.load_weights()
        y = s.update_parameters(x, t)
        s.save_weights()
        y = s(x, decode=True)
    except IOError:
        y = s.update_parameters(x, t)
        s.save_weights()

    plt.figure("weights", figsize=(30, 10))
    for i, coef in enumerate(s.w):
        plt.subplot(131 + i)
        plt.imshow(coef, cmap='coolwarm',
                   vmin=-1, vmax=1
                   )
        # plt.colorbar()
    plt.show()

    t_dec = np.array([decode_sun(t0) for t0 in t])
    mse = MSE(y, t_dec)
    mse_lon = MSE(y, t_dec, theta=False)
    mse_lat = MSE(y, t_dec, phi=False)

    print "MSE:", mse, "MSE-longitude:", mse_lon, "MSE-latitude:", mse_lat
