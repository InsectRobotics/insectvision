import numpy as np
from sensor import CompassSensor, NB_EN, decode_sun
from compoundeye import CompoundEye
from learn import get_loss
from sky import get_seville_observer, SkyModel
from datetime import datetime, timedelta
from code.compass import encode_sph, decode_sph
import os


MSE = get_loss("ad3")
__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"
nb_lenses = [4, 12, 60, 112, 176, 368, 840]
fovs = [(14, 4), (29, 12), (60, 60), (90, 112), (120, 176), (150, 272), (180, 368)]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameters
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "normal"

    fov_deg = 120
    nb_lens = 176
    fov = np.deg2rad(fov_deg)
    mse, mse_lon, mse_lat = [], [], []

    md = '' if mode == 'normal' else mode + '-'
    name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d" % (md, fov_deg, nb_lens, NB_EN, nb_months, delta.seconds)
    data = np.load(__datadir__ + "%s.npz" % name)
    dates = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['m']

    s = CompassSensor(nb_lenses=nb_lens, fov=np.deg2rad(fov_deg), mode=mode)
    eye = CompoundEye(
        ommatidia=np.array([s.theta_global, s.phi_global]).T,
        central_microvili=(0, 0),
        noise_factor=.0,
        activate_dop_sensitivity=False
    )
    eye._channel_filters.pop("g")
    eye._channel_filters.pop("b")

    # create and generate a sky instance
    observer.date = datetime(2018, 6, 21, 8, 0, 0)
    s.sky.obs = observer
    s.sky.generate()
    eye.sky.obs = observer
    eye.sky.generate()

    x, t = data['x'], data['t']
    t = np.array([decode_sun(t0) for t0 in t])
    y = s.update_parameters(x=data['x'], t=data['t'])
    mse.append(MSE(y, t))
    mse_lon.append(MSE(y, t, theta=False))
    mse_lat.append(MSE(y, t, phi=False))
    print "MSE:", mse[-1], "MSE-longitude:", mse_lon[-1],  "MSE-latitude:", mse_lat[-1]

    sL = s.w_whitening.dot(s.w).T
    # plt.figure("whitening-weights", figsize=(10, 10))
    # plt.imshow(s.w_whitening, vmin=-20, vmax=20, cmap='coolwarm')
    # # plt.imshow(np.rot90(np.rot90(s.w_whitening)), vmin=-20, vmax=20, cmap='coolwarm')
    # # plt.colorbar()
    #
    # plt.figure("weights", figsize=(10, 10))
    # plt.imshow(s.w.T, vmin=-.06, vmax=.06, cmap='coolwarm')
    # # plt.imshow(s.w[::-1].T, vmin=-.06, vmax=.06, cmap='coolwarm')
    # # plt.colorbar()
    #
    # plt.figure("weights-total", figsize=(10, 10))
    # plt.imshow(s.w_whitening.dot(s.w).T, vmin=-1.5, vmax=1.5, cmap='coolwarm')
    # # plt.imshow(s.w_whitening.dot(s.w)[::-1].T, vmin=-1.5, vmax=1.5, cmap='coolwarm')
    # # plt.colorbar()
    #
    # s.visualise(s, sL=np.arange(nb_lens) / np.float32(nb_lens), sides=False, colormap='viridis')
    #
    # phi = []
    # for x in sL.T:
    #     lat, lon = decode_sph(x)
    #     phi.append(lon)
    # phi = (np.array(phi) % (2 * np.pi)) / (2 * np.pi)
    # s.visualise(s, sL=phi)
    #
    # sL = np.abs(sL)
    #
    # for sl in sL:
    #     sl = sl / sL.max()
    #     s.visualise(s, sL=sl, sides=False)
    #
    # sL = np.sqrt(np.square(sL).sum(axis=0))
    # sL = sL / sL.max()
    # s.visualise(s, sL=sL, sides=False)







    for i in xrange(13):
        s.refresh()
        aop = s.AOP  # % np.pi
        aop = (aop - aop.min()) / (aop.max() - aop.min())
        s.visualise(s, sL=aop, show_sun=True, colormap="hsv", title="AOP-%03d" % (i * 45))
        s.rotate(yaw=np.pi/6)
    s.rotate(yaw=-np.pi/2)
    # for i in xrange(3):
    #     s.refresh()
    #     aop_filter = (s._aop_filter - s._aop_filter.min()) / (s._aop_filter.max() - s._aop_filter.min())
    #     s.visualise(s, sL=aop_filter, show_sun=True, colormap="hsv", title="AOP-filter-%03d" % (i * 45))
    #     s.rotate(roll=np.pi/6)
    # s.rotate(roll=-np.pi/2)
    # for i in xrange(9):
    #     s.refresh()
    #     f = eye.L.flatten() / (1. - eye.DOP)
    #     d = (eye.AOP + eye.yaw - eye._aop_filter) % (2 * np.pi)
    #     E1 = np.array([
    #         np.cos(d),
    #         np.sin(d)
    #     ]) * np.sqrt(f)
    #     E2 = np.array([
    #         np.cos(d + np.pi / 2),
    #         np.sin(d + np.pi / 2)
    #     ]) * np.sqrt(f) * (1. - eye.DOP)
    #     E1[1] *= 0.
    #     E2[1] *= 0.
    #     E = np.array([np.sqrt(np.square(E1).sum(axis=0)), np.sqrt(np.square(E2).sum(axis=0))])
    #     # f = np.sqrt(np.square(E1).sum(axis=0)) * np.sqrt(np.square(E2).sum(axis=0))
    #     f = np.sqrt(np.square(E).sum(axis=0))
    #     # f = d
    #     print f.shape, f.min(), f.max()
    #     f = (f - f.min()) / (f.max() - f.min())
    #     s.visualise(s, sL=f, show_sun=True, sides=False,  # colormap="hsv",
    #                 title="cosine-%03d" % (i * 45))
    #     eye.rotate(yaw=np.pi/4)
    #     s.rotate(yaw=np.pi/4)
