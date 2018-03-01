import numpy as np
from sensor import CompassSensor, NB_EN, decode_sun
from learn import get_loss
from sky import get_seville_observer, SkyModel
from datetime import datetime, timedelta
import os


MSE = get_loss("ad3")
__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"
nb_lenses = [4, 12, 60, 112, 176, 368, 840]
fovs = [(14, 4), (30, 12), (60, 60), (90, 112), (120, 176), (150, 272), (180, 368)]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameters
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "cross"
    fov_deg = 60
    fov = np.deg2rad(fov_deg)
    generate_weights = True

    plt.figure("MSE", figsize=(15, 10))
    mse, mse_lon, mse_lat = [], [], []
    for nb_lens in nb_lenses:
        md = '' if mode == 'normal' else mode + '-'
        name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d" % (md, fov_deg, nb_lens, NB_EN, nb_months, delta.seconds)
        data = np.load(__datadir__ + "%s.npz" % name)
        dates = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['m']

        s = CompassSensor(nb_lenses=nb_lens, fov=np.deg2rad(fov_deg), mode=mode)

        # create and generate a sky instance
        observer.date = datetime.now()
        sky = SkyModel(observer=observer, nside=1)
        sky.generate()

        x, t = data['x'], data['t']
        t = np.array([decode_sun(t0) for t0 in t])
        if generate_weights:
            y = s.update_parameters(x=data['x'])  # , t=data['t'])
            s.save_weights()
        else:
            s.load_weights()
            y = s(data['x'], decode=True)
        mse.append(MSE(y, t))
        mse_lon.append(MSE(y, t, theta=False))
        mse_lat.append(MSE(y, t, phi=False))
        print nb_lens, "MSE:", mse[-1], "MSE-longitude:", mse_lon[-1],  "MSE-latitude:", mse_lat[-1]

    plt.subplot(121)
    plt.plot(nb_lenses, mse, label="MSE")
    plt.plot(nb_lenses, mse_lon, label="MSE-lon")
    plt.plot(nb_lenses, mse_lat, label="MSE-lat")
    plt.semilogx()
    plt.legend()
    plt.title("Number of Lenses (FOV = 60 degrees)")
    plt.xticks(nb_lenses, nb_lenses)
    plt.xlim([0, 850])
    plt.ylim([0, 15])
    plt.ylabel("MSE (degrees)")
    plt.xlabel("Number of Lenses")

    # plt.figure("MSE - Field of view", figsize=(15, 10))
    mse, mse_lon, mse_lat, fovs_ = [],  [], [], []
    for fov_deg_, nb_lens_ in fovs:
        md = '' if mode == 'normal' else mode + '-'
        name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d" % (md, fov_deg_, nb_lens_, NB_EN, nb_months, delta.seconds)
        data = np.load(__datadir__ + "%s.npz" % name)
        dates = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['m']

        s = CompassSensor(nb_lenses=nb_lens_, fov=np.deg2rad(fov_deg_), mode=mode)

        # create and generate a sky instance
        observer.date = datetime.now()
        sky = SkyModel(observer=observer, nside=1)
        sky.generate()

        x, t = data['x'], data['t']
        t = np.array([decode_sun(t0) for t0 in t])
        if generate_weights:
            y = s.update_parameters(x=data['x'])  # , t=data['t'])
            s.save_weights()
        else:
            s.load_weights()
            y = s(data['x'], decode=True)
        mse.append(MSE(y, t))
        mse_lon.append(MSE(y, t, theta=False))
        mse_lat.append(MSE(y, t, phi=False))
        fovs_.append(fov_deg_)
        print fov_deg, nb_lens, "MSE:", mse[-1], "MSE-longitude:", mse_lon[-1],  "MSE-latitude:", mse_lat[-1]

    plt.subplot(122)
    plt.plot(fovs_, mse, label="MSE")
    plt.plot(fovs_, mse_lon, label="MSE-lon")
    plt.plot(fovs_, mse_lat, label="MSE-lat")
    plt.legend()
    plt.title("Field of View (similar density)")
    plt.xticks(fovs_)
    plt.xlim([0, 180])
    plt.ylim([0, 15])
    plt.xlabel("Field of View (degrees)")

    plt.show()



