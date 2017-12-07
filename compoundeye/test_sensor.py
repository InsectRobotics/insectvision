import numpy as np
from sensor import CompassSensor, NB_EN, decode_sun, mse
from sky import get_seville_observer, ChromaticitySkyModel
from datetime import datetime, timedelta
import os


__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameters
    nb_lenses = 60
    fov = 60
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "monthly"
    # mode = "hourly"
    # mode = 6

    name = "seville-F%03d-I%03d-O%03d-M%02d-D%04d" % (fov, nb_lenses, NB_EN, nb_months, delta.seconds)
    data = np.load(__datadir__ + "%s.npz" % name)
    dates = np.load(__datadir__ + "M%02d-D%04d.npz" % (nb_months, delta.seconds))['m']

    s = CompassSensor(nb_lenses=nb_lenses, fov=np.deg2rad(fov))

    # create and generate a sky instance
    observer.date = datetime.now()
    sky = ChromaticitySkyModel(observer=observer, nside=1)
    sky.generate()

    x, t = data['x'], data['t']
    t = np.array([decode_sun(t0) for t0 in t])
    y = s.update_parameters(x=data['x'], t=data['t'])
    s.save_weights()

    print "MSE:", mse(y, t)
    print "MSE-longitude:", mse(y, t, theta=False)
    print "MSE-latitude:", mse(y, t, phi=False)

    S = 360
    step = 10

    if isinstance(mode, basestring) and mode == "monthly":
        plt.figure("Monthly", figsize=(15, 10))
        for j in xrange(12):
            i = np.array([True if date[0].month == j+1 and 8 <= date[0].hour <= 10 else False for ii, date in enumerate(dates)])
            ax = plt.subplot(3, 4, j + 1, polar=True)
            ax.set_theta_zero_location("N")
            plt.scatter(
                y[i][:S:step, 0],
                np.rad2deg(y[i][:S:step, 1]), label="prediction")
            plt.scatter(
                t[i][:S:step, 0],
                np.rad2deg(t[i][:S:step, 1]), label="target")
            for y0, t0 in zip(y[i][:S:step], t[i][:S:step]):
                plt.plot([y0[0], t0[0]], [np.rad2deg(y0[1]), np.rad2deg(t0[1])], 'k-')
            plt.xlim([-180, 180])
            plt.ylim([0, 90])
            # plt.legend()
    elif isinstance(mode, int):
        plt.figure(datetime(year=2017, month=mode, day=1).strftime("%B %Y"), figsize=(15, 10))
        i = np.array([True if date[0].month == mode else False for ii, date in enumerate(dates)])
        hours = i.sum() / S
        for j in xrange(min(hours, 12)):
            ax = plt.subplot(3, 4, j + 1, polar=True)
            ax.set_theta_zero_location("N")
            plt.scatter(
                y[i][j*S:(j+1)*S:step, 0],
                np.rad2deg(y[i][j*S:(j+1)*S:step, 1]), label="prediction")
            plt.scatter(
                t[i][j*S:(j+1)*S:step, 0],
                np.rad2deg(t[i][j*S:(j+1)*S:step, 1]), label="target")
            for y0, t0 in zip(y[i][j*S:(j+1)*S:step], t[i][j*S:(j+1)*S:step]):
                plt.plot([y0[0], t0[0]], [np.rad2deg(y0[1]), np.rad2deg(t0[1])], 'k-')
            plt.xlim([-180, 180])
            plt.ylim([0, 90])
            if j < 4:
                plt.title("%d:00" % dates[i][j*S][0].hour)
    elif isinstance(mode, basestring) and mode == "hourly":
        plt.figure("hourly-binned-mse", figsize=(15, 5))
        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        for h in xrange(24):
            i = np.array([True if date[0].month == 6 and date[0].hour == h else False for ii, date in enumerate(dates)])
            if i.sum() == 0:
                continue
            mse_x.append(h)
            mse_y.append(mse(y[i], t[i]))
            mse_y_lon.append(mse(y[i], t[i], theta=False))
            mse_y_lat.append(mse(y[i], t[i], phi=False))
        plt.subplot(131)
        plt.plot(mse_x, mse_y, label="MSE")
        plt.plot(mse_x, mse_y_lon, label="MSE-lon")
        plt.plot(mse_x, mse_y_lat, label="MSE-lat")
        plt.legend()
        plt.xlim([5, 19])
        plt.ylim([0, 22])
        plt.xlabel("Time (hour)")
        plt.ylabel("MSE")
        plt.title("June")

        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        for h in xrange(24):
            i = np.array([True if date[0].month == 9 and date[0].hour == h else False for ii, date in enumerate(dates)])
            if i.sum() == 0:
                continue
            mse_x.append(h)
            mse_y.append(mse(y[i], t[i]))
            mse_y_lon.append(mse(y[i], t[i], theta=False))
            mse_y_lat.append(mse(y[i], t[i], phi=False))
        plt.subplot(132)
        plt.plot(mse_x, mse_y, label="MSE")
        plt.plot(mse_x, mse_y_lon, label="MSE-lon")
        plt.plot(mse_x, mse_y_lat, label="MSE-lat")
        plt.legend()
        plt.xlim([5, 19])
        plt.ylim([0, 22])
        plt.xlabel("Time (hour)")
        plt.ylabel("MSE")
        plt.title("September")

        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        for h in xrange(24):
            i = np.array([True if date[0].month == 12 and date[0].hour == h else False for ii, date in enumerate(dates)])
            if i.sum() == 0:
                continue
            mse_x.append(h)
            mse_y.append(mse(y[i], t[i]))
            mse_y_lon.append(mse(y[i], t[i], theta=False))
            mse_y_lat.append(mse(y[i], t[i], phi=False))
        plt.subplot(133)
        plt.plot(mse_x, mse_y, label="MSE")
        plt.plot(mse_x, mse_y_lon, label="MSE-lon")
        plt.plot(mse_x, mse_y_lat, label="MSE-lat")
        plt.legend()
        plt.xlim([5, 19])
        plt.ylim([0, 22])
        plt.xlabel("Time (hour)")
        plt.ylabel("MSE")
        plt.title("December")
    plt.show()

    lon, lat = sky.lon, sky.lat
    print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    s.update_parameters(x=data['x'], t=data['t'])[0]
    s.facing_direction = 0
    lon, lat = s(sky)
    print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    # CompassSensor.visualise(s)

