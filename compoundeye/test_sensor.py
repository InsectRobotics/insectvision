import numpy as np
from sensor import CompassSensor, NB_EN, decode_sun
from learn import get_loss
from sky import get_seville_observer, SkyModel
from datetime import datetime, timedelta
from learn.whitening import pca, zca
import os


__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"
mse = get_loss("ad3")
std = get_loss("astd3")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameters
    md = 'cross'
    fibonacci = False
    tilting = True
    nb_lenses = 12
    fov = 60
    kernel = zca
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)
    mode = "monthly"
    # mode = "monthly-scatter"
    # mode = "hourly"
    # mode = 6

    s = CompassSensor(nb_lenses=nb_lenses, fov=np.deg2rad(fov), kernel=kernel, mode=md, fibonacci=fibonacci)

    md = '' if md == 'normal' else md + '-'
    name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d" % (md, fov, nb_lenses, NB_EN, nb_months, delta.seconds)
    if tilting:
        name += "-tilt"
        dates = np.load(__datadir__ + "%sM%02d-D%04d-tilted.npz" % (md, nb_months, delta.seconds))['m']
    else:
        dates = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['m']
    if fibonacci:
        name += "-fibonacci"
    data = np.load(__datadir__ + "%s.npz" % name)

    # create and generate a sky instance
    observer.date = datetime.now()
    s.sky.obs = observer
    s.sky.generate()

    x, t = data['x'], data['t']
    # x[np.isnan(x)] = 0.
    t = np.array([decode_sun(t0) for t0 in t])
    y = s.update_parameters(x=x, t=data['t'])
    s.save_weights()
    # s.load_weights()
    # y = s(x, decode=True)

    print "MSE:", mse(y, t)
    print "MSE-longitude:", mse(y, t, theta=False)
    print "MSE-latitude:", mse(y, t, phi=False)

    print "SE:", std(y, t) / np.sqrt(y.shape[0])
    print "SE-longitude:", std(y, t, theta=False) / np.sqrt(y.shape[0])
    print "SE-latitude:", std(y, t, phi=False) / np.sqrt(y.shape[0])

    S = 360
    step = 10

    if isinstance(mode, basestring) and mode == "monthly-scatter":
        import calendar

        plt.figure("Monthly-12pm", figsize=(15, 10))
        for j in xrange(12):
            i = np.array([True if date[0].month == j+1 and 11 <= date[0].hour <= 13 else False for ii, date in enumerate(dates)])
            if i.sum() == 0:
                i = np.array([True if date[0].month == 11-j and 11 <= date[0].hour <= 13 else False for ii, date in enumerate(dates)])
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
            plt.text(0, 0, calendar.month_name[j+1], fontsize=16,
                     bbox=dict(facecolor='white', alpha=.5),
                     horizontalalignment='center', verticalalignment='center')
            plt.xlim([-180, 180])
            plt.ylim([0, 90])
            # plt.legend()
    elif isinstance(mode, basestring) and mode == "monthly":
        plt.figure("Monthly", figsize=(15, 10))
        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        std_x, std_y, std_y_lon, std_y_lat = [], [], [], []
        ylim = 20

        for j in xrange(12):
            i = np.array([True if date[0].month == j+1 else False for ii, date in enumerate(dates)])
            n = i.sum()
            if n == 0:
                continue
            mse_x.append(j+1)
            std_x.append(j+1)
            mse_y.append(mse(y[i], t[i]))
            mse_y_lon.append(mse(y[i], t[i], theta=False))
            mse_y_lat.append(mse(y[i], t[i], phi=False))
            std_y.append(std(y[i], t[i]) / np.sqrt(n))
            std_y_lon.append(std(y[i], t[i], theta=False) / np.sqrt(n))
            std_y_lat.append(std(y[i], t[i], phi=False) / np.sqrt(n))

        for j in xrange(12):
            if j+1 not in mse_x:
                mse_y.append(mse_y[mse_x.index(11 - j)])
                mse_y_lon.append(mse_y_lon[mse_x.index(11 - j)])
                mse_y_lat.append(mse_y_lat[mse_x.index(11 - j)])
                mse_x.append(j+1)
            if j+1 not in std_x:
                std_y.append(std_y[std_x.index(11 - j)])
                std_y_lon.append(std_y_lon[std_x.index(11 - j)])
                std_y_lat.append(std_y_lat[std_x.index(11 - j)])
                std_x.append(j+1)

        i = np.argsort(mse_x)
        mse_x = np.array(mse_x)[i]
        mse_y = np.array(mse_y)[i]
        mse_y_lon = np.array(mse_y_lon)[i]
        mse_y_lat = np.array(mse_y_lat)[i]
        i = np.argsort(std_x)
        std_x = np.array(std_x)[i]
        std_y = np.array(std_y)[i]
        std_y_lon = np.array(std_y_lon)[i]
        std_y_lat = np.array(std_y_lat)[i]
        scale = 3.

        # plt.subplot(131)
        plt.fill_between(std_x, mse_y - scale * std_y, mse_y + scale * std_y,
                         facecolor="#3F5D7D", alpha=.5)
        plt.plot(mse_x, mse_y, label="MSE")
        plt.fill_between(std_x, mse_y_lon - scale * std_y_lon, mse_y_lon + scale * std_y_lon,
                         facecolor="#7D3F5D", alpha=.5)
        plt.plot(mse_x, mse_y_lon, label="MSE-lon")
        plt.fill_between(std_x, mse_y_lat - scale * std_y_lat, mse_y_lat + scale * std_y_lat,
                         facecolor="#3F7D5D", alpha=.5)
        plt.plot(mse_x, mse_y_lat, label="MSE-lat")
        plt.legend()
        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 12))
        plt.ylim([0, ylim])
        plt.xlabel("Time (month)")
        plt.ylabel("MSE (degrees)")
        plt.title("Monthly")
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
            k = ((j + 2) / 6) % 2
            plt.text(k * np.pi, k * 60, "%d:00" % dates[i][j*S][0].hour,
                     bbox=dict(facecolor='white', alpha=.5), fontsize=16,
                     horizontalalignment='center', verticalalignment='center')
            # plt.title("%d:00" % dates[i][j*S][0].hour)
    elif isinstance(mode, basestring) and mode == "hourly":
        plt.figure("hourly-binned-mse", figsize=(15, 5))
        ylim = 50

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
        plt.ylim([0, ylim])
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
        plt.ylim([0, ylim])
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
        plt.ylim([0, ylim])
        plt.xlabel("Time (hour)")
        plt.ylabel("MSE")
        plt.title("December")
    plt.show()

    s.rotate(-s.yaw, -s.pitch, -s.roll)
    lon, lat = (s.sky.lon + np.pi) % (2 * np.pi) - np.pi, (s.sky.lat + np.pi) % (2 * np.pi) - np.pi
    print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    lon, lat = s(s.sky, decode=True).flatten()
    lon, lat = (lon + np.pi) % (2 * np.pi) - np.pi, (lat + np.pi) % (2 * np.pi) - np.pi
    print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    # CompassSensor.visualise(s)

