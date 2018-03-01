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
    noise = .0
    fibonacci = False
    wtilting = False
    tilting = False
    nb_lenses = 60
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
    # mode = "tilts"
    # mode = "lon"
    # mode = "lat"
    # mode = 6

    s = CompassSensor(nb_lenses=nb_lenses, fov=np.deg2rad(fov), kernel=kernel, mode=md, fibonacci=fibonacci)

    md = '' if md == 'normal' else md + '-'
    name = "%sseville-F%03d-I%03d-O%03d-M%02d-D%04d" % (md, fov, nb_lenses, NB_EN, nb_months, delta.seconds)
    if tilting:
        name += "-tilt"
        dates = np.load(__datadir__ + "%sM%02d-D%04d-tilted.npz" % (md, nb_months, delta.seconds))['m']
        r = np.load(__datadir__ + "%sM%02d-D%04d-tilted.npz" % (md, nb_months, delta.seconds))['r']
    else:
        dates = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['m']
        r = np.load(__datadir__ + "%sM%02d-D%04d.npz" % (md, nb_months, delta.seconds))['r']
    if fibonacci:
        name += "-fibonacci"

    # r_deg = np.rad2deg(r)
    # for yaw, pitch, roll, lon, lat in r_deg:
    #     print "% 3d % 3d % 3d % 3d % 3d" % (yaw, pitch, roll, lon, lat)

    data = np.load(__datadir__ + "%s.npz" % name)

    wname = "%ssensor-L%03d-V%03d" % (md, nb_lenses, np.rad2deg(np.deg2rad(fov)))
    if fibonacci or nb_lenses >= 100:
        wname += "-fibonacci"
    if wtilting:
        wname += "-tilt"
    print wname

    # create and generate a sky instance
    observer.date = datetime.now()
    s.sky.obs = observer
    s.sky.generate()

    x, t = data['x'], data['t']
    # x[np.isnan(x)] = 0.
    t = np.array([decode_sun(t0) for t0 in t])
    # y = s.update_parameters(x=x, t=data['t'])
    # s.save_weights(name=wname)
    s.load_weights(name=wname)
    n = np.random.randn(*x.shape) * noise
    print "Noise level:", n.min(), n.max()
    y = s(x + n, decode=True)

    print "MSE:", mse(y, t)
    print "MSE-longitude:", mse(y, t, theta=False)
    print "MSE-latitude:", mse(y, t, phi=False)

    print "SE:", std(y, t) / np.sqrt(y.shape[0])
    print "SE-longitude:", std(y, t, theta=False) / np.sqrt(y.shape[0])
    print "SE-latitude:", std(y, t, phi=False) / np.sqrt(y.shape[0])

    if tilting:
        tilts = 25
        skip = 0
    else:
        tilts = 1
        skip = 0
    S = 360
    step = 10

    if isinstance(mode, basestring) and mode == "monthly-scatter":
        import calendar

        if tilting:
            pi = 0.
            ro = np.pi/6
        else:
            pi = ro = 0.

        def isvalid(date, j, pitch, roll):
            return date.month == j+1 and 10 <= date.hour <= 11 and np.isclose(pitch, pi) and np.isclose(roll, ro)

        plt.figure("Monthly-12pm-%03d-%03d" % (np.rad2deg(pi), np.rad2deg(ro)), figsize=(15, 10))
        for j in xrange(12):
            i = np.array([isvalid(date[0], j, pitch, roll)
                          for ii, (date, pitch, roll) in enumerate(zip(dates, r[:, 1], r[:, 2]))])
            if i.sum() == 0:
                i = np.array([isvalid(date[0], 11-j, pitch, roll)
                              for ii, (date, pitch, roll) in enumerate(zip(dates, r[:, 1], r[:, 2]))])
            ax = plt.subplot(3, 4, j + 1, polar=True)
            ax.set_theta_zero_location("N")
            plt.scatter(
                y[i][skip*S:(skip+1)*S:step, 0],
                np.rad2deg(y[i][skip*S:(skip+1)*S:step, 1]), label="prediction")
            plt.scatter(
                t[i][skip*S:(skip+1)*S:step, 0],
                np.rad2deg(t[i][skip*S:(skip+1)*S:step, 1]), label="target")
            for y0, t0 in zip(y[i][skip*S:(skip+1)*S:step], t[i][skip*S:(skip+1)*S:step]):
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
        if tilting:
            ylim = 90
        else:
            ylim = 11

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
        scale = 1.

        # plt.subplot(131)
        plt.fill_between(std_x, mse_y - scale * std_y, mse_y + scale * std_y,
                         facecolor="C0", alpha=.5)
        plt.plot(mse_x, mse_y, color="C0", label="MSE")
        plt.fill_between(std_x, mse_y_lon - scale * std_y_lon, mse_y_lon + scale * std_y_lon,
                         facecolor="C1", alpha=.5)
        plt.plot(mse_x, mse_y_lon, color="C1", label="MSE-lon")
        plt.fill_between(std_x, mse_y_lat - scale * std_y_lat, mse_y_lat + scale * std_y_lat,
                         facecolor="C2", alpha=.5)
        plt.plot(mse_x, mse_y_lat, color="C2", label="MSE-lat")
        plt.legend()
        plt.xlim([1, 12])
        plt.xticks(np.arange(1, 12))
        plt.ylim([0, ylim])
        plt.xlabel("Time (month)")
        plt.ylabel("MSE (degrees)")
        plt.title("Monthly")
    elif isinstance(mode, int):

        if tilting:
            pi = 0.
            ro = np.pi/6
        else:
            pi = ro = 0.

        def isvalid(date, pitch, roll):
            return date.month == mode and np.isclose(pitch, pi) and np.isclose(roll, ro)


        i = np.array([isvalid(date[0], pitch, roll)
                      for ii, (date, pitch, roll) in enumerate(zip(dates, r[:, 1], r[:, 2]))])
        plt.figure(datetime(year=2017, month=mode, day=1).strftime("%B %Y"), figsize=(15, 10))
        hours = i.sum() / S
        if tilting:
            hours /= 9
        for j in xrange(min(hours, 12)):
            ax = plt.subplot(3, 4, j + 1, polar=True)
            ax.set_theta_zero_location("N")
            yj = y[i].reshape((-1, tilts, S, 2))[j, skip, ::step]
            tj = t[i].reshape((-1, tilts, S, 2))[j, skip, ::step]
            plt.scatter(yj[:, 0], np.rad2deg(yj[:, 1]), label="prediction")
            plt.scatter(tj[:, 0], np.rad2deg(tj[:, 1]), label="target")
            for y0, t0 in zip(yj, tj):
                plt.plot([y0[0], t0[0]], [np.rad2deg(y0[1]), np.rad2deg(t0[1])], 'k-')
            plt.xlim([-180, 180])
            plt.ylim([0, 90])
            k = ((j + 2) / 6) % 2
            plt.text(k * np.pi, k * 60, "%d:00" % dates[i][j*tilts*S][0].hour,
                     bbox=dict(facecolor='white', alpha=.5), fontsize=16,
                     horizontalalignment='center', verticalalignment='center')
            # plt.title("%d:00" % dates[i][j*S][0].hour)
    elif isinstance(mode, basestring) and mode == "hourly":
        plt.figure("hourly-binned-mse", figsize=(15, 5))
        if tilting:
            ylim = 60
        else:
            ylim = 15

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
    elif isinstance(mode, basestring) and mode == "tilts" and tilting:
        plt.figure("tilting-mse", figsize=(15, 5))
        ylim = 180
        scale = 1.

        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        std_x, std_y, std_y_lon, std_y_lat = [], [], [], []
        angles = np.array([-np.pi/3, -np.pi/6, 0., np.pi/6, np.pi/3])
        for i, pitch in enumerate(angles):
            for k, roll in enumerate(angles):
                angle = np.sqrt(np.square(pitch) + np.square(roll))
                if np.rad2deg(angle) in mse_x:
                    continue

                print np.rad2deg(angle)

                j = np.array([np.isclose(np.sqrt(np.square(ph) + np.square(rl)), angle) for ph, rl in r[:, 1:3]])
                yj = y[j]
                tj = t[j]

                mse_x.append(np.rad2deg(angle))
                std_x.append(np.rad2deg(angle))
                mse_y.append(mse(yj, tj))
                mse_y_lon.append(mse(yj, tj, theta=False))
                mse_y_lat.append(mse(yj, tj, phi=False))
                std_y.append(std(yj, tj) / np.sqrt(yj.shape[0]))
                std_y_lon.append(std(yj, tj, theta=False) / np.sqrt(yj.shape[0]))
                std_y_lat.append(std(yj, tj, phi=False) / np.sqrt(yj.shape[0]))

        mse_x = np.array(mse_x)
        mse_y = np.array(mse_y)
        mse_y_lon = np.array(mse_y_lon)
        mse_y_lat = np.array(mse_y_lat)
        std_x = np.array(std_x)
        std_y = np.array(std_y)
        std_y_lon = np.array(std_y_lon)
        std_y_lat = np.array(std_y_lat)

        plt.subplot(111)
        # plt.fill_between(std_x, mse_y - scale * std_y, mse_y + scale * std_y,
        #                  facecolor="C0", alpha=.5)
        # plt.plot(mse_x, mse_y, color="C0", label="MSE")
        plt.fill_between(std_x, mse_y_lon - scale * std_y_lon, mse_y_lon + scale * std_y_lon,
                         facecolor="C1", alpha=.5)
        plt.plot(mse_x, mse_y_lon, color="C1", label="MSE-lon")
        # plt.fill_between(std_x, mse_y_lat - scale * std_y_lat, mse_y_lat + scale * std_y_lat,
        #                  facecolor="C2", alpha=.5)
        # plt.plot(mse_x, mse_y_lat, color="C2", label="MSE-lat")
        # plt.legend()
        plt.xlim([mse_x.min(), mse_x.max()])
        plt.ylim([0, ylim])
        plt.xticks(mse_x)
        plt.xlabel("Tilting (degrees)")
        plt.ylabel("MSE (degrees)")
    elif isinstance(mode, basestring) and mode == "lon":
        plt.figure("sun-lon-mse", figsize=(15, 5))
        ylim = 180
        scale = 1.

        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        std_x, std_y, std_y_lon, std_y_lat = [], [], [], []
        if tilting:
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        else:
            angles = np.linspace(0, np.pi, 180, endpoint=False)
        for angle in angles:
            if np.rad2deg(angle) in mse_x:
                continue

            j = np.array([phi - np.pi/360. < angle <= phi + np.pi/360. for phi in r[:, 3]])
            if j.sum() == 0:
                continue
            yj = y[j]
            tj = t[j]

            mse_x.append(np.rad2deg(angle))
            std_x.append(np.rad2deg(angle))
            mse_y.append(mse(yj, tj))
            mse_y_lon.append(mse(yj, tj, theta=False))
            mse_y_lat.append(mse(yj, tj, phi=False))
            std_y.append(std(yj, tj) / np.sqrt(yj.shape[0]))
            std_y_lon.append(std(yj, tj, theta=False) / np.sqrt(yj.shape[0]))
            std_y_lat.append(std(yj, tj, phi=False) / np.sqrt(yj.shape[0]))

        mse_x = np.array(mse_x)
        mse_y = np.array(mse_y)
        mse_y_lon = np.array(mse_y_lon)
        mse_y_lat = np.array(mse_y_lat)
        std_x = np.array(std_x)
        std_y = np.array(std_y)
        std_y_lon = np.array(std_y_lon)
        std_y_lat = np.array(std_y_lat)

        plt.subplot(111)
        plt.fill_between(std_x, mse_y - scale * std_y, mse_y + scale * std_y,
                         facecolor="C0", alpha=.5)
        plt.plot(mse_x, mse_y, color="C0", label="MSE")
        plt.fill_between(std_x, mse_y_lon - scale * std_y_lon, mse_y_lon + scale * std_y_lon,
                         facecolor="C1", alpha=.5)
        plt.plot(mse_x, mse_y_lon, color="C1", label="MSE-lon")
        plt.fill_between(std_x, mse_y_lat - scale * std_y_lat, mse_y_lat + scale * std_y_lat,
                         facecolor="C2", alpha=.5)
        plt.plot(mse_x, mse_y_lat, color="C2", label="MSE-lat")
        plt.legend()
        plt.xlim([mse_x.min(), mse_x.max()])
        plt.ylim([0, ylim])
        # plt.xticks(mse_x)
        plt.xlabel("Sun longitude (degrees)")
        plt.ylabel("MSE (degrees)")
    elif isinstance(mode, basestring) and mode == "lat":
        plt.figure("sun-lat-mse", figsize=(15, 5))
        ylim = 180
        scale = 1.

        mse_x, mse_y, mse_y_lon, mse_y_lat = [], [], [], []
        std_x, std_y, std_y_lon, std_y_lat = [], [], [], []
        if tilting:
            angles = np.linspace(0, np.pi, 180, endpoint=False)
        else:
            angles = np.linspace(0, np.pi/2, 90, endpoint=False)
        for angle in angles:
            if np.rad2deg(angle) in mse_x:
                continue

            j = np.array([phi - np.pi/360. < angle <= phi + np.pi/360. for phi in r[:, 4]])
            if j.sum() == 0:
                continue

            yj = y[j]
            tj = t[j]

            mse_x.append(np.rad2deg(angle))
            std_x.append(np.rad2deg(angle))
            mse_y.append(mse(yj, tj))
            mse_y_lon.append(mse(yj, tj, theta=False))
            mse_y_lat.append(mse(yj, tj, phi=False))
            std_y.append(std(yj, tj) / np.sqrt(yj.shape[0]))
            std_y_lon.append(std(yj, tj, theta=False) / np.sqrt(yj.shape[0]))
            std_y_lat.append(std(yj, tj, phi=False) / np.sqrt(yj.shape[0]))

        mse_x = np.array(mse_x)
        mse_y = np.array(mse_y)
        mse_y_lon = np.array(mse_y_lon)
        mse_y_lat = np.array(mse_y_lat)
        std_x = np.array(std_x)
        std_y = np.array(std_y)
        std_y_lon = np.array(std_y_lon)
        std_y_lat = np.array(std_y_lat)

        plt.subplot(111)
        plt.fill_between(std_x, mse_y - scale * std_y, mse_y + scale * std_y,
                         facecolor="C0", alpha=.5)
        plt.plot(mse_x, mse_y, color="C0", label="MSE")
        plt.fill_between(std_x, mse_y_lon - scale * std_y_lon, mse_y_lon + scale * std_y_lon,
                         facecolor="C1", alpha=.5)
        plt.plot(mse_x, mse_y_lon, color="C1", label="MSE-lon")
        plt.fill_between(std_x, mse_y_lat - scale * std_y_lat, mse_y_lat + scale * std_y_lat,
                         facecolor="C2", alpha=.5)
        plt.plot(mse_x, mse_y_lat, color="C2", label="MSE-lat")
        plt.legend()
        plt.xlim([mse_x.min(), mse_x.max()])
        plt.ylim([0, ylim])
        # plt.xticks(mse_x)
        plt.xlabel("Sun latitude (degrees)")
        plt.ylabel("MSE (degrees)")

    plt.show()

    s.rotate(-s.yaw, -s.pitch, -s.roll)
    lon, lat = (s.sky.lon + np.pi) % (2 * np.pi) - np.pi, (s.sky.lat + np.pi) % (2 * np.pi) - np.pi
    print "Reality: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    lon, lat = s(s.sky, decode=True).flatten()
    lon, lat = (lon + np.pi) % (2 * np.pi) - np.pi, (lat + np.pi) % (2 * np.pi) - np.pi
    print "Prediction: Lon = %.2f, Lat = %.2f" % (np.rad2deg(lon), np.rad2deg(lat))
    # CompassSensor.visualise(s)

