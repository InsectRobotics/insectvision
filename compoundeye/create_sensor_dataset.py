import numpy as np
from sensor import CompassSensor, NB_EN, encode_sun
from sky import get_seville_observer, ChromaticitySkyModel
from datetime import datetime, timedelta
import os

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"


if __name__ == "__main__":

    # parameters
    nb_lenses = 100
    fov = 60
    observer = get_seville_observer()
    nb_months = 7
    start_month = 6
    start_day = 21
    delta = timedelta(hours=1)

    # data-set
    x = np.empty((0, nb_lenses), dtype=np.float32)
    t = np.empty((0, NB_EN), dtype=np.float32)
    m = np.empty((0, 1), dtype='datetime64[s]')
    name = "seville-F%03d-I%03d-O%03d-M%02d-D%04d" % (fov, nb_lenses, NB_EN, nb_months, delta.seconds)
    print name

    sensor = CompassSensor(nb_lenses=nb_lenses, fov=np.deg2rad(fov))

    months = (np.arange(start_month-1, start_month + nb_months - 1, 1) % 12) + 1
    for month in months:
        observer.date = datetime(year=2017, month=month, day=start_day, hour=0, minute=0, second=0)
        sky = ChromaticitySkyModel(observer=observer, nside=1)
        rising = observer.next_rising(sky.sun).datetime() + timedelta(hours=1)
        setting = observer.next_setting(sky.sun).datetime() - timedelta(hours=1)

        observer.date = rising
        while observer.date.datetime() < setting:
            print "Date:", observer.date,
            sky = ChromaticitySkyModel(observer=observer, nside=1)
            sky.generate()

            r = sensor.facing_direction
            for i in xrange(360):
                sensor.rotate(np.deg2rad(1))
                sensor.set_sky(sky)
                lon = (sky.lon + sensor.facing_direction) % (2 * np.pi)
                lat = sky.lat

                x = np.vstack([x, sensor.L.flatten()])
                t = np.vstack([t, encode_sun(lon, lat)])
                m = np.vstack([m, observer.date.datetime()])
                if i % 10 == 0:
                    print '.',
            print " ", x.shape, t.shape

            sensor.facing_direction = r
            observer.date = observer.date.datetime() + delta

    np.savez_compressed(__datadir__ + "%s.npz" % name, x=x, t=t)
    np.savez_compressed(__datadir__ + "M%02d-D%04d.npz" % (nb_months, delta.seconds), m=m)
