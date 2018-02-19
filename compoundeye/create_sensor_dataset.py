import numpy as np
import healpy as hp
import os

from sensor import CompassSensor, NB_EN, encode_sun
from sky import get_seville_observer, SkyModel
from datetime import datetime, timedelta

__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/datasets/"


if __name__ == "__main__":

    # parameters
    mode = 'cross'
    fibonacci = True
    tilting = False
    nb_lenses = 60
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
    r = np.empty((0, 5), dtype=np.float32)
    name = "seville-F%03d-I%03d-O%03d-M%02d-D%04d" % (fov, nb_lenses, NB_EN, nb_months, delta.seconds)
    if tilting:
        name += "-tilt"
    if fibonacci:
        name += "-fibonacci"
    print name

    sensor = CompassSensor(nb_lenses=nb_lenses, fov=np.deg2rad(fov), mode=mode, fibonacci=fibonacci)

    months = (np.arange(start_month-1, start_month + nb_months - 1, 1) % 12) + 1
    for month in months:
        observer.date = datetime(year=2017, month=month, day=start_day, hour=0, minute=0, second=0)
        sensor.sky.obs = observer
        rising = observer.next_rising(sensor.sky.sun).datetime() + timedelta(hours=1)
        setting = observer.next_setting(sensor.sky.sun).datetime() - timedelta(hours=1)

        sensor.sky.obs.date = rising
        while sensor.sky.obs.date.datetime() < setting:
            sensor.sky.generate()

            if tilting:
                tilt = [-60, -30, 0, 30, 60]
                # tilt = [-60, -30, -15, -5, 0, 5, 15, 30, 60]
            else:
                tilt = [0]

            print "Date:", sensor.sky.obs.date,
            for k in tilt:
                sensor.rotate(roll=np.deg2rad(k)-sensor.roll)
                for j in tilt:
                    sensor.rotate(pitch=np.deg2rad(j)-sensor.pitch)
                    for i in xrange(0, 360):
                        sensor.refresh()

                        # calculate relative sun position
                        lat, lon = hp.Rotator(rot=(
                            np.rad2deg(-sensor.yaw), np.rad2deg(-sensor.pitch), np.rad2deg(-sensor.roll)
                        ))(sensor.sky.lat, np.pi - sensor.sky.lon)
                        lat, lon = lat % (2 * np.pi), (2 * np.pi - lon) % (2 * np.pi) - np.pi

                        # update the data-set
                        x = np.vstack([x, sensor.L])
                        t = np.vstack([t, encode_sun(lon, lat)])
                        m = np.vstack([m, sensor.sky.obs.date.datetime()])
                        r = np.vstack([r, [sensor.yaw, sensor.pitch, sensor.roll, lon, lat]])
                        sensor.rotate(yaw=np.deg2rad(1))
                        # if i % 10 == 0:
                    print '.',  # "% 3d % 3d % 3d" % tuple(np.rad2deg(sensor.yaw_pitch_roll))
                    sensor.rotate(yaw=-sensor.yaw)
                sensor.rotate(pitch=-sensor.pitch)
            sensor.rotate(roll=-sensor.roll)
            #
            # if tilting:
            #     for j in tilt:
            #         sensor.rotate(roll=np.deg2rad(j)-sensor.roll)
            #         for i in xrange(360):
            #             sensor.refresh()
            #
            #             # calculate relative sun position
            #             lat, lon = hp.Rotator(rot=(
            #                 np.rad2deg(-sensor.yaw), np.rad2deg(-sensor.pitch), np.rad2deg(-sensor.roll)
            #             ))(sensor.sky.lat, np.pi - sensor.sky.lon)
            #             lat, lon = lat % (2 * np.pi), (2 * np.pi - lon) % (2 * np.pi) - np.pi
            #
            #             # update the data-set
            #             x = np.vstack([x, sensor.L])
            #             t = np.vstack([t, encode_sun(lon, lat)])
            #             m = np.vstack([m, sensor.sky.obs.date.datetime()])
            #             tlt = np.vstack([tlt, [0., np.deg2rad(j), lon, lat]])
            #             sensor.rotate(yaw=np.deg2rad(1))
            #             # if i % 10 == 0:
            #         print '.',
            #     sensor.rotate(roll=-sensor.roll)
            sensor.refresh()
            print " ", x.shape, t.shape

            sensor.sky.obs.date = sensor.sky.obs.date.datetime() + delta

    if mode == 'normal':
        mode = ''
    else:
        mode += '-'
    np.savez_compressed(__datadir__ + "%s%s.npz" % (mode, name), x=x, t=t)
    if tilting:
        dates_name = __datadir__ + "%sM%02d-D%04d-tilted.npz"
    else:
        dates_name = __datadir__ + "%sM%02d-D%04d.npz"
    np.savez_compressed(dates_name % (mode, nb_months, delta.seconds), m=m, r=r)
