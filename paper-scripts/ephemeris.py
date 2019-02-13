from environment import get_seville_observer
# from environment.utils import sun2lonlat
from compoundeye.geometry import fibonacci_sphere
from compoundeye.evaluation import evaluate

from ephem import Sun
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # mode = "ephemeris"
    mode = "res2ele"
    # mode = "res2azi"

    sun = Sun()
    obs = get_seville_observer()
    obs.date = datetime.strptime("2018-06-21", "%Y-%m-%d")
    dt = 10
    delta = timedelta(minutes=dt)

    cur = obs.next_rising(sun).datetime() + delta
    end = obs.next_setting(sun).datetime()
    if cur > end:
        cur = obs.previous_rising(sun).datetime() + delta

    if mode is "res2ele":
        samples = 1000

        ele, azi, azi_diff, res = [], [], [], []

        theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
        phi_s = phi_s[theta_s <= np.pi / 2]
        theta_s = theta_s[theta_s <= np.pi / 2]
        phi_s = phi_s[theta_s > np.pi / 18]
        theta_s = theta_s[theta_s > np.pi / 18]
        samples = theta_s.size

        for e, a in zip(theta_s, phi_s):
            d_err, d_eff, tau, _, _ = evaluate(sun_azi=a, sun_ele=e, tilting=False, noise=0.)
            azi.append(a)
            ele.append(e)
            res.append(tau.flatten())
            # res.append(d_eff.flatten())

        ele = np.rad2deg(ele).flatten()
        # res = np.array(res).flatten() - np.pi/2  # Min: 0.02, Max: 2.04; Min: 18.22, Max: 66.91
        res = (np.array(res).flatten() - 1.06) * 7 / 4
        # res = (np.array(res).flatten() - 1) * 35 / 20
        # res = (np.array(res).flatten() - 2.12) * 7 / 8
        # res = np.array(res)
        ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15  # + np.random.randn(res.size)

        plt.figure("tau2ele", figsize=(5, 5))
        i_min = np.argmin(res)
        i_max = np.argmax(res)
        print "Elevation -- Min: %.2f, Max: %.2f" % (ele[i_min], ele[i_max])
        print "Response  -- Min: %.2f, Max: %.2f" % (res[i_min], res[i_max])
        plt.subplot(111)
        plt.scatter(res, ele, c='black', marker='.')
        plt.scatter(res, ele_pred, c='red', marker='.')
        plt.plot([-.5, 3*np.pi/4], [18.75, 18.75], "k--")
        plt.plot([-.5, 3*np.pi/4], [65.98, 65.98], "k--")
        plt.ylabel(r'$\epsilon (\circ)$')
        plt.xlabel(r'$\tau$')
        plt.xlim([-.5, 3*np.pi/4])
        plt.ylim([90, 0])
        # plt.xticks([0, 90, 180, 270, 360])
        plt.show()

    elif mode is "res2azi":
        azi, azi_diff, ele, res = [], [], [], []

        for month in xrange(12):
            obs.date = datetime(year=2018, month=month+1, day=13)

            cur = obs.next_rising(sun).datetime() + delta
            end = obs.next_setting(sun).datetime()
            if cur > end:
                cur = obs.previous_rising(sun).datetime() + delta

            while cur <= end:
                obs.date = cur
                sun.compute(obs)
                a, e = sun.az, np.pi/2 - sun.alt
                if len(azi) > 0:
                    d = 60. / dt * np.absolute((a - azi[-1] + np.pi) % (2 * np.pi) - np.pi)
                    if d > np.pi/2:
                        azi_diff.append(0.)
                    else:
                        azi_diff.append(d)
                else:
                    azi_diff.append(0.)

                d_err, d_eff, tau, _, _ = evaluate(sun_azi=a, sun_ele=e, tilting=False, noise=0.)
                azi.append(a % (2 * np.pi))
                ele.append(e)
                res.append(tau.flatten())
                # increase the current time
                cur = cur + delta

        ele = np.rad2deg(ele)
        azi = np.rad2deg(azi)
        azi_diff = np.rad2deg(azi_diff)
        res = np.array(res).flatten()
        azi = azi[ele < 90]
        azi_diff = azi_diff[ele < 90]
        res = res[ele < 90]
        ele = ele[ele < 90]
        ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15

        ele_nonl = np.exp(.1 * (54 - ele_pred))

        x = np.array([9 + np.exp(.1 * (54 - ele_pred)), np.ones_like(azi_diff)]).T

        # w = np.linalg.pinv(x).dot(azi_diff)
        w = np.array([1., 0.])
        print w

        y = x.dot(w)
        error = np.absolute(y - azi_diff)
        print "Error: %.4f +/- %.4f" % (np.nanmean(error), np.nanstd(error) / np.sqrt(len(error))),
        print "| N = %d" % len(error)

        plt.figure(figsize=(10, 10))

        plt.subplot(221)
        plt.scatter(azi, ele_pred, c=azi_diff, cmap='Reds', marker='.')
        plt.ylabel(r'$\theta_s\' (\circ)$')
        plt.xlim([0, 360])
        plt.ylim([85, 0])
        plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360], [""] * 9)
        plt.yticks([0, 15, 30, 45, 60, 75])

        plt.subplot(222)
        plt.scatter(azi_diff, ele_pred, c=azi, cmap='coolwarm', marker='.')
        xx = np.linspace(0, 90, 100, endpoint=True)
        yy = 9 + np.exp(.1 * (54 - xx))
        plt.plot(yy, xx, 'k-')
        plt.ylim([85, 0])
        plt.xlim([7, 60])
        plt.xticks([10, 20, 30, 40, 50, 60], [""] * 6)
        plt.yticks([0, 15, 30, 45, 60, 75], [""] * 6)

        plt.subplot(223)
        plt.scatter(azi, x[:, 0], c=azi_diff, cmap='Reds', marker='.')
        plt.xlabel(r'$\phi_s (\circ)$')
        plt.ylabel(r'$\Delta\phi_s (\circ/h)$ -- prediction')
        plt.xlim([0, 360])
        plt.ylim([7, 65])
        plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        plt.yticks([10, 20, 30, 40, 50, 60])

        plt.subplot(224)
        plt.scatter(azi_diff, x[:, 0], c=azi, cmap='coolwarm', marker='.')
        plt.plot([w[0] * 7 + w[1], w[0] * 65 + w[1]], [7, 65], 'k-')
        plt.xlabel(r'$\Delta\phi_s (\circ/h)$ -- true')
        plt.xlim([7, 60])
        plt.xticks([10, 20, 30, 40, 50, 60])
        plt.ylim([7, 65])
        plt.yticks([10, 20, 30, 40, 50, 60], [""] * 6)

        plt.show()

    elif mode is "ephemeris":
        azi, azi_diff, ele = [], [], []

        for month in xrange(12):
            obs.date = datetime(year=2018, month=month+1, day=13)

            cur = obs.next_rising(sun).datetime() + delta
            end = obs.next_setting(sun).datetime()
            if cur > end:
                cur = obs.previous_rising(sun).datetime() + delta

            print cur, end
            while cur <= end:
                obs.date = cur
                sun.compute(obs)
                a, e = sun.az, np.pi/2 - sun.alt
                if len(azi) > 0:
                    d = 60. / dt * np.absolute((a - azi[-1] + np.pi) % (2 * np.pi) - np.pi)
                    if d > np.pi/2:
                        azi_diff.append(0.)
                    else:
                        azi_diff.append(d)
                else:
                    azi_diff.append(0.)
                azi.append(a % (2 * np.pi))
                ele.append(e)
                # increase the current time
                cur = cur + delta

        ele = np.rad2deg(ele)
        azi = np.rad2deg(azi)
        azi_diff = np.rad2deg(azi_diff)
        azi = azi[ele < 90]
        azi_diff = azi_diff[ele < 90]
        ele = ele[ele < 90]

        ele_nonl = np.exp(.1 * (54 - ele))

        x = np.array([9 + np.exp(.1 * (54 - ele)), np.ones_like(azi_diff)]).T

        # w = np.linalg.pinv(x).dot(azi_diff)
        w = np.array([1., 0.])
        print w

        y = x.dot(w)
        error = np.absolute(y - azi_diff)
        print "Error: %.4f +/- %.4f" % (error.mean(), error.std() / np.sqrt(len(error))),
        print "| N = %d" % len(error)

        plt.figure(figsize=(10, 10))

        plt.subplot(221)
        plt.scatter(azi, ele, c=azi_diff, cmap='Reds', marker='.')
        plt.ylabel(r'$\theta_s (\circ)$')
        plt.xlim([0, 360])
        plt.ylim([85, 0])
        plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360], [""] * 9)
        plt.yticks([0, 15, 30, 45, 60, 75])

        plt.subplot(222)
        plt.scatter(azi_diff, ele, c=azi, cmap='coolwarm', marker='.')
        xx = np.linspace(0, 90, 100, endpoint=True)
        yy = 9 + np.exp(.1 * (54 - xx))
        plt.plot(yy, xx, 'k-')
        plt.ylim([85, 0])
        plt.xlim([7, 60])
        plt.xticks([10, 20, 30, 40, 50, 60], [""] * 6)
        plt.yticks([0, 15, 30, 45, 60, 75], [""] * 6)

        plt.subplot(223)
        plt.scatter(azi, x[:, 0], c=azi_diff, cmap='Reds', marker='.')
        plt.xlabel(r'$\phi_s (\circ)$')
        plt.ylabel(r'$\Delta\phi_s (\circ/h)$ -- prediction')
        plt.xlim([0, 360])
        plt.ylim([7, 65])
        plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        plt.yticks([10, 20, 30, 40, 50, 60])

        plt.subplot(224)
        plt.scatter(azi_diff, x[:, 0], c=azi, cmap='coolwarm', marker='.')
        plt.plot([w[0] * 7 + w[1], w[0] * 65 + w[1]], [7, 65], 'k-')
        plt.xlabel(r'$\Delta\phi_s (\circ/h)$ -- true')
        plt.xlim([7, 60])
        plt.xticks([10, 20, 30, 40, 50, 60])
        plt.ylim([7, 65])
        plt.yticks([10, 20, 30, 40, 50, 60], [""] * 6)

        plt.show()
