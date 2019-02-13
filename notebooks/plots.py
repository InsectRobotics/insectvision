from compoundeye.geometry import fibonacci_sphere
from compoundeye.evaluation import evaluate

from ephem import Sun
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


def plot_sky(phi, theta, y, p, a, noise=0.):
    ax = plt.subplot(131, polar=True)
    ax.scatter(phi, theta, s=10, marker='.', c=y, cmap="Blues_r", vmin=-0., vmax=7.)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([])
    plt.xticks([])

    ax = plt.subplot(132, polar=True)
    ax.scatter(phi, theta, s=10, marker='.', c=p, cmap="Greys", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([])
    plt.xticks([])

    ax = plt.subplot(133, polar=True)
    ax.scatter(phi, theta, s=10, marker='.', c=a, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([])
    plt.xticks([])

    return plt


def plot_pol_neurons_rotating_linear_polariser(s_1, s_2, r_1, r_2, r_z, r_pol, save_figs=False):
    fontsize = 20

    ax = plt.subplot2grid((2, 2), (0, 0))
    plt.plot([0, 0], [0, 1.2], "k-", lw=1)
    plt.plot(s_2[0], s_2[1], label=r'$s_\perp$')
    plt.plot(s_1[0], s_1[1], label=r'$s_\parallel$')
    ax.annotate(r'$s_\parallel$', xy=(90, 1), xytext=(110, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    ax.annotate(r'$s_\perp$', xy=(0, 1), xytext=(20, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    plt.ylabel("stimulus", fontsize=fontsize - 1)
    plt.ylim([0, 1.2])
    plt.yticks([0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "", "", "", ""], fontsize=fontsize)
    if save_figs:
        plt.savefig("stimuli.eps")

    ax = plt.subplot2grid((2, 2), (1, 0))
    plt.plot([0, 0], [0, 1.2], "k-", lw=1)
    plt.plot(r_2[0], r_2[1], label=r'$r_\perp$')
    plt.plot(r_1[0], r_1[1], label=r'$r_\parallel$')
    ax.annotate(r'$r_\parallel$', xy=(90, 1), xytext=(110, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    ax.annotate(r'$r_\perp$', xy=(0, 1), xytext=(20, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    plt.ylabel("response", fontsize=fontsize - 1)
    plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([0, 1.2])
    plt.yticks([0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)
    if save_figs:
        plt.savefig("response.eps")

    ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2, sharex=ax)
    plt.plot([0, 0], [-1.2, 1.2], "k-", lw=1)
    plt.plot([-45, 135], [0, 0], "k-", lw=1)
    plt.plot(r_z[0], r_z[1], "k--", label=r'$\frac{r_\parallel - r_\perp}{z}$')
    plt.plot(r_pol[0], r_pol[1], "k-", label=r'$\frac{r_\parallel - r_\perp}{r_\parallel + r_\perp}$')
    plt.legend(fontsize=fontsize)
    plt.ylabel("POL-neuron response", fontsize=fontsize)
    plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([-1.2, 1.2])
    plt.yticks([-1, 0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)

    plt.tight_layout()
    if save_figs:
        plt.savefig("POL-neurons.eps")
    return plt


def plot_gate_ring(sigma=np.deg2rad(13), shift=np.deg2rad(40), theta_t=0., phi_t=0., subplot=111):
    theta, phi = fibonacci_sphere(samples=1000, fov=161)
    d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
             np.cos(shift - theta) * np.sin(theta_t) *
             np.cos(phi - phi_t))
    gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)

    ax = plt.subplot(subplot, polar=True)
    ax.scatter(phi, theta, s=10, marker='o', c=gate, cmap="Reds", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([np.deg2rad(28)], [r'$28^\circ$'])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$135^\circ$',
                        r'$180^\circ$', r'$-135^\circ$', r'$-90^\circ$', r'$-45^\circ$'])

    return plt


def plot_res2ele(samples=1000, noise=0., subplot=111):

    ele, azi, azi_diff, res = [], [], [], []

    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    phi_s = phi_s[theta_s <= np.pi / 2]
    theta_s = theta_s[theta_s <= np.pi / 2]
    phi_s = phi_s[theta_s > np.pi / 18]
    theta_s = theta_s[theta_s > np.pi / 18]
    samples = theta_s.size

    for e, a in zip(theta_s, phi_s):
        d_err, d_eff, tau, _, _ = evaluate(sun_azi=a, sun_ele=e, tilting=False, noise=noise)
        azi.append(a)
        ele.append(e)
        res.append(tau.flatten())

    ele = np.rad2deg(ele).flatten()
    res = (np.array(res).flatten() - 1.06) * 7 / 4
    ele = ele[res <= 2]
    res = res[res <= 2]
    ele_pred = 26 * (1 - 2 * np.arcsin(1 - res) / np.pi) + 15  # + np.random.randn(res.size)

    plt.subplot(subplot)
    plt.scatter(res, ele, c='black', marker='.')
    plt.scatter(res, ele_pred, c='red', marker='.')
    plt.plot([-.5, 3 * np.pi / 4], [18.75, 18.75], "k--")
    plt.plot([-.5, 3 * np.pi / 4], [65.98, 65.98], "k--")
    plt.ylabel(r'$\epsilon (\circ)$')
    plt.xlabel(r'$\tau$')
    plt.xlim([-.5, 3 * np.pi / 4])
    plt.ylim([90, 0])
    # plt.xticks([0, 90, 180, 270, 360])
    return plt


def plot_ephemeris(obs, dt=10):
    sun = Sun()
    delta = timedelta(minutes=dt)

    azi, azi_diff, ele = [], [], []

    for month in xrange(12):
        obs.date = datetime(year=2018, month=month + 1, day=13)

        cur = obs.next_rising(sun).datetime() + delta
        end = obs.next_setting(sun).datetime()
        if cur > end:
            cur = obs.previous_rising(sun).datetime() + delta

        while cur <= end:
            obs.date = cur
            sun.compute(obs)
            a, e = sun.az, np.pi / 2 - sun.alt
            if len(azi) > 0:
                d = 60. / dt * np.absolute((a - azi[-1] + np.pi) % (2 * np.pi) - np.pi)
                if d > np.pi / 2:
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

    return plt


def plot_snapshot(theta, phi, r_pol, r_sol, r_tcl, w_sol=None, w_tcl=None, phi_sol=None, phi_tcl=None,
                  theta_t=0., phi_t=0., sun_ele=None, sun_azi=None, subplot=111):
    x = np.linspace(0, 2 * np.pi, 721)
    nb_pol = r_pol.size
    nb_sol = r_sol.size
    nb_tcl = r_tcl.size
    if phi_sol is None:
        phi_sol = np.linspace(0., 2 * np.pi, nb_sol, endpoint=False)
    if phi_tcl is None:
        phi_tcl = phi_sol

    s = subplot // 100
    u = (subplot % 100) // 10
    b = subplot % 10

    if w_sol is not None:
        ax = plt.subplot(6, u, u * 3 + b)
        plt.imshow(5 * w_sol.T, cmap="coolwarm", vmin=-1, vmax=1)
        plt.yticks([0, 7], ["1", "8"])
        plt.xticks([0, 59], ["1", "60"])

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

    if w_tcl is not None:
        ax = plt.subplot(12, u, u * 8 + b)
        plt.imshow(w_tcl.T, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks([0, 7], ["1", "8"])
        plt.yticks([0, 7], ["1", "8"])

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)

    ax = plt.subplot(2, u, b, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(False)

    # POL
    ax.scatter(phi, theta, s=90, c=r_pol, marker='o', edgecolor='black', cmap="coolwarm", vmin=-1, vmax=1)

    # SOL
    y = np.deg2rad(37.5)
    sy = np.deg2rad(15)
    ax.fill_between(x, np.full_like(x, y - sy / 2), np.full_like(x, y + sy / 2),
                    facecolor="C1", alpha=.5, label="SOL")
    ax.scatter(phi_sol, np.full(nb_sol, np.deg2rad(37.5)), s=600,
               c=r_sol, marker='o', edgecolor='red', cmap="coolwarm", vmin=-1, vmax=1)

    for ii, pp in enumerate(phi_sol):
        ax.text(pp - np.pi / 13, y, "%d" % (ii + 1), ha="center", va="center", size=10,
                bbox=dict(boxstyle="circle", fc="w", ec="k"))
        ax.arrow(pp, np.deg2rad(33), 0, np.deg2rad(4), fc='k', ec='k',
                 head_width=.1, head_length=.1, overhang=.3)

    # TCL
    y = np.deg2rad(52.5)
    sy = np.deg2rad(15)
    ax.fill_between(x, np.full_like(x, y - sy / 2), np.full_like(x, y + sy / 2),
                    facecolor="C2", alpha=.5, label="TCL")
    ax.scatter(phi_tcl, np.full_like(phi_tcl, y), s=600,
               c=r_tcl, marker='o', edgecolor='green', cmap="coolwarm", vmin=-1, vmax=1)
    for ii, pp in enumerate(phi_tcl):
        ax.text(pp + np.pi / 18, y, "%d" % (ii + 1), ha="center", va="center", size=10,
                bbox=dict(boxstyle="circle", fc="w", ec="k"))
        dx, dy = np.deg2rad(4) * np.sin(0.), np.deg2rad(4) * np.cos(0.)
        ax.arrow(pp - dx, y - dy / 2 - np.deg2rad(2.5), dx, dy, fc='k', ec='k',
                 head_width=.07, head_length=.1, overhang=.3)

    # Sun position
    if sun_ele is not None and sun_azi is not None:
        ax.scatter(sun_azi, sun_ele, s=500, marker='o', edgecolor='black', facecolor='yellow')
        ax.scatter(phi_t, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')

    # Decoded TCL
    R = r_tcl.dot(np.exp(-np.arange(nb_tcl) * (0. + 1.j) * 2. * np.pi / float(nb_tcl)))
    a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)

    ax.plot([0, a_pred], [0, np.pi / 2], 'k--', lw=1)
    ax.arrow(a_pred, 0, 0, np.deg2rad(20),
             fc='k', ec='k', head_width=.3, head_length=.2, overhang=.3)

    ax.legend(ncol=2, loc=(.15, -2.), fontsize=16)

    ax.set_ylim([0, np.deg2rad(60)])
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels([r'N', r'E', r'S', r'W'])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    plt.subplots_adjust(left=.07, bottom=.0, right=.93, top=.96)

    return plt
