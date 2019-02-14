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


def plot_accuracy(save=None, repeats=10, **kwargs):

    plt.subplot(121)
    sun_ele = np.linspace(0, np.pi/2, 91)
    sun_azi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    sun_ele = kwargs.get('sun_ele', sun_ele)
    d_mean = np.zeros_like(sun_ele)
    d_se = np.zeros_like(sun_ele)
    tau = np.zeros_like(sun_ele)
    kwargs['sun_azi'] = kwargs.get('sun_azi', sun_azi)
    kwargs['tilting'] = kwargs.get('tilting', False)
    kwargs['verbose'] = kwargs.get('verbose', False)

    for j, noise in enumerate(np.linspace(0, 1, 5, endpoint=False)):
        kwargs['noise'] = noise
        for i, theta_s in enumerate(sun_ele):
            kwargs['sun_ele'] = np.full_like(sun_azi, theta_s)
            d_err, d_eff, t, a_ret, tb1 = evaluate(**kwargs)
            d_mean[i] = np.nanmean(d_err)
            d_se[i] = np.nanstd(d_err) / np.sqrt(np.sum(~np.isnan(d_err)))
            tau[i] += np.nanmean(t)
        plt.fill_between(np.rad2deg(sun_ele), d_mean - d_se, d_mean + d_se, facecolor='C%d' % j, alpha=.5)
        plt.plot(np.rad2deg(sun_ele), d_mean, color='C%d' % j, label=r'$\eta = %.1f$' % noise)

    tau /= 5
    plt.plot(np.rad2deg(sun_ele), tau * 45, 'k--')
    plt.legend()
    plt.yticks([0, 10, 20, 90])
    plt.ylim([0, 90])
    plt.xticks([0, 30, 60, 90])
    plt.xlim([0, 90])
    plt.ylabel(r'MSE ($J_s$) [$^\circ$]')
    plt.xlabel(r'sun elevation ($\theta_s$) [$^\circ$]')

    ax1 = plt.subplot(122)
    kwargs['sun_ele'] = None
    kwargs['sun_azi'] = None
    etas = np.linspace(0, 1, 21)
    taus = np.zeros_like(etas)
    means = np.zeros_like(etas)
    ses = np.zeros_like(etas)
    for i in xrange(repeats):
        noise = np.ones(60, int)
        x = np.argsort(np.absolute(np.random.randn(noise.size)))
        for ii, eta in enumerate(etas):
            noise[:] = 0
            noise[x[:int(eta * float(noise.size))]] = 1
            kwargs['noise'] = noise
            d_err, d_eff, tau, _, _ = evaluate(**kwargs)
            means[ii] = (means[ii] * i + d_err.mean()) / (i + 1)
            ses[ii] = (ses[ii] * i + d_err.std() / np.sqrt(d_err.size)) / (i + 1)
            taus[ii] = (taus[ii] * i + tau.mean()) / (i + 1)

    # print " Disturbance         Cost        "
    # print "---------------------------------"
    # for i, eta in enumerate(etas):
    #     print "   % 3.2f%%    % 2.2f +/- %.4f " % (eta * 100, means[i], ses[i])

    ax1.fill_between(etas * 100, means-ses, means+ses, facecolor="grey")
    ax1.plot(etas * 100, means, color="black", linestyle="-", label=r'$J_s$')
    ax1.set_ylim([0, 90])
    ax1.set_yticks([])
    ax1.set_xlim([0, 100])
    ax1.set_xlabel(r'disturbance ($\eta$) [%]')

    ax2 = ax1.twinx()
    ax2.plot(etas * 100, taus, color="black", linestyle="--", label=r'$\tau$')
    ax2.set_ylim([0, 2])
    ax2.set_yticks([0, .5, 1, 1.5, 2])
    plt.legend()

    if save:
        plt.savefig(save)

    return plt


def plot_gate_optimisation(load="data/gate-costs.npz", save=None, **kwargs):

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    sigmas = np.linspace(np.pi/180, np.pi/2, 90)
    shifts = np.linspace(0, 2*np.pi, 91)

    if load is not None:
        data = np.load(load)
        shifts, sigmas, means = data["shifts"], data["sigmas"], data["costs"]
    else:
        # TODO parametrise this to work in batches so that I can run it in multiple processors
        sigmas, shifts = np.meshgrid(sigmas, shifts)
        means = np.zeros(sigmas.size)
        for ii, sigma, shift in zip(np.arange(sigmas.size), sigmas.flatten(), shifts.flatten()):
            d_err, d_eff, tau, _, _ = evaluate(sigma=sigma, shift=shift, verbose=False, **kwargs)
            means[ii] = d_err.mean()
            print "Sigma: %d, Shift: %d, Cost: %.2f" % (np.rad2deg(sigma), np.rad2deg(shift), d_err.mean())

        means = means.reshape(shifts.shape)
        np.savez_compressed(save, shifts=shifts, sigmas=sigmas, costs=means)

    ii = np.argmin(means.flatten())
    sigma_min = sigmas.flatten()[ii]
    shift_min = shifts.flatten()[ii]
    means_min = means.flatten()[ii]
    print 'Minimum cost (%.2f) for Sigma = %.2f, Shift = %.2f' % (
        means_min, np.rad2deg(sigma_min), np.rad2deg(shift_min))

    with plt.rc_context({'ytick.color': 'white'}):
        plt.pcolormesh(shifts, sigmas, means, cmap="Reds", vmin=0, vmax=90)
        plt.scatter(shift_min, sigma_min, s=20, c='yellowgreen', marker='o')
        plt.yticks([0, np.pi/6, np.pi/3, np.pi/2],
                   [r"$0$", r"$30^\circ$", r"$60^\circ$", r"$90^\circ$"])
        plt.xticks(np.linspace(0, 2*np.pi, 8, endpoint=False),
                   [r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$135^\circ$',
                    r'$180^\circ$', r'$-135^\circ$', r'$-90^\circ$', r'$-45^\circ$'])
        plt.ylim([0, np.pi/2])
        ax.grid(alpha=0.2)

    return plt


def plot_gate_cost(samples=500, **kwargs):
    d_err, d_eff, tau, _, _ = evaluate(samples=samples, tilting=True, **kwargs)

    tau = np.rad2deg(tau)
    d_mean = np.nanmean(d_err)
    d_se = d_err.std() / np.sqrt(d_err.size)
    print "Tilt             overall              0 deg              30 deg             60 deg     "
    print "---------------------------------------------------------------------------------------"
    print "Mean cost    %.2f +/- %.4f" % (d_mean, d_se),

    if samples == 1000:
        samples /= 2
    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    phi_s = phi_s[theta_s <= np.pi / 2]
    theta_s = theta_s[theta_s <= np.pi / 2]

    d_00 = d_err[:, 0]
    d_30 = np.nanmean(d_err[:, 1:9], axis=1)
    d_60 = np.nanmean(d_err[:, 9:], axis=1)
    print "   %.2f +/- %.4f" % (np.nanmean(d_00), np.nanstd(d_00) / d_00.size),
    print "   %.2f +/- %.4f" % (np.nanmean(d_err[:, 1:9]), np.nanstd(d_err[:, 1:9]) / d_err[:, 1:9].size),
    print "   %.2f +/- %.4f" % (np.nanmean(d_err[:, 9:]), np.nanstd(d_err[:, 9:]) / d_err[:, 9:].size)

    plt.figure("Tilts", figsize=(10, 3))
    for i, ang, dd in zip(range(3), [0, np.pi/6, np.pi/3], [d_00, d_30, d_60]):
        ax = plt.subplot2grid((1, 10), (0, i * 3), colspan=3, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.scatter(phi_s, np.rad2deg(theta_s), marker=".", c=dd, cmap="Reds", vmin=0, vmax=90)
        plt.scatter(np.pi, np.rad2deg(ang), marker="o", c="yellowgreen", edgecolors="black")
        plt.text(-np.deg2rad(50), 145, ["A", "B", "C"][i], fontsize=12)
        plt.axis("off")
    plt.subplot2grid((3, 10), (1, 9))
    plt.imshow(np.array([np.arange(0, np.pi / 2, np.pi / 180)] * 10).T, cmap="Reds")
    plt.xticks([])
    plt.yticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])

    return plt


def plot_structure_optimisation(save=None, mode=0, **kwargs):
        print "Running structure test:", kwargs

        plt.figure("Structure", figsize=(10, 5))

        ns = np.linspace(0, 360, 91)
        ns[0] = 1
        omegas = np.linspace(1, 180, 180)
        # ns = np.array([4, 12, 60, 112, 176, 272, 368, 840])
        # omegas = np.array([14, 30, 60, 90, 120, 150, 180])

        filename = "structure-costs.npz"
        if mode < 2:
            plt.subplot(121)
            means = np.zeros_like(ns)
            ses = np.zeros_like(ns)
            n_default = kwargs.pop('n', 360)
            omega_default = kwargs.pop('omega', 56)
            for ii, n in enumerate(ns.astype(int)):
                d_err, d_eff, tau, _, _ = evaluate(n=n, omega=omega_default, verbose=False, **kwargs)
                means[ii] = np.mean(d_err)
                ses[ii] = d_err.std() / np.sqrt(d_err.size)
                print 'N = % 3d, Omega = %.2f | Mean cost: %.2f +/- %.4f' % (n, omega_default, means[ii], ses[ii])

            means = means.reshape(ns.shape)

            plt.fill_between(ns, means - ses, means + ses, facecolor="grey")
            plt.plot(ns, means, color="black", label=r'$n$')
            plt.ylim([0, 60])
            plt.xlim([1, 360])
            plt.yticks([0, 15, 30, 45, 60], [r'%d$^\circ$' % o for o in [0, 15, 30, 45, 60]])
            plt.xticks([4, 12, 60, 112, 176, 272, 360])
            plt.xlabel(r'units ($n$)')
            plt.ylabel(r'MSE ($^\circ$)')

            plt.subplot(122)
            means = np.zeros_like(omegas)
            ses = np.zeros_like(omegas)
            for ii, omega in enumerate(omegas):
                d_err, d_eff, tau, _, _ = evaluate(n=n_default, omega=omega, verbose=False, **kwargs)
                means[ii] = np.mean(d_err)
                ses[ii] = d_err.std() / np.sqrt(d_err.size)
                print 'N = % 3d, Omega = %.2f | Mean cost: %.2f +/- %.4f' % (n_default, omega, means[ii], ses[ii])

            means = means.reshape(omegas.shape)

            plt.fill_between(omegas, means - ses, means + ses, facecolor="grey", alpha=.5)
            plt.plot(omegas, means, color="black", label=r'$\omega$')
            plt.ylim([0, 60])
            plt.xlim([0, 180])
            plt.yticks([0, 15, 30, 45, 60], [r'%d$^\circ$' % o for o in [0, 15, 30, 45, 60]])
            plt.xticks(np.linspace(0, 180, 7, endpoint=True),
                       [r'%d$^\circ$' % o for o in np.linspace(0, 180, 7, endpoint=True)])
            plt.xlabel(r'receptive field ($\omega$)')
            plt.ylabel(r'MSE ($^\circ$)')
            # plt.legend()
        else:
            ax = plt.subplot(111, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            if mode > 2:
                data = np.load(filename)
                ns, omegas, means = data["ns"], data["omegas"], data["costs"]
            else:
                ns, omegas = np.meshgrid(ns, omegas)
                means = np.zeros(omegas.size)
                for ii, omega, n in zip(np.arange(omegas.size), omegas.flatten(), ns.flatten()):
                    kwargs["n"] = n
                    kwargs["omega"] = omega
                    d_err, d_eff, tau, _, _ = evaluate(verbose=False, **kwargs)
                    means[ii] = np.mean(d_err)
                    se = d_err.std() / np.sqrt(d_err.size)
                    print 'N = % 3d, Omega = %.2f | Mean cost: %.2f +/- %.4f' % (n, omega, means[ii], se)

                means = means.reshape(omegas.shape)
                np.savez_compressed(filename, omegas=omegas, ns=ns, costs=means)

            ii = np.nanargmin(means, axis=0)
            jj = np.nanargmin(means[ii, np.arange(91)])
            omega_min = omegas[ii, np.arange(91)]
            n_min = ns[ii, np.arange(91)]
            means_min = means[ii, np.arange(91)]
            print means_min
            print n_min
            print omega_min
            print 'Minimum cost (%.2f) for N = %d, Omega = %.2f' % (means_min[jj], n_min[jj], omega_min[jj])
            print 'Mean omega %.2f +/- %.4f' % (omega_min.mean(), omega_min.std() / np.sqrt(omega_min.size))

            with plt.rc_context({'ytick.color': 'white'}):
                plt.pcolormesh(np.deg2rad(omegas), ns, means, cmap="Reds", vmin=0, vmax=90)
                # plt.scatter(np.deg2rad(omega_min), n_min, s=20, c='yellowgreen', marker='o')
                plt.plot(np.deg2rad(omega_min), n_min, 'g-')
                plt.yticks([4, 12, 60, 112, 176, 272, 360], [""] * 7)
                plt.xticks(np.deg2rad([14, 30, 60, 90, 120, 150, 180]))
                plt.ylim([4, 360])
                plt.xlim([0, 180])
                ax.grid(alpha=0.2)

            if save:
                plt.savefig(save)

        return plt


if __name__ == "__main__":
    # plot_accuracy().show()
    plot_gate_optimisation(save="data/gate-costs-2.npz", load=None)
    # plot_structure_optimisation(tilting=True, mode=3, n=60, omega=56, weighted=True).show()
