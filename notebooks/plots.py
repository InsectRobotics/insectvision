#!/usr/bin/env python

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


from compoundeye.geometry import fibonacci_sphere
from compoundeye.evaluation import evaluate
from environment import Sun
from results import get_noise

from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import circmean
import numpy as np
import matplotlib.pyplot as plt


def plot_sky(phi, theta, y, p, a):
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
    res = np.array(res).flatten()
    ele_pred = 26 * (1 - 2 * np.arcsin(np.clip(2.855 - 3.5 * res, -1, 1)) / np.pi) + 15

    plt.subplot(subplot)
    plt.scatter(res, ele, c='black', marker='.')
    plt.scatter(np.clip(res, 0, 4), ele_pred, c='red', marker='.')
    plt.plot([-.5, 3 * np.pi / 4], [18.75, 18.75], "k--")
    plt.plot([-.5, 3 * np.pi / 4], [65.98, 65.98], "k--")
    plt.ylabel(r'$\epsilon (\circ)$')
    plt.xlabel(r'$\tau$')
    plt.xticks([0.5, 0.75, 1, 1.25])
    plt.xlim([.43, 1.21])
    plt.ylim([90, 0])
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


def plot_accuracy(save=None, repeats=10, verbose=False, **kwargs):

    ax1 = plt.subplot(132)
    ax3 = plt.subplot(131)
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
            tau[i] += np.nanmean(np.clip(t, 0., 2.)) / 5.
            # print tau
            # if j == 0:
            #     tau[i] = np.nanmean(t)
        ax1.fill_between(np.rad2deg(sun_ele), d_mean - d_se, d_mean + d_se, facecolor='C%d' % j, alpha=.5)
        ax1.plot(np.rad2deg(sun_ele), d_mean, color='C%d' % j, label=r'$\eta = %.1f$' % noise)
        ax3.fill_between(np.rad2deg(sun_ele), d_mean - d_se, d_mean + d_se, facecolor='C%d' % j, alpha=.5)
        ax3.plot(np.rad2deg(sun_ele), d_mean, color='C%d' % j, label=r'$\eta = %.1f$' % noise)

    plt.legend()
    ax1.set_yticks([0, 10, 20, 30])
    ax1.set_ylim([0, 30])
    ax1.set_xticks([0, 30, 60, 90])
    ax1.set_xlim([0, 90])
    ax1.set_ylabel(r'MSE ($J_s$) [$^\circ$]')
    ax1.set_xlabel(r'sun elevation ($\theta_s$) [$^\circ$]')

    ax2 = ax1.twinx()
    ax2.plot(np.rad2deg(sun_ele), tau, 'k--')
    ax2.set_ylim([0, 1.2])
    ax2.set_yticks([0, .4, .8, 1.2])

    ax3.set_yticks([0, .03, .06, .09])
    ax3.set_ylim([0, .09])
    ax3.set_xticks([0, 30, 60, 90])
    ax3.set_xlim([0, 90])
    ax3.set_ylabel("")
    ax3.set_xlabel(r'sun elevation ($\theta_s$) [$^\circ$]')

    ax1 = plt.subplot(133)
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
            taus[ii] = (taus[ii] * i + np.maximum(tau, 0).mean()) / (i + 1)

    if verbose:
        print " Disturbance         Cost          Confidence        "
        print "-----------------------------------------------------"
        for i, eta in enumerate(etas):
            print "   % 3.2f%%    % 2.2f +/- %.4f    % 2.2f" % (eta * 100, means[i], ses[i], taus[i])

    ax1.fill_between(etas * 100, means-ses, means+ses, facecolor="grey")
    ax1.plot(etas * 100, means, color="black", linestyle="-", label=r'$J_s$')
    ax1.plot(etas * 100, tau2sigma(taus), color="black", linestyle="--", label=r'$\sigma$')
    ax1.set_ylim([0, 30])
    ax1.set_yticks([])
    ax1.set_xlim([0, 100])
    ax1.set_xlabel(r'disturbance ($\eta$) [%]')

    ax2 = ax1.twinx()
    ax2.plot(etas * 100, taus, color="grey", linestyle="--", label=r'$\tau$')
    ax2.set_ylim([0, 1.2])
    ax2.set_yticks([0, .4, .8, 1.2])
    plt.legend()

    if save:
        plt.savefig(save)

    return plt


def plot_gate_optimisation(load="data/gate-costs.npz", save=None, **kwargs):

    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    sigmas = np.linspace(np.pi/180, np.pi/2, 90)
    shifts = np.linspace(0, 2*np.pi, 361)

    if load is not None:
        data = np.load(load)
        shifts, sigmas, means = data["shifts"], data["sigmas"], data["costs"]
    else:
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


def plot_structure_optimisation(load="data/structure-costs.npz", save=None, **kwargs):
        ns = np.linspace(0, 360, 91)
        ns[0] = 1
        omegas = np.linspace(1, 180, 180)

        plt.subplot2grid((1, 5), (0, 0), colspan=2)
        means = np.zeros_like(ns)
        ses = np.zeros_like(ns)
        n_default = kwargs.pop('n', 360)
        omega_default = kwargs.pop('omega', 56)
        for ii, n in enumerate(ns.astype(int)):
            d_err, d_eff, tau, _, _ = evaluate(nb_pol=n, omega=omega_default, verbose=False, **kwargs)
            means[ii] = np.mean(d_err)
            ses[ii] = d_err.std() / np.sqrt(d_err.size)

        means = means.reshape(ns.shape)

        plt.fill_between(ns, means - ses, means + ses, facecolor="grey")
        plt.plot(ns, means, color="black", label=r'$n$')
        plt.ylim([0, 60])
        plt.xlim([1, 360])
        plt.yticks([0, 15, 30, 45, 60], [r'%d$^\circ$' % o for o in [0, 15, 30, 45, 60]])
        plt.xticks([4, 12, 60, 112, 176, 272, 360])
        plt.xlabel(r'units ($n$)')
        plt.ylabel(r'MSE ($^\circ$)')

        plt.subplot2grid((1, 5), (0, 2), colspan=2)
        means = np.zeros_like(omegas)
        ses = np.zeros_like(omegas)
        for ii, omega in enumerate(omegas):
            d_err, d_eff, tau, _, _ = evaluate(nb_pol=n_default, omega=omega, verbose=False, **kwargs)
            means[ii] = np.mean(d_err)
            ses[ii] = d_err.std() / np.sqrt(d_err.size)

        means = means.reshape(omegas.shape)

        plt.fill_between(omegas, means - ses, means + ses, facecolor="grey", alpha=.5)
        plt.plot(omegas, means, color="black", label=r'$\omega$')
        plt.ylim([0, 60])
        plt.xlim([0, 180])
        plt.yticks([0, 15, 30, 45, 60], [r'%d$^\circ$' % o for o in [0, 15, 30, 45, 60]])
        plt.xticks(np.linspace(0, 180, 7, endpoint=True),
                   [r'%d$^\circ$' % o for o in np.linspace(0, 180, 7, endpoint=True)])
        plt.xlabel(r'receptive field ($\omega$)')

        ax = plt.subplot2grid((1, 5), (0, 4), polar=True)
        # ax = plt.subplot(133, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)

        if load is not None:
            data = np.load(load)
            ns, omegas, means = data["ns"], data["omegas"], data["costs"]
        else:
            ns, omegas = np.meshgrid(ns, omegas)
            means = np.zeros(omegas.size)
            kwargs["verbose"] = False
            for ii, omega, n in zip(np.arange(omegas.size), omegas.flatten(), ns.flatten()):
                kwargs["nb_pol"] = n
                kwargs["omega"] = omega
                d_err, d_eff, tau, _, _ = evaluate(**kwargs)
                means[ii] = np.mean(d_err)
                se = d_err.std() / np.sqrt(d_err.size)
                # print 'N = % 3d, Omega = %.2f | Mean cost: %.2f +/- %.4f' % (n, omega, means[ii], se)

            means = means.reshape(omegas.shape)
            if save is not None:
                np.savez_compressed(save, omegas=omegas, ns=ns, costs=means)

        ii = np.nanargmin(means, axis=0)
        jj = np.nanargmin(means[ii, np.arange(91)])
        omega_min = omegas[ii, np.arange(91)]
        n_min = ns[ii, np.arange(91)]
        means_min = means[ii, np.arange(91)]

        print 'Minimum cost (%.2f) for N = %d, Omega = %.2f' % (means_min[jj], n_min[jj], omega_min[jj])
        print 'Mean omega %.2f +/- %.4f' % (omega_min.mean(), omega_min.std() / np.sqrt(omega_min.size))

        with plt.rc_context({'ytick.color': 'white'}):
            plt.pcolormesh(np.deg2rad(omegas), ns, means, cmap="Reds", vmin=0, vmax=90)
            plt.scatter(np.deg2rad(omega_min), n_min, s=1, c='yellowgreen', marker='o')
            plt.plot(np.deg2rad(omega_min), n_min, 'g-')
            plt.yticks([4, 12, 60, 112, 176, 272, 360], [""] * 7)
            plt.xticks(np.deg2rad([14, 30, 60, 90, 120, 150, 180]))
            plt.ylim([4, 360])
            plt.xlim([0, 180])
            ax.grid(alpha=0.2)

        ax.set_thetamin(0)
        ax.set_thetamax(180)

        return plt


def plot_disturbance(phi, theta, r_pol, subplot=111):
    ax = plt.subplot(subplot, polar=True)
    ax.scatter(phi, theta, s=100, marker='.', c=r_pol, cmap="coolwarm", vmin=-.6, vmax=.6)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(30)])
    plt.yticks([])
    plt.xticks([])

    return plt


def plot_terrain(terrain, max_altitude=.5):
    ax = plt.gca()
    im = ax.imshow(terrain, cmap="PRGn", extent=[0, 10, 0, 10], vmin=-max_altitude, vmax=max_altitude)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)

    plt.colorbar(im, cax=cax)
    return plt


def plot_route(opath, ipath, id=None, label=None, subplot=111, xlim=None):
    if type(subplot) is not int:
        plt.subplot(*subplot)
    else:
        plt.subplot(subplot)
    if id is not None:
        plt.plot(opath[:, 0], opath[:, 1], 'C%d' % id, alpha=.5)
        plt.plot(ipath[:, 0], ipath[:, 1], 'C%d' % id, label=label)
        # plt.plot(ipath[:, 0], ipath[:, 1], 'C%d' % id, label=r'$\eta = %.1f$' % noise)
    else:
        plt.plot(opath[:, 0], opath[:, 1], 'r-')
        plt.plot(ipath[:, 0], ipath[:, 1], 'k--')
    # plt.xlim([4, 7] if xlim is None else xlim)
    # plt.ylim([-1, 9])

    return plt


def plot_tortuosity(d_c, d_x, id=0, label=None, subplot=111):
    d_x_mean = np.mean(d_x, axis=0)
    d_x_se = np.std(d_x, axis=0) / np.sqrt(len(d_x))

    plt.subplot(subplot)
    plt.fill_between(d_c[-1], d_x_mean - 3 * d_x_se, d_x_mean + 3 * d_x_se, facecolor='C%d' % id, alpha=.5)
    plt.plot(d_c[-1], d_x_mean, 'C%d' % id, label=label)
    plt.ylim([0, 100])
    plt.xlim([0, 200])
    # plt.ylabel(r"Distance from home [%]")
    # plt.xlabel(r"Distance travelled / Turning point distance [%]")

    return plt


def plot_circ_response(phi, r_mean, r_std, phi_mean, baseline=.5, uniform=True, subplot=(1, 1, 1)):
    ax = plt.subplot(*subplot, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    y_min, y_max = -.3, 1.1
    plt.bar(phi, baseline + r_mean, .1, yerr=r_std, facecolor='black')
    plt.plot(np.linspace(-np.pi, np.pi, 361), np.full(361, baseline), 'k-')
    if uniform:
        x_mean = [phi_mean, phi_mean, phi_mean + np.pi, phi_mean + np.pi]
        y_mean = [y_max, y_min, y_min, y_max]
    else:
        x_mean = [phi_mean + np.pi, phi_mean + np.pi]
        y_mean = [y_max, y_min]
    plt.plot(x_mean, y_mean, 'r-.')
    plt.yticks([])
    plt.xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False),
               [r'%d$^\circ$' % x for x in ((np.linspace(0, 360, 8, endpoint=False) + 180) % 360 - 180)])
    plt.ylim([y_min, y_max])

    return plt


def plot_summary_response(phi_mean, phi_max, fit_line=True, subplot=111):
    tb1_names = []
    x, y = [], []

    plt.subplot(subplot)
    for col, (phi_mean_i, phi_max_i) in enumerate(zip(phi_mean, phi_max)):
        if not np.any(np.isnan(phi_max_i)):
            x.append([(phi_max_i + np.pi/18) % (2 * np.pi) - np.pi/18, 1])
            y.append(col)
        tb1_names.append('L%d/R%d' % (8 - col, col + 1))
        plt.scatter([col] * len(phi_mean_i), np.rad2deg(phi_mean_i) % 360, s=20, c='black')
        plt.scatter([col] * len(phi_mean_i), np.rad2deg(phi_mean_i) % 360 + 360, s=20, c='black')
        plt.scatter([col] * len(phi_mean_i), np.rad2deg(phi_mean_i) % 360 - 360, s=20, c='black')
        plt.scatter(col, np.rad2deg(phi_max_i) % 360, s=50, c='red', marker='*')
        plt.scatter(col, np.rad2deg(phi_max_i) % 360 + 360, s=50, c='red', marker='*')
        plt.scatter(col, np.rad2deg(phi_max_i) % 360 + 360, s=50, c='red', marker='*')
    x = np.array(x)
    y = np.array(y)
    if fit_line:
        print x.shape, y.shape
        a, b = np.linalg.pinv(x[:-1]).dot(y[:-1])

        plt.plot([-1, 8], np.rad2deg([(-1 - b) / a, (8 - b) / a]), 'r-.')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
               [tb1_names[0], '', tb1_names[2], '', tb1_names[4], '', tb1_names[6], ''])
    plt.yticks([-90, -45, 0, 45, 90, 135, 180, 225, 270, 315, 360],
               ['-90', '', '0', '', '90', '', '180', '', '270', '', '360'])
    plt.ylim([-20, 380])
    plt.xlim([-1, 8])

    return plt


def tau2sigma(tau):
    return 4. / tau - 2


def sigma2tau(sigma):
    return 4. / (sigma + 2.)


if __name__ == "__main__":

    from astropy.stats import rayleightest

    # plt.figure("res2ele", figsize=(5, 5))
    # plot_res2ele(noise=0.).show()
    # plot_accuracy(verbose=True).show()
    # plot_gate_optimisation(save="data/gate-costs-2.npz", load=None)
    # plot_structure_optimisation(load="../data/structure-costs.npz", save=None).show()

    phi_tb1 = 3 * np.pi / 2 - np.linspace(np.pi, 0, 8)

    def responses(sun_ele=np.pi / 3, uniform=True, noise=.5, bl=.5):
        sun_azi = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        sun_ele = np.full_like(sun_azi, sun_ele)

        phi_maxs = [[]] * 8
        r_means = [[]] * 8
        r_stds = [[]] * 8
        p_values = [[]] * 8
        tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

        for n_tb1 in np.arange(8):
            tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

            for _ in np.linspace(0, 1, 100):
                d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_polariser=uniform,
                                                       sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=noise)
                tb1s = np.vstack([tb1s, np.transpose(r_tb1, axes=(1, 0, 2))])

            r_mean = np.median(tb1s[..., n_tb1], axis=0)
            z = r_mean.max() - r_mean.min()

            r_mean = (r_mean - r_mean.min()) / z - bl
            r_means[n_tb1] = r_mean
            r_stds[n_tb1] = tb1s[..., n_tb1].std(axis=0) / np.sqrt(z)

            p_values[n_tb1] = rayleightest(sun_azi, weights=r_mean + bl)
            phi_max = circmean(sun_azi, weights=np.power(r_mean + bl, 50))
            phi_maxs[n_tb1] = phi_max

        z = tb1s.max() - tb1s.min()
        tb1s = (tb1s - tb1s.min()) / z
        phis = np.transpose(np.array([[sun_azi] * 100] * 8), axes=(1, 2, 0))
        phi_means = circmean(phis, axis=1, weights=np.power(tb1s, 50)).T

        return np.array(phi_maxs)[::-1], phi_means[::-1], np.array(r_means)[::-1], np.array(r_stds)[::-1], np.array(
            p_values)[::-1]


    tcl_phi_max, _, _, _, _ = responses(uniform=True, noise=0.)
    _, tcl_phi_mean, tcl_r_mean, tcl_r_std, tcl_p_values = responses(uniform=True, noise=0.1)

    phi = np.linspace(np.deg2rad(90), np.deg2rad(270), 8)

    for i in range(len(phi)):
        d_000 = (tcl_phi_max[i] - phi[i] + np.pi) % (2 * np.pi) - np.pi
        d_180 = (tcl_phi_max[i] - phi[i]) % (2 * np.pi) - np.pi
        if np.absolute(d_000) < np.absolute(d_180):
            tcl_phi_max[i] = (tcl_phi_max[i] + np.pi) % (2 * np.pi) - np.pi
        else:
            tcl_phi_max[i] = tcl_phi_max[i] % (2 * np.pi) - np.pi
    #
    #     d_000 = (tcl_phi_mean[i] - tcl_phi_max[i] + np.pi) % (2 * np.pi) - np.pi
    #     d_180 = (tcl_phi_mean[i] - tcl_phi_max[i]) % (2 * np.pi) - np.pi
    #
    #     tcl_phi_mean[i][np.absolute(d_000) < np.absolute(d_180)] = \
    #         (tcl_phi_mean[i][np.absolute(d_000) < np.absolute(d_180)] + np.pi) % (2 * np.pi) - np.pi
    #
    #     tcl_phi_mean[i][np.absolute(d_000) >= np.absolute(d_180)] = \
    #         tcl_phi_mean[i][np.absolute(d_000) >= np.absolute(d_180)] % (2 * np.pi) - np.pi
    #
    #     j = np.random.permutation(np.arange(tcl_phi_mean[i].shape[0]))
    #     tcl_phi_mean[i][j[:10]] = tcl_phi_mean[i][j[:10]] % (2 * np.pi) - np.pi

    plt.figure("heinze-uni-fig-1F", figsize=(5, 5))
    plot_summary_response(tcl_phi_mean, tcl_phi_max, fit_line=False).show()

    # plt.figure("heinze-fig-1F", figsize=(5, 5))
    # tcl_sky_phi_max, _, _, _, _= responses(uniform=False, noise=0.)
    # _, tcl_sky_phi_mean, tcl_sky_r_mean, tcl_sky_r_std, tcl_sky_p_values = responses(uniform=False, noise=0.9)
    # plot_summary_response(tcl_sky_phi_mean, tcl_sky_phi_max, fit_line=True).show()

    # sigma = np.array([0.28, 2.32, 2.80, 3.67, 4.63, 5.15, 5.70, 6.58, 7.85, 7.86, 8.47,
    #                   9.01, 10.58, 11.44, 12.77, 14.01, 16.82, 19.71, 29.40, 46.86, 90])
    # tau = np.array([0.91, 0.86, 0.83, 0.79, 0.74, 0.69, 0.64, 0.58, 0.54, 0.51, 0.46,
    #                 0.41, 0.37, 0.34, 0.29, 0.24, 0.20, 0.17, 0.11, 0.07, 0.00])
    #
    # f_sigma = lambda x: 4./x - 2.
    # f_tau = lambda x: 1./8. * np.log(90./x)
    # eta = np.linspace(0, 1, sigma.size)
    # plt.plot(eta, sigma, 'C0-')
    # plt.plot(eta, tau, 'C1-')
    # plt.plot(eta, f_sigma(tau), 'C0--')
    # plt.plot(eta, f_tau(sigma), 'C1--')
    # plt.plot(sigma, tau)
    # plt.plot(sigma, f_tau(sigma))
    # plt.plot(f_sigma(tau), tau)
    # plt.ylim([0, 30])
    # plt.show()

