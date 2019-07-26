from compoundeye.geometry import angles_distribution, fibonacci_sphere
from compoundeye.evaluation import evaluate
from notebooks.results import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from astropy.stats import circmean, circvar, rayleightest

tb1_names = ['L8/R1', 'L7/R2', 'L6/R3', 'L5/R4', 'L4/R5', 'L3/R6', 'L2/R7', 'L1/R8']


def nb_neurons_test(save=None, mode=0, **kwargs):
    print "Running number of neurons test:", kwargs

    plt.figure("Structure", figsize=(10, 5))

    nb_tb1s = np.linspace(4, 360, 90)
    nb_cl1s = np.linspace(4, 360, 90)

    filename = "structure-costs.npz"
    if mode < 2:
        plt.subplot(121)
        means = np.zeros_like(nb_tb1s)
        ses = np.zeros_like(nb_tb1s)
        nb_tb1_default = kwargs.pop('nb_tb1', 8)
        nb_cl1_default = kwargs.pop('nb_cl1', 16)
        for ii, nb_tb1 in enumerate(nb_tb1s):
            d_err, d_eff, tau, _, _ = evaluate(nb_tb1=nb_tb1, nb_cl1=nb_cl1_default, verbose=False, **kwargs)
            means[ii] = (d_err * tau / tau.sum()).sum()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print 'TB1 = %03d, CL1 = %03d | Mean cost: %.2f +/- %.4f' % (nb_tb1, nb_cl1_default, means[ii], ses[ii])

        means = means.reshape(nb_tb1s.shape)

        plt.fill_between(nb_tb1s, means - ses, means + ses, facecolor="grey")
        plt.plot(nb_tb1s, means, color="black", label=r'$n$')
        plt.ylim([0, 20])
        plt.xlim([0, 360])
        plt.yticks([0, 5, 10, 15, 20], [r'%d$^\circ$' % o for o in [0, 5, 10, 15, 20]])
        plt.xticks([8, 24, 60, 90, 180, 270, 360])
        plt.xlabel(r'TB1 neurons')
        plt.ylabel(r'MSE ($^\circ$)')

        plt.subplot(122)
        means = np.zeros_like(nb_cl1s)
        ses = np.zeros_like(nb_cl1s)
        for ii, nb_cl1 in enumerate(nb_cl1s):
            d_err, d_eff, tau, _, _ = evaluate(nb_tb1=nb_tb1_default, nb_cl1=nb_cl1, verbose=False, **kwargs)
            means[ii] = (d_err * tau / tau.sum()).sum()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print 'TB1 = %03d, CL1 = %03d | Mean cost: %.2f +/- %.4f' % (nb_tb1_default, nb_cl1, means[ii], ses[ii])

        means = means.reshape(nb_cl1s.shape)

        plt.fill_between(nb_cl1s, means - ses, means + ses, facecolor="grey", alpha=.5)
        plt.plot(nb_cl1s, means, color="black", label=r'$\omega$')
        plt.ylim([0, 20])
        plt.xlim([0, 360])
        plt.yticks([0, 5, 10, 15, 20], [r'%d$^\circ$' % o for o in [0, 5, 10, 15, 20]])
        plt.xticks([8, 24, 60, 90, 180, 270, 360])
        plt.xlabel(r'CL1 neurons')
        plt.ylabel(r'MSE ($^\circ$)')
        # plt.legend()
    else:
        ax = plt.subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if mode > 2:
            data = np.load(filename)
            nb_tb1s, nb_cl1s, means = data["nb_tb1"], data["nb_cl1"], data["costs"]
        else:
            nb_tb1s, nb_cl1s = np.meshgrid(nb_tb1s, nb_cl1s)
            means = np.zeros(nb_cl1s.size)
            for ii, nb_tb1, nb_cl1 in zip(np.arange(nb_cl1s.size), nb_tb1s.flatten(), nb_cl1s.flatten()):
                d_err, d_eff, tau, _, _ = evaluate(nb_tb1=nb_tb1, nb_cl1=nb_cl1, verbose=False, **kwargs)
                means[ii] = (d_err * tau / tau.sum()).sum()
                se = d_err.std() / np.sqrt(d_err.size)
                print 'TB1 = %03d, CL1 = %03d | Mean cost: %.2f +/- %.4f' % (nb_tb1, nb_cl1, means[ii], se)

            means = means.reshape(nb_cl1s.shape)
            np.savez_compressed(filename, nb_tb1=nb_tb1s, nb_cl1=nb_cl1s, costs=means)

        ii = np.nanargmin(means, axis=0)
        tb1_min = nb_tb1s[ii, np.arange(91)]
        cl1_min = nb_cl1s[ii, np.arange(91)]
        means_min = means[ii, np.arange(91)]
        print 'Minimum cost (%.2f) for %d TB1, CL1 = %d neurons' % (means_min[15], tb1_min[15], cl1_min[30])
        print 'Mean TB1 %.2f +/- %.4f' % (tb1_min.mean(), tb1_min.std() / np.sqrt(tb1_min.size))

        with plt.rc_context({'ytick.color': 'white'}):
            plt.pcolormesh(np.deg2rad(nb_tb1s), nb_cl1s, means, cmap="Reds", vmin=0, vmax=90)
            plt.plot(np.deg2rad(tb1_min), cl1_min, 'g-')
            plt.yticks([4, 12, 60, 112, 176, 272, 360], [""] * 7)
            plt.xticks(np.deg2rad([14, 30, 60, 90, 120, 150, 180]))
            plt.ylim([4, 360])
            plt.xlim([0, 180])
            ax.grid(alpha=0.2)

        if save:
            plt.savefig(save)

    plt.show()


def noise_test(save=None, mode=0, repeats=10, **kwargs):
    from sphere.transform import sph2vec

    print "Running noise test:", kwargs
    modes = ['uniform', 'corridor', 'canopy']
    print "Mode:", modes[mode]

    plt.figure("noise-%s" % modes[mode], figsize=(4, 3))
    n, omega = 60, 56
    etas = np.linspace(0, 1, 21)
    taus = np.zeros_like(etas)
    means = np.zeros_like(etas)
    ses = np.zeros_like(etas)
    data = []
    theta, phi, fit = angles_distribution(n, float(omega))
    for i in xrange(repeats):
        for ii, eta in enumerate(etas):
            noise = get_noise(theta, phi, eta, mode=modes[mode])
            d_err, d_eff, tau, _, _ = evaluate(
                n=n, omega=omega, noise=noise, verbose=False, tilting=True, **kwargs
            )
            data.append([tau, d_err])
            means[ii] = (means[ii] * i + d_err.mean()) / (i + 1)
            ses[ii] = (ses[ii] * i + d_err.std() / np.sqrt(d_err.size)) / (i + 1)
            taus[ii] = (taus[ii] * i + tau.mean()) / (i + 1)

            print "Noise level: %.2f (%03d) | Mean cost: %.2f +/- %.4f | tau: %.2f --> sigma: %2f" % (
                eta, np.sum(noise), means[ii], ses[ii], taus[ii],  6. / taus[ii] + 6)

        np.savez_compressed("../data/noise-%s.npz" % modes[mode], x=np.array(data)[:, 0], y=np.array(data)[:, 1])

    sigmas = 6. / taus + 6.
    plt.fill_between(etas * 100, means-ses, means+ses, facecolor="grey")
    plt.plot(etas * 100, means, color="red", linestyle="-", label="tilting")
    plt.plot(etas * 100, taus * 45, color="red", linestyle="--", label="tau-tilting")
    plt.plot(etas * 100, sigmas, color="red", linestyle="--", label="sigma-tilting")

    taus = np.zeros_like(etas)
    means = np.zeros_like(etas)
    ses = np.zeros_like(etas)
    for i in xrange(repeats):
        for ii, eta in enumerate(etas):
            noise = get_noise(theta, phi, eta, mode=modes[mode])
            d_err, d_eff, tau, _, _ = evaluate(
                n=n, omega=omega, noise=noise, verbose=False, tilting=False, **kwargs
            )
            data.append([tau, d_err])
            means[ii] = (means[ii] * i + d_err.mean()) / (i + 1)
            ses[ii] = (ses[ii] * i + d_err.std() / np.sqrt(d_err.size)) / (i + 1)
            taus[ii] = (taus[ii] * i + tau.mean()) / (i + 1)
            print "Noise level: %.2f (%03d) | Mean cost: %.2f +/- %.4f | tau: %.2f --> sigma: %2f" % (
                eta, np.sum(noise), means[ii], ses[ii], taus[ii],  4. / taus[ii] + 2)

        np.savez_compressed("../data/noise-%s.npz" % modes[mode], x=np.array(data)[:, 0], y=np.array(data)[:, 1])

    sigmas = 4. / taus - 2.
    plt.fill_between(etas * 100, means-ses, means+ses, facecolor="grey", alpha=.5)
    plt.plot(etas * 100, means, color="black", linestyle="-", label="plane")
    plt.plot(etas * 100, taus * 45, color="black", linestyle="--", label="tau-plane")
    plt.plot(etas * 100, sigmas, color="black", linestyle="--", label="sigma-plane")

    plt.ylim([0, 90])
    plt.yticks([0, 30, 60, 90], [r'%d$^\circ$' % o for o in [0, 30, 60, 90]])
    plt.xlim([0, 100])
    plt.xlabel(r'noise ($\eta$)')
    plt.ylabel("MAE ($^\circ$)")
    # plt.legend()

    if save:
        plt.savefig(save)

    plt.show()


def noise2disturbance_plot(n=60, samples=1000):

    etas = np.linspace(0, 2, 21)
    noise_lvl = np.zeros_like(etas)
    z = float(n) / 100.
    for ii, eta in enumerate(etas):
        for _ in xrange(samples):
            noise_lvl[ii] += np.sum(np.absolute(np.random.randn(n)) < eta) / z
        noise_lvl[ii] /= float(samples)
        print "Noise: %.2f -- %2.2f" % (eta, noise_lvl[ii])

    plt.figure("noise2level", figsize=(5, 2))
    plt.plot(etas, noise_lvl, color="black")

    plt.ylim([0, 100])
    plt.yticks([0, 25, 50, 75, 100])
    plt.xlim([0, 2])
    plt.xlabel(r'noise ($\eta$)')
    plt.ylabel(r'disturbance (%)')
    # plt.grid()

    plt.show()


def gate_test(save=None, mode=2, filename="gate-costs.npz", **kwargs):
    print "Running gating test:", kwargs

    plt.figure(filename[:-4], figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    sigmas = np.linspace(np.pi/90, np.pi/2, 45)
    shifts = np.linspace(0, 2*np.pi, 91)

    if mode < 2:
        means = np.zeros_like(sigmas)
        ses = np.zeros_like(sigmas)
        for ii, sigma in enumerate(sigmas):
            d_err, d_eff, _, _, _ = evaluate(sigma_pol=sigma, verbose=False, **kwargs)

            means[ii] = d_err.mean()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print "Sigma: %.2f | Mean cost: %.2f +/- %.4f" % (np.rad2deg(sigma), means[ii], ses[ii])

        plt.fill_between(sigmas, means - ses, means + ses, facecolor="C1", alpha=.5)
        plt.plot(sigmas, means, color="C1", label="sigma")

        means = np.zeros_like(shifts)
        ses = np.zeros_like(shifts)
        for ii, shift in enumerate(shifts):
            d_err, d_eff, _, _, _ = evaluate(shift_pol=shift, verbose=False, **kwargs)

            means[ii] = d_err.mean()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print "Shift: %.2f | Mean cost: %.2f +/- %.4f" % (np.rad2deg(shift), means[ii], ses[ii])

        plt.fill_between(shifts, means - ses, means + ses, facecolor="C2", alpha=.5)
        plt.plot(shifts, means, color="C2", label="shift")

        plt.ylim([0, 60])
        # plt.xlim([0, np.rad2deg(shifts.max())])
        plt.yticks([10, 30, 60])
        plt.xticks(np.linspace(-3 * np.pi / 4, 5 * np.pi / 4, 8, endpoint=False))
        # plt.xlabel(r'Variance ($\sigma$)')
        # plt.ylabel(r'MSE ($^\circ$)')
        plt.legend()
    else:
        if mode > 2:
            data = np.load(filename)
            shifts, sigmas, means = data["shifts"], data["sigmas"], data["costs"]
        else:
            # TODO parametrise this to work in batches so that I can run it in multiple processors
            sigmas_pol, shifts_pol, sigmas_sol, shifts_sol = np.meshgrid(sigmas, shifts, sigmas, shifts)
            means = np.zeros(sigmas_pol.size)
            for ii, sigma_pol, shift_pol, sigma_sol, shift_sol in zip(np.arange(sigmas_pol.size),
                                                                      sigmas_pol.flatten(), shifts_pol.flatten(),
                                                                      sigmas_sol.flatten(), shifts_sol.flatten()):
                d_err, d_eff, tau, _, _ = evaluate(sigma_pol=sigma_pol, shift_pol=shift_pol,
                                                   sigma_sol=sigma_sol, shift_sol=shift_sol, verbose=False, **kwargs)
                means[ii] = d_err.mean()
                se = np.rad2deg(d_err.std() / np.sqrt(d_err.size))
                print r'S_p = % 3.2f, T_p = % 3.2f, S_s = % 3.2f, T_s = % 3.2f | Mean cost: %.2f +/- %.4f' % (
                    np.rad2deg(sigma_pol), np.rad2deg(shift_pol),
                    np.rad2deg(sigma_sol), np.rad2deg(shift_sol), means[ii], se)

            means = means.reshape(shifts_pol.shape)
            np.savez_compressed(filename, shifts=shifts_pol, sigmas=sigmas_pol, costs=means)

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

    if save:
        plt.savefig(save)

    plt.show()


def structure_test(save=None, mode=0, **kwargs):
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

    plt.show()


def tilt_test(samples=500, **kwargs):
    d_err, d_eff, tau, _, _ = evaluate(samples=samples, tilting=True, **kwargs)

    tau = np.rad2deg(tau)
    d_mean = np.nanmean(d_err)
    tau_mean = np.nanmean(tau)
    d_se = d_err.std() / np.sqrt(d_err.size)
    print "Mean cost: %.2f +/- %.4f, Certainty: %.2f" % (d_mean, d_se, tau_mean)

    if samples == 1000:
        samples /= 2
    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    phi_s = phi_s[theta_s <= np.pi / 2]
    theta_s = theta_s[theta_s <= np.pi / 2]

    d_00 = d_err[:, 0]
    tau_00 = tau[:, 0]
    d_30 = np.nanmean(d_err[:, 1:9], axis=1)
    tau_30 = np.nanmean(tau[:, 1:9], axis=1)
    d_60 = np.nanmean(d_err[:, 9:], axis=1)
    tau_60 = np.nanmean(tau[:, 9:], axis=1)
    print "Mean cost (00): %.2f +/- %.4f, Certainty: %.2f" % (
        np.nanmean(d_00), np.nanstd(d_00) / d_00.size, np.nanmean(tau_00))
    print "Mean cost (30): %.2f +/- %.4f, Certainty: %.2f" % (
        np.nanmean(d_err[:, 1:9]), np.nanstd(d_err[:, 1:9]) / d_err[:, 1:9].size, np.nanmean(tau[:, 1:9]))
    print "Mean cost (60): %.2f +/- %.4f, Certainty: %.2f" % (
        np.nanmean(d_err[:, 9:]), np.nanstd(d_err[:, 9:]) / d_err[:, 9:].size, np.nanmean(tau[:, 9:]))

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

    # plt.figure("Tilts-taus", figsize=(10, 3))
    # for i, ang, dd in zip(range(3), [0, np.pi/6, np.pi/3], [tau_00, tau_30, tau_60]):
    #     ax = plt.subplot2grid((1, 10), (0, i * 3), colspan=3, polar=True)
    #     ax.set_theta_zero_location("N")
    #     ax.set_theta_direction(-1)
    #     plt.scatter(phi_s, np.rad2deg(theta_s), marker=".", c=dd, cmap="Reds", vmin=0, vmax=90)
    #     plt.scatter(np.pi, np.rad2deg(ang), marker="o", c="yellowgreen", edgecolors="black")
    #     plt.text(-np.deg2rad(50), 145, ["A", "B", "C"][i], fontsize=12)
    #     plt.axis("off")
    # plt.subplot2grid((3, 10), (1, 9))
    # plt.imshow(np.array([np.arange(0, np.pi / 2, np.pi / 180)] * 10).T, cmap="Reds")
    # plt.xticks([])
    # plt.yticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])

    plt.show()


def gate_ring(sigma=np.deg2rad(13), shift=np.deg2rad(40), theta_t=0., phi_t=0.):
    theta, phi = fibonacci_sphere(samples=1000, fov=161)
    d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
             np.cos(shift - theta) * np.sin(theta_t) *
             np.cos(phi - phi_t))
    gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)

    plt.figure("gate-ring", figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    ax.scatter(phi, theta, s=10, marker='o', c=gate, cmap="Reds", vmin=0, vmax=1)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([0, np.deg2rad(90)])
    plt.yticks([np.deg2rad(28)], [r'$28^\circ$'])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels([r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$135^\circ$',
                        r'$180^\circ$', r'$-135^\circ$', r'$-90^\circ$', r'$-45^\circ$'])
    # ax.set_title("POL Response")
    plt.show()


def heinze_experiment(n_tb1=0, eta=.0, sun_ele=np.pi/2, absolute=False, uniform=False):
    sun_azi = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    sun_ele = np.full_like(sun_azi, sun_ele)
    tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

    for _ in np.linspace(0, 1, 100):
        d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_polariser=uniform,
                                               sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=eta)
        tb1s = np.vstack([tb1s, np.transpose(r_tb1, axes=(1, 0, 2))])

    if absolute:
        tb1s = np.absolute(tb1s)
    bl = .5
    r_mean = np.median(tb1s[..., n_tb1], axis=0)
    z = r_mean.max() - r_mean.min()

    r_mean = (r_mean - r_mean.min()) / z - bl
    r_std = tb1s[..., n_tb1].std(axis=0) / np.sqrt(z)

    p_value = rayleightest(sun_azi, weights=r_mean + bl)
    if uniform:
        # phi_mean_00 = circmean((sun_azi - np.pi / 2) % np.pi + np.pi / 2, weights=np.power(r_mean + bl, 8))
        # phi_var_00 = circvar((sun_azi - np.pi / 2) % np.pi + np.pi / 2, weights=np.power(r_mean + bl, 8))
        # phi_mean_90 = circmean(sun_azi % np.pi, weights=np.power(r_mean + bl, 8))
        # phi_var_90 = circvar(sun_azi % np.pi, weights=np.power(r_mean + bl, 8))
        # phi_mean = phi_mean_00 if phi_var_00 < phi_var_90 else phi_mean_90
        phi_mean = circmean(sun_azi, weights=np.power(r_mean + bl, 50))
    else:
        phi_mean = circmean(sun_azi, weights=np.power(r_mean + bl, 50))
    # phi_max[j].append(phi_mean)

    plt.figure("heinze-%s%s" % ("abs-" if absolute else "uni-" if uniform else "", tb1_names[n_tb1]), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    y_min, y_max = -.3, 1.1
    plt.bar((sun_azi + np.pi) % (2 * np.pi) - np.pi, bl + r_mean, .1, yerr=r_std, facecolor='black')
    plt.plot(np.linspace(-np.pi, np.pi, 361), np.full(361, bl), 'k-')
    if uniform:
        x_mean = [phi_mean, phi_mean, phi_mean + np.pi, phi_mean + np.pi]
        plt.plot(x_mean, [y_max, y_min, y_min, y_max], 'r-.')
    else:
        plt.plot([phi_mean, phi_mean], [y_max, y_min], 'r-.')
    plt.yticks([])
    plt.xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False),
               [r'%d$^\circ$' % x for x in ((np.linspace(0, 360, 8, endpoint=False) + 180) % 360 - 180)])
    plt.ylim([y_min, y_max])
    # plt.savefig("heinze-%s%d.eps" % ("abs-" if absolute else "uni-" if uniform else "", n_tb1))
    plt.show()


def heinze_1f(eta=.5, uniform=False):
    from astropy.stats import circmean

    phi_tb1 = 3 * np.pi / 2 - np.linspace(np.pi, 0, 8)
    # phi_tb1 = np.pi / 2 + np.linspace(0, np.pi, 8)

    sun_azi = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    sun_ele = np.full_like(sun_azi, np.pi/2)
    # phi_tb1 = np.linspace(0., 2 * np.pi, 8, endpoint=False)  # TB1 preference angles
    tb1_ids = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)
    tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

    for _ in np.linspace(0, 1, 100):
        d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_polariser=uniform,
                                               sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=eta)
        tb1s = np.vstack([tb1s, np.transpose(r_tb1, axes=(1, 0, 2))])
        tb1_ids = np.vstack([tb1_ids, np.array([sun_azi] * 8).T.reshape((1, 36, 8))])
    z = tb1s.max() - tb1s.min()
    tb1s = (tb1s - tb1s.min()) / z

    phis = np.transpose(np.array([[sun_azi] * 100] * 8), axes=(1, 2, 0))
    d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_polariser=uniform,
                                           sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=0.)

    tb1 = np.transpose(r_tb1, axes=(1, 0, 2)) * 1e+15
    if uniform:
        phi_mean = circmean(phis, axis=1, weights=np.power(tb1s, 50))

        for i in xrange(phi_mean.shape[1]):
            d_00 = np.absolute((phi_mean[:, i] - phi_tb1[-1 - i] + np.pi) % (2 * np.pi) - np.pi)
            d_pi = np.absolute((phi_mean[:, i] - phi_tb1[-1 - i]) % (2 * np.pi) - np.pi)
            phi_mean[d_00 > d_pi, i] += np.pi

        phi_max = circmean(phis[0][np.newaxis], axis=1, weights=np.power(tb1, 50)).flatten()
        for i, phi_max_i in enumerate(phi_max):
            d_00 = np.absolute((phi_max_i - phi_tb1[-1 - i] + np.pi) % (2 * np.pi) - np.pi)
            d_pi = np.absolute((phi_max_i - phi_tb1[-1 - i]) % (2 * np.pi) - np.pi)
            if d_00 > d_pi:
                phi_max[i] += np.pi
    else:
        phi_mean = circmean(phis, axis=1, weights=np.power(tb1s, 50))
        phi_max = circmean(phi_mean, axis=0)

    x, y = [], []
    for i, phi_max_i in enumerate(phi_max):
        x.append([(phi_max[i] + np.pi/18) % (2 * np.pi) - np.pi/18, 1])
        y.append(i)

    # for i, phi_mean_i in enumerate(phi_mean.T):
    #     for phi_mean_j in phi_mean_i:
    #         x.append([(phi_mean_j + np.pi/18) % (2 * np.pi) - np.pi/18, 1])
    #         y.append(i)

    x = np.array(x)
    y = np.array(y)
    a, b = np.linalg.pinv(x).dot(y)

    plt.figure("heinze-%sfig-1F" % ("uni-" if uniform else ""), figsize=(5, 5))
    # phi = circmean(tb1_ids, weights=tb1s, axis=1)
    plt.scatter([0, 1, 2, 3, 4, 5, 6, 7][::-1] * 100, np.rad2deg(phi_mean) % 360, s=20, c='black')
    plt.scatter([0, 1, 2, 3, 4, 5, 6, 7][::-1], np.rad2deg(phi_max) % 360, s=50, c='red', marker='*')
    plt.plot([-1, 8][::-1], np.rad2deg([(-1 - b) / a, (8 - b) / a]), 'r-.')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
               [tb1_names[0], '', tb1_names[1], '', tb1_names[2], '', tb1_names[3], ''])
    plt.yticks([0, 45, 90, 135, 180, 225, 270, 315, 360],
               ['0', '', '90', '', '180', '', '270', '', '360'])
    plt.ylim([-20, 380])
    plt.xlim([-1, 8])
    plt.show()


def heinze_real(mode=1, n_tb1=0):

    columns = [
        ['fz1028'],  # L8/R1
        ['060126', '060131', '050105a', 'fz1049'],  # L7/R2
        ['050329', '050309a', '050124b', '041215', 'fz1020', 'fz1038'],  # L6/R3
        ['060124', '060206a', '060206b', 'fz1016'],  # L5/R4
        ['050520', '040604b', 'fz1040'],  # L4/R5
        [],  # L3/R6
        ['050309b', '050222'],  # L2/R7
        ['041209', 'fz1051'],  # L1/R8
    ]

    phi_tb1 = 3*np.pi/2 - np.linspace(0, np.pi, 8)
    # phi_tb1 = np.linspace(0, np.pi, 8) + np.pi/2
    phi = np.linspace(np.deg2rad(5), np.deg2rad(355), 36)
    phi_max = []

    for j, filenames in enumerate(columns if n_tb1 is None else [columns[n_tb1]]):
        col = j if n_tb1 is None else n_tb1
        phi_max.append([])

        for i, filename in enumerate(filenames):
            if 'fz' in filename:
                continue
            tb1s = loadmat("../data/TB1_neurons/mean_rotation_%s.mat" % filename)['mean_rotation'][:, ::2]
            tb1s = tb1s.reshape((-1, 2)).mean(axis=1).reshape((1, -1))

            z = tb1s.max() - tb1s.min()
            r_std = tb1s.std(axis=0) / np.sqrt(z)
            bl = .5
            r_mean = tb1s.flatten() / tb1s.max() - bl
            p_value = rayleightest(phi, weights=r_mean + bl)
            phi_mean_00 = circmean((phi - np.pi /2) % np.pi + np.pi/2, weights=np.power(r_mean + bl, 8))
            phi_var_00 = circvar((phi - np.pi /2) % np.pi + np.pi/2, weights=np.power(r_mean + bl, 8))
            phi_mean_90 = circmean(phi % np.pi, weights=np.power(r_mean + bl, 8))
            phi_var_90 = circvar(phi % np.pi, weights=np.power(r_mean + bl, 8))
            phi_mean = phi_mean_00 if phi_var_00 < phi_var_90 else phi_mean_90
            d_00 = np.absolute((phi_mean - phi_tb1[-1 - i] + np.pi) % (2 * np.pi) - np.pi)
            d_pi = np.absolute((phi_mean - phi_tb1[-1 - i]) % (2 * np.pi) - np.pi)
            if d_00 > d_pi:
                phi_mean += np.pi
            phi_max[j].append(phi_mean)
            print "Col %d - %s, mean: % 3.2f, p = %.4f" % (col, filename, np.rad2deg(phi_mean), p_value)

            if mode == 1:
                plt.figure("heinze-L%d-R%d-%s" % (8 - col, col + 1, filename), figsize=(3, 3))
                ax = plt.subplot(111, polar=True)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)

                y_min, y_max = -.3, 1.1
                plt.bar(phi, bl + r_mean, .1, yerr=r_std, facecolor='black')
                plt.plot(np.linspace(-np.pi, np.pi, 361), np.full(361, bl), 'k-')
                x_mean = [phi_mean, phi_mean, phi_mean + np.pi, phi_mean + np.pi]
                plt.plot(x_mean, [y_max, y_min, y_min, y_max], 'r-.')
                plt.yticks([])
                plt.xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False),
                           [r'%d$^\circ$' % x for x in ((np.linspace(0, 360, 8, endpoint=False) + 180) % 360 - 180)])
                plt.ylim([y_min, y_max])
                # plt.savefig("heinze-%s%d.eps" % ("abs-" if absolute else "uni-" if uniform else "", n_tb1))
                plt.show()

    if mode == 2:
        plt.figure("heinze-real-fig-1F", figsize=(5, 5))
        tb1_names = []
        x, y = [], []
        for i, phi_max_i in enumerate(phi_max):
            col = i if n_tb1 is None else n_tb1
            for j, phi_max_j in enumerate(phi_max_i):
                d_00 = np.absolute((phi_max_j - phi_tb1[-1-i] + np.pi) % (2 * np.pi) - np.pi)
                d_pi = np.absolute((phi_max_j - phi_tb1[-1-i]) % (2 * np.pi) - np.pi)
                if d_00 > d_pi:
                    phi_max_i[j] += np.pi

            phi_mean_i = circmean(np.array(phi_max_i)) % (2 * np.pi)
            if not np.isnan(phi_mean_i):
                x.append([phi_mean_i, 1])
                y.append(col)
            tb1_names.append('L%d/R%d' % (8 - col, col + 1))
            plt.scatter([col] * len(phi_max_i), np.rad2deg(phi_max_i) % 360, s=20, c='black')
            plt.scatter(col, np.rad2deg(phi_mean_i) % 360, s=50, c='red', marker='*')
        x = np.array(x)
        y = np.array(y)
        a, b = np.linalg.pinv(x).dot(y)
        plt.plot([-1, 8], np.rad2deg([(-1 - b) / a, (8 - b) / a]), 'r-.')
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
                   [tb1_names[0], '', tb1_names[2], '', tb1_names[4], '', tb1_names[6], ''])
        plt.yticks([0, 45, 90, 135, 180, 225, 270, 315, 360],
                   ['0', '', '90', '', '180', '', '270', '', '360'])
        plt.ylim([-20, 380])
        plt.xlim([-1, 8])
        plt.show()


def tilt_ephem_test(**kwargs):
    print "Running simple tilt and ephemeris test:", kwargs
    kwargs['samples'] = kwargs.get('samples', 1)
    kwargs['show_plots'] = kwargs.get('show_plots', True)
    kwargs['show_structure'] = kwargs.get('show_structure', False)
    kwargs['sun_azi'] = kwargs.get('sun_azi', -np.pi/3)
    kwargs['sun_ele'] = kwargs.get('sun_ele', np.pi/3)
    # kwargs['tilting'] = kwargs.get('tilting', False)
    kwargs['tilting'] = kwargs.get('tilting', (np.pi/9, np.pi/2 + np.pi/3))
    kwargs['ephemeris'] = kwargs.get('ephemeris', True)

    d_err, d_eff, t, _, _ = evaluate(**kwargs)
    d_mean = np.nanmean(d_err)
    d_se = np.nanstd(d_err) / np.sqrt(d_err.size)
    print "Mean cost: %.2f +/- %.4f -- Certainty: %.2f" % (d_mean, d_se, np.nanmean(np.rad2deg(t)))


def one_test(**kwargs):
    print "Running single test:", kwargs

    d_err, d_eff, t, _, _ = evaluate(**kwargs)
    d_mean = np.nanmean(d_err)
    d_se = np.nanstd(d_err) / np.sqrt(d_err.size)
    print "Mean cost: %.2f +/- %.4f -- Certainty: %.2f" % (d_mean, d_se, np.nanmean(np.rad2deg(t)))


def elevation_test(**kwargs):
    print "Running elevation test:", kwargs

    sun_ele = np.linspace(0, np.pi/2, 91)
    sun_azi = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    sun_ele = kwargs.get('sun_ele', sun_ele)
    d_mean = np.zeros_like(sun_ele)
    d_se = np.zeros_like(sun_ele)
    tau = np.zeros_like(sun_ele)
    kwargs['sun_azi'] = kwargs.get('sun_azi', sun_azi)
    kwargs['tilting'] = kwargs.get('tilting', False)
    kwargs['weighted'] = kwargs.get('weighted', True)

    plt.figure("elevation", figsize=(4.5, 3))
    for j, noise in enumerate(np.linspace(0, 2, 5)):
        # plt.figure("elevation-%02d" % (10 * noise), figsize=(5, 2))
        # kwargs['noise'] = kwargs.get('noise', noise)
        kwargs['noise'] = noise

        d = np.zeros((sun_azi.shape[0], sun_ele.shape[0]))
        for i, theta_s in enumerate(sun_ele):
            kwargs['sun_ele'] = np.full_like(sun_azi, theta_s)
            d_err, d_eff, t, a_ret, tb1 = evaluate(**kwargs)
            d_mean[i] = np.nanmean(d_err)
            d_se[i] = np.nanstd(d_err) / np.sqrt(np.sum(~np.isnan(d_err)))
            tau[i] += np.rad2deg(np.nanmean(t))
            print "Mean cost: %.2f +/- %.4f -- Certainty: %.2f -- ele: %.2f" % (d_mean[i], d_se[i], tau[i], np.rad2deg(theta_s))
        plt.fill_between(np.rad2deg(sun_ele), d_mean - d_se, d_mean + d_se, facecolor='C%d' % j, alpha=.5)
        # plt.semilogy(np.rad2deg(sun_ele), d_mean, color='C%d' % j, label=r'$\eta = %.1f$' % noise)
        plt.plot(np.rad2deg(sun_ele), d_mean, color='C%d' % j, label=r'$\eta = %.1f$' % noise)
    tau /= 5.
    plt.plot(np.rad2deg(sun_ele), tau, 'k--')
    plt.legend()
    # plt.yticks([0.01, 0.1, 1, 10, 100])
    plt.yticks([0, 30, 60, 90])
    # plt.ylim([0.001, 120])
    plt.ylim([0, 90])
    plt.xticks([0, 30, 60, 90])
    plt.xlim([0, 90])
    plt.ylabel(r'MSE [$\circ$]')
    plt.xlabel(r'sun elevation [$\circ$]')

    # kwargs.pop('sun_azi')
    # kwargs.pop('sun_ele')
    # noises = np.linspace(0, 2, 100)
    # d_mean = np.zeros_like(noises)
    # d_se = np.zeros_like(noises)
    # tau = np.zeros_like(noises)
    #
    # plt.figure("noise", figsize=(3, 3))
    # for j, noise in enumerate(noises):
    #     kwargs['noise'] = noise
    #     d_err, d_eff, t, a_ret, tb1 = evaluate(**kwargs)
    #     d_mean[j] = np.nanmean(d_err)
    #     d_se[j] = np.nanstd(d_err) / np.sqrt(np.sum(~np.isnan(d_err)))
    #     tau[j] = np.rad2deg(np.nanmean(t))
    #     print "Mean cost: %.2f +/- %.4f -- Certainty: %.2f -- noise: %.2f" % (d_mean[j], d_se[j], tau[j], noise)
    # plt.fill_between(noises, d_mean - d_se, d_mean + d_se, facecolor='black', alpha=.5)
    # # plt.semilogy(noises, d_mean, 'k-%s' % ('' if weighted else '-'),
    # #              label=r'%s' % ('weighted' if weighted else ' simple'))
    # plt.plot(noises, d_mean, 'k-', label=r'$E[J]$')
    # plt.plot(noises, tau, 'k--', label=r'$E[\tau]$')
    #
    # plt.legend()
    # # plt.yticks([0.01, 0.1, 1, 10, 100])
    # plt.yticks([0, 30, 60, 90])
    # # plt.ylim([0.001, 120])
    # plt.ylim([0, 90])
    # plt.xticks([0, 0.5, 1, 1.5, 2])
    # plt.xlim([0, 2])
    # plt.xlabel(r'noise ($\eta$)')
    plt.show()


if __name__ == "__main__":
    noise_test(mode=0, repeats=100)
    # nb_neurons_test(mode=2, tilting=True, weighted=False, noise=.0)
    # gate_ring(sigma=np.deg2rad(13), shift=np.deg2rad(40))
    # noise2disturbance_plot()
    # gate_test(tilting=True, mode=3, filename="gate-costs.npz")
    # tilt_test(weighted=True, use_default=False)
    # tilt_ephem_test()
    # structure_test(tilting=True, mode=3, n=60, omega=56, weighted=True)
    # for n_tb1 in xrange(8):
    #     heinze_experiment(n_tb1=n_tb1, sun_ele=np.deg2rad(91), absolute=False, uniform=False)
    # heinze_1f(eta=.5, uniform=True)
    # heinze_real(mode=2, n_tb1=None)
    # one_test(n=60, omega=56, sigma_pol=np.deg2rad(13), shift_pol=np.deg2rad(40), use_default=False, weighted=True,
    #          show_plots=True, show_structure=False, verbose=True, samples=1, tilting=False, noise=.0)
    # elevation_test()
