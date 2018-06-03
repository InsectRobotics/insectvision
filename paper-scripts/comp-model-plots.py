from compoundeye.geometry import angles_distribution, fibonacci_sphere
from sphere import azidist
from sphere.transform import tilt
from learn.loss_function import SensorObjective

import matplotlib.pyplot as plt
import numpy as np


def evaluate(n=60, omega=60,
             noise=0.,
             nb_cl1=16, sigma=np.deg2rad(13), shift=np.deg2rad(40),
             nb_tb1=8,
             use_default=False,

             # data parameters
             tilting=False, samples=1000, show_plots=False, verbose=False):

    # default parameters
    tau_L = 2.
    c1 = .6
    c2 = 4.
    eps = np.finfo(float).eps
    AA, BB, CC, DD, EE = SensorObjective.T_L.dot(np.array([tau_L, 1.]))  # sky parameters
    T_T = np.linalg.pinv(SensorObjective.T_L)
    tau_L, c = T_T.dot(np.array([AA, BB, CC, DD, EE]))
    tau_L /= c  # turbidity correction

    # Prez. et. al. Luminance function
    def L(cchi, zz):
        ii = zz < (np.pi/2)
        ff = np.zeros_like(zz)
        if zz.ndim > 0:
            ff[ii] = (1. + AA * np.exp(BB / (np.cos(zz[ii]) + eps)))
        elif ii:
            ff = (1. + AA * np.exp(BB / (np.cos(zz) + eps)))
        pphi = (1. + CC * np.exp(DD * cchi) + EE * np.square(np.cos(cchi)))
        return ff * pphi

    if tilting:
        angles = np.array([
            [0., 0.],
            [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
            [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4],
            [np.pi / 6, 7 * np.pi / 4],
            [np.pi / 3, 0.], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
            [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4],
            [np.pi / 3, 7 * np.pi / 4]
        ])  # 17
        if samples == 1000:
            samples /= 2
    else:
        angles = np.array([[0., 0.]])  # 1

    # generate the different sun positions
    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    phi_s = phi_s[theta_s <= np.pi / 2]
    theta_s = theta_s[theta_s <= np.pi / 2]
    samples = theta_s.size

    # generate the properties of the sensor
    theta, phi, _ = angles_distribution(n, omega)
    alpha = (phi + np.pi/2) % (2 * np.pi) - np.pi

    # computational model parameters
    phi_cl1 = np.linspace(0., 4 * np.pi, nb_cl1, endpoint=False)  # CL1 preference angles
    phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles

    # initialise lists for the statistical data
    d = np.zeros((samples, angles.shape[0]))
    d_eff = np.zeros((samples, angles.shape[0]))

    # iterate through the different tilting angles
    for j, (theta_t, phi_t) in enumerate(angles):
        # transform relative coordinates
        theta_s_, phi_s_ = tilt(theta_t, phi_t, theta=theta_s, phi=phi_s)
        theta_, phi_ = tilt(theta_t, phi_t + np.pi, theta=theta, phi=phi)
        _, alpha_ = tilt(theta_t, phi_t + np.pi, theta=np.pi / 2, phi=alpha)

        for i, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):

            # SKY INTEGRATION
            gamma = np.arccos(np.cos(theta_) * np.cos(e_org) + np.sin(theta_) * np.sin(e_org) * np.cos(phi_ - a_org))
            # Intensity
            I_prez, I_00, I_90 = L(gamma, theta_), L(0., e_org), L(np.pi / 2, np.absolute(e_org - np.pi / 2))
            # influence of sky intensity
            I = (1. / (I_prez + eps) - 1. / (I_00 + eps)) * I_00 * I_90 / (I_00 - I_90 + eps)
            chi = (4. / 9. - tau_L / 120.) * (np.pi - 2 * e_org)
            Y_z = (4.0453 * tau_L - 4.9710) * np.tan(chi) - 0.2155 * tau_L + 2.4192
            Y = np.maximum(Y_z * I_prez / (I_00 + eps), 0.)  # Illumination

            # Degree of Polarisation
            M_p = np.exp(-(tau_L - c1) / (c2 + eps))
            LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
            P = np.clip(2. / np.pi * M_p * LP * (theta_ * np.cos(theta_) + (np.pi/2 - theta_) * I), 0., 1.)
            # P = LP

            # Angle of polarisation
            _, A = tilt(e_org, a_org + np.pi, theta_, phi_)

            # create cloud disturbance
            if noise > 0:
                eta = np.absolute(np.random.randn(*P.shape)) < noise
                if verbose:
                    print "Noise level: %.4f (%.2f %%)" % (noise, 100. * eta.sum() / float(eta.size))
                P[eta] = 0.  # destroy the polarisation pattern
            else:
                eta = np.zeros(1)

            # COMPUTATIONAL MODEL

            # Input (POL) layer -- Photo-receptors
            s_1 = 15. * (np.square(np.sin(A - alpha_)) + np.square(np.cos(A - alpha_)) * np.square(1. - P))
            s_2 = 15. * (np.square(np.cos(A - alpha_)) + np.square(np.sin(A - alpha_)) * np.square(1. - P))
            r_1, r_2 = np.sqrt(s_1), np.sqrt(s_2)
            # r_1, r_2 = np.log(s_1 + 1.), np.log(s_2 + 1.)
            r_pol = (r_1 - r_2) / (r_1 + r_2 + eps)

            # Tilting (CL1) layer
            d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
                     np.cos(shift - theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)
            w_cl1 = float(nb_cl1) / float(n) * np.sin(alpha[:, np.newaxis] - phi_cl1[np.newaxis]) * gate[:,
                                                                                                          np.newaxis]
            r_cl1 = r_pol.dot(w_cl1)

            # Output (TB1) layer
            w_tb1 = float(nb_tb1) / float(2 * nb_cl1) * np.cos(phi_tb1[np.newaxis] - phi_cl1[:, np.newaxis])

            r_tb1 = r_cl1.dot(w_tb1)

            if use_default:
                w = -float(nb_tb1) / (2. * float(n)) * np.sin(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * gate[:,
                                                                                                            np.newaxis]
                r_tb1 = r_pol.dot(w)

            # decode response - FFT
            R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * np.pi / 4.))
            e_pred = np.absolute(R) % (2. * np.pi)  # sun elevation (prediction)
            a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)

            d[i, j] = np.absolute(azidist(np.array([e, a]), np.array([e_pred, a_pred])))

            # effective degree of polarisation
            M = r_cl1.max() - r_cl1.min()
            p = np.power(10, M/2.)
            d_eff[i, j] = np.mean((p - 1.) / (p + 1.))

            if show_plots:
                plt.figure("sensor-noise-%.2f" % (100. * eta.sum() / float(eta.size)), figsize=(18, 4.5))

                ax = plt.subplot(1, 12, 10)
                plt.imshow(w_cl1, cmap="coolwarm", vmin=-1, vmax=1)
                plt.xlabel("CL1")
                plt.xticks([0, 15], ["1", "16"])
                plt.yticks([0, 59], ["1", "60"])

                ax = plt.subplot(1, 6, 6)  # , sharey=ax)
                plt.imshow(w_tb1, cmap="coolwarm", vmin=-1, vmax=1)
                plt.xlabel("TB1")
                plt.xticks([0, 7], ["1", "8"])
                plt.yticks([0, 15], ["1", "16"])
                cbar = plt.colorbar(ticks=[-1, 0, 1])
                cbar.ax.set_yticklabels([r'$\leq$ -1', r'0', r'$\geq$ 1'])

                ax = plt.subplot(1, 4, 1, polar=True)
                ax.scatter(phi, theta, s=150, c=r_pol, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
                ax.scatter(a, e, s=100, marker='o', edgecolor='black', facecolor='yellow')
                ax.scatter(phi_t + np.pi, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.deg2rad(50)])
                ax.set_yticks([])
                ax.set_xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
                ax.set_title("POL Response")

                ax = plt.subplot(1, 4, 2, polar=True)
                ax.scatter(phi, theta, s=150, c=r_pol * gate, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
                ax.scatter(a, e, s=100, marker='o', edgecolor='black', facecolor='yellow')
                ax.scatter(phi_t + np.pi, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.deg2rad(50)])
                ax.set_yticks([])
                ax.set_xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
                ax.set_title("Gated Response")

                ax = plt.subplot(1, 4, 3, polar=True)
                x = np.linspace(0, 2 * np.pi, 721)

                # CL1
                ax.fill_between(x, np.full_like(x, np.deg2rad(60)), np.full_like(x, np.deg2rad(90)),
                                facecolor="C1", alpha=.5, label="CL1")
                ax.scatter(phi_cl1[:nb_cl1/2] - np.pi/24, np.full(nb_cl1/2, np.deg2rad(75)), s=600,
                           c=r_cl1[:nb_cl1/2], marker='o', edgecolor='red', cmap="coolwarm", vmin=-1, vmax=1)
                ax.scatter(phi_cl1[nb_cl1/2:] + np.pi/24, np.full(nb_cl1/2, np.deg2rad(75)), s=600,
                           c=r_cl1[nb_cl1/2:], marker='o', edgecolor='green', cmap="coolwarm", vmin=-1, vmax=1)

                for ii, pp in enumerate(phi_cl1[:nb_cl1/2] - np.pi/24):
                    ax.text(pp - np.pi/20, np.deg2rad(75), "%d" % (ii + 1), ha="center", va="center", size=10,
                            bbox=dict(boxstyle="circle", fc="w", ec="k"))
                for ii, pp in enumerate(phi_cl1[nb_cl1/2:] + np.pi/24):
                    ax.text(pp + np.pi/20, np.deg2rad(75), "%d" % (ii + 9), ha="center", va="center", size=10,
                            bbox=dict(boxstyle="circle", fc="w", ec="k"))
                # TB1
                ax.fill_between(x, np.full_like(x, np.deg2rad(30)), np.full_like(x, np.deg2rad(60)),
                                facecolor="C2", alpha=.5, label="TB1")
                ax.scatter(phi_tb1, np.full_like(phi_tb1, np.deg2rad(45)), s=600,
                           c=r_tb1, marker='o', edgecolor='blue', cmap="coolwarm", vmin=-1, vmax=1)
                for ii, pp in enumerate(phi_tb1):
                    ax.text(pp, np.deg2rad(35), "%d" % (ii + 1), ha="center", va="center", size=10,
                            bbox=dict(boxstyle="circle", fc="w", ec="k"))
                    ax.arrow(pp, np.deg2rad(35), 0, np.deg2rad(10), fc='k', ec='k', head_width=.1, overhang=.3)

                # Sun position
                ax.scatter(a, e, s=500, marker='o', edgecolor='black', facecolor='yellow')

                # Decoded TB1
                # ax.plot([0, a_pred], [0, e_pred], 'k--', lw=1)
                ax.plot([0, a_pred], [0, np.pi/2], 'k--', lw=1)
                ax.arrow(a_pred, 0, 0, np.deg2rad(20),
                         fc='k', ec='k', head_width=.3, head_length=.2, overhang=.3)

                ax.legend(ncol=2, loc=(-.35, -.1))
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.pi/2])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Sensor Response")

                plt.subplots_adjust(left=.02, bottom=.12, right=.98, top=.88)

                plt.show()

    d_deg = np.rad2deg(d)
    return d_deg, d_eff


def noise_test(save=None, **kwargs):
    print "Running noise test:", kwargs

    plt.figure("Noise", figsize=(5, 5))

    etas = np.linspace(0, 2, 21)
    means = np.zeros_like(etas)
    ses = np.zeros_like(etas)
    for ii, eta in enumerate(etas):
        d_err, d_eff = evaluate(
            noise=eta, verbose=False, sigma=np.deg2rad(13), **kwargs
        )

        means[ii] = d_err.mean()
        ses[ii] = d_err.std() / np.sqrt(d_err.size)
        print "Noise level: %.2f | Mean cost: %.2f +/- %.4f" % (eta, means[ii], ses[ii])

    plt.fill_between(etas, means-ses, means+ses, color="C1", alpha=.5)
    plt.plot(etas, means, color="C1", label="gate")

    for ii, eta in enumerate(etas):
        d_err, d_eff = evaluate(
            noise=eta, verbose=False, sigma=np.deg2rad(360), **kwargs
        )

        means[ii] = d_err.mean()
        ses[ii] = d_err.std() / np.sqrt(d_err.size)
        print "Noise level: %.2f | Mean cost: %.2f +/- %.4f" % (eta, means[ii], ses[ii])

    plt.fill_between(etas, means-ses, means+ses, color="C2", alpha=.5)
    plt.plot(etas, means, color="C2", label="smooth")

    plt.ylim([0, 90])
    plt.xlim([0, 2])
    plt.xlabel(r'noise ($\eta$)')
    plt.ylabel("MSE ($^\circ$)")
    plt.legend()

    if save:
        plt.savefig(save)

    plt.show()


def gate_test(save=None, mode=2, **kwargs):
    print "Running gating test:", kwargs

    plt.figure("Gating", figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    sigmas = np.linspace(0, np.pi/2, 91)
    shifts = np.linspace(0, 2*np.pi, 361)

    if mode < 2:
        means = np.zeros_like(sigmas)
        ses = np.zeros_like(sigmas)
        for ii, sigma in enumerate(sigmas):
            d_err, d_eff = evaluate(sigma=sigma, verbose=False, **kwargs)

            means[ii] = d_err.mean()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print "Sigma: %.2f | Mean cost: %.2f +/- %.4f" % (np.rad2deg(sigma), means[ii], ses[ii])

        plt.fill_between(sigmas, means - ses, means + ses, facecolor="C1", alpha=.5)
        plt.plot(sigmas, means, color="C1", label="sigma")

        means = np.zeros_like(shifts)
        ses = np.zeros_like(shifts)
        for ii, shift in enumerate(shifts):
            d_err, d_eff = evaluate(shift=shift, verbose=False, **kwargs)

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
        sigmas, shifts = np.meshgrid(sigmas, shifts)
        means = np.zeros(sigmas.size)
        for ii, sigma, shift in zip(np.arange(sigmas.size), sigmas.flatten(), shifts.flatten()):
            d_err, d_eff = evaluate(sigma=sigma, shift=shift, verbose=False, **kwargs)
            means[ii] = d_err.mean()
            se = d_err.std() / np.sqrt(d_err.size)
            print 'Sigma = %.2f, Shift = %.2f | Mean cost: %.2f +/- %.4f' % (
                np.rad2deg(sigma), np.rad2deg(shift), means[ii], se)

        ii = np.argmin(means)
        sigma_min = sigmas.flatten()[ii]
        shift_min = shifts.flatten()[ii]
        means_min = means[ii]
        print 'Minimum cost (%.2f) for Sigma = %.2f, Shift = %.2f' % (
            means_min, np.rad2deg(sigma_min), np.rad2deg(shift_min))

        with plt.rc_context({'ytick.color': 'white'}):
            plt.pcolormesh(shifts, sigmas, means.reshape(shifts.shape), cmap="Reds", vmin=0, vmax=90)
            plt.scatter(shift_min, sigma_min, s=20, c='yellowgreen', marker='o')
            plt.yticks([0, np.pi/6, np.pi/3, np.pi/2],
                       [r"$0$", r"$30^\circ$", r"$60^\circ$", r"$90^\circ$"])
            plt.xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
            plt.ylim([0, np.pi/2])
            ax.grid(alpha=0.2)

    if save:
        plt.savefig(save)

    plt.show()


def tilt_test(samples=1000, **kwargs):
    d_err, d_eff = evaluate(samples=samples, tilting=True, **kwargs)

    print "Mean cost: %.2f +/- %.4f" % (d_err.mean(), d_err.std() / np.sqrt(d_err.size))

    if samples == 1000:
        samples /= 2
    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    phi_s = phi_s[theta_s <= np.pi / 2]
    theta_s = theta_s[theta_s <= np.pi / 2]

    d_00 = d_err[:, 0]
    d_30 = d_err[:, 1:9].mean(axis=1)
    d_60 = d_err[:, 9:].mean(axis=1)
    print d_00.shape, d_30.shape, d_60.shape

    plt.figure("Tilts", figsize=(10, 3))
    for i, ang, dd in zip(range(3), [0, np.pi/6, np.pi/3], [d_00, d_30, d_60]):
        ax = plt.subplot2grid((1, 10), (0, i * 3), colspan=3, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.scatter(phi_s, np.rad2deg(theta_s), marker="o", c=dd, cmap="Reds", vmin=0, vmax=90)
        plt.scatter(np.pi, np.rad2deg(ang), marker="o", c="yellowgreen", edgecolors="black")
        plt.text(-np.deg2rad(50), 145, ["A", "B", "C"][i], fontsize=12)
        plt.axis("off")
    plt.subplot2grid((3, 10), (1, 9))
    plt.imshow(np.array([np.arange(0, np.pi / 2, np.pi / 180)] * 10).T, cmap="Reds")
    plt.xticks([])
    plt.yticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])
    plt.show()


def one_test(**kwargs):
    print "Running single test:", kwargs

    d_err, d_eff = evaluate(**kwargs)
    print "Mean cost: %.2f +/- %.4f" % (d_err.mean(), d_err.std() / np.sqrt(d_err.size))


if __name__ == "__main__":
    # noise_test(save="noise-tilt.eps", tilting=True)
    gate_test(tilting=True)
    # tilt_test(sigma=np.deg2rad(13))
    # one_test(sigma=np.deg2rad(13), shift=np.deg2rad(40), use_default=False,
    #          show_plots=False, verbose=True, samples=5, tilting=True, noise=.1)
