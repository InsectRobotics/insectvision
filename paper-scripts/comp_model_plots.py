from compoundeye.geometry import angles_distribution, fibonacci_sphere
from sphere import azidist
from sphere.transform import tilt
from learn.loss_function import SensorObjective

import matplotlib.pyplot as plt
import numpy as np

tb1_names = ['L5/R4', 'L6/R3', 'L7/R2', 'L8/R1', 'L1/R8', 'L2/R7', 'L3/R6', 'L4/R5']


def evaluate(n=60, omega=56,
             noise=0.,
             nb_cl1=8, sigma=np.deg2rad(13), shift=np.deg2rad(40),
             nb_tb1=8,
             use_default=False,
             weighted=True,
             fibonacci=False,
             simple_pol=False, uniform_poliriser=False,

             # single evaluation
             sun_azi=None, sun_ele=None,

             # data parameters
             tilting=True, samples=1000, show_plots=False, show_structure=False, verbose=False):

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
    if sun_azi is not None or sun_ele is not None:
        theta_s = sun_ele if type(sun_ele) is np.ndarray else np.array([sun_ele])
        phi_s = sun_azi if type(sun_azi) is np.ndarray else np.array([sun_azi])
    else:
        theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
        phi_s = phi_s[theta_s <= np.pi / 2]
        theta_s = theta_s[theta_s <= np.pi / 2]
    samples = theta_s.size

    # generate the properties of the sensor
    try:
        theta, phi, fit = angles_distribution(n, float(omega))
    except ValueError:
        theta = np.empty(0, dtype=np.float32)
        phi = np.empty(0, dtype=np.float32)
        fit = False

    if not fit or n > 100 or fibonacci:
        theta, phi = fibonacci_sphere(n, float(omega))
    # theta, phi, fit = angles_distribution(n, omega)
    # if not fit:
    #     print theta.shape, phi.shape
    theta = (theta - np.pi) % (2 * np.pi) - np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    alpha = (phi + np.pi/2) % (2 * np.pi) - np.pi

    # computational model parameters
    phi_cl1 = np.linspace(0., 2 * np.pi, nb_cl1, endpoint=False)  # CL1 preference angles
    phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles

    # initialise lists for the statistical data
    d = np.zeros((samples, angles.shape[0]), dtype=np.float32)
    t = np.zeros_like(d)
    d_eff = np.zeros((samples, angles.shape[0]), dtype=np.float32)
    a_ret = np.zeros_like(t)
    tb1 = np.zeros((samples, angles.shape[0], nb_tb1), dtype=np.float32)

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
            if uniform_poliriser:
                Y = np.maximum(np.full_like(I_prez, Y_z), 0.)
            else:
                Y = np.maximum(Y_z * I_prez / (I_00 + eps), 0.)  # Illumination

            # Degree of Polarisation
            M_p = np.exp(-(tau_L - c1) / (c2 + eps))
            LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
            if uniform_poliriser:
                P = np.ones_like(LP)
            elif simple_pol:
                P = np.clip(2. / np.pi * M_p * LP, 0., 1.)
            else:
                P = np.clip(2. / np.pi * M_p * LP * (theta_ * np.cos(theta_) + (np.pi/2 - theta_) * I), 0., 1.)

            # Angle of polarisation
            if uniform_poliriser:
                A = np.full_like(P, a_org + np.pi)
            else:
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
            s_1 = Y * (np.square(np.sin(A - alpha_)) + np.square(np.cos(A - alpha_)) * np.square(1. - P))
            s_2 = Y * (np.square(np.cos(A - alpha_)) + np.square(np.sin(A - alpha_)) * np.square(1. - P))
            r_1, r_2 = np.sqrt(s_1), np.sqrt(s_2)
            # r_1, r_2 = np.log(s_1 + 1.), np.log(s_2 + 1.)
            r_op, r_po = r_1 - r_2, r_1 + r_2
            r_pol = r_op / (r_po + eps)

            # Tilting (SOL) layer
            d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
                     np.cos(shift - theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            d_sun = (np.sin(-theta) * np.cos(theta_t) +
                     np.cos(-theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)
            gate_po = np.power(np.exp(-np.square(d_sun) / (2. * np.square(sigma))), 1)
            w_cl1_po = float(nb_cl1) / float(n) * np.sin(phi_cl1[np.newaxis] - alpha[:, np.newaxis]) * gate_po[:, np.newaxis]
            w_cl1 = -float(nb_cl1) / float(n) * np.sin(phi_cl1[np.newaxis] - alpha[:, np.newaxis]) * gate[:, np.newaxis]
            # w_cl1 = float(nb_cl1) / float(n) * np.sin(alpha[:, np.newaxis] - phi_cl1[np.newaxis]) * gate[:,
            #                                                                                               np.newaxis]
            # r_pol = Y
            r_cl1 = Y.dot(w_cl1_po)
            # r_cl1 = r_pol.dot(w_cl1)
            # r_cl1 = 11./12. * r_pol.dot(w_cl1) + 1./12. * Y.dot(w_cl1_po)

            # Output (TCL) layer
            # w_tb1 = np.eye(nb_tb1)
            w_tb1 = float(nb_tb1) / float(nb_cl1) * np.cos(phi_tb1[np.newaxis] - phi_cl1[:, np.newaxis])

            r_tb1 = r_cl1.dot(w_tb1)

            if use_default:
                w = -float(nb_tb1) / (2. * float(n)) * np.sin(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * gate[:,
                                                                                                            np.newaxis]
                r_tb1 = r_pol.dot(w)

            # decode response - FFT
            R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * np.pi / (float(nb_tb1) / 2.)))
            a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)
            tau_pred = np.maximum(np.absolute(R) - np.pi/2, 0)  # certainty of prediction

            d[i, j] = np.absolute(azidist(np.array([e, a]), np.array([0., a_pred])))
            t[i, j] = tau_pred if weighted else 1.
            a_ret[i, j] = a_pred
            tb1[i, j] = r_tb1

            # effective degree of polarisation
            M = r_cl1.max() - r_cl1.min()
            # M = t[i, j] * 2.
            p = np.power(10, M/2.)
            d_eff[i, j] = np.mean((p - 1.) / (p + 1.))

            if show_plots:
                plt.figure("sensor-noise-%2d" % (100. * eta.sum() / float(eta.size)), figsize=(18, 4.5))

                ax = plt.subplot(1, 12, 10)
                plt.imshow(w_cl1, cmap="coolwarm", vmin=-1, vmax=1)
                plt.xlabel("CBL", fontsize=16)
                plt.xticks([0, 15], ["1", "16"])
                plt.yticks([0, 59], ["1", "60"])

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(1, 6, 6)  # , sharey=ax)
                plt.imshow(w_tb1, cmap="coolwarm", vmin=-1, vmax=1)
                plt.xlabel("TB1", fontsize=16)
                plt.xticks([0, 7], ["1", "8"])
                plt.yticks([0, 15], ["1", "16"])
                cbar = plt.colorbar(ticks=[-1, 0, 1])
                cbar.ax.set_yticklabels([r'$\leq$ -1', r'0', r'$\geq$ 1'])

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(1, 4, 1, polar=True)
                ax.scatter(phi, theta, s=150, c=r_pol, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
                ax.scatter(a, e, s=100, marker='o', edgecolor='black', facecolor='yellow')
                ax.scatter(phi_t + np.pi, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.deg2rad(40)])
                ax.set_yticks([])
                ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
                ax.set_xticklabels([r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$135^\circ$',
                                    r'$180^\circ$', r'$-135^\circ$', r'$-90^\circ$', r'$-45^\circ$'])
                ax.set_title("POL Response", fontsize=16)

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(1, 4, 2, polar=True)
                ax.scatter(phi, theta, s=150, c=r_pol * gate, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
                ax.scatter(a, e, s=100, marker='o', edgecolor='black', facecolor='yellow')
                ax.scatter(phi_t + np.pi, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.deg2rad(40)])
                ax.set_yticks([])
                ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
                ax.set_xticklabels([r'$0^\circ$', r'$45^\circ$', r'$90^\circ$', r'$135^\circ$',
                                    r'$180^\circ$', r'$-135^\circ$', r'$-90^\circ$', r'$-45^\circ$'])
                ax.set_title("Gated Response", fontsize=16)

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(1, 4, 3, polar=True)
                x = np.linspace(0, 2 * np.pi, 721)

                # CBL
                ax.fill_between(x, np.full_like(x, np.deg2rad(60)), np.full_like(x, np.deg2rad(90)),
                                facecolor="C1", alpha=.5, label="CBL")
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

                ax.legend(ncol=2, loc=(-.55, -.1), fontsize=16)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim([0, np.pi/2])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Sensor Response", fontsize=16)

                plt.subplots_adjust(left=.02, bottom=.12, right=.98, top=.88)

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                plt.show()

    d_deg = np.rad2deg(d)

    if show_structure:
        plt.figure("sensor-structure", figsize=(4.5, 4.5))
        ax = plt.subplot(111, polar=True)
        ax.scatter(phi, theta, s=150, c="black", marker='o')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim([0, np.deg2rad(50)])
        ax.set_yticks([])
        ax.set_xticks(np.linspace(-3 * np.pi / 4, 5 * np.pi / 4, 8, endpoint=False))
        ax.set_title("POL Response")
        plt.show()

    return d_deg, d_eff, t, a_ret, tb1


def nb_neurons_test(save=None, mode=0, **kwargs):
    print "Running number of neurons test:", kwargs

    plt.figure("Structure", figsize=(10, 5))

    nb_tb1s = np.linspace(4, 360, 90)
    nb_cl1s = np.linspace(4, 360, 90)

    filename = "nb-neurons-costs.npz"
    if mode < 2:
        plt.subplot(121)
        means = np.zeros_like(nb_tb1s)
        ses = np.zeros_like(nb_tb1s)
        nb_tb1_default = kwargs.pop('nb_tb1', 8)
        nb_cl1_default = kwargs.pop('nb_cl1', 16)
        for ii, nb_tb1 in enumerate(nb_tb1s):
            d_err, d_eff, tau = evaluate(nb_tb1=nb_tb1, nb_cl1=nb_cl1_default, verbose=False, **kwargs)
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
            d_err, d_eff, tau = evaluate(nb_tb1=nb_tb1_default, nb_cl1=nb_cl1, verbose=False, **kwargs)
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
                d_err, d_eff, tau = evaluate(nb_tb1=nb_tb1, nb_cl1=nb_cl1, verbose=False, **kwargs)
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


def noise_test(save=None, **kwargs):
    print "Running noise test:", kwargs

    plt.figure("Noise", figsize=(5, 5))

    etas = np.linspace(0, 2, 21)
    means = np.zeros_like(etas)
    ses = np.zeros_like(etas)
    for ii, eta in enumerate(etas):
        d_err, d_eff, tau = evaluate(
            noise=eta, verbose=False, tilting=True, **kwargs
        )
        means[ii] = (d_err * tau / tau.sum()).sum()
        ses[ii] = d_err.std() / np.sqrt(d_err.size)
        print "Noise level: %.2f | Mean cost: %.2f +/- %.4f" % (eta, means[ii], ses[ii])

    plt.fill_between(etas, means-ses, means+ses, facecolor="grey")
    plt.plot(etas, means, color="black", label="tilting")

    for ii, eta in enumerate(etas):
        d_err, d_eff, tau = evaluate(
            noise=eta, verbose=False, tilting=False, **kwargs
        )
        means[ii] = (d_err * tau / tau.sum()).sum()
        ses[ii] = d_err.std() / np.sqrt(d_err.size)
        print "Noise level: %.2f | Mean cost: %.2f +/- %.4f" % (eta, means[ii], ses[ii])

    plt.fill_between(etas, means-ses, means+ses, facecolor="grey", alpha=.5)
    plt.plot(etas, means, color="black", linestyle="--", label="plane")

    plt.ylim([0, 60])
    plt.yticks([0, 15, 30, 45, 60], [r'%d$^\circ$' % o for o in [0, 15, 30, 45, 60]])
    plt.xlim([0, 2])
    plt.xlabel(r'noise ($\eta$)')
    plt.ylabel("MSE ($^\circ$)")
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
            d_err, d_eff, _, _, _ = evaluate(sigma=sigma, verbose=False, **kwargs)

            means[ii] = d_err.mean()
            ses[ii] = d_err.std() / np.sqrt(d_err.size)
            print "Sigma: %.2f | Mean cost: %.2f +/- %.4f" % (np.rad2deg(sigma), means[ii], ses[ii])

        plt.fill_between(sigmas, means - ses, means + ses, facecolor="C1", alpha=.5)
        plt.plot(sigmas, means, color="C1", label="sigma")

        means = np.zeros_like(shifts)
        ses = np.zeros_like(shifts)
        for ii, shift in enumerate(shifts):
            d_err, d_eff, _, _, _ = evaluate(shift=shift, verbose=False, **kwargs)

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
            sigmas, shifts = np.meshgrid(sigmas, shifts)
            means = np.zeros(sigmas.size)
            for ii, sigma, shift in zip(np.arange(sigmas.size), sigmas.flatten(), shifts.flatten()):
                d_err, d_eff, tau, _, _ = evaluate(sigma=sigma, shift=shift, verbose=False, **kwargs)
                means[ii] = d_err.mean()
                se = np.rad2deg(d_err.std() / np.sqrt(d_err.size))
                print 'Sigma = %.2f, Shift = %.2f | Mean cost: %.2f +/- %.4f' % (
                    np.rad2deg(sigma), np.rad2deg(shift), means[ii], se)

            means = means.reshape(shifts.shape)
            np.savez_compressed(filename, shifts=shifts, sigmas=sigmas, costs=means)

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
            plt.xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
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
        n_default = kwargs.pop('n', 60)
        omega_default = kwargs.pop('omega', 56)
        for ii, n in enumerate(ns.astype(int)):
            d_err, d_eff, tau = evaluate(n=n, omega=omega_default, verbose=False, **kwargs)
            means[ii] = (d_err * tau / tau.sum()).sum()
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
            d_err, d_eff, tau = evaluate(n=n_default, omega=omega, verbose=False, **kwargs)
            means[ii] = (d_err * tau / tau.sum()).sum()
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
                d_err, d_eff, tau = evaluate(n=n, omega=omega, verbose=False, **kwargs)
                means[ii] = (d_err * tau / tau.sum()).sum()
                se = d_err.std() / np.sqrt(d_err.size)
                print 'N = % 3d, Omega = %.2f | Mean cost: %.2f +/- %.4f' % (n, omega, means[ii], se)

            means = means.reshape(omegas.shape)
            np.savez_compressed(filename, omegas=omegas, ns=ns, costs=means)

        ii = np.nanargmin(means, axis=0)
        omega_min = omegas[ii, np.arange(91)]
        n_min = ns[ii, np.arange(91)]
        means_min = means[ii, np.arange(91)]
        print 'Minimum cost (%.2f) for N = %d, Omega = %.2f' % (means_min[15], n_min[15], omega_min[30])
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


def heinze_experiment(n_tb1=0, eta=.0, absolute=False, uniform=False):
    sun_azi = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    sun_ele = np.full_like(sun_azi, np.pi/2)
    tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

    for _ in np.linspace(0, 1, 100):
        d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_poliriser=uniform,
                                               sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=eta)
        tb1s = np.vstack([tb1s, np.transpose(r_tb1, axes=(1, 0, 2))])

    plt.figure("heinze-%s%s" % ("abs-" if absolute else "uni-" if uniform else "", tb1_names[n_tb1]), figsize=(3, 3))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if absolute:
        tb1s = np.absolute(tb1s)
    r_mean = np.median(tb1s[..., n_tb1], axis=0)
    z = r_mean.max() - r_mean.min()

    r_mean = (r_mean - r_mean.min()) / z - .5
    r_std = tb1s[..., n_tb1].std(axis=0) / np.sqrt(z)
    bl = .5
    plt.bar((sun_azi + np.pi) % (2 * np.pi) - np.pi, bl + r_mean, .1, yerr=r_std, facecolor='black')
    plt.plot(np.linspace(-np.pi, np.pi, 361), np.full(361, bl), 'k-')
    plt.yticks([])
    plt.xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False),
               [r'%d$^\circ$' % x for x in ((np.linspace(0, 360, 8, endpoint=False) + 180) % 360 - 180)])
    plt.ylim([-.3, 1.1])
    # plt.savefig("heinze-%s%d.eps" % ("abs-" if absolute else "uni-" if uniform else "", n_tb1))
    plt.show()


def heinze_1f(eta=.5, uniform=False):
    from astropy.stats import circmean

    sun_azi = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    sun_ele = np.full_like(sun_azi, np.pi/2)
    phi_tb1 = np.linspace(0., 2 * np.pi, 8, endpoint=False)  # TB1 preference angles
    tb1_ids = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)
    tb1s = np.empty((0, sun_azi.shape[0], 8), dtype=sun_azi.dtype)

    for _ in np.linspace(0, 1, 100):
        d_deg, d_eff, t, phi, r_tb1 = evaluate(uniform_poliriser=uniform,
                                               sun_azi=sun_azi, sun_ele=sun_ele, tilting=False, noise=eta)
        tb1s = np.vstack([tb1s, np.transpose(r_tb1, axes=(1, 0, 2))])
        tb1_ids = np.vstack([tb1_ids, np.array([sun_azi] * 8).T.reshape((1, 36, 8))])
    z = tb1s.max() - tb1s.min()
    tb1s = (tb1s - tb1s.min()) / z
    plt.figure("heinze-%sfig-1F" % ("uni-" if uniform else ""), figsize=(5, 5))
    phi = circmean(tb1_ids, weights=tb1s, axis=1)
    # plt.boxplot(phi)
    print phi.shape
    plt.scatter([4, 5, 6, 7, 0, 1, 2, 3] * 100, np.rad2deg(phi), s=20, c='black')
    plt.scatter([4, 5, 6, 7, 0, 1, 2, 3], np.rad2deg(circmean(phi, axis=0)), s=50, c='red')
    print np.rad2deg(circmean(phi, axis=0))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
               [tb1_names[4], '', tb1_names[6], '', tb1_names[0], '', tb1_names[2], ''])
    plt.yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180],
               ['-180', '', '-90', '', '0', '', '90', '', '180'])
    plt.ylim([-200, 200])
    plt.show()


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
    # noise_test()
    # nb_neurons_test(mode=2, tilting=True, weighted=False, noise=.0)
    gate_ring(sigma=np.deg2rad(26), shift=np.deg2rad(0))
    # noise2disturbance_plot()
    # gate_test(tilting=True, mode=2, filename="gate-costs-po.npz")
    # tilt_test(weighted=True, use_default=False)
    # structure_test(tilting=True, mode=1, n=60, omega=52, weighted=True)
    # for n_tb1 in xrange(8):
    #     heinze_experiment(n_tb1=n_tb1, absolute=False, uniform=True)
    # heinze_1f(eta=.0, uniform=False)
    # one_test(n=60, omega=56, sigma=np.deg2rad(13), shift=np.deg2rad(40), use_default=False, weighted=True,
    #          show_plots=False, show_structure=False, verbose=True, samples=1000, tilting=True, noise=.0)
    # elevation_test()
