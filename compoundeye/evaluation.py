from compoundeye.geometry import angles_distribution, fibonacci_sphere
from sphere.transform import tilt
from environment import Sky, eps, spectrum_influence, spectrum
from world.sky import T_L
from compoundeye import POLCompassDRA
from sphere import azidist

import numpy as np
import matplotlib.pyplot as plt


def evaluate_slow(n=60, omega=56,
             noise=0.,
             nb_cl1=8, sigma_pol=np.deg2rad(13), shift_pol=np.deg2rad(40),
             nb_tb1=8, sigma_sol=np.deg2rad(13), shift_sol=np.deg2rad(40),
             use_default=False,
             weighted=True,
             fibonacci=False,
             uniform_polariser=False,

             # single evaluation
             sun_azi=None, sun_ele=None,

             # data parameters
             tilting=True, samples=1000, show_plots=False, show_structure=False, verbose=False):

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
        _, alpha_ = tilt(theta_t, phi_t + np.pi, theta=np.pi / 2, phi=alpha)

        for i, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):

            sky = Sky(theta_s=e_org, phi_s=a_org, theta_t=theta_t, phi_t=phi_t)
            sky.verbose = verbose

            # COMPUTATIONAL MODEL

            # Input (POL) layer -- Photo-receptors

            dra = POLCompassDRA(n=n, omega=omega)
            dra.theta_t = theta_t
            dra.phi_t = phi_t
            r_pol = dra(sky, noise=noise, uniform_polariser=uniform_polariser)
            r_sol = dra.r_po

            # Tilting (SOL) layer
            d_pol = (np.sin(shift_pol - theta) * np.cos(theta_t) +
                     np.cos(shift_pol - theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            gate_pol = np.power(np.exp(-np.square(d_pol) / (2. * np.square(sigma_pol))), 1)
            z_pol = -float(nb_cl1) / float(n)
            w_cl1_pol = z_pol * np.sin(phi_cl1[np.newaxis] - alpha[:, np.newaxis]) * gate_pol[:, np.newaxis]

            d_sol = (np.sin(shift_sol - theta) * np.cos(theta_t) +
                     np.cos(shift_sol - theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            gate_sol = np.power(np.exp(-np.square(d_sol) / (2. * np.square(sigma_sol))), 1)
            z_sol = float(nb_cl1) / float(n)
            w_cl1_sol = z_sol * np.sin(phi_cl1[np.newaxis] - alpha[:, np.newaxis]) * gate_sol[:, np.newaxis]

            o = 1./64.
            f_pol, f_sol = .5 * np.power(2*theta_t/np.pi, o), .5 * (1 - np.power(2*theta_t/np.pi, o))
            r_cl1_pol = r_pol.dot(w_cl1_pol)
            r_cl1_sol = r_sol.dot(w_cl1_sol)
            r_cl1 = f_pol * r_cl1_pol + f_sol * r_cl1_sol
            # r_cl1 = r_cl1_sol

            # Output (TCL) layer
            # w_tb1 = np.eye(nb_tb1)
            w_tb1 = float(nb_tb1) / float(nb_cl1) * np.cos(phi_tb1[np.newaxis] - phi_cl1[:, np.newaxis])

            r_tb1 = r_cl1.dot(w_tb1)

            if use_default:
                w = -float(nb_tb1) / (2. * float(n)) * np.sin(
                    phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * gate_pol[:, np.newaxis]
                r_tb1 = r_pol.dot(w)

            # decode response - FFT
            R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * np.pi / (float(nb_tb1) / 2.)))
            a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)
            tau_pred = np.maximum(np.absolute(R), 0)  # certainty of prediction

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
                plt.figure("sensor-noise-%2d" % (100. * sky.eta.sum() / float(sky.eta.size)), figsize=(18, 4.5))

                ax = plt.subplot(1, 12, 10)
                plt.imshow(w_cl1_pol, cmap="coolwarm", vmin=-1, vmax=1)
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
                ax.scatter(phi, theta, s=150, c=r_pol * gate_pol, marker='o', cmap="coolwarm", vmin=-1, vmax=1)
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


def evaluate(n=60, omega=56,
             noise=0.,
             nb_cl1=8, sigma_pol=np.deg2rad(13), shift_pol=np.deg2rad(40),
             nb_tb1=8, sigma_sol=np.deg2rad(13), shift_sol=np.deg2rad(40),
             use_default=False,
             weighted=True,
             fibonacci=False,
             uniform_polariser=False,

             # single evaluation
             sun_azi=None, sun_ele=None,

             # data parameters
             tilting=True, ephemeris=False,
             samples=1000, show_plots=False, show_structure=False, verbose=False):

    # default parameters
    tau_L = 2.
    c1 = .6
    c2 = 4.
    AA, BB, CC, DD, EE = T_L.dot(np.array([tau_L, 1.]))  # sky parameters
    T_T = np.linalg.pinv(T_L)
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

    if tilting is not None and type(tilting) is tuple:
        angles = np.array([tilting])  # 1
    elif tilting:
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
            if uniform_polariser:
                Y = np.maximum(np.full_like(I_prez, Y_z), 0.)
            else:
                Y = np.maximum(Y_z * I_prez / (I_00 + eps), 0.)  # Illumination

            # Degree of Polarisation
            M_p = np.exp(-(tau_L - c1) / (c2 + eps))
            LP = np.square(np.sin(gamma)) / (1 + np.square(np.cos(gamma)))
            if uniform_polariser:
                P = np.ones_like(LP)
            else:
                P = np.clip(2. / np.pi * M_p * LP * (theta_ * np.cos(theta_) + (np.pi/2 - theta_) * I), 0., 1.)

            # Angle of polarisation
            if uniform_polariser:
                A = np.full_like(P, a_org + np.pi)
            else:
                _, A = tilt(e_org, a_org + np.pi, theta_, phi_)

            # create cloud disturbance
            if type(noise) is np.ndarray:
                if noise.size == P.size:
                    # print "yeah!"
                    eta = np.array(noise, dtype=bool)
                else:
                    eta = np.zeros_like(theta, dtype=bool)
                    eta[:noise.size] = noise
            elif noise > 0:
                eta = np.argsort(np.absolute(np.random.randn(*P.shape)))[:int(noise * P.shape[0])]
                # eta = np.array(np.absolute(np.random.randn(*P.shape)) < noise, dtype=bool)
                if verbose:
                    print "Noise level: %.4f (%.2f %%)" % (noise, 100. * eta.sum() / float(eta.size))
            else:
                eta = np.zeros_like(theta, dtype=bool)
            P[eta] = 0.  # destroy the polarisation pattern

            # COMPUTATIONAL MODEL

            # Input (POL) layer -- Photo-receptors
            y = spectrum_influence(Y, spectrum["uv"])
            s_1 = y * (np.square(np.sin(A - alpha_)) + np.square(np.cos(A - alpha_)) * np.square(1. - P))
            s_2 = y * (np.square(np.cos(A - alpha_)) + np.square(np.sin(A - alpha_)) * np.square(1. - P))
            r_1, r_2 = np.sqrt(s_1), np.sqrt(s_2)
            # r_1, r_2 = np.log(s_1 + 1.), np.log(s_2 + 1.)
            r_opo, r_sol = r_1 - r_2, r_1 + r_2
            r_pol = r_opo / (r_sol + eps)
            # r_sol = 2. * r_sol / np.max(r_sol) - 1.

            # Tilting (SOL) layer
            d_pol = (np.sin(shift_pol - theta) * np.cos(theta_t) +
                     np.cos(shift_pol - theta) * np.sin(theta_t) *
                     np.cos(phi - phi_t))
            gate_pol = np.power(np.exp(-np.square(d_pol) / (2. * np.square(sigma_pol))), 1)
            z_pol = float(nb_cl1) / float(n)
            w_cl1_pol = z_pol * np.sin(alpha[:, np.newaxis] - phi_cl1[np.newaxis]) * gate_pol[:, np.newaxis]

            # d_sol = (np.sin(shift_sol - theta) * np.cos(theta_t) +
            #          np.cos(shift_sol - theta) * np.sin(theta_t) *
            #          np.cos(phi - phi_t))
            # gate_sol = np.power(np.exp(-np.square(d_sol) / (2. * np.square(sigma_sol))), 1)
            # z_sol = float(nb_cl1) / float(n)
            # w_cl1_sol = z_sol * np.sin(phi_cl1[np.newaxis] - alpha[:, np.newaxis]) * gate_sol[:, np.newaxis]

            # o = 1./64.
            r_cl1_pol = r_pol.dot(w_cl1_pol)
            # r_cl1_sol = r_sol.dot(w_cl1_sol)
            # f_pol, f_sol = .5 * np.power(2*theta_t/np.pi, o), .5 * (1 - np.power(2*theta_t/np.pi, o))
            # r_cl1 = f_pol * r_cl1_pol + f_sol * r_cl1_sol
            r_cl1 = r_cl1_pol

            # Output (TB1) layer
            eph = (a - np.pi/3) if ephemeris else 0.
            w_tb1 = float(nb_tb1) / float(nb_cl1 * 2.) * np.cos(phi_tb1[np.newaxis] - phi_cl1[:, np.newaxis] + eph)

            # r_tb1 = r_cl1.dot(w_tb1)
            r_tb1 = r_cl1.dot(w_tb1)

            if use_default:
                w = -float(nb_tb1) / (2. * float(n)) * np.sin(
                    phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * gate_pol[:, np.newaxis]
                r_tb1 = r_pol.dot(w)

            # decode response - FFT
            R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * 2. * np.pi / float(nb_tb1)))
            a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)
            tau_pred = np.absolute(R)  # certainty of prediction

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
                plt.figure("sensor-noise-%2d" % (100. * eta.sum() / float(eta.size)), figsize=(4.5, 9))
                x = np.linspace(0, 2 * np.pi, 721)

                ax = plt.subplot(6, 1, 4)
                plt.imshow(5 * w_cl1_pol.T, cmap="coolwarm", vmin=-1, vmax=1)
                plt.yticks([0, 7], ["1", "8"])
                plt.xticks([0, 59], ["1", "60"])

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(12, 1, 9)  # , sharey=ax)
                plt.imshow(w_tb1.T, cmap="coolwarm", vmin=-1, vmax=1)
                plt.xticks([0, 7], ["1", "8"])
                plt.yticks([0, 7], ["1", "8"])
                # cbar = plt.colorbar(ticks=[-1, 0, 1], orientation='vertical')
                # cbar.ax.set_yticklabels([r'$\leq$ -1', r'0', r'$\geq$ 1'])

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                ax = plt.subplot(2, 1, 1, polar=True)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.grid(False)

                # POL
                ax.scatter(phi, theta, s=90, c=r_pol, marker='o', edgecolor='black', cmap="coolwarm", vmin=-1, vmax=1)

                # SOL
                y = np.deg2rad(37.5)
                sy = np.deg2rad(15)
                ax.fill_between(x, np.full_like(x, y - sy/2), np.full_like(x, y + sy/2),
                                facecolor="C1", alpha=.5, label="SOL")
                ax.scatter(phi_cl1, np.full(nb_cl1, np.deg2rad(37.5)), s=600,
                           c=r_cl1, marker='o', edgecolor='red', cmap="coolwarm", vmin=-1, vmax=1)

                for ii, pp in enumerate(phi_cl1):
                    ax.text(pp - np.pi/13, y, "%d" % (ii + 1), ha="center", va="center", size=10,
                            bbox=dict(boxstyle="circle", fc="w", ec="k"))
                    ax.arrow(pp, np.deg2rad(33), 0, np.deg2rad(4), fc='k', ec='k',
                             head_width=.1, head_length=.1, overhang=.3)

                # TCL
                y = np.deg2rad(52.5)
                sy = np.deg2rad(15)
                ax.fill_between(x, np.full_like(x, y - sy/2), np.full_like(x, y + sy/2),
                                facecolor="C2", alpha=.5, label="TCL")
                ax.scatter(phi_tb1, np.full_like(phi_tb1, y), s=600,
                           c=r_tb1, marker='o', edgecolor='green', cmap="coolwarm", vmin=-1, vmax=1)
                for ii, pp in enumerate(phi_tb1):
                    ax.text(pp + np.pi/18, y, "%d" % (ii + 1), ha="center", va="center", size=10,
                            bbox=dict(boxstyle="circle", fc="w", ec="k"))
                    pref = a if ephemeris else 0.
                    print "Sun:", np.rad2deg(pref)
                    dx, dy = np.deg2rad(4) * np.sin(0.), np.deg2rad(4) * np.cos(0.)
                    ax.arrow(pp - dx, y - dy/2 - np.deg2rad(2.5), dx, dy, fc='k', ec='k',
                             head_width=.07, head_length=.1, overhang=.3)

                # Sun position
                ax.scatter(a, e, s=500, marker='o', edgecolor='black', facecolor='yellow')
                ax.scatter(phi_t, theta_t, s=200, marker='o', edgecolor='black', facecolor='yellowgreen')

                # Decoded TB1
                # ax.plot([0, a_pred], [0, e_pred], 'k--', lw=1)
                ax.plot([0, a_pred], [0, np.pi/2], 'k--', lw=1)
                ax.arrow(a_pred, 0, 0, np.deg2rad(20),
                         fc='k', ec='k', head_width=.3, head_length=.2, overhang=.3)

                ax.legend(ncol=2, loc=(.15, -1.), fontsize=16)

                ax.set_ylim([0, np.deg2rad(60)])
                ax.set_yticks([])
                ax.set_xticks(np.linspace(0, 2*np.pi, 4, endpoint=False))
                ax.set_xticklabels([r'N', r'E', r'S', r'W'])
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='both', which='minor', labelsize=16)

                plt.subplots_adjust(left=.07, bottom=.0, right=.93, top=.96)

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


def evaluate_old(n=60, omega=56,
             noise=0.,
             nb_cl1=16, sigma=np.deg2rad(13), shift=np.deg2rad(40),
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
    AA, BB, CC, DD, EE = T_L.dot(np.array([tau_L, 1.]))  # sky parameters
    T_T = np.linalg.pinv(T_L)
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
    phi_cl1 = np.linspace(0., 4 * np.pi, nb_cl1, endpoint=False)  # CL1 preference angles
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
            R = r_tb1.dot(np.exp(-np.arange(nb_tb1) * (0. + 1.j) * np.pi / (float(nb_tb1) / 2.)))
            a_pred = (np.pi - np.arctan2(R.imag, R.real)) % (2. * np.pi) - np.pi  # sun azimuth (prediction)
            tau_pred = np.absolute(R)  # certainty of prediction

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
                ax.set_xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
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
                ax.set_xticks(np.linspace(-3*np.pi/4, 5*np.pi/4, 8, endpoint=False))
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
