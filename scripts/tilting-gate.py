
from compoundeye.geometry import fibonacci_sphere, angles_distribution
from sphere import azidist
from sphere.transform import tilt as transtilt
from learn import SensorObjective

import numpy as np


def evaluate(D=60, samples=1000, tilt=True, noise=0.,
             # nb_tl2=16, nb_cl1=16, nb_tb1=8,
             gate_shift=np.pi/4, gate_order=5, scatter=False, plot=False):
    theta, phi, fit = angles_distribution(D, 60)
    alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi

    # if tilt:
    #     angles = np.array([
    #         [0., 0.],
    #         [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
    #         [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4], [np.pi / 6, 7 * np.pi / 4],
    #         [np.pi / 3, 0.], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
    #         [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4], [np.pi / 3, 7 * np.pi / 4]
    #     ])  # 17
    #     if samples == 1000:
    #         samples /= 2
    # else:
    #     angles = np.array([[0., 0.]])  # 1
    #
    # theta_s, phi_s = fibonacci_sphere(samples=samples, fov=161)
    # phi_s = phi_s[theta_s <= np.pi/2]
    # theta_s = theta_s[theta_s <= np.pi/2]
    # samples = theta_s.size
    # d = np.zeros((samples, angles.shape[0]))
    # phi_tl2 = np.linspace(0., 4 * np.pi, nb_tl2, endpoint=False)  # TL2 preference angles
    # phi_cl1 = np.linspace(0., 4 * np.pi, nb_cl1, endpoint=False)  # CL1 preference angles
    # phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles
    #
    # for j, (theta_t, phi_t) in enumerate(angles):
    #     theta_s_, phi_s_ = transtilt(theta_t, phi_t, theta=theta_s, phi=phi_s)
    #     theta_, phi_ = transtilt(theta_t, phi_t + np.pi, theta=theta, phi=phi)
    #     _, alpha_ = transtilt(theta_t, phi_t + np.pi, theta=np.pi / 2, phi=alpha)
    #
    #     if before:
    #         # Gate + Shift
    #         g = np.power(np.sqrt(1 - np.square(
    #             np.cos(theta + shift) * np.cos(theta_t) +
    #             np.sin(theta + shift) * np.sin(theta_t) * np.cos(phi_t - phi))), gate_order)
    #     else:
    #         # alternative form
    #         g = np.power(np.sin(theta_ + shift), gate_order)  # dynamic gating
    #
    #     # w = -8. / (2. * 60.) * np.cos(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * g[:, np.newaxis]
    #     w_tl2 = -1. / 60. * np.cos(phi_tl2[np.newaxis] - phi[:, np.newaxis]) * g[..., np.newaxis]
    #
    #     for k, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):
    #         _, dop, aop = SensorObjective.encode(e_org, a_org, theta_, phi_)
    #         r = SensorObjective.opticalencode(dop, aop, alpha_, noise=noise)
    #         ele, azi = SensorObjective.decode(r, theta, phi, w_tl2=w_tl2)
    #         d[k, j] = np.absolute(azidist(np.array([e, a]), np.array([ele, azi])))
    # d_deg = np.rad2deg(d)

    d_deg, _, _ = SensorObjective._fitness(
        theta, phi, alpha, gate_shift=gate_shift, gate_order=gate_order, activation="linear",
        tilt=tilt, error=azidist, samples=samples, noise=noise, return_mean=False)

    if plot or scatter:
        import matplotlib.pyplot as plt

    if scatter:
        # plt.figure("Tilts", figsize=(10, 15))
        # for i, ang, dd in zip(range(angles.shape[0]), angles, d.T):
        #     ax = plt.subplot2grid((5, 4), ((i-1) // 4, (i-1) % 4), polar=True)
        #     ax.set_theta_zero_location("N")
        #     ax.set_theta_direction(-1)
        #     plt.scatter(phi_s, np.rad2deg(theta_s), marker=".", c=dd, cmap="Reds", vmin=0, vmax=np.pi/2)
        #     plt.scatter(ang[1]+np.pi, np.rad2deg(ang[0]), marker="o", c="yellow", edgecolors="black")
        #     plt.title(r"$\epsilon_t=%03d, \alpha_t=%03d$" % tuple(np.rad2deg(ang)))
        #     plt.axis("off")
        # plt.subplot2grid((5, 4), (4, 0), colspan=3)
        # plt.imshow([np.arange(0, np.pi/2, np.pi/180)] * 3, cmap="Reds")
        # plt.yticks([])
        # plt.xticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])
        # plt.show()

        plt.figure("Tilts", figsize=(10, 3))
        for i, ang, dd in zip(range(3), angles[[0, 1, 9]], d.T[[0, 1, 9]]):
            ax = plt.subplot2grid((1, 10), (0, i * 3), colspan=3, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            plt.scatter(phi_s, np.rad2deg(theta_s), marker="o", c=dd, cmap="Reds", vmin=0, vmax=np.pi/2)
            plt.scatter(ang[1]+np.pi, np.rad2deg(ang[0]), marker="o", c="yellowgreen", edgecolors="black")
            plt.text(-np.deg2rad(50), 145, ["A.", "B.", "C."][i], fontsize=12)
            plt.axis("off")
        plt.subplot2grid((3, 10), (1, 9))
        plt.imshow(np.array([np.arange(0, np.pi/2, np.pi/180)] * 10).T, cmap="Reds")
        plt.xticks([])
        plt.yticks([0, 45, 89], [r"0", r"$\frac{\pi}{4}$", r"$\geq\frac{\pi}{2}$"])
        plt.show()

    if plot:
        plt.figure("cost-function")
        w = np.bartlett(10)
        w /= w.sum()
        d_000 = np.convolve(w, np.rad2deg(d[:, 0]), mode="same")
        plt.plot(np.rad2deg(theta_s), d_000, label=r"$0^\circ$")
        if angles.shape[0] > 1:
            d_030 = np.convolve(w, np.rad2deg(d[:, 1]), mode="same")
            d_060 = np.convolve(w, np.rad2deg(d[:, 9]), mode="same")
            plt.plot(np.rad2deg(theta_s), d_030, label=r"$30^\circ$")
            plt.plot(np.rad2deg(theta_s), d_060, label=r"$60^\circ$")
            plt.legend()
        plt.xlim([0, 90])
        plt.ylim([0, 180])
        plt.xlabel(r"sun elevation ($^\circ$)")
        plt.ylabel(r"cost ($^\circ$)")
        plt.show()

    return d_deg.mean(), d_deg.std() / np.sqrt(d_deg.size)


if __name__ == "__main__":
    scale = 1.
    before = False
    a_shift = np.deg2rad(19)
    b_shift = np.deg2rad(54)
    a_order = 10
    b_order = 15
    mode = "heatmap"  # "shift", "order"
    mode = "heatmap"  # "shift", "order"
    mode = "heatmap"  # "shift", "order"

    shifts = np.linspace(0, 2 * np.pi, 721, endpoint=True)
    orders = np.linspace(1, 15, 29, endpoint=True)
    s, o = np.meshgrid(shifts, orders)

    cost = []
    if mode == "order":
        shift = b_shift if before else a_shift
        for order in orders:
            cost[i].append(evaluate(gate_order=order, shift=shift, before=before))
            print "Order %02.1f :  %.2f +- %.2f deg" % (order, cost[i][-1][0], cost[i][-1][1])
    elif mode == "shift":
        order = b_order if before else a_order
        for shift in shifts:
            cost[i].append(evaluate(shift=shift, gate_order=order, before=before))
            print "Shift %03d :  %.2f +- %.2f deg" % (np.rad2deg(shift), cost[i][-1][0], cost[i][-1][1])
    else:
        # try:
        #     data = np.load("cost-%s.npz" % mode)
        #     cost = data["cost"]
        #     # cost = np.vstack([cost, [cost[-1]] * 14])
        #     orders = data["orders"]
        #     shifts = np.linspace(0, 2 * np.pi, 721, endpoint=True)
        # except IOError:
        for order, shift in zip(o.flatten(), s.flatten()):
            if shift / np.pi < 1:
                c = evaluate(gate_order=order, gate_shift=shift)
                print "Order %02.1f, Shift: %03.1f ---" % (order, np.rad2deg(shift)),
                print "%.2f +- %.2f deg" % (c[0], c[1])
            else:
                c = cost[int(np.round(2 * np.rad2deg(shift % np.pi)))]
            cost.append(c)
        cost = np.array(cost)
        print cost.shape
        np.savez_compressed("cost-%s.npz" % mode, cost=cost, orders=orders, shifts=shifts)

        s, o = np.meshgrid(shifts, orders)
        i = np.argmin(cost[..., 0])
        s_opt = s.flatten()[i]
        o_opt = o.flatten()[i]
        print "Optimal -- Shift: %03.2f; Order: %03.2f" % (np.rad2deg(s_opt), o_opt)

    import matplotlib.pyplot as plt

    if mode == "order":
        plt.figure("Gate-Order", figsize=(7, 5))
        for i in xrange(cost.shape[0]):
            plt.fill_between(orders,
                             cost[i, :, 0] - scale * cost[i, :, 1],
                             cost[i, :, 0] + scale * cost[i, :, 1],
                             color="C%d" % (i+1), alpha=.5)
            plt.plot(orders, cost[i, :, 0], color="C%d" % (i+1), label="before" if i % 2 == 0 else "after")
        plt.xlim([orders[0], orders[-1]])
        plt.ylim([0, 60])
        plt.xlabel(r"gate order")
        plt.ylabel(r"cost ($^\circ$)")
        if cost.shape[0] > 1:
            plt.legend()
        plt.show()
    elif mode == "shift":
        plt.figure("Shift", figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for i in xrange(cost.shape[0]):
            plt.fill_between(shifts,
                             cost[i, :, 0] - scale * cost[i, :, 1],
                             cost[i, :, 0] + scale * cost[i, :, 1], color="C%d" % (i+1), alpha=.5)
            plt.plot(shifts, cost[i, :, 0], color="C%d" % (i+1), label="before" if i % 2 == 0 else "after")
        plt.ylim([0, 120])
        plt.yticks([0, 30, 60, 90])
        plt.xticks((np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi) % (2 * np.pi) - np.pi)
        if cost.shape[0] > 1:
            plt.legend()
        plt.show()
    else:
        plt.figure("heatmap", figsize=(10, 5))
        with plt.rc_context({'ytick.color': 'white'}):
            ax = plt.subplot(121, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            plt.pcolormesh(s, o, cost[..., 0].reshape(s.shape), cmap="hot", vmin=0, vmax=110)
            plt.scatter(s_opt, o_opt, s=10, c="blue", marker='o')
            plt.xticks((np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi) % (2 * np.pi) - np.pi)
            plt.ylim([0, orders.max()])
            plt.yticks([0, 7.5, 15])
            ax.grid(alpha=0.2)

        with plt.rc_context({'ytick.color': 'white'}):
            ax = plt.subplot(122, polar=True)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            plt.pcolormesh(s, o, cost[..., 1].reshape(s.shape), cmap="hot", vmin=0, vmax=0.6)
            plt.scatter(s_opt, o_opt, s=10, c="blue", marker='o')
            plt.xticks((np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi) % (2 * np.pi) - np.pi)
            plt.ylim([0, orders.max()])
            plt.yticks([0, 7.5, 15])
            ax.grid(alpha=0.2)
        plt.show()

