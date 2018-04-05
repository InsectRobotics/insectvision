
from compoundeye.geometry import fibonacci_sphere, angles_distribution
from sphere import sph2vec, vec2sph, azidist
from sphere.transform import point2rotmat
from learn import SensorObjective

import numpy as np


def evaluate(D=60, N=8, samples=1000, tilt=True, shift=np.pi/4, gate_order=5, before=True, scatter=False, plot=False):
    theta, phi, fit = angles_distribution(D, 60)
    alpha = (phi - np.pi/2) % (2 * np.pi) - np.pi

    if tilt:
        angles = np.array([
            [0., 0.],
            [np.pi / 6, 0.], [np.pi / 6, np.pi / 4], [np.pi / 6, 2 * np.pi / 4], [np.pi / 6, 3 * np.pi / 4],
            [np.pi / 6, 4 * np.pi / 4], [np.pi / 6, 5 * np.pi / 4], [np.pi / 6, 6 * np.pi / 4], [np.pi / 6, 7 * np.pi / 4],
            [np.pi / 3, 0.], [np.pi / 3, np.pi / 4], [np.pi / 3, 2 * np.pi / 4], [np.pi / 3, 3 * np.pi / 4],
            [np.pi / 3, 4 * np.pi / 4], [np.pi / 3, 5 * np.pi / 4], [np.pi / 3, 6 * np.pi / 4], [np.pi / 3, 7 * np.pi / 4]
        ])  # 17
        if samples == 1000:
            samples /= 2
    else:
        angles = np.array([[0., 0.]])  # 1

    theta_s, phi_s = fibonacci_sphere(samples=samples, fov=180)
    d = np.zeros((samples, angles.shape[0]))
    phi_tb1 = np.linspace(0., 2 * np.pi, N, endpoint=False) + np.pi / 2  # TB1 preference angles

    for j, (theta_t, phi_t) in enumerate(angles):
        v_t = sph2vec(theta_t, phi_t, zenith=True)
        v_s = sph2vec(theta_s, phi_s, zenith=True)
        v = sph2vec(theta, phi, zenith=True)
        v_a = sph2vec(np.full(alpha.shape[0], np.pi / 2), alpha, zenith=True)
        R = point2rotmat(v_t)
        theta_s_, phi_s_, _ = vec2sph(R.dot(v_s), zenith=True)
        # theta_s_, phi_s_ = transtilt(-theta_t, -phi_t, theta=theta_s, phi=phi_s)

        theta_, phi_, _ = vec2sph(R.T.dot(v), zenith=True)
        # theta_, phi_ = transtilt(theta_t, phi_t, theta=theta, phi=phi)
        _, alpha_, _ = vec2sph(R.T.dot(v_a), zenith=True)
        # _, alpha_ = transtilt(theta_t, phi_t, theta=np.pi/2, phi=alpha)

        if before:
            # Gate + Shift
            g = np.power(np.sqrt(1 - np.square(
                np.cos(theta + shift) * np.cos(theta_t) +
                np.sin(theta + shift) * np.sin(theta_t) * np.cos(phi_t - phi))), gate_order)
        else:
            # alternative form
            g = np.power(np.sin(theta_ + shift), gate_order)  # dynamic gating

        # other versions
        # g = np.sin(theta)  # static gating
        # g = np.ones_like(theta)  # uniform - no gating

        w = -8. / (2. * 60.) * np.cos(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * g[:, np.newaxis]

        for i, (e, a, e_org, a_org) in enumerate(zip(theta_s_, phi_s_, theta_s, phi_s)):
            _, dop, aop = SensorObjective.encode(e_org, a_org, theta_, phi_)
            ele, azi = SensorObjective.decode(dop, aop, alpha_, w=w)
            d[i, j] = np.absolute(azidist(np.array([e, a]), np.array([ele, azi])))

    d_deg = np.rad2deg(d)

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

    shifts = np.linspace(0, 2*np.pi, 361, endpoint=True)
    orders = np.linspace(0, 20, 101, endpoint=True)
    s, o = np.meshgrid(shifts, orders)

    cost = []
    for i, before in enumerate([True, False]):
        cost.append([])
        print "Before:" if before else "After:"

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
            for order, shift in zip(o.flatten(), s.flatten()):
                c = evaluate(gate_order=order, shift=shift, before=before)
                cost[i].append(c[0])
                print "Order %02.1f, Shift: %03d ---" % (order, np.rad2deg(shift)),
                print "%.2f +- %.2f deg" % (c[0], c[1])

    cost = np.array(cost)
    print cost.shape
    np.savez_compressed("cost-%s.npz" % mode, cost=cost, orders=orders, shifts=shifts)

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
        for i in xrange(cost.shape[0]):
            with plt.rc_context({'ytick.color': 'white'}):
                ax = plt.subplot(121 + i, polar=True)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                plt.pcolormesh(s, o, cost[i].reshape(s.shape), cmap="hot", vmin=0, vmax=110)
                plt.xticks((np.linspace(0., 2 * np.pi, 8, endpoint=False) + np.pi) % (2 * np.pi) - np.pi)
                plt.ylim([0, orders.max()])
                plt.yticks([0, 9, 15, 20])
                ax.grid(alpha=0.2)
        plt.show()

