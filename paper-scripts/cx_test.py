from world import load_routes, Hybrid
from environment import Sky
from compoundeye.geometry import angles_distribution, fibonacci_sphere
from code.compass import decode_sph
from sphere.transform import sph2vec, vec2sph, tilt
from net import CX
from notebooks.results import get_noise

from datetime import datetime
import ephem
import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(float).eps

# create sky
sky = Sky()

# create random terrain
x_terrain = np.linspace(0, 10, 1001, endpoint=True)
y_terrain = np.linspace(0, 10, 1001, endpoint=True)
x_terrain, y_terrain = np.meshgrid(x_terrain, y_terrain)
try:
    z_terrain = np.load("terrain-%.2f.npz" % 0.6)["terrain"] * 1000 * .5
except IOError:
    z_terrain = np.random.randn(*x_terrain.shape) / 50

print z_terrain.max(), z_terrain.min()


def encode(theta, phi, Y, P, A, theta_t=0., phi_t=0., nb_tb1=8, sigma=np.deg2rad(13), shift=np.deg2rad(40)):
    alpha = (phi + np.pi/2) % (2 * np.pi) - np.pi
    phi_tb1 = np.linspace(0., 2 * np.pi, nb_tb1, endpoint=False)  # TB1 preference angles

    # Input (POL) layer -- Photo-receptors
    s_1 = Y * (np.square(np.sin(A - alpha)) + np.square(np.cos(A - alpha)) * np.square(1. - P))
    s_2 = Y * (np.square(np.cos(A - alpha)) + np.square(np.sin(A - alpha)) * np.square(1. - P))
    r_1, r_2 = np.sqrt(s_1), np.sqrt(s_2)
    r_pol = (r_1 - r_2) / (r_1 + r_2 + eps)

    # Tilting (CL1) layer
    d_cl1 = (np.sin(shift - theta) * np.cos(theta_t) +
             np.cos(shift - theta) * np.sin(theta_t) *
             np.cos(phi - phi_t))
    gate = np.power(np.exp(-np.square(d_cl1) / (2. * np.square(sigma))), 1)
    w = -float(nb_tb1) / (2. * float(n)) * np.sin(phi_tb1[np.newaxis] - alpha[:, np.newaxis]) * gate[:, np.newaxis]
    r_tb1 = r_pol.dot(w)

    return r_tb1


def turn(x, y, yaw, theta, phi, theta_s, phi_s, flow, noise=0.):
    sky.theta_s, sky.phi_s = theta_s, phi_s
    Y, P, A = sky(theta, phi + yaw, noise=noise)
    # Y, P, A = get_sky_cues(theta, phi + yaw, theta_s, phi_s, noise=noise)
    r_tb1 = encode(theta, phi, Y, P, A)[::-1]
    _, yaw = decode_sph(r_tb1)
    # yaw = phi_s - yaw
    # motor = net(r_tb1, flow)
    net(yaw, flow)
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx


def get_3d_direction(x, y, yaw, tau=.06):
    # create point cloud
    i = np.sqrt(np.square(x - x_terrain) + np.square(y - y_terrain)) < tau
    points = np.array([x_terrain[i], y_terrain[i], z_terrain[i]]).T
    if points.shape[0] == 0:
        return 0., 0.
    # print points.shape,
    # compute centred coordinates and run SVD
    _, _, vh = np.linalg.svd(points - points.mean(axis=0))
    # unitary normal vector
    u = vh.conj().transpose()[:, -1]
    p = sph2vec(np.pi / 2, yaw, zenith=True)
    pp = p - p.dot(u) / np.square(np.linalg.norm(u)) * u
    theta_p, phi_p, _ = vec2sph(pp, zenith=True)
    return theta_p - np.pi/2, phi_p - yaw


if __name__ == "__main__":
    from notebooks.plots import plot_sky

    # sensor design
    n = 60
    omega = 56
    theta, phi, fit = angles_distribution(n, float(omega))
    theta_t, phi_t = 0., 0.

    # sun position
    seville = ephem.Observer()
    seville.lat = '37.392509'
    seville.lon = '-5.983877'
    seville.date = datetime(2018, 6, 21, 9, 0, 0)
    sun = ephem.Sun()
    sun.compute(seville)
    theta_s = np.array([np.pi/2 - sun.alt])
    phi_s = np.array([(sun.az + np.pi) % (2 * np.pi) - np.pi])

    theta_sky, phi_sky = fibonacci_sphere(1000, 180)

    # ant-world
    noise_type = "canopy"
    mode = "uneven"
    noise = 0.0
    ttau = .06
    dx = 1e-02
    # world = load_world()
    routes = load_routes()
    flow = dx * np.ones(2) / np.sqrt(2)
    max_theta = 0.

    for ni, noise in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    # for ni, noise in enumerate([0.0]):

        # stats
        d_x = []  # logarithmic distance
        d_c = []
        tau = []  # tortuosity
        ri = 0

        print "Noise:", noise
        sky.theta_s, sky.phi_s = np.pi/6, np.pi
        eta = get_noise(theta_sky, phi_sky, noise, noise_type)
        Y, P, A = sky(theta_sky, phi_sky, noise=eta)
        plt.figure("Sky-%d-%s" % (noise * 10, noise_type), figsize=(7.5, 2.5))
        plot_sky(phi_sky, theta_sky, Y, P, A)

        # for route in routes[::2]:
        for route in [routes[0]]:
            net = CX(noise=0., pontin=False)
            net.update = True

            # outward route
            route.condition = Hybrid(tau_x=dx)
            oroute = route.reverse()
            x, y, yaw = [(x0, y0, yaw0) for x0, y0, _, yaw0 in oroute][0]
            opath = [[x, y, yaw]]

            v = np.zeros(2)
            tb1 = []
            # print np.rad2deg(phi_s)

            # plt.figure("yaws")
            for _, _, _, yaw in oroute:
                if mode == "uneven":
                    theta_t, phi_t = get_3d_direction(opath[-1][0], opath[-1][1], yaw, tau=ttau)
                    max_theta = max_theta if max_theta > np.absolute(theta_t) else np.absolute(theta_t)
                theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                sky.theta_s, sky.phi_s = theta_s, phi_s
                eta = get_noise(theta, phi, noise, noise_type)

                Y, P, A = sky(theta_n, phi_n, noise=eta)

                r_tb1 = encode(theta, phi, Y, P, A)
                yaw0 = yaw
                _, yaw = np.pi - decode_sph(r_tb1) + phi_s
                # plt.plot(yaw0 % (2*np.pi), yaw % (2*np.pi), 'k.')

                # motor = net(r_tb1, flow)
                net(yaw, flow)
                yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                opath.append([opath[-1][0] + v[0], opath[-1][1] + v[1], yaw])
                tb1.append(net.tb1)
            # plt.xlabel("org")
            # plt.ylabel("com")
            # plt.xlim([0, 2*np.pi])
            # plt.ylim([0, 2*np.pi])
            # plt.show()
            opath = np.array(opath)

            yaw -= phi_s
            # plt.figure(figsize=(10, 3))
            # plt.imshow(np.array(tb1).T)
            # plt.show()

            # inward route
            ipath = [[opath[-1][0], opath[-1][1], opath[-1][2]]]
            L = 0.  # straight distance to the nest
            C = 0.  # distance towards the nest that the agent has covered
            SL = 0.
            TC = 0.
            tb1 = []
            tau.append([])
            d_x.append([])
            d_c.append([])

            while C < 15:
                if mode == "uneven":
                    theta_t, phi_t = get_3d_direction(ipath[-1][0], ipath[-1][1], yaw, tau=ttau)
                theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                sky.theta_s, sky.phi_s = theta_s, phi_s
                Y, P, A = sky(theta_n, phi_n, noise=noise)

                r_tb1 = encode(theta, phi, Y, P, A)
                _, yaw = np.pi - decode_sph(r_tb1) + phi_s
                # motor = net(r_tb1, flow)
                motor = net(yaw, flow)
                yaw = (ipath[-1][2] + motor + np.pi) % (2 * np.pi) - np.pi
                v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                ipath.append([ipath[-1][0] + v[0], ipath[-1][1] + v[1], yaw])
                tb1.append(net.tb1)
                L = np.sqrt(np.square(opath[0][0] - ipath[-1][0]) + np.square(opath[0][1] - ipath[-1][1]))
                C += route.dx
                # d_x[-1].append(np.power(10, 1 + (2 * L / 20)))  # following (Stone et al., 2017)
                d_x[-1].append(L)
                d_c[-1].append(C)
                tau[-1].append(L / C)
                if C <= route.dx:
                    SL = L
                if TC == 0. and len(d_x[-1]) > 50 and d_x[-1][-1] > d_x[-1][-2]:
                    TC = C

            ipath = np.array(ipath)
            d_x[-1] = np.array(d_x[-1]) / SL * 100
            d_c[-1] = np.array(d_c[-1]) / TC * 100
            tau[-1] = np.array(tau[-1])

            if ri == ni * 10 or True:
                plt.figure("cx-%s-%02d-%s" % (mode, noise * 10, noise_type), figsize=(1.5, 5))
                plt.plot(opath[:, 0], opath[:, 1], 'C%d' % ni, alpha=.5)
                plt.plot(ipath[:, 0], ipath[:, 1], 'C%d' % ni, label=r'$\eta = %.1f$' % noise)
                # plt.plot(opath[:, 0], opath[:, 1], 'r-')
                # plt.plot(ipath[:, 0], ipath[:, 1], 'k--')
                plt.xlim([4, 7])
                plt.ylim([-1, 9])
                plt.legend()

            ri += 1

        print "Maximum tilting: %.2f deg" % np.rad2deg(max_theta)

        d_x_mean = np.mean(d_x, axis=0)
        d_x_se = np.std(d_x, axis=0) / np.sqrt(len(d_x))
        d_c_mean = np.mean(d_c, axis=0)
        tau_mean = np.mean(tau, axis=0)
        tau_se = np.std(tau, axis=0) / np.sqrt(len(tau))

        # plt.figure("distance-%s" % mode, figsize=(3, 3))
        # plt.fill_between(d_c[-1], d_x_mean - 3 * d_x_se, d_x_mean + 3 * d_x_se, facecolor='C%d' % ni, alpha=.5)
        # plt.plot(d_c[-1], d_x_mean, 'C%d' % ni, label=r'$\eta = %.1f$' % noise)
        # plt.ylim([0, 100])
        # plt.xlim([0, 200])
        # plt.legend()
        # # plt.ylabel(r"Distance from home [%]")
        # # plt.xlabel(r"Distance travelled / Turning point distance [%]")
        #
        # plt.figure("tortuosity-%s" % mode, figsize=(3, 3))
        # plt.fill_between(d_c[-1], tau_mean - 3 * tau_se, tau_mean + 3 * tau_se, facecolor='C%d' % ni, alpha=.5)
        # plt.semilogy(d_c[-1], tau_mean, 'C%d' % ni, label=r'$\eta = %.1f$' % noise)
        # plt.ylim([0, 1000])
        # plt.xlim([0, 200])
        # plt.legend()
        # # plt.ylabel(r"Tortuosity of homebound route")
        # # plt.xlabel(r"Distance travelled / Turning point distance [%]")

        print "Noise:", noise
    plt.show()

if __name__ == "__main__2":
    tau = .6

    try:
        # terrain = np.load("terrain-%.2f.npz" % tau)["terrain"]
        terrain = z_terrain
    except IOError:
        terrain = np.zeros_like(z_terrain)
        for i in xrange(terrain.shape[0]):
            print "%04d / %04d" % (i + 1, terrain.shape[0]),
            for j in xrange(terrain.shape[1]):
                k = np.sqrt(np.square(x_terrain[i, j] - x_terrain) + np.square(y_terrain[i, j] - y_terrain)) < tau
                terrain[i, j] = z_terrain[k].mean()
                if j % 20 == 0:
                    print ".",
            print ""

        print terrain.min(), terrain.max()
        np.savez_compressed("terrain-%.2f.npz" % tau, terrain=terrain)

    plt.figure("terrain", figsize=(5, 5))
    plt.imshow(terrain, cmap="coolwarm", extent=[0, 10, 0, 10], vmin=-.5, vmax=.5)
    plt.colorbar()
    plt.show()
