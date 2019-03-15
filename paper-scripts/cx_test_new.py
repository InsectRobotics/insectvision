from compoundeye.geometry import angles_distribution
from environment import Sun, Sky, get_seville_observer, eps, load_routes, Hybrid
from sphere.transform import sph2vec, vec2sph, tilt
from code.compass import decode_sph
from net import CX

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# create sun
sun = Sun()
# create sky
sky = Sky()
# create observer
seville_obs = get_seville_observer()

# create terrain
tau = .6
x_terrain = np.linspace(0, 10, 1001, endpoint=True)
y_terrain = np.linspace(0, 10, 1001, endpoint=True)
x_terrain, y_terrain = np.meshgrid(x_terrain, y_terrain)
try:
    z_terrain = np.load("../data/terrain-%.2f.npz" % tau)["terrain"] * 1000
except IOError:
    z_terrain = np.random.randn(*x_terrain.shape) / 50
    terrain = np.zeros_like(z_terrain)

    for i in xrange(terrain.shape[0]):
        print "%04d / %04d" % (i + 1, terrain.shape[0]),
        for j in xrange(terrain.shape[1]):
            k = np.sqrt(np.square(x_terrain[i, j] - x_terrain) + np.square(y_terrain[i, j] - y_terrain)) < tau
            terrain[i, j] = z_terrain[k].mean()
            if j % 20 == 0:
                print ".",
        print ""

    np.savez_compressed("terrain-%.2f.npz" % tau, terrain=terrain)
    z_terrain = terrain


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


def get_3d_direction(x, y, yaw, z_terrain, tau=.06):
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


# sensor design
n = 60
omega = 56
theta, phi, fit = angles_distribution(n, float(omega))
theta_t, phi_t = 0., 0.

# sun position
seville_obs.date = datetime(2018, 6, 21, 9, 0, 0)
sun.compute(seville_obs)
theta_s = np.array([np.pi / 2 - sun.alt])
phi_s = np.array([(sun.az + np.pi) % (2 * np.pi) - np.pi])

# ant-world
noise = 0.0
ttau = .06
dx = 1e-02
# world = load_world()
routes = load_routes()
flow = dx * np.ones(2) / np.sqrt(2)
max_theta = 0.

stats = {
    "max_alt": [],
    "noise": [],
    "opath": [],
    "ipath": [],
    "d_x": [],
    "d_c": [],
    "tau": []
}

terrain = z_terrain.copy()


def create_dataset():
    for max_altitude in [.0, .1, .2, .3, .4, .5]:
        z_terrain = terrain * max_altitude
        print "Maximum altitude has been set to %.2f m (variance [-%.2f m, %.2f m])" % tuple([max_altitude] * 3)
        for ni, noise in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, .97]):
            print "Disturbance level has been set to %.2f %%" % noise

            # stats
            d_x = []  # logarithmic distance
            d_c = []
            tau = []  # tortuosity
            ri = 0

            print "Routes: ",
            for route in routes[::2]:
                net = CX(noise=0., pontin=False)
                net.update = True

                # outward route
                route.condition = Hybrid(tau_x=dx)
                oroute = route.reverse()
                x, y, yaw = [(x0, y0, yaw0) for x0, y0, _, yaw0 in oroute][0]
                opath = [[x, y, yaw]]

                v = np.zeros(2)
                tb1 = []

                for _, _, _, yaw in oroute:
                    theta_t, phi_t = get_3d_direction(opath[-1][0], opath[-1][1], yaw, z_terrain, tau=ttau)
                    max_theta = max_theta if max_theta > np.absolute(theta_t) else np.absolute(theta_t)
                    theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                    sky.theta_s, sky.phi_s = theta_s, phi_s
                    Y, P, A = sky(theta_n, phi_n, noise=noise)

                    r_tb1 = encode(theta, phi, Y, P, A)
                    yaw0 = yaw
                    _, yaw = np.pi - decode_sph(r_tb1) + phi_s

                    net(yaw, flow)
                    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                    v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                    opath.append([opath[-1][0] + v[0], opath[-1][1] + v[1], yaw])
                    tb1.append(net.tb1)
                opath = np.array(opath)

                yaw -= phi_s

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
                    theta_t, phi_t = get_3d_direction(ipath[-1][0], ipath[-1][1], yaw, z_terrain, tau=ttau)
                    theta_n, phi_n = tilt(theta_t, phi_t, theta, phi + yaw)

                    sky.theta_s, sky.phi_s = theta_s, phi_s
                    Y, P, A = sky(theta_n, phi_n, noise=noise)

                    r_tb1 = encode(theta, phi, Y, P, A)
                    _, yaw = np.pi - decode_sph(r_tb1) + phi_s
                    motor = net(yaw, flow)
                    yaw = (ipath[-1][2] + motor + np.pi) % (2 * np.pi) - np.pi
                    v = np.array([np.sin(yaw), np.cos(yaw)]) * route.dx
                    ipath.append([ipath[-1][0] + v[0], ipath[-1][1] + v[1], yaw])
                    tb1.append(net.tb1)
                    L = np.sqrt(np.square(opath[0][0] - ipath[-1][0]) + np.square(opath[0][1] - ipath[-1][1]))
                    C += route.dx
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

                ri += 1

                stats["max_alt"].append(max_altitude)
                stats["noise"].append(noise)
                stats["opath"].append(opath)
                stats["ipath"].append(ipath)
                stats["d_x"].append(d_x)
                stats["d_c"].append(d_c)
                stats["tau"].append(tau)
                print ".",
            print ""

    np.savez_compressed("../data/pi-stats.npz", **stats)


if __name__ == '__main__':
    from plots import plot_route
    stats = np.load("../data/pi-stats.npz")

    ipaths = stats["ipath"]
    opaths = stats["opath"]
    d_xs = stats["d_x"]
    d_cs = stats["d_c"]
    max_alts = stats["max_alt"]
    noises = stats["noise"]
    un_max_alts = np.sort(np.unique(max_alts))
    un_noises = np.sort(np.unique(noises))

    plt.figure("Inclinations", figsize=(15, 30))
    for j, max_alt in enumerate(un_max_alts):
        for id, noise in enumerate(un_noises):
            ipath_f = ipaths[np.all([max_alt == max_alts, noise == noises], axis=0)]
            opath_f = opaths[np.all([max_alt == max_alts, noise == noises], axis=0)]
            for ipath, opath in zip(ipath_f[::2], opath_f[::2]):
                plot_route(opath, ipath, subplot=(len(un_noises), len(un_max_alts), 1 + len(un_noises) * j + id))
    # plt.legend()
    plt.savefig("cx-detailed.svg")
    plt.show()
