import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, ConnectionPatch
from matplotlib.cm import get_cmap

from code.compass import decode_sph
from sphere import sph2vec


def visualise_compnet(sensor, sL=None, interactive=True, cmap="coolwarm", vmin=0, vmax=1, title=None):
    """

    :param sensor:
    :type sensor: CompassSensor
    :return:
    """

    if interactive and not plt.isinteractive():
        plt.ion()

    if isinstance(cmap, basestring):
        cmap = get_cmap(cmap)

    xyz = sph2vec(np.pi/2 - sensor.theta_local, np.pi + sensor.phi_local, sensor.R_c)
    xyz[0] *= -1

    lat, lon = hp.Rotator(rot=(
        np.rad2deg(-sensor.yaw), np.rad2deg(-sensor.pitch), np.rad2deg(-sensor.roll)
    ))(sensor.sky.lat, np.pi-sensor.sky.lon)
    xyz_sun = sph2vec(np.pi/2 - lat, np.pi + lon, sensor.R_c)
    xyz_sun[0] *= -1

    fig = plt.figure("Compass-CompModel" if title is None or interactive else title, figsize=(12, 7))
    fig.clear()
    ax_t = plt.subplot2grid((1, 12), (0, 0), colspan=10)

    outline = Ellipse(xy=np.zeros(2),
                      width=2 * sensor.R_c,
                      height=2 * sensor.R_c)
    sensor_outline = Ellipse(xy=np.zeros(2),
                             width=2 * sensor.alpha + 2 * sensor.R_l,
                             height=2 * sensor.alpha + 2 * sensor.R_l)
    ax_t.add_patch(outline)
    outline.set_clip_box(ax_t.bbox)
    outline.set_alpha(.2)
    outline.set_facecolor("grey")

    stheta, sphi = sensor.theta_local, np.pi + sensor.phi_local
    sL = np.clip(sL if sL is not None else sensor.L, 0., 1.)
    for k, ((x, y, z), th, ph, L) in enumerate(zip(xyz.T, stheta, sphi, sL)):
        for j, tl2 in enumerate(sensor.tl2):
            line = ConnectionPatch([x, y], [sensor.R_c + 5, j * sensor.R_c / 8. - sensor.R_c + 1],
                                   "data", "data", lw=.5, color=cmap(np.asscalar(L) * sensor.w_tl2[k, j] + .5))
            ax_t.add_patch(line)

    ax_t.add_patch(sensor_outline)
    sensor_outline.set_clip_box(ax_t.bbox)
    sensor_outline.set_alpha(.5)
    sensor_outline.set_facecolor("grey")

    for k, ((x, y, z), th, ph, L) in enumerate(zip(xyz.T, stheta, sphi, sL)):
        lens = Ellipse(xy=[x, y], width=1.5 * sensor.r_l, height=1.5 * np.cos(th) * sensor.r_l,
                       angle=np.rad2deg(ph))
        ax_t.add_patch(lens)
        lens.set_clip_box(ax_t.bbox)
        lens.set_facecolor(cmap(np.asscalar(L)))

    scale = 5.
    for j, tl2 in enumerate(sensor.tl2):
        x = sensor.R_c + 5
        y = j * sensor.R_c / 8. - sensor.R_c + 1
        for k, cl1 in enumerate(sensor.cl1):
            line = ConnectionPatch([x, y], [sensor.R_c + 10, k * sensor.R_c / 8. - sensor.R_c + 1],
                                   "data", "data", lw=.5, color=cmap(scale * tl2 * sensor.w_cl1[j, k] + .5))
            ax_t.add_patch(line)
        neuron = Ellipse(xy=[x, y], width=.1 * sensor.R_c, height=.1 * sensor.R_c)
        ax_t.add_artist(neuron)
        neuron.set_clip_box(ax_t.bbox)
        neuron.set_facecolor(cmap(scale * tl2 + .5))

    for j, cl1 in enumerate(sensor.cl1):
        x = sensor.R_c + 10
        y = j * sensor.R_c / 8. - sensor.R_c + 1
        for k, tb1 in enumerate(sensor.tb1):
            line = ConnectionPatch([x, y], [sensor.R_c + 15, k * sensor.R_c / 8. - sensor.R_c / 2. + 1],
                                   "data", "data", lw=.5, color=cmap(scale * cl1 * sensor.w_tb1[j, k] + .5))
            ax_t.add_patch(line)
        neuron = Ellipse(xy=[x, y], width=.1 * sensor.R_c, height=.1 * sensor.R_c)
        ax_t.add_artist(neuron)
        neuron.set_clip_box(ax_t.bbox)
        neuron.set_facecolor(cmap(scale * cl1 + .5))

    for j, tb1 in enumerate(sensor.tb1):
        x = sensor.R_c + 15
        y = j * sensor.R_c / 8. - sensor.R_c / 2 + 1
        neuron = Ellipse(xy=[x, y], width=.1 * sensor.R_c, height=.1 * sensor.R_c)
        ax_t.add_artist(neuron)
        neuron.set_clip_box(ax_t.bbox)
        neuron.set_facecolor(cmap(scale * tb1 + .5))

    ax_t.text(0, sensor.R_c - sensor.r_l, "0", fontsize=15, verticalalignment='center', horizontalalignment='center')
    ax_t.text(-sensor.R_c + sensor.r_l, 0, "90", fontsize=15, verticalalignment='center', horizontalalignment='center')
    ax_t.text(0, -sensor.R_c + sensor.r_l, "180", fontsize=15, verticalalignment='center', horizontalalignment='center')
    ax_t.text(sensor.R_c - sensor.r_l, 0, "-90", fontsize=15, verticalalignment='center', horizontalalignment='center')

    norm = np.sqrt(np.square(xyz_sun[0]) + np.square(xyz_sun[1])) / (sensor.R_c - 3 * sensor.r_l)
    ax_t.plot([-xyz_sun[0]/norm, xyz_sun[0]/norm], [-xyz_sun[1]/norm, xyz_sun[1]/norm], "k-")
    ax_t.plot(xyz_sun[0], xyz_sun[1],
              marker='o',
              fillstyle='full',
              markeredgecolor='black',
              markerfacecolor='yellow',
              markersize=15)

    ax_t.text(0, 0, "z", fontsize=15, verticalalignment='center', horizontalalignment='center',
              color='red', fontweight='bold')

    ax_t.set_xlim(-sensor.R_c - 2, sensor.R_c + 17)
    ax_t.set_ylim(-sensor.R_c - 2, sensor.R_c + 2)
    ax_t.set_xticklabels([])
    ax_t.set_yticklabels([])

    plt.axis('off')

    plt.subplot2grid((4, 12), (1, 10), colspan=2, rowspan=2)
    theta, phi = decode_sph(sensor.tb1)
    alpha = np.linspace(0, 2*np.pi, 100)
    plt.plot(theta * np.sin(phi + alpha + np.pi/2), alpha)
    plt.ylabel("phase (degrees)")
    plt.ylim([0, 2*np.pi])
    plt.xlim([theta+np.pi/360, -theta-np.pi/360])
    plt.xticks([-theta, 0, theta], ["%d" % np.rad2deg(-theta), "0", "%d" % np.rad2deg(theta)])
    plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ["0", "90", "180", "270", "360"])

    plt.tight_layout(pad=0.)

    if interactive:
        plt.draw()
        plt.pause(.1)
    else:
        plt.show()


if __name__ == "__main__":
    from sensor import CompassSensor
    from sky import get_seville_observer
    from datetime import datetime

    s = CompassSensor(fov=np.deg2rad(60), nb_lenses=60, mode="cross", fibonacci=False)
    observer = get_seville_observer()
    observer.date = datetime(2018, 6, 21, 8, 0, 0)
    s.sky.obs = observer

    for i, angle in enumerate(np.linspace(0, 2 * np.pi, 13, endpoint=True)):
        s.refresh()
        y = s()
        sL = (s.L - .5) * 20 + .5
        visualise_compnet(s, sL=sL, interactive=False, title="20180621-0800-network-%03d" % np.rad2deg(angle))
        s.rotate(yaw=np.pi/6)
