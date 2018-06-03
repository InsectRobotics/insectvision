import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    I = 1500.
    pol = .9  # degree of polarasation
    fontsize = 20

    aop = np.linspace(-np.pi/4, 3*np.pi/4, 361)

    # interneuron1 = np.log(I * (np.square(np.sin(aop)) + np.square(np.cos(aop)) * np.square(1. - pol)) + 1) / np.log(2)
    # interneuron2 = np.log(I * (np.square(np.cos(aop)) + np.square(np.sin(aop)) * np.square(1. - pol)) + 1) / np.log(2)
    interneuron1 = np.sqrt(I) * np.sqrt(np.square(np.sin(aop)) + np.square(np.cos(aop)) * np.square(1. - pol))
    interneuron2 = np.sqrt(I) * np.sqrt(np.square(np.cos(aop)) + np.square(np.sin(aop)) * np.square(1. - pol))

    I = max(interneuron1.max(), interneuron2.max())
    plt.figure("photoreceptors-%02d" % (pol * 10), figsize=(5, 9))
    ax = plt.subplot2grid((3, 1), (0, 0))
    plt.plot([0, 0], [0, 1.2], "k-", lw=1)
    plt.plot(np.rad2deg(aop), interneuron1 / I, label=r'$r_\parallel$')
    plt.plot(np.rad2deg(aop), interneuron2 / I, label=r'$r_\perp$')
    ax.annotate(r'$r_\parallel$', xy=(90, 1), xytext=(110, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    ax.annotate(r'$r_\perp$', xy=(0, 1), xytext=(20, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    # plt.legend(fontsize=fontsize)
    plt.ylabel("response", fontsize=fontsize-1)
    # plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([0, 1.2])
    plt.yticks([0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)
    # plt.savefig("photoreceptors-%02d.eps" % (pol * 10))
    # plt.figure("Interneurons", figsize=(5, 8))
    # plt.plot([0, 0], [-1, 2], "k-", lw=1)
    # plt.plot([-45, 135], [0, 0], "k-", lw=1)
    # plt.plot(np.rad2deg(aop), interneuron1 - interneuron2, label=r'$r_\parallel - r_\perp$')
    # plt.plot(np.rad2deg(aop), interneuron1 + interneuron2, label=r'$r_\parallel + r_\perp$')
    # plt.legend(fontsize=14)
    # plt.ylabel("response")
    # plt.xlabel("e-vector (degrees)")
    # plt.ylim([-1, 2])
    # plt.xlim([-np.pi/4, 3*np.pi/4])
    # plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""])

    # sahaneuron = np.log(interneuron1) - np.log(interneuron2)
    polneuron = (interneuron1 - interneuron2) / (interneuron1 + interneuron2)
    # polneuron = (np.log(interneuron2) - np.log(interneuron1)) / (np.log(interneuron1) + np.log(interneuron2))

    # plt.figure("POL-neurons-%02d" % (pol * 10), figsize=(5, 6))
    ax = plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=ax)
    plt.plot([0, 0], [-1.2, 1.2], "k-", lw=1)
    plt.plot([-45, 135], [0, 0], "k-", lw=1)
    plt.plot(np.rad2deg(aop), (interneuron1 - interneuron2) / I, "k--", label=r'$\frac{r_\parallel - r_\perp}{z}$')
    plt.plot(np.rad2deg(aop), polneuron, "k-", label=r'$\frac{r_\parallel - r_\perp}{r_\parallel + r_\perp}$')
    # plt.plot(np.rad2deg(aop), sahaneuron, label=r'$\log(r_\parallel) - \log(r_\perp)$')
    # plt.plot(np.rad2deg(aop), sahaneuron / sahaneuron.max(), label=r'$\eta \cdot [\log(r_\parallel) - \log(r_\perp)]$')
    plt.legend(fontsize=fontsize)
    plt.ylabel("POL-neuron response", fontsize=fontsize)
    plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([-1.2, 1.2])
    plt.yticks([-1, 0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)

    plt.tight_layout()
    plt.savefig("POL-neurons-%02d.eps" % (pol * 10))

    plt.show()
