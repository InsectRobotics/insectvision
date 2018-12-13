import numpy as np
import matplotlib.pyplot as plt


def plot_pol_neurons_rotating_linear_polariser(s_1, s_2, r_1, r_2, r_z, r_pol, save_figs=False):
    fontsize = 20

    plt.figure("photoreceptors", figsize=(9, 5))

    ax = plt.subplot2grid((2, 2), (0, 0))
    plt.plot([0, 0], [0, 1.2], "k-", lw=1)
    plt.plot(s_1[0], s_1[1], label=r'$s_\parallel$')
    plt.plot(s_2[0], s_2[1], label=r'$s_\perp$')
    ax.annotate(r'$s_\parallel$', xy=(90, 1), xytext=(110, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    ax.annotate(r'$s_\perp$', xy=(0, 1), xytext=(20, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    plt.ylabel("stimulus", fontsize=fontsize - 1)
    plt.ylim([0, 1.2])
    plt.yticks([0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "", "", "", ""], fontsize=fontsize)
    if save_figs:
        plt.savefig("stimuli.eps")

    ax = plt.subplot2grid((2, 2), (1, 0))
    plt.plot([0, 0], [0, 1.2], "k-", lw=1)
    plt.plot(r_1[0], r_1[1], label=r'$r_\parallel$')
    plt.plot(r_2[0], r_2[1], label=r'$r_\perp$')
    ax.annotate(r'$r_\parallel$', xy=(90, 1), xytext=(110, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    ax.annotate(r'$r_\perp$', xy=(0, 1), xytext=(20, 1.05),
                arrowprops=dict(facecolor='black', arrowstyle="-|>"), fontsize=fontsize)
    plt.ylabel("response", fontsize=fontsize - 1)
    plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([0, 1.2])
    plt.yticks([0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)
    if save_figs:
        plt.savefig("response.eps")

    ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2, sharex=ax)
    plt.plot([0, 0], [-1.2, 1.2], "k-", lw=1)
    plt.plot([-45, 135], [0, 0], "k-", lw=1)
    plt.plot(r_z[0], r_z[1], "k--", label=r'$\frac{r_\parallel - r_\perp}{z}$')
    plt.plot(r_pol[0], r_pol[1], "k-", label=r'$\frac{r_\parallel - r_\perp}{r_\parallel + r_\perp}$')
    plt.legend(fontsize=fontsize)
    plt.ylabel("POL-neuron response", fontsize=fontsize)
    plt.xlabel("e-vector (degrees)", fontsize=fontsize)
    plt.ylim([-1.2, 1.2])
    plt.yticks([-1, 0, 1], fontsize=fontsize)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)
    plt.xlim([-45, 135])
    plt.xticks([-45, 0, 45, 90, 135], ["", "0", "", "90", ""], fontsize=fontsize)

    plt.tight_layout()
    if save_figs:
        plt.savefig("POL-neurons.eps")
    return plt
