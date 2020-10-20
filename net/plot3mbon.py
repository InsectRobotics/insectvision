import numpy as np
import matplotlib.pyplot as plt


def plot_hist_old(**kwargs):
    fig = plt.figure("6-neuron-responses-old", figsize=(15, 7))
    plt.clf()
    labels = ['Odour A response', 'Odour B response']
    labels_c = ['Odour A weight', 'Odour B weight']

    ax = plt.subplot(231)
    ax.plot(kwargs["m1"])
    for i in [0, 1]:
        if 'm1s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m1s'][:, i], "C%d:" % i, label=labels[i])
        if 'm1c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m1c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot(np.arange(17)[i::2], kwargs["m1"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [15] * 7], 'r-')
    ax.set_ylim([-1, 15])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels(["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
    ax.set_xlim([-2./3., 16 + 1./3.])
    plt.grid(axis='x')

    if 'm1c' in kwargs.keys():
        ax = fig.add_axes([0.2, 0.8, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m1c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    ax = plt.subplot(232)
    ax.plot(kwargs["m2"])
    for i in [0, 1]:
        if 'm2s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m2s'][:, i], "C%d:" % i, label=labels[i])
        if 'm2c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m2c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot(np.arange(17)[i::2], kwargs["m2"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [15] * 7], 'r-')
    ax.set_ylim([-1, 15])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels(["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
    ax.set_xlim([-2./3., 16 + 1./3.])
    plt.grid(axis='x')

    if 'm2c' in kwargs.keys():
        ax = fig.add_axes([0.53, 0.8, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m2c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    ax = plt.subplot(233)
    ax.plot(kwargs["m3"])
    for i in [0, 1]:
        if 'm3s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m3s'][:, i], "C%d:" % i, label=labels[i])
        if 'm3c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m3c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot(np.arange(17)[i::2], kwargs["m3"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [15] * 7], 'r-')
    ax.set_ylim([-1, 15])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels(["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
    ax.set_xlim([-2./3., 16 + 1./3.])
    plt.grid(axis='x')

    if 'm3c' in kwargs.keys():
        ax = fig.add_axes([0.86, 0.8, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['m3c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    ax = plt.subplot(234)
    ax.plot(kwargs["d1"])
    for i in [0, 1]:
        if 'd1s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d1s'][:, i], "C%d:" % i, label=labels[i])
        if 'd1c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d1c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot(np.arange(17)[i::2], kwargs["d1"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [25] * 7], 'r-')
    ax.set_ylim([-1, 25])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels([""] * 17)
    ax.set_xlim([-2./3., 16 + 1./3.])
    ax.xaxis.tick_top()
    ax.legend()
    plt.grid(axis='x')

    if 'd1c' in kwargs.keys():
        ax = fig.add_axes([0.2, 0.3, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d1c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    ax = plt.subplot(235)
    ax.plot(kwargs["d2"])
    for i in [0, 1]:
        if 'd2s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d2s'][:, i], "C%d:" % i, label=labels[i])
        if 'd2c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d2c'][:, i], "C%d:" % (i + 2), label=['CS-', 'CS+'][i])
        ax.plot(np.arange(17)[i::2], kwargs["d2"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [25] * 7], 'r-')
    ax.set_ylim([-1, 25])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels([""] * 17)
    ax.set_xlim([-2./3., 16 + 1./3.])
    ax.xaxis.tick_top()
    plt.grid(axis='x')

    if 'd2c' in kwargs.keys():
        ax = fig.add_axes([0.53, 0.3, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d2c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    ax = plt.subplot(236)
    ax.plot(kwargs["d3"])
    for i in [0, 1]:
        if 'd3s' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d3s'][:, i], "C%d:" % i, label=labels[i])
        if 'd3c' in kwargs.keys():
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d3c'][:, i], "C%d:" % (i + 2), label=['CS-', 'CS+'][i])
        ax.plot(np.arange(17)[i::2], kwargs["d3"][i::2, i], "C%d." % i)
    ax.plot([3, 5, 7, 9, 11, 14, 16], [-1] * 7, 'r*')
    ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-1] * 7, [25] * 7], 'r-')
    ax.set_ylim([-1, 25])
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels([""] * 17)
    ax.set_xlim([-2./3., 16 + 1./3.])
    ax.xaxis.tick_top()
    plt.grid(axis='x')

    if 'd3c' in kwargs.keys():
        ax = fig.add_axes([0.86, 0.3, 0.1, 0.15])
        for i in [0, 1]:
            ax.plot(np.arange(-2./3., 16, 1./3.), kwargs['d3c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
        ax.plot([[3, 5, 7, 9, 11, 14, 16]] * 2, [[-15] * 7, [15] * 7], 'r-', lw=1)
        ax.set_ylim([-15, 15])
        ax.set_xticks(np.arange(17))
        ax.set_xticklabels([""] * 17)
        ax.set_xlim([-2./3., 16 + 1./3.])
        plt.grid()

    plt.tight_layout()
    plt.show()


def plot_hist(**kwargs):
    fig = plt.figure("6-neuron-responses", figsize=(15, 7))
    plt.clf()
    labels = ['Odour A response', 'Odour B response']
    labels_c = ['Odour A weight', 'Odour B weight']

    tau = kwargs.get('tau', 5.)
    neurons = ["m1", "m2", "m3", "d1", "d2", "d3"]
    subplots = [231, 232, 233, 234, 235, 236]
    ylims = [15., 15., 15., 25., 25., 25.]
    inset_recs = [[0.20, 0.80, 0.10, 0.15],
                  [0.53, 0.80, 0.10, 0.15],
                  [0.86, 0.80, 0.10, 0.15],
                  [0.20, 0.30, 0.10, 0.15],
                  [0.53, 0.30, 0.10, 0.15],
                  [0.86, 0.30, 0.10, 0.15]]

    for n, p, ylim, rect in zip(neurons, subplots, ylims, inset_recs):
        ax = fig.add_subplot(p)
        x = np.arange(tau, 340 + tau, tau)
        x_in = np.arange(tau, 340 + tau, 20) + 5
        x_us = np.arange(tau, 340 + tau, 20) + 9
        x_out = np.arange(tau, 340 + tau, 20) + 10
        x_all = np.array([x_in, x_out]).T.flatten()
        x_all = np.array([x_all, x_all]).T.flatten()
        xa = x_all.reshape((-1, 4))[::2].flatten()
        xb = x_all.reshape((-1, 4))[1::2].flatten()
        y1a = np.full_like(xa, -1)
        y1b = np.full_like(xb, -1)
        y2a = np.full_like(xa, ylim)
        y2b = np.full_like(xb, ylim)
        y2a[::4] = -1
        y2b[::4] = -1
        y2a[3::4] = -1
        y2b[3::4] = -1
        ax.fill_between(xa, y1a, y2a, color='C0', alpha=0.1)
        ax.fill_between(xb, y1b, y2b, color='C1', alpha=0.1)
        ax.plot(x_us, kwargs[n])
        for i in [0, 1]:
            if n + 's' in kwargs.keys():
                ax.plot(x, kwargs[n + 's'][:, i], "C%d:" % i, label=labels[i])
            if n + 'c' in kwargs.keys():
                ax.plot(x, kwargs[n + 'c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
            ax.plot(x_us[i::2], kwargs[n][i::2, i], "C%d." % i)
        ax.plot(x_us[[3, 5, 7, 9, 11, 14, 16]], [-1] * 7, 'r*')
        ax.plot([x_us[[3, 5, 7, 9, 11, 14, 16]].tolist()] * 2, [[-1] * 7, [ylim] * 7], 'r-')
        ax.text(0.75, 7. * ylim / 8., kwargs["names"][n], fontsize=14, fontweight='light', backgroundcolor='w')
        ax.set_ylim([-1, ylim])
        ax.set_xticks(np.arange(-10, 350, 20))
        ax.set_xticklabels([""] + ["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
        ax.set_xlim([-25, 345])
        if 'd1' in n:
            ax.legend(loc='upper right')
        plt.grid(axis='x')

        if n + 'c' in kwargs.keys():
            vmax = 150
            ax = fig.add_axes(rect)
            for i in [0, 1]:
                ax.plot(x, kwargs[n + 'c'][:, i], "C%d:" % (i + 2), label=labels_c[i])
            ax.plot([x_us[[3, 5, 7, 9, 11, 14, 16]].tolist()] * 2, [[-vmax] * 7, [vmax] * 7], 'r-', lw=1)
            ax.set_ylim([-vmax, vmax])
            ax.set_xticks(np.arange(-10, 350, 20))
            ax.set_xticklabels([""] * 18)
            ax.set_xlim([-25, 345])
            plt.grid()

    plt.tight_layout()
    plt.show()
