import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

from fruitfly import DataFrame

eps = np.finfo(float).eps


def plot_matrix(M, title="", labels1=None, labels2=None, vmin=-1., vmax=1., verbose=False, figsize=(10.7, 10)):
    if verbose:
        print "M_max: %.2f, M_min: %.2f" % (M.max(), M.min())
    plt.figure(title, figsize=figsize)
    ax1 = plt.gca()
    if labels2 is not None:
        ax2 = ax1.twinx()

    ax1.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", origin='lower', aspect="equal")
    plt.xlim([-.5, M.shape[1]-.5])
    plt.ylim([-.5, M.shape[1]-.5])

    if labels2 is not None:
        ax2.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", origin='lower', aspect="equal")
        plt.xlim([-.5, M.shape[1]-.5])
        plt.ylim([-.5, M.shape[1]-.5])

    types = [""] if labels2 is None else np.unique(labels2)
    names = np.unique(labels1)
    tp_ticks, nm_ticks = [], []
    for tp in types:
        for nm in names:
            if labels2 is None:
                q = np.argwhere(labels1 == nm)
            else:
                q = np.argwhere(np.all([labels1 == nm, labels2 == tp], axis=0))
            if len(q) == 0:
                continue
            if labels2 is not None:
                x0 = np.max(q) + .5
                ax2.plot([-.5, M.shape[1]-.5], [x0, x0], 'b--', lw=.5)
                ax2.plot([x0, x0], [-.5, M.shape[1]-.5], 'b--', lw=.5)
            nm_ticks.append([q.mean(), nm])
        if labels2 is not None:
            q = np.argwhere(labels2 == tp)
            x0 = np.max(q) + .5
            ax2.plot([-.5, M.shape[1]-.5], [x0, x0], 'k-', lw=1)
            ax2.plot([x0, x0], [-.5, M.shape[1]-.5], 'k-', lw=1)
        tp_ticks.append([q.mean(), tp])

    tp_ticks = np.array(tp_ticks)
    nm_ticks = np.array(nm_ticks)
    ax1.xaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.xaxis.set_ticklabels(nm_ticks[:, 1], rotation='vertical')
    ax1.xaxis.set_tick_params(which='major', labelsize=10)
    ax1.yaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.yaxis.set_ticklabels(nm_ticks[:, 1])
    ax1.yaxis.set_tick_params(which='major', labelsize=10)

    if labels2 is not None:
        ax2.xaxis.set_ticks(np.float32(nm_ticks[:, 0]))
        ax2.xaxis.set_ticklabels(nm_ticks[:, 1], rotation='vertical')
        ax2.xaxis.set_tick_params(which='major', labelsize=10)
        ax2.yaxis.set_ticks(np.float32(tp_ticks[:, 0]))
        ax2.yaxis.set_ticklabels(tp_ticks[:, 1])
        ax2.yaxis.set_tick_params(which='major', labelsize=10)

    plt.tight_layout()


def corr_matrix(df, mode="all", avg=False, abs=False, diff=False, shock=True, show=True, figsize=(10.7, 10)):

    df = DataFrame.normalise(df.astype(float))

    if diff:
        df1 = df.T[:-200].T
        df2 = df.T[200:].T
        df1.columns = [i for i, x in enumerate(df1.columns)]
        df2.columns = [i for i, x in enumerate(df2.columns)]
        df = df2.sub(df1)

    if avg:
        df = df.groupby(['type', 'name', 'genotype'], axis=0).mean()

    mask = np.zeros(df.T.shape[0], dtype=bool)
    if type(mode) is not list:
        mode = [mode]
    if None in mode or "all" in mode:
        mask[:] = 1
    if "training" in mode:
        mask[-1500:-300] = 1
    if "pretrain" in mode or "trial-1" in mode:
        mask[:-1500] = 1
    if "trial-2" in mode:
        mask[-1500:-1300] = 1
    if "trial-3" in mode:
        mask[-1300:-1100] = 1
    if "trial-4" in mode:
        mask[-1100:-900] = 1
    if "trial-5" in mode:
        mask[-900:-700] = 1
    if "trial-6" in mode:
        mask[-700:-500] = 1
    if "trial-7" in mode:
        mask[-500:-300] = 1
    if "reversal" in mode or "trial-8" in mode:
        mask[-200:] = 1
    if "iter-1" in mode and not diff:
        mask[:-1600] = 1
    if "iter-2" in mode and not diff:
        mask[-1600:-1500] = 1
    if "iter-3" in mode:
        mask[-1500:-1400] = 1
    if "iter-4" in mode:
        mask[-1400:-1300] = 1
    if "iter-5" in mode:
        mask[-1300:-1200] = 1
    if "iter-6" in mode:
        mask[-1200:-1100] = 1
    if "iter-7" in mode:
        mask[-1100:-1000] = 1
    if "iter-8" in mode:
        mask[-1000:-900] = 1
    if "iter-9" in mode:
        mask[-900:-800] = 1
    if "iter-10" in mode:
        mask[-800:-700] = 1
    if "iter-11" in mode:
        mask[-700:-600] = 1
    if "iter-12" in mode:
        mask[-600:-500] = 1
    if "iter-13" in mode:
        mask[-500:-400] = 1
    if "iter-14" in mode:
        mask[-400:-300] = 1
    if "iter-15" in mode:
        mask[-300:-200] = 1
    if "iter-16" in mode:
        mask[-200:-100] = 1
    if "iter-17" in mode:
        mask[-100:] = 1
    if not shock:
        cols = np.array(([1] * 44 + [0] * 56) * 17, dtype=bool)
        if diff:
            cols = cols[:-200]
        mask = np.all([mask, cols], axis=0)
    print mask

    names = df.index.levels[1][df.index.codes[1]]
    types = df.index.levels[0][df.index.codes[0]]

    corr = df.T.astype(float).loc[mask].corr()
    if abs:
        corr = corr.abs()
    corr.columns = names
    corr.index = names

    if show:
        plot_matrix(corr, title="cc-matrix-%s%s%s%s%s" % (
            mode[0],
            "-avg" if avg else "",
            "-abs" if abs else "",
            "" if shock else "-noshock",
            "-diff" if diff else ""),
                    vmin=-1., vmax=1.,
                    labels1=names.values.astype('unicode'),
                    labels2=types.values.astype('unicode'), figsize=figsize)
        plt.show()

    return corr, names.values.astype('unicode'), types.values.astype('unicode')


def cross_corr_matrix(df, mode1="all", mode2="all", avg=False, diff=False, shock=True, show=True):

    df = DataFrame.normalise(df.astype(float))

    if diff:
        df1 = df.T[:-200].T
        df2 = df.T[200:].T
        df1.columns = [i for i, x in enumerate(df1.columns)]
        df2.columns = [i for i, x in enumerate(df2.columns)]
        df = df2.sub(df1)

    if avg:
        df = df.groupby(['type', 'name', 'genotype'], axis=0).mean()

    mask1 = np.zeros(df.T.shape[0], dtype=bool)
    mask2 = np.zeros_like(mask1)

    for mode, mask in zip([mode1, mode2], [mask1, mask2]):
        if mode is not list:
            mode = [mode]
        if None in mode or "all" in mode:
            mask[:] = 1
        if "training" in mode:
            mask[-1500:-300] = 1
        if "pretrain" in mode or "trial-1" in mode:
            mask[:-1500] = 1
        if "trial-2" in mode:
            mask[-1500:-1300] = 1
        if "trial-3" in mode:
            mask[-1300:-1100] = 1
        if "trial-4" in mode:
            mask[-1100:-900] = 1
        if "trial-5" in mode:
            mask[-900:-700] = 1
        if "trial-6" in mode:
            mask[-700:-500] = 1
        if "trial-7" in mode:
            mask[-500:-300] = 1
        if "reversal" in mode or "trial-8" in mode:
            mask[-200:] = 1
        if "iter-1" in mode and not diff:
            mask[:-1600] = 1
        if "iter-2" in mode and not diff:
            mask[-1600:-1500] = 1
        if "iter-3" in mode:
            mask[-1500:-1400] = 1
        if "iter-4" in mode:
            mask[-1400:-1300] = 1
        if "iter-5" in mode:
            mask[-1300:-1200] = 1
        if "iter-6" in mode:
            mask[-1200:-1100] = 1
        if "iter-7" in mode:
            mask[-1100:-1000] = 1
        if "iter-8" in mode:
            mask[-1000:-900] = 1
        if "iter-9" in mode:
            mask[-900:-800] = 1
        if "iter-10" in mode:
            mask[-800:-700] = 1
        if "iter-11" in mode:
            mask[-700:-600] = 1
        if "iter-12" in mode:
            mask[-600:-500] = 1
        if "iter-13" in mode:
            mask[-500:-400] = 1
        if "iter-14" in mode:
            mask[-400:-300] = 1
        if "iter-15" in mode:
            mask[-300:-200] = 1
        if "iter-16" in mode:
            mask[-200:-100] = 1
        if "iter-17" in mode:
            mask[-100:] = 1

    if not shock:
        cols = np.array(([1] * 44 + [0] * 56) * 17, dtype=bool)
        if diff:
            cols = cols[:-200]
        mask1 = np.all([mask1, cols], axis=0)
        mask2 = np.all([mask2, cols], axis=0)

    names = df.index.levels[1][df.index.codes[1]]
    types = df.index.levels[0][df.index.codes[0]]

    df1 = df.T.astype(float).loc[mask1]  # type: pd.DataFrame
    df2 = df.T.astype(float).loc[mask2]  # type: pd.DataFrame
    df1.columns = names
    df1.index = np.arange(100) if shock else np.arange(44)
    df2.columns = names
    df2.index = np.arange(100) if shock else np.arange(44)
    corr = pd.concat([df1, df2], axis=1).corr()
    corr = corr[df1.shape[1]:].T[df2.shape[1]:].T

    # corr.columns = names
    # corr.index = names

    if show:
        plot_matrix(corr, title="cc-matrix-%s-vs-%s%s%s" % (
            mode1, mode2,
            "-avg" if avg else "",
            "-diff" if diff else ""),
                    vmin=-1., vmax=1.,
                    labels1=names.values.astype('unicode'),
                    labels2=types.values.astype('unicode'))
        plt.show()

    return corr


def plot_covariance(df, plot_pca_2d=False):

    xs = df.T[5:].astype(float)
    x_max = xs.max(axis=1)
    xs = (xs.T / (x_max + eps)).T

    C = xs.T.dot(xs) / xs.shape[0]
    v = .02
    plot_matrix(C, title="fly-covariance-matrix",
                labels1=df['name'], labels2=df['type'], vmin=-v, vmax=v, verbose=True)

    if plot_pca_2d:
        from sklearn.decomposition import PCA

        pca = PCA(xs.shape[1], whiten=False)
        pca.fit(xs)
        x_proj = pca.transform(xs)

        types = np.unique(df['type'])

        plt.figure("pca-types", figsize=(9, 9))
        colours = {
            "KC": "black",
            "MBON-ND": "grey",
            "MBON-glu": "green",
            "MBON-gaba": "blue",
            "MBON-ach": "red",
            "PAM": "cyan",
            "PPL1": "magenta"
        }
        for t in types:
            x0 = x_proj.T[df['type'] == t]
            plt.scatter(x0[:, 0], x0[:, 1], color=colours[t], marker=".", label=t)
        plt.xlim([-.25, .25])
        plt.ylim([-.25, .25])
        plt.legend()

        plt.figure("pca-location", figsize=(9, 9))
        colours = {
            u"\u03b1": "red",
            u"\u03b2": "green",
            u"\u03b1'": "pink",
            u"\u03b2'": "greenyellow",
            u"\u03b3": "blue"
        }

        for loc in np.sort(colours.keys()):
            q = np.where([loc in name for name in df['name']])
            x0 = x_proj.T[q]
            plt.scatter(x0[:, 0], x0[:, 1], color=colours[loc], marker=".", label=loc)
        plt.xlim([-.25, .25])
        plt.ylim([-.25, .25])
        plt.legend()

    plt.show()


def plot_mutual_information(df, diff=False):
    from sklearn.feature_selection import mutual_info_regression

    if diff:
        df1 = df.T[:-200].T
        df2 = df.T[200:].T
        df1.columns = [i for i, x in enumerate(df1.columns)]
        df2.columns = [i for i, x in enumerate(df2.columns)]
        df = df2.sub(df1)

    xs = df.T.astype(float)

    names = df.index.levels[1][df.index.codes[1]]
    types = df.index.levels[0][df.index.codes[0]]
    MI = np.zeros((xs.shape[1], xs.shape[1]), dtype=float)
    try:
        MI = np.load("mi%s.npz" % ("-diff" if diff else ""))['MI']
    except IOError:
        for i, x in enumerate(xs.T.values):
            print i,
            for j, y in enumerate(xs.T.values):
                m = mutual_info_regression(x[..., np.newaxis], y)[0]
                print "%.2f" % m,
                MI[i, j] = m
            print ''
        np.savez_compressed("mi%s.npz" % ("-diff" if diff else ""), MI=MI)

    MI = pd.DataFrame(MI, index=names, columns=types)

    v = .5
    plot_matrix(MI, title="fly-mutual-information%s" % ("-diff" if diff else ""),
                labels1=names.values.astype('unicode'),
                labels2=types.values.astype('unicode'), vmin=-v, vmax=v)
    plt.show()


def plot_f_score(df, diff=False):
    from sklearn.feature_selection import f_regression

    if diff:
        df1 = df.T[:-200].T
        df2 = df.T[200:].T
        df1.columns = [i for i, x in enumerate(df1.columns)]
        df2.columns = [i for i, x in enumerate(df2.columns)]
        df = df2.sub(df1)

    xs = df.T.astype(float)

    names = df.index.levels[1][df.index.codes[1]]
    types = df.index.levels[0][df.index.codes[0]]
    F = np.zeros((xs.shape[1], xs.shape[1]), dtype=float)
    try:
        F = np.load("f-score%s.npz" % ("-diff" if diff else ""))['F']
    except IOError:
        for i, x in enumerate(xs.T.values):
            print i,
            for j, y in enumerate(xs.T.values):
                f = f_regression(x[..., np.newaxis], y)[0]
                print "%.2f" % f,
                F[i, j] = f
            print ''
        np.savez_compressed("f-score%s.npz" % ("-diff" if diff else ""), F=F)

    F = pd.DataFrame(F, index=names, columns=types)

    v = 2000.
    plot_matrix(F, title="fly-f-score%s" % ("-diff" if diff else ""),
                labels1=names.values.astype('unicode'),
                labels2=types.values.astype('unicode'), vmin=-v, vmax=v)
    plt.show()


def pairplot(df, cols=None):
    if cols is None or not cols:
        cols = []
        types = df['type'].unique().astype('unicode')
        for tp in types:
            cols.append(df[df['type'].values.astype('unicode') == tp].index[0])

    types = df['type'].unique().astype('unicode')

    x = df.T[cols][5:].astype(float)  # type: pd.DataFrame
    x.columns = types

    pp = sns.pairplot(x, size=1.8, aspect=1.8, plot_kws=dict(edgecolor='k', linewidth=0.5),
                      diag_kind='kde', diag_kws=dict(shade=True))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    fig.show()

    plt.show()


def plot_corr_over_time(df, shock=True):

    cors = []
    for i in xrange(0, 17):
        c = corr_matrix(df.sort_index(axis=0, level=['type', 'name']), mode="iter-%d" % (i + 1),
                        shock=shock, show=False)
        # cors.append(np.sqrt(np.sum(np.sum(np.square(c)))))
        cors.append(np.sum(np.sum(np.square(c)))/float(c.size))

    plt.figure("corr-over-time%s" % ("" if shock else "-noshock"), figsize=(10, 10))

    cors = np.array(cors)

    plt.plot(np.arange(9), cors[0::2], "C0-", lw=2, label="CS-")
    plt.plot(np.arange(8) + .5, cors[1::2], "C1-", lw=2, label="CS+")
    plt.plot([1.5, 2.5, 3.5, 4.5, 5.5, 7, 8], cors[[3, 5, 7, 9, 11, 14, 16]], 'ro', lw=1)

    cors = []
    for i in xrange(2, 17):
        c = corr_matrix(df.sort_index(axis=0, level=['type', 'name']), mode="iter-%d" % (i + 1),
                        shock=shock, diff=True, show=False)
        # cors.append(np.sqrt(np.sum(np.sum(np.square(c)))))
        cors.append(np.sum(np.sum(np.square(c)))/float(c.size))

    plt.figure("corr-over-time%s" % ("" if shock else "-noshock"), figsize=(10, 10))

    cors = np.array(cors)

    plt.plot(np.arange(1, 9), cors[0::2], "C0--", lw=2, label="CS- (change)")
    plt.plot(np.arange(1, 8) + .5, cors[1::2], "C1--", lw=2, label="CS+ (change)")
    plt.plot([1.5, 2.5, 3.5, 4.5, 5.5, 7, 8], cors[[1, 3, 5, 7, 9, 12, 14]], 'ro', lw=1)

    plt.xticks(np.arange(0, 8.5, .5), [
        "1-", "1+", "2-", "2+", "3-", "3+", "4-", "4+", "5-", "5+", "6-", "6+", "7-", "7+", "8-", "8+", "9-"])
    plt.ylim([0, .5])
    plt.xlim([-.5, 8.5])
    plt.xlabel("Trial")
    plt.ylabel(r'$\frac{1}{N^2}\sum{C^2}$')
    plt.legend()


def plot_cross_corr_over_time(df, shock=True):

    cors = []
    for i in xrange(0, 15):
        c = cross_corr_matrix(df.sort_index(axis=0, level=['type', 'name']), shock=shock,
                              mode1="iter-%d" % (i + 1), mode2="iter-%d" % (i + 3), show=False)
        # cors.append(np.sqrt(np.sum(np.sum(np.square(c)))))
        cors.append(np.sum(np.sum(np.square(c)))/float(c.size))

    plt.figure("cross-corr-over-time%s" % ("" if shock else "-noshock"), figsize=(10, 10))

    cors = np.array(cors)

    plt.plot(np.arange(8), cors[0::2], "C0-", lw=2, label="CS-")
    plt.plot(np.arange(7) + .5, cors[1::2], "C1-", lw=2, label="CS+")
    plt.plot([0.5, 1.5, 2.5, 3.5, 4.5, 6, 7], cors[[1, 3, 5, 7, 9, 12, 14]], 'ro', lw=1)

    cors = []
    for i in xrange(2, 15):
        c = cross_corr_matrix(df.sort_index(axis=0, level=['type', 'name']), diff=True, shock=shock,
                              mode1="iter-%d" % (i + 1), mode2="iter-%d" % (i + 3), show=False)
        # cors.append(np.sqrt(np.sum(np.sum(np.square(c)))))
        cors.append(np.sum(np.sum(np.square(c)))/float(c.size))

    cors = np.array(cors)

    plt.plot(np.arange(1, 8), cors[0::2], "C0--", lw=2, label="CS- (change)")
    plt.plot(np.arange(1, 7) + .5, cors[1::2], "C1--", lw=2, label="CS+ (change)")
    plt.plot([1.5, 2.5, 3.5, 4.5, 6, 7], cors[[1, 3, 5, 7, 10, 12]], 'ro', lw=1)

    plt.xticks(np.arange(0, 7.5, .5), [
        "1- vs 2-", "1+ vs 2+", "2- vs 3-", "2+ vs 3+", "3- vs 4-", "3+ vs 4+", "4- vs 5-", "4+ vs 5+", "5- vs 6-",
        "5+ vs 6+", "6- vs 7-", "6+ vs 7+", "7- vs 8-", "7+ vs 8+", "8- vs 9-"], rotation="vertical")
    plt.ylim([0, .5])
    plt.xlim([-.5, 7.5])
    plt.xlabel("Trial")
    plt.ylabel(r'$\frac{1}{N^2}\sum{C^2}$')
    plt.legend()


def plot_iter_corr_matrix(df, sort_by=None, ascending=None, diff=False, shock=True):
    # df = DataFrame.normalise(df.astype(float))

    if sort_by is None:
        sort_by = ['trial', 'condition']
    if ascending is None:
        ascending = [True, False]

    if diff:
        df1 = df.T[:-200].T
        df2 = df.T[200:].T
        df1.columns = [i for i, x in enumerate(df1.columns)]
        df2.columns = [i for i, x in enumerate(df2.columns)]
        df = df2.sub(df1)

    if not shock:
        cols = np.array(([1] * 44 + [0] * 56) * 17, dtype=bool)
        if diff:
            cols = cols[:-200]
        df = df.T.loc[cols].T

    df = df.T.reset_index(level='shock', drop=True)
    df = df.reorder_levels(['trial', 'condition', 'time'], axis=0)
    df = df.unstack([-1]).sort_index(axis=0, by=sort_by, ascending=ascending)

    trials = df.index.levels[0][df.index.codes[0]]
    conditions = df.index.levels[1][df.index.codes[1]]
    corr = df.T.astype(float).corr()

    plt.figure("iter-corr-matrix%s" % ("" if shock else "-noshock"), figsize=(5.4, 5))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", origin='lower', aspect="equal")
    if sort_by[0] == 'trial':
        plt.plot([-.5, 16.5], [3, 3], 'r--')
        plt.plot([-.5, 16.5], [5, 5], 'r--')
        plt.plot([-.5, 16.5], [7, 7], 'r--')
        plt.plot([-.5, 16.5], [9, 9], 'r--')
        plt.plot([-.5, 16.5], [11, 11], 'r--')
        plt.plot([-.5, 16.5], [14, 14], 'r--')
        plt.plot([-.5, 16.5], [16, 16], 'r--')
        plt.plot([3, 3], [-.5, 16.5], 'r--')
        plt.plot([5, 5], [-.5, 16.5], 'r--')
        plt.plot([7, 7], [-.5, 16.5], 'r--')
        plt.plot([9, 9], [-.5, 16.5], 'r--')
        plt.plot([11, 11], [-.5, 16.5], 'r--')
        plt.plot([14, 14], [-.5, 16.5], 'r--')
        plt.plot([16, 16], [-.5, 16.5], 'r--')
        plt.plot([-.5, 16.5], [1.5, 1.5], 'k-')
        plt.plot([-.5, 16.5], [11.5, 11.5], 'k-')
        plt.plot([-.5, 16.5], [12.5, 12.5], 'k-')
        plt.plot([1.5, 1.5], [-.5, 16.5], 'k-')
        plt.plot([11.5, 11.5], [-.5, 16.5], 'k-')
        plt.plot([12.5, 12.5], [-.5, 16.5], 'k-')
        plt.plot([11.5, 12.5] * 9, np.linspace(-.5, 16.5, 18), 'k--')
        plt.plot(np.linspace(-.5, 16.5, 18), [12.5, 11.5] * 9, 'k--')
    elif sort_by[0] == 'condition':
        plt.plot([-.5, 16.5], [10, 10], 'r--')
        plt.plot([-.5, 16.5], [11, 11], 'r--')
        plt.plot([-.5, 16.5], [12, 12], 'r--')
        plt.plot([-.5, 16.5], [13, 13], 'r--')
        plt.plot([-.5, 16.5], [14, 14], 'r--')
        plt.plot([-.5, 16.5], [8, 8], 'r--')
        plt.plot([-.5, 16.5], [7, 7], 'r--')
        plt.plot([10, 10], [-.5, 16.5], 'r--')
        plt.plot([11, 11], [-.5, 16.5], 'r--')
        plt.plot([12, 12], [-.5, 16.5], 'r--')
        plt.plot([13, 13], [-.5, 16.5], 'r--')
        plt.plot([14, 14], [-.5, 16.5], 'r--')
        plt.plot([8, 8], [-.5, 16.5], 'r--')
        plt.plot([7, 7], [-.5, 16.5], 'r--')
        plt.plot([8.5, 8.5], [-.5, 16.5], 'k-')
        plt.plot([-.5, 16.5], [8.5, 8.5], 'k-')
        plt.plot([6.5, 6.5], [-.5, 16.5], 'k--')
        plt.plot([5.5, 5.5], [-.5, 16.5], 'k--')
        plt.plot([0.5, 0.5], [-.5, 16.5], 'k--')
        plt.plot([9.5, 9.5], [-.5, 16.5], 'k--')
        plt.plot([14.5, 14.5], [-.5, 16.5], 'k--')
        plt.plot([-.5, 16.5], [6.5, 6.5], 'k--')
        plt.plot([-.5, 16.5], [5.5, 5.5], 'k--')
        plt.plot([-.5, 16.5], [0.5, 0.5], 'k--')
        plt.plot([-.5, 16.5], [9.5, 9.5], 'k--')
        plt.plot([-.5, 16.5], [14.5, 14.5], 'k--')
        plt.plot([6.5, 5.5] * 9, np.linspace(-.5, 16.5, 18), 'k--')
        plt.plot(np.linspace(-.5, 16.5, 18), [5.5, 6.5] * 9, 'k--')

    plt.yticks(range(17), ["%d %s" % (trial, condition) for trial, condition in zip(trials, conditions)])
    plt.xticks(range(17), ["%d %s" % (trial, condition) for trial, condition in zip(trials, conditions)],
               rotation="vertical")

    print corr


def plot_traces(df, title="traces", vmin=-20, vmax=20, normalise=False, avg=False, diff=False, verbose=False):
    if verbose:
        print "M_max: %.2f, M_min: %.2f" % (df.max(), df.min())

    # set the ticks of the left (1) axis to the names of the neurons
    labels1 = df.index.levels[1][df.index.codes[1]].values.astype('unicode')
    # set the ticks of the right (2) axis to the types of the neurons
    labels2 = df.index.levels[0][df.index.codes[0]].values.astype('unicode')

    plt.figure(title, figsize=(20, 10))
    ax1 = plt.gca()  # create main axis
    ax2 = ax1.twinx()  # create secondary axis

    img = df.astype(float)
    if normalise:
        img = DataFrame.normalise(img)
        vmin = -1
        vmax = 1
    if diff:
        img1 = img.T[:-200].T
        img2 = img.T[200:].T
        img1.columns = [i for i, x in enumerate(img1.columns)]
        img2.columns = [i for i, x in enumerate(img2.columns)]
        img = img2.sub(img1)
    if avg:
        img = img.groupby(['type', 'name', 'genotype'], axis=0).mean()

    ax1.imshow(img, vmin=vmin, vmax=vmax, cmap="coolwarm",
               interpolation='nearest', origin="lower", aspect="auto")
    plt.xlim([-.5, img.shape[1]-.5])
    plt.ylim([-.5, img.shape[0]-.5])

    ax2.imshow(img, vmin=vmin, vmax=vmax, cmap="coolwarm",
               interpolation='nearest', origin="lower", aspect="auto")
    plt.xlim([-.5, img.shape[1]-.5])
    plt.ylim([-.5, img.shape[0]-.5])

    types = np.unique(labels2)
    names = np.unique(labels1)
    tp_ticks, nm_ticks = [], []
    for tp in types:
        for nm in names:
            q = np.argwhere(np.all([labels1 == nm, labels2 == tp], axis=0))
            if len(q) == 0:
                continue
            x0 = np.max(q) + .5
            ax2.plot([0, img.shape[1]], [x0, x0], 'b--', lw=.5)
            nm_ticks.append([q.mean(), nm])
        q = np.argwhere(labels2 == tp)
        x0 = np.max(q) + .5
        ax2.plot([0, img.shape[1]], [x0, x0], 'k-', lw=1)
        tp_ticks.append([q.mean(), tp])

    x_ticks = []
    shock = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    if diff:
        shock = shock[2:]
    for trial, s in zip(xrange(img.shape[1] / 100), shock):
        plt.plot([trial * 100, trial * 100], [-.5, img.shape[0]], 'k-', lw=.5)
        if s:
            plt.plot([trial * 100 + 45, trial * 100 + 45], [-.5, img.shape[0]], 'r-', lw=.5)

        trial1 = trial / 2 + 1
        trial2 = trial / 2 + 2
        if diff:
            x_ticks.append([trial * 100 + 50, "Trial %d - %d\nCS%s" % (trial2, trial1, '-' if trial % 2 == 0 else '+')])
        else:
            x_ticks.append([trial * 100 + 50, "Trial %d\nCS%s" % (trial1, '-' if trial % 2 == 0 else '+')])

    tp_ticks = np.array(tp_ticks)
    nm_ticks = np.array(nm_ticks)
    x_ticks = np.array(x_ticks)
    # plt.yticks(np.float32(nm_ticks[:, 0]), nm_ticks[:, 1])
    # plt.xticks(np.float32(x_ticks[:, 0]), x_ticks[:, 1])

    ax1.xaxis.set_ticks(np.float32(x_ticks[:, 0]))
    ax1.xaxis.set_ticklabels(x_ticks[:, 1])
    ax1.xaxis.set_tick_params(which='major', labelsize=10)
    ax1.yaxis.set_ticks(np.float32(nm_ticks[:, 0]))
    ax1.yaxis.set_ticklabels(nm_ticks[:, 1])
    ax1.yaxis.set_tick_params(which='major', labelsize=10)

    ax2.xaxis.set_ticks(np.float32(x_ticks[:, 0]))
    ax2.xaxis.set_ticklabels(x_ticks[:, 1])
    ax2.xaxis.set_tick_params(which='major', labelsize=10)
    ax2.yaxis.set_ticks(np.float32(tp_ticks[:, 0]))
    ax2.yaxis.set_ticklabels(tp_ticks[:, 1])
    ax2.yaxis.set_tick_params(which='major', labelsize=10)

    plt.tight_layout()


m_count = 0


def plot_traces_over_time(df, group=None, types=["type"], normalise=False, shock=True, diff=False, merge=0):
    global m_count

    ymax = 5
    ymin = 0
    if normalise:
        df = DataFrame.normalise(df)
        ymax = .15
    if diff:
        ymax = 3.5
        ymin = -3.5
        if normalise:
            ymax = .5
            ymin = -.5

    if diff:
        columns = df.columns
        df1 = pd.concat([df.T[:200]]*9)[:-100].T  # type: pd.DataFrame
        df1.columns = columns
        df2 = df
        df = df2.sub(df1)

    if not shock:
        cols = np.array(([1] * 44 + [0] * 56) * 17, dtype=bool)
        df = df.T.astype(float).loc[cols].T

    dff = df.T.groupby(by=["trial", "condition"]).mean().T  # type: pd.DataFrame

    cond = []
    if group is not None:
        print group
        if type(group) is not list:
            group = [group]
        for g in group:
            if g == "MBON":
                group.append("MBON-ACh")
                group.append("MBON-Glu")
                group.append("MBON-GABA")
                group.append("MBON-ND")
                continue
            elif g == "DAN":
                group.append("PAM")
                group.append("PPL1")
                continue

            if type(g) is list:
                c = np.all([dff.index.get_level_values(t) == gg for t, gg in zip(types, g)], axis=0)
            else:
                c = dff.index.get_level_values(types[0]) == g
            cond.append(c)
        dfff = dff.iloc[np.any(cond, axis=0)]
        print dfff

    dff_csm = dfff.T.iloc[dfff.columns.get_level_values('condition') == 'CS-'].T.mean()
    dff_csp = dfff.T.iloc[dfff.columns.get_level_values('condition') == 'CS+'].T.mean()

    plt.figure("trails-over-time%s%s" % (
        "" if shock else "-noshock",
        "" if group is None or merge > 0 else "-%s" % str.join("|", group if type(g) is not list else group[0])),
               figsize=(10, 2.5))

    par = ""
    if merge > 0:
        par += str.join('|', group)
        if diff:
            par += "; "
    # if diff:
    #     par += "change"

    print m_count, par
    plt.plot(np.arange(0, 9), dff_csm,
             "C%d--" % m_count, lw=2, label="CS- (%s)" % par if par is not None else "CS-")
    plt.plot(np.arange(0, 8) + .5, dff_csp,
             "C%d-" % m_count, lw=2, label="CS+ (%s)" % par if par is not None else "CS+")
    plt.plot([1.5, 2.5, 3.5, 4.5, 5.5, 7, 8],
             np.append(dff_csp[[1, 2, 3, 4, 5]], dff_csm[[7, 8]]),
             'ro', lw=1)

    plt.xticks(np.arange(0, 8.5, .5), [
        "1-", "1+", "2-", "2+", "3-", "3+", "4-", "4+", "5-", "5+", "6-", "6+", "7-", "7+", "8-", "8+", "9-"])
    plt.ylim([ymin, ymax])
    plt.xlim([-.5, 8.5])
    plt.xlabel("Trial")
    plt.ylabel(r'$\frac{1}{T}\sum{r(t)}$')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    m_count += 1
    if m_count >= merge:
        m_count = 0


def plot_overall_response(df, title="traces", vmin=-20, vmax=20, normalise=False, diff=False, verbose=False):
    if verbose:
        print "M_max: %.2f, M_min: %.2f" % (df.max(), df.min())

    plt.figure(title, figsize=(20, 10))

    # img = df.T[5:].T.astype(float)  # type: DataFrame
    img = df.astype(float)  # type: DataFrame
    if normalise:
        img = DataFrame.normalise(img)  # type: DataFrame
    if diff:
        img1 = img.T[:-200].T
        img2 = img.T[200:].T
        img1.columns = [i for i, x in enumerate(img1.columns)]
        img2.columns = [i for i, x in enumerate(img2.columns)]
        img = img2.sub(img1)  # type: DataFrame

    # img = img.groupby(['type', 'name', 'genotype']).mean()
    img = img.groupby(['type']).mean()

    x_ticks = []
    shock = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    if diff:
        shock = shock[2:]
    for trial, s in zip(xrange(img.shape[1] / 100), shock):
        plt.plot([trial, trial], [-.1, 1.1], 'k-', lw=1)
        if s:
            plt.plot([trial + .45, trial + .45], [-.1, 1.1], 'r-', lw=1)

        trial1 = trial / 2 + 1
        trial2 = trial / 2 + 2
        if diff:
            x_ticks.append([trial + .5, "Trial %d - %d" % (trial2, trial1), 'CS-' if trial % 2 == 0 else 'CS+'])
        else:
            x_ticks.append([trial + .5, "Trial %d" % trial1, 'CS-' if trial % 2 == 0 else 'CS+'])

    x_ticks = np.array(x_ticks)
    img_mean = img.T.groupby(['trial', 'condition']).mean().T
    img_std = img.T.groupby(['trial', 'condition']).std().T / np.sqrt(100.)

    colours = {
        'KC': ['black', 'black'],
        'MBON-ND': ['grey', 'grey'],
        'MBON-ACh': ['red', 'red'],
        'MBON-GABA': ['blue', 'blue'],
        'MBON-Glu': ['green', 'green'],
        'PAM': ['cyan', 'cyan'],
        'PPL1': ['magenta', 'magenta']
    }
    for label, y, y_err in zip(img_mean.index.values, img_mean.values, img_std.values):
        x = np.arange(.5, 17.5, 1)
        if diff:
            x = x[:-2]

        plt.fill_between(x[0::2], y[0::2] + y_err[0::2], y[0::2] - y_err[0::2], color=colours[label][0], alpha=.5)
        plt.plot(x[0::2], y[0::2], color=colours[label][0], label=label + " - CS-")
        plt.fill_between(x[1::2], y[1::2] + y_err[1::2], y[1::2] - y_err[1::2], color=colours[label][1], alpha=.5)
        plt.plot(x[1::2], y[1::2], color=colours[label][1], label=label + " - CS+")
    plt.legend()

    plt.xticks(np.float32(x_ticks[:, 0]), x_ticks[:, 1])
    plt.ylim([-.1, .2])

    plt.tight_layout()


def plot_3_mbon_traces(df):
    dan_types = [r"PPL1", r"PAM", r"PAM"]
    dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
    mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]

    cond = []
    for group in [[dan_types[0], dan_names[0]], [mbon_types[0], mbon_names[0]],
                  [dan_types[1], dan_names[1]], [mbon_types[1], mbon_names[1]],
                  [dan_types[2], dan_names[2]], [mbon_types[2], mbon_names[2]]]:
        c = np.all([df.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
        cond.append(c)
    data_i = df.iloc[np.any(cond, axis=0)]
    data_avg = data_i.groupby(["type", "name"], axis=0).mean()  # type: pd.DataFrame
    data_se = data_i.groupby(["type", "name"], axis=0).std() / np.sqrt(
        data_i.groupby(["type", "name"], axis=0).count())  # type: pd.DataFrame
    # print data_i

    plt.figure("traces", figsize=(10, 5))
    for i in xrange(3):
        plt.subplot(231 + i)
        t = np.linspace(0, 340, 1700, endpoint=False)
        m = data_avg.T[mbon_types[i], mbon_names[i]].T.to_numpy(dtype=float)
        e = data_se.T[mbon_types[i], mbon_names[i]].T.to_numpy(dtype=float)
        mm = m.reshape((-1, 100))
        me = e.reshape((-1, 100))
        sm = mm[:, 44:49].mean(axis=1)
        se = me[:, 44:49].mean(axis=1)
        mmin = mm.min()
        mmax = mm.max()
        # sm = 20. * (sm - mmin) / (mmax - mmin) + 5.
        # se = 20. * (se - mmin) / (mmax - mmin)
        me = (me.sum(axis=1) - me[:, 44:49].sum(axis=1)) / 95.
        # me = 20. * (me - mmin) / (mmax - mmin)
        mm = (mm.sum(axis=1) - mm[:, 44:49].sum(axis=1)) / 35.
        # mm = 20. * (mm - mmin) / (mmax - mmin) + 5.
        plt.fill_between(t, m - e, m + e, color='grey', alpha=0.5)
        plt.plot(t, m, 'k-')
        plt.fill_between(t[40::200], mm[::2] - me[::2], mm[::2] + me[::2], color='blue', alpha=.2)
        plt.plot(t[40::200], mm[::2], 'bo-')
        plt.fill_between(t[140::200], mm[1::2] - me[1::2], mm[1::2] + me[1::2], color='red', alpha=.2)
        plt.plot(t[140::200], mm[1::2], 'ro-')
        plt.fill_between(t[40::200], sm[::2] - se[::2], sm[::2] + se[::2], color='blue', alpha=.1)
        plt.plot(t[40::200], sm[::2], 'b^--')
        plt.fill_between(t[140::200], sm[1::2] - se[1::2], sm[1::2] + se[1::2], color='red', alpha=.1)
        plt.plot(t[140::200], sm[1::2], 'r^--')
        # plt.errorbar(t[40::200], mm[::2], me[::2], c='blue', marker='o', linestyle='-')
        # plt.errorbar(t[140::200], mm[1::2], me[1::2], c='red', marker='o', linestyle='-')
        # plt.errorbar(t[40::200], sm[::2], se[::2], c='blue', marker='^', linestyle='--')
        # plt.errorbar(t[140::200], sm[1::2], se[1::2], c='red', marker='^', linestyle='--')
        plt.xticks(range(0, 340, 20), [""] * 17)
        plt.xlim([0, 340])
        plt.ylim([-1, 15])
        plt.title("%s-%s" % ("MBON", mbon_names[i]))

        plt.subplot(234 + i)
        d = data_avg.T[dan_types[i], dan_names[i]].T.to_numpy(dtype=float)
        e = data_se.T[dan_types[i], dan_names[i]].T.to_numpy(dtype=float)
        md = d.reshape((-1, 100))
        me = e.reshape((-1, 100))
        sd = md[:, 44:49].mean(axis=1)
        se = me[:, 44:49].mean(axis=1)
        dmin = md.min()
        dmax = md.max()
        # sd = 15. * (sd - dmin) / (dmax - dmin) + 5.
        # se = 15. * (se - dmin) / (dmax - dmin)
        me = (me.sum(axis=1) - me[:, 44:49].sum(axis=1)) / 95.
        # me = 15. * (me - md.min()) / (md.max() - md.min())
        md = (md.sum(axis=1) - md[:, 44:49].sum(axis=1)) / 35.
        # md = 15. * (md - md.min()) / (md.max() - md.min()) + 5.
        plt.fill_between(t, d - e, d + e, color='grey', alpha=0.5)
        plt.plot(t, d, 'k-')
        plt.fill_between(t[40::200], md[::2] - me[::2], md[::2] + me[::2], color='blue', alpha=.2)
        plt.plot(t[40::200], md[::2], 'bo-')
        plt.fill_between(t[140::200], md[1::2] - me[1::2], md[1::2] + me[1::2], color='red', alpha=.2)
        plt.plot(t[140::200], md[1::2], 'ro-')
        plt.fill_between(t[40::200], sd[::2] - se[::2], sd[::2] + se[::2], color='blue', alpha=.1)
        plt.plot(t[40::200], sd[::2], 'b^--')
        plt.fill_between(t[140::200], sd[1::2] - se[1::2], sd[1::2] + se[1::2], color='red', alpha=.1)
        plt.plot(t[140::200], sd[1::2], 'r^--')
        plt.xticks(range(0, 340, 20), [s % (j // 2 + 1) for j, s in enumerate(["%d-", "%d+"] * 8 + ["%d-"])])
        plt.xlim([0, 340])
        plt.ylim([-1, 25])
        plt.title("%s-%s" % (dan_types[i], dan_names[i]))
    plt.tight_layout()


def plot_3_mbon_shock(df):
    dan_types = [r"PPL1", r"PAM", r"PAM"]
    dan_names = [u"\u03b31pedc", u"\u03b2'2m", u"\u03b2'2a"]
    mbon_types = [r"MBON-GABA", r"MBON-Glu", r"MBON-Glu"]
    mbon_names = [u"\u03b31pedc", u"\u03b2'2mp", u"\u03b35\u03b2'2a"]

    cond = []
    for group in [[dan_types[0], dan_names[0]], [mbon_types[0], mbon_names[0]],
                  [dan_types[1], dan_names[1]], [mbon_types[1], mbon_names[1]],
                  [dan_types[2], dan_names[2]], [mbon_types[2], mbon_names[2]]]:
        c = np.all([df.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
        cond.append(c)
    data_i = df.iloc[np.any(cond, axis=0)]

    data_shock = data_i.T.groupby(["condition", "trial", "shock"], axis=0
                                  ).mean().T.groupby(["type", "name", "id"], axis=0).mean()
    print data_shock
    data_shock = (data_shock.T.query('shock == True').reset_index(level='shock', drop=True) -
                  data_shock.T.query('shock == False').reset_index(level='shock', drop=True)).T

    plt.figure("shock-boxes", figsize=(15, 5))
    for i in xrange(3):
        plt.subplot(231 + i)

        m = data_shock.T[mbon_types[i], mbon_names[i]].T.to_numpy(dtype=float)
        m /= (max(m.max(), -m.min()) * 1.2)
        plt.fill_between([-1, 9.5], [-1, -1], [1, 1], color='blue', alpha=.2)
        plt.fill_between([9.5, 18], [-1, -1], [1, 1], color='red', alpha=.2)
        plt.plot([-1, 18], [0, 0], 'k-', alpha=.2)
        plt.boxplot(np.concatenate([m[:, 8:], m[:, :8]], axis=1), sym='k+',
                    labels=['%d-' % (j + 1) for j in range(9)] + ['%d+' % (j + 1) for j in range(8)])
        plt.ylim([-.4, 1])
        plt.title("%s-%s" % ("MBON", mbon_names[i]))

        plt.subplot(234 + i)

        d = data_shock.T[dan_types[i], dan_names[i]].T.to_numpy(dtype=float)
        d /= (max(d.max(), -d.min()) * 1.2)
        plt.fill_between([-1, 9.5], [-1, -1], [1, 1], color='blue', alpha=.2)
        plt.fill_between([9.5, 18], [-1, -1], [1, 1], color='red', alpha=.2)
        plt.plot([-1, 18], [0, 0], 'k-', alpha=.2)
        plt.boxplot(np.concatenate([d[:, 8:], d[:, :8]], axis=1), sym='k+',
                    labels=['%d-' % (j + 1) for j in range(9)] + ['%d+' % (j + 1) for j in range(8)])
        plt.ylim([-.2, 1])
        plt.title("%s-%s" % (dan_types[i], dan_names[i]))
    plt.tight_layout()

