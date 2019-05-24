from base import Network, RNG
from willshawnet import generate_pn2kc_weights
from data.visualisation import plot_matrix

import numpy as np
import pandas as pd
import yaml
import os


# get path of the script
__dir__ = os.path.dirname(os.path.abspath(__file__))
# load parameters
with open(os.path.join(__dir__, 'Aso2014_simplified.yaml'), 'rb') as f:
    params = yaml.safe_load(f)


class MushroomBody(Network):

    def __init__(self, **kwargs):
        super(MushroomBody, self).__init__(**kwargs)

        # initialised PN values
        types = sorted(params['PN'].keys())
        self.pn = pd.concat([
            pd.concat([pd.DataFrame(np.zeros(n, dtype=float)) for _, n in params['PN'][t]],
                      keys=[s for s, _ in params['PN'][t]],
                      names=["name", "index"], axis=0) for t in types],
            keys=types, names=["cluster", "name", "index"], axis=0
        ).T  # type:pd.DataFrame
        self.nb_pn = self.pn.shape[1]

        # KCs
        types = sorted(params['KC'].keys())
        self.kc = pd.concat([
            pd.concat([pd.DataFrame(np.zeros(n, dtype=float)) for _, n in params['KC'][t]],
                      keys=[s for s, _ in params['KC'][t]],
                      names=["name", "index"], axis=0) for t in types],
            keys=types, names=["cluster", "name", "index"], axis=0
        ).T  # type:pd.DataFrame
        self.nb_kc = self.kc.shape[1]

        # MBONs
        types = sorted(params['MBON'].keys())
        self.mbon = pd.concat([
            pd.concat([pd.DataFrame(np.zeros(n, dtype=float)) for _, n in params['MBON'][t]],
                      keys=[s for s, _ in params['MBON'][t]],
                      names=["transmitter", "index"], axis=0) for t in types],
            keys=types, names=["type", "name", "index"], axis=0
        ).T  # type:pd.DataFrame
        self.nb_mbon = self.mbon.shape[1]

        # DANs
        types = sorted(params['DAN'].keys())[:2]
        self.dan = pd.concat([
            pd.concat([pd.DataFrame(np.zeros(n, dtype=float)) for _, n in params['DAN'][t]],
                      keys=[s for s, _ in params['DAN'][t]],
                      names=["name", "index"], axis=0) for t in types],
            keys=types, names=["cluster", "name", "index"]
        ).T  # type:pd.DataFrame
        self.nb_dan = self.dan.shape[1]

        locs = sorted(params['LOC2DAN'].keys())

        # PN2KC
        pn_names = self.pn.columns.levels[1][self.pn.columns.codes[1]]
        self.w_pn2kc = generate_pn2kc_weights(self.nb_pn, self.nb_kc, min_pn=5, max_pn=21, dtype=float)
        self.w_pn2kc = pd.DataFrame(self.w_pn2kc, index=pn_names, columns=locs, dtype=float)
        self.w_pn2kc.index = self.pn.columns

        # KC2LOC
        kc_names = self.kc.columns.levels[1][self.kc.columns.codes[1]]
        self.w_kc2loc = np.zeros((self.nb_kc, len(locs)), dtype=float)
        for i, key in enumerate(kc_names):
            for loc in params['KC2LOC'][key]:
                j = locs.index(loc)
                self.w_kc2loc[i, j] = 1.
        self.w_kc2loc = pd.DataFrame(self.w_kc2loc, index=kc_names, columns=locs, dtype=float)
        self.w_kc2loc.index = self.kc.columns

        # DAN2LOC
        # self.dan = self.dan.sort_index(axis=1, level=[0, 1])
        dan_names = self.dan.columns.levels[1][self.dan.columns.codes[1]]
        self.w_dan2loc = np.zeros((self.dan.shape[1], len(locs)), dtype=float)
        for i, key in enumerate(dan_names):
            for loc in params['DAN2LOC'][key]:
                j = locs.index(loc)
                self.w_dan2loc[i, j] = 1.
        self.w_dan2loc = pd.DataFrame(self.w_dan2loc, index=dan_names, columns=locs, dtype=float)
        self.w_dan2loc.index = self.dan.columns

        # MBON2LOC
        # self.mbon = self.mbon.sort_index(axis=1, level=[0, 1])
        mbon_names = self.mbon.columns.levels[1][self.mbon.columns.codes[1]]
        self.w_mbon2loc = np.zeros((self.mbon.shape[1], len(locs)), dtype=float)
        for i, key in enumerate(mbon_names):
            for loc in params['MBON2LOC'][key]:
                j = locs.index(loc)
                self.w_mbon2loc[i, j] = 1.
        self.w_mbon2loc = pd.DataFrame(self.w_mbon2loc, index=mbon_names, columns=locs, dtype=float)
        self.w_mbon2loc.index = self.mbon.columns

        # LOC2DAN
        self.w_loc2dan = np.zeros((len(locs), self.dan.shape[1]), dtype=float)
        for loc, keys in params['LOC2DAN'].iteritems():
            j = locs.index(loc)
            for key in keys:
                self.w_loc2dan[j, dan_names == key] = 1.
        self.w_loc2dan = pd.DataFrame(self.w_loc2dan, index=locs, columns=dan_names, dtype=float)
        self.w_loc2dan.columns = self.dan.columns

        # LOC2MBON
        self.w_loc2mbon = np.zeros((len(locs), self.mbon.shape[1]), dtype=float)
        for loc, keys in params['LOC2MBON'].iteritems():
            j = locs.index(loc)
            for key in keys:
                self.w_loc2mbon[j, mbon_names == key] = 1.
        self.w_loc2mbon = pd.DataFrame(self.w_loc2mbon, index=locs, columns=mbon_names, dtype=float)
        self.w_loc2mbon.columns = self.mbon.columns

        self.f_pn = lambda x: x
        self.f_kc = lambda x: x
        self.f_mbon = lambda x: x
        self.f_dan = lambda x: x

    def __call__(self, *args, **kwargs):
        return None

    def _fprop(self, pn):
        a_pn = self.f_pn(pn)

        kc = a_pn.dot(self.w_pn2kc)
        # TODO: Add connections from CZ to KC
        a_kc = self.f_kc(kc)

        loc = a_kc.dot(self.w_kc2loc) + self.dan.dot(self.w_dan2loc) + self.mbon.dot(self.w_mbon2loc)

        mbon = loc.dot(self.w_loc2mbon)
        a_mbon = self.f_mbon(mbon)

        dan = loc.dot(self.w_loc2dan)
        a_dan = self.f_dan(dan)

        return a_pn, a_kc, a_dan, a_mbon

    def _update(self, kc):
        pass


if __name__ == "__main__":
    mb = MushroomBody()
    print mb.kc


if __name__ == "__main__" and False:
    import matplotlib.pyplot as plt

    mb = MushroomBody()
    # print np.ones_like(mb.mbon).dot(mb.w_mbon2loc)

    print "KCs", mb.nb_kc
    print mb.kc

    print "MBONs", mb.nb_mbon
    print mb.mbon

    print "DANs", mb.nb_dan
    print mb.dan

    inp = pd.concat([mb.w_mbon2loc, mb.w_dan2loc], axis=0).dot(mb.w_kc2loc.T)
    out = mb.w_kc2loc.dot(pd.concat([mb.w_loc2mbon, mb.w_loc2dan], axis=1))
    M = inp + out.T  # type: pd.DataFrame
    M = M.T.corr()

    plt.figure("model-matrix", figsize=(10, 10))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.imshow(M, vmin=-1, vmax=1, cmap="coolwarm", origin='lower', aspect="equal")
    plt.xlim([-.5, M.shape[0] - .5])
    plt.ylim([-.5, M.shape[1] - .5])

    ax2.imshow(M, vmin=-1, vmax=1, cmap="coolwarm", origin='lower', aspect="equal")
    plt.xlim([-.5, M.shape[0] - .5])
    plt.ylim([-.5, M.shape[1] - .5])

    codes0 = np.array(M.columns.codes[0])
    levels0 = np.array(M.columns.levels[0])

    codes1 = np.array(M.columns.codes[1])
    levels1 = np.array(M.columns.levels[1])

    x0, x1 = [], []
    labels0, labels1 = [], []
    for label0 in levels0:
        for label1 in levels1:
            q = np.argwhere(np.all([levels0[codes0] == label0, levels1[codes1] == label1], axis=0))
            if len(q) == 0:
                continue
            x_ = np.max(q) + .5
            ax2.plot([-.5, M.shape[1] - .5], [x_, x_], 'b--', lw=.5)
            ax2.plot([x_, x_], [-.5, M.shape[1] - .5], 'b--', lw=.5)
            x1.append(q.mean())
            labels1.append(label1)
        q = np.argwhere(levels0[codes0] == label0)
        if len(q) == 0:
            continue
        x_ = np.max(q) + .5
        ax2.plot([-.5, M.shape[1] - .5], [x_, x_], 'k-', lw=1)
        ax2.plot([x_, x_], [-.5, M.shape[1] - .5], 'k-', lw=1)
        x0.append(q.mean())
        labels0.append(label0)

    ax1.xaxis.set_ticks(x1)
    ax1.xaxis.set_ticklabels(labels1, rotation='vertical')
    ax1.xaxis.set_tick_params(which='major', labelsize=10)
    ax1.yaxis.set_ticks(x1)
    ax1.yaxis.set_ticklabels(labels1)
    ax1.yaxis.set_tick_params(which='major', labelsize=10)

    ax2.imshow(M, vmin=-1, vmax=1, cmap="coolwarm", origin='lower', aspect="equal")
    ax2.xaxis.set_ticks(x1)
    ax2.xaxis.set_ticklabels(labels1, rotation='vertical')
    ax2.xaxis.set_tick_params(which='major', labelsize=10)
    ax2.yaxis.set_ticks(x0)
    ax2.yaxis.set_ticklabels(labels0)
    ax2.yaxis.set_tick_params(which='major', labelsize=10)

    plt.tight_layout()
    plt.show()
    # print mb.dan.columns.levels[1]
