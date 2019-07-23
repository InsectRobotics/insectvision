from base import Network, RNG

import numpy as np
import yaml
import os

# get path of the script
__dir__ = os.path.dirname(os.path.abspath(__file__))
# load parameters
with open(os.path.join(__dir__, 'Ardin2016.yaml'), 'rb') as f:
    params = yaml.safe_load(f)

GAIN = params['gain']
LEARNING_RATE = params['learning-rate']
KC_THRESHOLD = params['kc-threshold']


class WillshawNet(Network):

    def __init__(self, learning_rate=LEARNING_RATE, tau=KC_THRESHOLD, nb_channels=1, **kwargs):
        """

        :param learning_rate: the rate with which the weights are changing
        :type learning_rate: float
        :param tau: the threshold after witch a KC is activated
        :type tau: float
        :param nb_channels: number of colour channels that can be interpreted
        :type nb_channels: int
        """
        super(WillshawNet, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self._tau = tau
        self.nb_channels = nb_channels

        self.nb_pn = params['mushroom-body']['PN'] * nb_channels
        self.nb_kc = params['mushroom-body']['KC'] * nb_channels
        self.nb_en = params['mushroom-body']['EN']

        self.w_pn2kc = generate_pn2kc_weights(self.nb_pn, self.nb_kc, dtype=self.dtype)
        self.w_kc2en = np.ones((self.nb_kc, self.nb_en), dtype=self.dtype)
        self.params = [self.w_pn2kc, self.w_kc2en]

        self.f_pn = lambda x: np.maximum(self.dtype(x) / self.dtype(255), 0)
        # self.f_pn = lambda x: np.maximum(self.dtype(self.dtype(x) / self.dtype(255) > .5), 0)
        self.f_kc = lambda x: self.dtype(x > tau)
        self.f_en = lambda x: np.maximum(x, 0)

        self.pn = np.zeros(self.nb_pn)
        self.kc = np.zeros(self.nb_kc)
        self.en = np.zeros(self.nb_en)

        self.__update = False

    def reset(self):
        super(WillshawNet, self).reset()

        self.pn = np.zeros(self.nb_pn)
        self.kc = np.zeros(self.nb_kc)
        self.en = np.zeros(self.nb_en)

        self.w_kc2en = np.ones((self.nb_kc, self.nb_en), dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        self.pn, self.kc, self.en = self._fprop(args[0])
        if self.__update:
            self._update(self.kc)
        return self.en

    def _fprop(self, pn):
        a_pn = self.f_pn(pn)
        kc = a_pn.dot(self.w_pn2kc)
        a_kc = self.f_kc(kc)
        en = a_kc.dot(self.w_kc2en)
        a_en = self.f_en(en)
        return a_pn, a_kc, a_en

    def _update(self, kc):
        """
            THE LEARNING RULE:
        ----------------------------

          KC  | KC2EN(t)| KC2EN(t+1)
        ______|_________|___________
           1  |    1    |=>   0
           1  |    0    |=>   0
           0  |    1    |=>   1
           0  |    0    |=>   0

        :param kc: the KC activation
        :return:
        """
        learning_rule = (kc >= self.w_kc2en[:, 0]).astype(bool)
        self.w_kc2en[:, 0][learning_rule] = np.maximum(self.w_kc2en[:, 0][learning_rule] - self.learning_rate, 0)


def generate_pn2kc_weights(nb_pn, nb_kc, min_pn=5, max_pn=21, aff_pn2kc=None, nb_trials=100000, baseline=25000,
                           rnd=RNG, dtype=np.float32):
    """
    Create the synaptic weights among the Projection Neurons (PNs) and the Kenyon Cells (KCs).
    Choose the first sample that has dispersion below the baseline (early stopping), or the
    one with the lower dispersion (in case non of the samples' dispersion is less than the
    baseline).

    :param nb_pn:       the number of the Projection Neurons (PNs)
    :param nb_kc:       the number of the Kenyon Cells (KCs)
    :param min_pn:
    :param max_pn:
    :param aff_pn2kc:   the number of the PNs connected to every KC (usually 28-34)
                        if the number is less than or equal to zero it creates random values
                        for each KC in range [28, 34]
    :param nb_trials:   the number of trials in order to find a acceptable sample
    :param baseline:    distance between max-min number of projections per PN
    :param rnd:
    :type rnd: np.random.RandomState
    :param dtype:
    """

    dispersion = np.zeros(nb_trials)
    best_pn2kc = None

    for trial in range(nb_trials):
        pn2kc = np.zeros((nb_pn, nb_kc), dtype=dtype)

        if aff_pn2kc is None or aff_pn2kc <= 0:
            vaff_pn2kc = rnd.randint(min_pn, max_pn + 1, size=nb_pn)
        else:
            vaff_pn2kc = np.ones(nb_pn) * aff_pn2kc

        # go through every kenyon cell and select a nb_pn PNs to make them afferent
        for i in range(nb_pn):
            pn_selector = rnd.permutation(nb_kc)
            pn2kc[i, pn_selector[:vaff_pn2kc[i]]] = 1

        # This selections mechanism can be used to restrict the distribution of random connections
        #  compute the sum of the elements in each row giving the number of KCs each PN projects to.
        pn2kc_sum = pn2kc.sum(axis=0)
        dispersion[trial] = pn2kc_sum.max() - pn2kc_sum.min()
        # pn_mean = pn2kc_sum.mean()

        # Check if the number of projections per PN is balanced (min max less than baseline)
        #  if the dispersion is below the baseline accept the sample
        if dispersion[trial] <= baseline: return pn2kc

        # cache the pn2kc with the least dispersion
        if best_pn2kc is None or dispersion[trial] < dispersion[:trial].min():
            best_pn2kc = pn2kc

    # if non of the samples have dispersion lower than the baseline,
    # return the less dispersed one
    return best_pn2kc


if __name__ == "__main__":
    from world import load_world, load_routes
    from agent.visualiser import Visualiser
    from world.conditions import Hybrid

    world = load_world()
    routes = load_routes()
    routes[0].condition = Hybrid(tau_x=.1, tau_phi=np.pi)
    world.add_route(routes[0])

    nn = WillshawNet(nb_channels=3)
    nn.update = True
    vis = Visualiser(mode="panorama")
    vis.reset()

    x, y, z = np.zeros(3)
    phi = 0.

    def world_snapshot(width=None, height=None):
        global x, y, z, phi
        return world.draw_panoramic_view(x, y, z, phi, update_sky=False, include_ground=.3, include_sky=1.,
                                         width=width, length=width, height=height)

    for x, y, z, phi in world.routes[-1]:

        if vis.is_quit():
            print "QUIT!"
            break

        img, _ = world.draw_panoramic_view(x, y, z, phi)
        inp = np.array(img).reshape((-1, 3))
        en = nn(inp.flatten())

        vis.update_main(world_snapshot, caption="PN: %3d, KC: %3d, EN: %3d" % (nn.pn.sum(), nn.kc.sum(), en.sum()))
