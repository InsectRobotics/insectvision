import numpy as np
from plot3mbon import plot_hist_old


class Neuron(object):
    def __init__(self, a=.5, b=0., c_shock=1., c_odour=1., u=0., d=2, gamma=0.1, utype="td"):
        """

        :param a: learning rate -- minus, plus
        :type a: np.ndarray, float
        :param b: bias -- background activity, rest response
        :type b: np.ndarray, float
        :param c_shock: shock affection constant
        :type c_shock: np.ndarray, float
        :param c_odour: odour affection constant
        :type c_odour: np.ndarray, float
        :param u: update of shock constant parameter
        :type u: np.ndarray, float
        :param d: dimensions of value
        :type d: int
        :param gamma: the discount value that sets the horizon
        :type gamma: float
        :param utype: the update type of the neuron
        :type utype: basestring
        """
        self._eta = np.array(a) if type(a) is not float and len(a) == d else np.full(d, a)
        self._c_shock = np.array(c_shock) if type(c_shock) is not float and len(c_shock) == d else np.full(2, c_shock)
        self._c_odour = np.array(c_odour) if type(c_odour) is not float and len(c_odour) == d else np.full(2, c_odour)
        self._u = np.array(u) if type(u) is not float and len(u) == d else np.full(2, u)
        self._b = b
        self._d = d
        self._v = np.zeros(d, dtype=float)  # the value of the neuron
        self._e = np.zeros_like(self._v)
        self._w_odour = np.zeros((2, d), dtype=float)
        self._w_shock = np.zeros((2, d), dtype=float)
        self.w_odour = np.eye(2, d, dtype=float)
        self.w_shock = np.eye(2, d, dtype=float)
        self.gamma = gamma
        assert utype.lower() in ["td", "mc", "dp"], "'utype' must be one of 'td' (TD-learning), 'mc' (Monte Carlo) " \
                                                    "or 'dp' (Dynamic Programming)."
        self.utype = utype.lower()
        self._f = lambda x: np.maximum(x)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v):
        self._v = v

    @property
    def w_shock(self):
        return self._c_shock * self._w_shock

    @w_shock.setter
    def w_shock(self, v):
        self._w_shock = v

    @property
    def w_odour(self):
        return self._c_odour * self._w_odour

    @w_odour.setter
    def w_odour(self, v):
        self._w_odour = v

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        return self.__fprop(odour=odour, shock=shock)
        # return self.__update(odour=odour, shock=shock)

    def __fprop(self, odour, shock):
        i = odour > 0
        isk = np.int32(odour * shock)

        # observations
        s = (shock * odour).dot(self.w_shock)  # allocate the shock obs to the corresponding odour identity
        o = odour.dot(self.w_odour)  # scale the odour obs

        # update the values
        eta = self._eta[isk][i]
        v0 = self.v[i]
        v1 = self.v[~i]

        # targets
        t = (o + s + self._b)[i]
        # t_pleasure, t_pain, t_desire, t_fear = t

        # errors
        e = self._e[i] = (t - v0)
        # e_pleasure, e_pain, e_desire, e_fear = e

        if self.utype == "td":  # TD learning
            self.v[i] += eta * (self.gamma * v1 + e.sum())
        elif self.utype == "mc":  # Monte Carlo
            self.v[i] += eta * e.sum()
        elif self.utype == "dp":  # Dynamic programming
            vpos = self.v - self.v.min()
            vpr = 1. if vpos.sum() == 0. else vpos[~i] / vpos.sum()
            self.v[i] += eta * (vpr * (t.sum() + self.gamma * self.v[~i])).sum()

        return self.v

    def __update(self, odour, shock):
        s = (shock * odour).dot(self.w_shock)
        o = odour.dot(self.w_odour)
        e = s - self.v * o + self._b
        self._v += odour * self._eta[np.int32(np.int32(s > 0) * odour)] * e
        self._v += self._u

        # THE update
        # self.w_odour += .07 * odour * self.v


class Network(object):
    def __init__(self, dan, mbon):
        self._dan = list(dan)
        self._mbon = list(mbon)
        self.neurons = self._dan + self._mbon

        self.w_dan2mbon = np.zeros((len(dan), len(mbon)), dtype=float)
        self.w_mbon2dan = np.zeros((len(mbon), len(dan)), dtype=float)
        self.w_mbon2mbon = np.zeros((len(mbon), len(mbon)), dtype=float)

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        self.__update(odour=odour, shock=shock)
        return [m.v for m in self._mbon] + [d.v for d in self._dan]

    def __update(self, odour, shock):
        i = odour > 0
        for odr, sck in zip([0. * odour, odour, odour], [0., 0., shock]):
            # odour activates KCs which drive activity in each MBON
            bm = np.array([m._b for m in self._mbon])
            m = np.array([m(odr, sck)[i] for m in self._mbon])

            # initial DAN activity is zero
            bd = np.array([d._b for d in self._dan])
            d = np.array([d(odr, sck)[i] for d in self._dan])

            print "M | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f || " % (m[0], m[1], m[2]),
            print "D | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f" % (d[0], d[1], d[2])

            # MBON2MBON reaction
            m = np.maximum(m[:, 0] + m.T.dot(self.w_mbon2mbon).T[:, 0], 0.)

            # MBON2DAN reaction
            d = np.maximum(d[:, 0] + m.T.dot(self.w_mbon2dan).T, 0.)
            dk = d.T.dot(self.w_dan2mbon).T

            for mm, mv, dd, dv, dkk in zip(self._mbon, m, self._dan, d, dk):
                mm._c_odour[i] = np.maximum(mm._c_odour[i] + dkk, 0.)
                mm.v[i] = mv
                dd.v[i] = dv
                mm._c_odour += 0.02 * (mm._b - mm._c_odour) * np.float32(1 - shock)

            print "M | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f || " % (m[0], m[1], m[2]),
            print "D | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f" % (d[0], d[1], d[2])
        print ""


def disconnected_neurons_network():
    mm1 = Neuron(a=np.array([1., 1.]), c_odour=np.array([7., 6.]), b=0.,
                 u=np.array([0., 0.]), c_shock=np.array([0., 0.]), utype="mc")
    mm2 = Neuron(a=np.array([1., 1.]), c_odour=np.array([3., 3.]), b=2.,
                 u=np.array([0., 0.]), c_shock=np.array([0., 0.]), utype="mc")
    mm3 = Neuron(a=np.array([1., 1.]), c_odour=np.array([0., 0.]),
                 u=np.array([0., 0.]), c_shock=np.array([0., 0.]), utype="mc")
    dd1 = Neuron(a=np.array([1., 1.]), c_odour=np.array([0., 0.]), b=1.,
                 u=np.array([0., 0.]), c_shock=np.array([5., 5.]), utype="mc")
    dd2 = Neuron(a=np.array([1., 1.]), c_odour=np.array([0., 0.]), b=1.,
                 u=np.array([0., 0.]), c_shock=np.array([2., 2.]), utype="mc")
    dd3 = Neuron(a=np.array([1., 1.]), c_odour=np.array([0., 0.]), b=7.,
                 u=np.array([0., 0.]), c_shock=np.array([3., 3.]), utype="mc")

    # initial values
    mm1.v = np.array([7., 7.])
    mm2.v = np.array([4., 4.])
    mm3.v = np.array([0., 0.])
    dd1.v = np.array([2., 2.])
    dd2.v = np.array([5., 2.])
    dd3.v = np.array([10., 10.])

    # the network
    nn = Network(dan=[dd1, dd2, dd3], mbon=[mm1, mm2, mm3])
    nn.w_mbon2mbon = np.array([
        [0., -.5, -.5],
        [0., 0., .1],
        [0., 0., 0.],
    ])
    nn.w_dan2mbon = np.array([
        [-1., 0., 0.],
        [0., -.5, 0.],
        [0., 0., .02],
    ])
    nn.w_mbon2dan = np.array([
        [-.8, 0., 0.],
        [0., -.1, -.5],
        [0., -.3, -.5],
    ])

    return nn


hist = {
    "m1": [], "m2": [], "m3": [],
    "d1": [], "d2": [], "d3": [],
}

net = disconnected_neurons_network()
# net = pairwise_neurons_network()
d1, d2, d3, m1, m2, m3 = net.neurons
# print d1.w_odour
# print d1.w_shock

# for i, s in enumerate([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#                        0., 0., 0., 0., 0., 0., 0.]):
for i, s in enumerate([0., 0., 0., 1., 0., 1., 0., 1., 0., 1.,
                       0., 1., 0., 0., 1., 0., 1.]):

    print ["%d CS-", "%d CS+"][i % 2] % (i // 2 + 1), ["", "shock"][int(s)]
    odours = [np.array([1., 0.]), np.array([0., 1.])]
    net(odour=odours[i % 2], shock=s)

    hist["m1"].append(m1.v.copy())
    hist["m2"].append(m2.v.copy())
    hist["m3"].append(m3.v.copy())

    hist["d1"].append(d1.v.copy())
    hist["d2"].append(d2.v.copy())
    hist["d3"].append(d3.v.copy())


hist["m1"] = np.array(hist["m1"])
hist["m2"] = np.array(hist["m2"])
hist["m3"] = np.array(hist["m3"])
hist["d1"] = np.array(hist["d1"])
hist["d2"] = np.array(hist["d2"])
hist["d3"] = np.array(hist["d3"])

plot_hist_old(**hist)
