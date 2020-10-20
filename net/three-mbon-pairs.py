import numpy as np
from plot3mbon import plot_hist_old


class Neuron(object):
    def __init__(self, a_odour=.5, a_shock=.5, b_odour=0., b_shock=0.,
                 c_shock=1., c_odour=1., u_shock=0., u_odour=0., d=2,
                 gamma=0.1, utype="td"):
        """

        :param a_odour: learning rate -- minus, plus
        :type a_odour: np.ndarray, float
        :param a_shock: learning rate -- minus, plus
        :type a_shock: np.ndarray, float
        :param b_odour: bias -- background activity, rest response
        :type b_odour: np.ndarray, float
        :param b_shock: bias -- background activity, rest response
        :type b_shock: np.ndarray, float
        :param c_shock: shock affection constant
        :type c_shock: np.ndarray, float
        :param c_odour: odour affection constant
        :type c_odour: np.ndarray, float
        :param u_shock: update of shock constant parameter
        :type u_shock: np.ndarray, float
        :param u_odour: update of the odour constant parameter
        :type u_odour: np.ndarray, float
        :param d: dimensions of value
        :type d: int
        :param gamma: the discount value that sets the horizon
        :type gamma: float
        :param utype: the update type of the neuron
        :type utype: basestring
        """
        self._a_shock = np.array(a_shock) if type(a_shock) is not float and len(a_shock) == d else np.full(d, a_shock)
        self._a_odour = np.array(a_odour) if type(a_odour) is not float and len(a_odour) == d else np.full(d, a_odour)
        self._c_shock = np.array(c_shock) if type(c_shock) is not float and len(c_shock) == d else np.full(2, c_shock)
        self._c_odour = np.array(c_odour) if type(c_odour) is not float and len(c_odour) == d else np.full(2, c_odour)
        self._u_shock = np.array(u_shock) if type(u_shock) is not float and len(u_shock) == d else np.full(2, u_shock)
        self._u_odour = np.array(u_odour) if type(u_odour) is not float and len(u_odour) == d else np.full(2, u_odour)
        self._b_odour = b_odour
        self._b_shock = b_shock
        self._d = d
        self.v_odour = np.zeros(d, dtype=float)  # the value of the neuron
        self.v_shock = np.zeros(d, dtype=float)  # the value of the neuron
        self._e_odour = np.zeros_like(self.v_odour)
        self._e_shock = np.zeros_like(self.v_shock)
        self._w_odour = np.zeros((2, d), dtype=float)
        self._w_shock = np.zeros((2, d), dtype=float)
        self.w_odour = np.eye(2, d, dtype=float)
        self.w_shock = np.eye(2, d, dtype=float)
        self.gamma = gamma
        assert utype.lower() in ["td", "mc", "dp"], "'utype' must be one of 'td' (TD-learning), 'mc' (Monte Carlo) " \
                                                    "or 'dp' (Dynamic Programming)."
        self.utype = utype.lower()

    @property
    def v(self):
        return self.v_odour + self.v_shock

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
        aod = self._a_odour[isk][i]
        aso = self._a_shock[isk][i]
        vo0 = self.v_odour[i]
        vo1 = self.v_odour[~i]
        vs0 = self.v_shock[i]
        vs1 = self.v_shock[~i]

        # targets
        t_odour = o
        t_shock = s

        # errors
        e_odour = self._e_odour[i] = (t_odour - vo0)[i]
        e_shock = self._e_shock[i] = (t_shock - vs0)[i]

        if self.utype == "td":  # TD learning
            self.v_odour[i] += aod * (self.gamma * vo1 + e_odour)
            self.v_shock[i] += aso * (self.gamma * vs1 + e_shock)
        elif self.utype == "mc":  # Monte Carlo
            self.v_odour[i] += aod * e_odour
            self.v_shock[i] += aso * e_shock
        elif self.utype == "dp":  # Dynamic programming
            vopos = self.v_odour - self.v_odour.min()
            vspos = self.v_shock - self.v_shock.min()
            vopr = 1. if vopos.sum() == 0. else vopos[~i] / vopos.sum()
            vspr = 1. if vspos.sum() == 0. else vspos[~i] / vspos.sum()
            self.v_odour[i] += aod * (vopr * (t_odour + self.gamma * self.v_odour[~i])).sum()
            self.v_shock[i] += aso * (vspr * (t_shock + self.gamma * self.v_shock[~i])).sum()

        s32 = np.int32(shock)
        self._c_odour[i] += self._u_odour[s32] * np.absolute(self._c_odour[i] + self._c_shock[i] - self._b_odour)
        self._c_shock[i] += self._u_shock[s32] * np.absolute(self._c_shock[i] + self._c_odour[i] - self._b_shock)
        print "shock", t_shock, "odour", t_odour, "vo", self.v_odour, "vs", self.v_shock

        return self.v_odour, self.v_shock

    def __update(self, odour, shock):
        s = (shock * odour).dot(self.w_shock)
        o = odour.dot(self.w_odour)
        e = s - self.v_odour * o + self._b_odour
        self.v_odour += odour * self._a[np.int32(np.int32(s > 0) * odour)] * e
        self.v_odour += self._u(self.v_odour)

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
        d, m = [], []

        # update MBON wrt the KCs
        i = 1
        for mbon in self._mbon:
            print "M%d" % i,
            i += 1
            m.append(mbon(odour=odour, shock=shock))
        m = np.array(m)

        # update DAN values wrt MBONs and collect DAN values
        md = m.T.dot(self.w_mbon2dan).T
        i = 1
        for dan, v in zip(self._dan, md):
            print "D%d" % i,
            i += 1
            d.append(dan(odour=odour, shock=shock))
            dan.v_odour += v[0]
            dan.v_shock += v[1]
            print "1", v[0], v[1]
        d = np.array(d)

        # update MBON wrt the MBON(t-1)
        m = np.array([[mm.v_odour, mm.v_shock] for mm in self._mbon])  # M[t-1]
        mm = m.T.dot(self.w_mbon2mbon).T
        for mbon, v in zip(self._mbon, mm):
            mbon.v_odour += v[0]
            mbon.v_shock += v[1]
            print "2", v[0], v[1]

        # modulate the  KC-to-MBON weights
        m = np.array([[mm.v_odour, mm.v_shock] for mm in self._mbon])  # M[t-1]
        dm = odour * d.T.dot(self.w_dan2mbon).T
        for mbon, v in zip(self._mbon, dm):
            mbon._c_odour += v[0]
            mbon._c_odour += v[1]
            print "3", v[0], v[1], (mbon.w_odour / mbon._c_odour).shape


def disconnected_neurons_network():
    mm1 = Neuron(a_odour=np.array([.3, .3]), c_odour=np.array([4., 4.]), u_odour=np.array([0., 0.]),
                 a_shock=np.array([.1, .9]), c_shock=np.array([2., 2.]), u_shock=np.array([0., -.7]),
                 utype="mc", gamma=0.1)
    mm2 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([2., 1.]), u_odour=np.array([.0, -.1]), b_odour=2.,
                 a_shock=np.array([.7, .7]), c_shock=np.array([-2., -2.]), u_shock=np.array([0., .5]), b_shock=2.,
                 utype="mc", gamma=0.1)
    mm3 = Neuron(a_odour=np.array([.1, .1]), c_odour=np.array([2., 2.]), u_odour=np.array([-.1, -.1]),
                 a_shock=np.array([.1, .2]), c_shock=np.array([4., 4.]), u_shock=np.array([0., 0.]),
                 utype="td", gamma=0.1)
    dd1 = Neuron(a_odour=np.array([.5, .5]), c_odour=np.array([1., 1.]), u_odour=np.array([-.01, 0.]),
                 a_shock=np.array([.9, .9]), c_shock=np.array([8., 8.]), u_shock=np.array([-.05, -.1]),
                 utype="mc", gamma=0.1)
    dd2 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([3., 1.]), u_odour=np.array([-.2, 0.]), b_odour=4.,
                 a_shock=np.array([.7, .7]), c_shock=np.array([2., 1.]), u_shock=np.array([0., 0.]), b_shock=-1.,
                 utype="mc", gamma=0.1)
    dd3 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([3., 10.]), u_odour=np.array([0., 0.]),
                 a_shock=np.array([.3, .1]), c_shock=np.array([1., 1.]), u_shock=np.array([0., -1.1]),
                 utype="mc", gamma=0.1)

    # initial values
    mm1.v_odour, mm1.v_shock = np.array([6.2, 5.8]), np.array([0., 0.])
    mm2.v_odour, mm2.v_shock = np.array([5.5, 1.0]), np.array([0., 0.])
    mm3.v_odour, mm3.v_shock = np.array([0.0, 0.0]), np.array([0., 0.])
    dd1.v_odour, dd1.v_shock = np.array([2.0, 0.5]), np.array([0., 0.])
    dd2.v_odour, dd2.v_shock = np.array([5.0, 2.0]), np.array([0., 0.])
    dd3.v_odour, dd3.v_shock = np.array([2.0, 10.]), np.array([0., 0.])

    # the network
    nn = Network(dan=[dd1, dd2, dd3], mbon=[mm1, mm2, mm3])
    nn.w_dan2mbon = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    nn.w_mbon2dan = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    nn.w_mbon2mbon = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])

    return nn


def pairwise_neurons_network():
    mm1 = Neuron(a_odour=np.array([.3, .3]), c_odour=np.array([4., 4.]), u_odour=np.array([0., 0.]),
                 a_shock=np.array([.1, .9]), c_shock=np.array([2., 2.]), u_shock=np.array([0., -.7]),
                 utype="mc", gamma=0.1)
    mm2 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([2., 1.]), u_odour=np.array([.0, -.1]), b_odour=2.,
                 a_shock=np.array([.7, .7]), c_shock=np.array([-2., -2.]), u_shock=np.array([0., .5]), b_shock=2.,
                 utype="mc", gamma=0.1)
    mm3 = Neuron(a_odour=np.array([.1, .1]), c_odour=np.array([2., 2.]), u_odour=np.array([-.1, -.1]),
                 a_shock=np.array([.1, .2]), c_shock=np.array([4., 4.]), u_shock=np.array([0., 0.]),
                 utype="td", gamma=0.1)
    dd1 = Neuron(a_odour=np.array([.5, .5]), c_odour=np.array([2., 2.]), u_odour=np.array([0., 0.]),
                 a_shock=np.array([.9, .9]), c_shock=np.array([8., 8.]), u_shock=np.array([-.05, -.1]),
                 utype="mc", gamma=0.1)
    dd2 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([3., 1.]), u_odour=np.array([-.2, 0.]), b_odour=4.,
                 a_shock=np.array([.7, .7]), c_shock=np.array([2., 1.]), u_shock=np.array([0., 0.]), b_shock=-1.,
                 utype="mc", gamma=0.1)
    dd3 = Neuron(a_odour=np.array([.7, .7]), c_odour=np.array([3., 10.]), u_odour=np.array([0., 0.]),
                 a_shock=np.array([.3, .1]), c_shock=np.array([1., 1.]), u_shock=np.array([0., -1.1]),
                 utype="mc", gamma=0.1)

    # initial values
    mm1.v_odour, mm1.v_shock = np.array([6.2, 5.8]), np.array([0., 0.])
    mm2.v_odour, mm2.v_shock = np.array([5.5, 1.0]), np.array([0., 0.])
    mm3.v_odour, mm3.v_shock = np.array([0.0, 0.0]), np.array([0., 0.])
    dd1.v_odour, dd1.v_shock = np.array([2.0, 2.0]), np.array([0., 0.])
    dd2.v_odour, dd2.v_shock = np.array([5.0, 2.0]), np.array([0., 0.])
    dd3.v_odour, dd3.v_shock = np.array([2.0, 10.]), np.array([0., 0.])

    # the network
    nn = Network(dan=[dd1, dd2, dd3], mbon=[mm1, mm2, mm3])

    nn.w_dan2mbon = np.array([
        [.1, 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    nn.w_mbon2dan = np.array([
        [-.1, 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    nn.w_mbon2mbon = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])

    return nn


hist = {
    "m1": [], "m2": [], "m3": [],
    "d1": [], "d2": [], "d3": [],
}

# net = disconnected_neurons_network()
net = pairwise_neurons_network()
d1, d2, d3, m1, m2, m3 = net.neurons
# print d1.w_odour
# print d1.w_shock

# for i, s in enumerate([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#                        0., 0., 0., 0., 0., 0., 0.]):
for i, s in enumerate([0., 0., 0., 1., 0., 1., 0., 1., 0., 1.,
                       0., 1., 0., 0., 1., 0., 1.]):

    odours = [np.array([1., 0.]), np.array([0., 1.])]
    net(odour=odours[i % 2], shock=s)

    print ["%d CS-", "%d CS+"][i % 2] % (i // 2 + 1), ["", "shock"][int(s)]
    print "M | 1: vp = %.2f, vm = %.2f; 2: vp = %.2f, vm = %.2f; 3: vp = %.2f, vm = %.2f" % (
        m1.v[1], m1.v[0], m2.v[1], m2.v[0], m3.v[1], m3.v[0]
    )
    print "D | 1: vp = %.2f, vm = %.2f; 2: vp = %.2f, vm = %.2f; 3: vp = %.2f, vm = %.2f" % (
        d1.v[1], d1.v[0], d2.v[1], d2.v[0], d3.v[1], d3.v[0]
    )
    print ""

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
