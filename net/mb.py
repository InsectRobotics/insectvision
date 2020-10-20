import numpy as np
import copy
import yaml
import os

__dir__ = os.path.dirname(os.path.realpath(__file__))


class Neuron(object):
    """
    Abstract class for the neuron model.
    """
    def __init__(self, a=1., b=0., u=0., d=2, gamma=0.1, verbose=False, utype="mc", name="Neuron"):
        """

        :param a: 2D learning rate -- [minus, plus]
        :type a: np.ndarray, float
        :param b: bias -- other feedback activity
        :type b: np.ndarray, float
        :param u: value descent
        :type u: np.ndarray, float
        :param d: dimensions of value
        :type d: int
        :param gamma: the discount value that sets the horizon (for TD learning)
        :type gamma: float
        :param verbose: how much text to print (for debugging)
        :type verbose: int, bool
        :param utype: the update type of the neuron
        :type utype: basestring
        :param name: the name of the neuron
        :type name: basestring
        """
        self._eta = (
            np.array(a)
            if type(a) is not float and len(a) == d
            else np.full((d if type(d) is not tuple else d[0], d[1]), a))
        self._u = (
            np.array(u)
            if type(u) is not float and len(u) == d
            else np.full((d if type(d) is not tuple else d[0], d[1]), u))
        self._b = (
            np.array(b)
            if type(b) is not float and len(b) == d
            else np.full(d if type(d) is not tuple else d[0], b))
        self._d = d
        self._v = np.zeros(d, dtype=float)  # the value of the neuron
        self._e = np.zeros_like(self._v)
        self.gamma = gamma
        assert utype.lower() in ["td", "mc", "dp"], "'utype' must be one of 'td' (TD-learning), 'mc' (Monte Carlo) " \
                                                    "or 'dp' (Dynamic Programming)."
        self.utype = utype.lower()
        self._f = lambda x: np.maximum(x, -self._b)
        self.name = name
        self._verbose = 0
        self.set_verbose(verbose)

    def set_verbose(self, v):
        self._verbose = np.clip(v if not isinstance(v, bool) else 3 if v else 0, 0, 3)

    @property
    def v(self):
        # return self._f(self._v)
        return self._f(self._v) + self._b

    @v.setter
    def v(self, v):
        self._v = v

    def __call__(self, *args, **kwargs):
        return self.v

    def __add__(self, other):
        assert type(other) == type(self) or type(other) == type(self._v), (
            "Value of type '%s' cannot be added to the neuron." % type(other)
        )
        v = other.v if isinstance(other, Neuron) else other
        n = copy.copy(self)
        n._v += n._eta * (v - n._v)
        return n

    def __sub__(self, other):
        assert type(other) == type(self) or type(other) == type(self._v), (
            "Value of type '%s' cannot be subtracted from the neuron." % type(other)
        )
        v = other.v if isinstance(other, Neuron) else other
        n = copy.copy(self)
        n._v -= n._eta * (v - n._v)
        return n

    def __mul__(self, other):
        assert type(other) == type(self) or type(other) == type(self._v), (
            "Value of type '%s' cannot be multiplied with the neuron." % type(other)
        )
        v = other.v if isinstance(other, Neuron) else other
        n = copy.copy(self)
        n._v *= v
        return n

    def __div__(self, other):
        assert type(other) == type(self) or type(other) == type(self._v), (
            "Value of type '%s' cannot divide the neuron." % type(other)
        )
        v = other.v if isinstance(other, Neuron) else other
        n = copy.copy(self)
        n._v /= v
        return n

    def __iter__(self):
        for v in self._v:
            yield v

    def __getitem__(self, item):
        return self._v[item]

    def __setitem__(self, key, value):
        # self._v[key] += self._eta[key] * (value - self._v[key])
        self._v[key] = value

    def __len__(self):
        return len(self._v)

    def __copy__(self):
        n = Neuron()
        n._eta = copy.copy(self._eta)
        n._u = copy.copy(self._u)
        n._b = copy.copy(self._b)
        n._d = copy.copy(self._d)
        n._v = copy.copy(self._v)
        n._e = copy.copy(self._e)
        n.gamma = copy.copy(self.gamma)
        n.utype = copy.copy(self.utype)
        n._f = copy.copy(self._f)
        return n

    def __deepcopy__(self, memodict={}):
        n = Neuron()
        n._eta = copy.deepcopy(self._eta, memodict)
        n._u = copy.deepcopy(self._u, memodict)
        n._b = copy.deepcopy(self._b, memodict)
        n._d = copy.deepcopy(self._d, memodict)
        n._v = copy.deepcopy(self._v, memodict)
        n._e = copy.deepcopy(self._e, memodict)
        n.gamma = copy.deepcopy(self.gamma, memodict)
        n.utype = copy.deepcopy(self.utype, memodict)
        n._f = copy.deepcopy(self._f, memodict)
        return n


class Network(object):
    """
    Abstract class for the network model.
    """
    def __init__(self, verbose=False):
        self._verbose = 0
        self.neurons = []
        self.set_verbose(verbose)

    def __call__(self, *args, **kwargs):
        return [n.v.sum() for n in self.neurons]

    def set_verbose(self, v):
        self._verbose = np.clip(v if not isinstance(v, bool) else 3 if v else 0, 0, 3)


class MANeuron(Neuron):
    """
    Mushroom Body Neuron model (A). This model is the first (complicated) model tried. It has a peculiar adaptation rule
    that we are not sure why it works, but it can represent the data quite well.
    """
    def __init__(self, c_odour=1., c_shock=1., **kwargs):
        """

        :param c_shock: shock affection constant
        :type c_shock: np.ndarray, float
        :param c_odour: odour affection constant
        :type c_odour: np.ndarray, float
        """
        kwargs['name'] = kwargs.get('name', 'MB-neuron')
        super(MANeuron, self).__init__(**kwargs)
        self._c_shock = np.array(
            c_shock) if type(c_shock) is not float and len(c_shock) == self._d else np.full(2, c_shock)
        self._c_odour = np.array(
            c_odour) if type(c_odour) is not float and len(c_odour) == self._d else np.full(2, c_odour)
        self._w_odour = np.zeros((2, self._d), dtype=float)
        self._w_shock = np.zeros((2, self._d), dtype=float)
        self.w_odour = np.eye(2, self._d, dtype=float)
        self.w_shock = np.eye(2, self._d, dtype=float)

    def _fprop(self, odour, shock, fb_mbon=None):
        i = odour > 0
        isk = np.int32(odour * shock)

        # observations
        s = (shock * odour).dot(self.w_shock)  # allocate the shock obs to the corresponding odour identity
        o = odour.dot(self.w_odour)  # scale the odour obs
        m = fb_mbon if fb_mbon is not None else 0.

        # update the values
        eta = self._eta[isk][i]
        v0 = self._v[i]
        v1 = self._v[~i]

        # targets
        t = (o + s + m)[i]
        # t = (o + s + self._b)[i]
        # t_pleasure, t_pain, t_desire, t_fear = t

        # errors
        e = self._e[i] = np.asscalar(t - v0) if len(t) > 0 else 0.
        # e_pleasure, e_pain, e_desire, e_fear = e

        if self.utype == "td":  # TD learning
            self._v[i] += eta * (self.gamma * v1 + e)
        elif self.utype == "mc":  # Monte Carlo
            self._v[i] += eta * e
        elif self.utype == "dp":  # Dynamic programming
            vpos = self.v - self.v.min()
            vpr = 1. if vpos.sum() == 0. else vpos[~i] / vpos.sum()
            self._v[i] += eta * (vpr * (t.sum() + self.gamma * self._v[~i])).sum()
        # self._v[i] -= self._u[isk][i] * self._v[i]
        self._c_odour[i] += eta * self._u[isk][i] * np.sign(e)
        j = ~i
        jsk = np.int32((1 - odour) * (1 - shock))
        eeta = self._eta[jsk][j]
        self._c_odour[j] += eeta * self._u[jsk][j] * np.sign(e)
        # self._c_odour[~i] += self._eta[1 - isk][~i] * self._u[1 - isk][~i] * self.v[~i] * self._c_odour[~i]
        # self._c_shock[i] += eta * self._u[isk][i] * self._v[i] * self._c_shock[i]
        if self._verbose > 2:
            print "c", self._c_odour[0], self._c_odour[1]
        return self.v

    @property
    def c_odour(self):
        return self._c_odour

    @c_odour.setter
    def c_odour(self, v):
        self._c_odour = self._f(v)

    @property
    def w_shock(self):
        return self._c_shock * self._w_shock

    @w_shock.setter
    def w_shock(self, v):
        self._w_shock = v

    @property
    def w_odour(self):
        return self.c_odour * self._w_odour

    @w_odour.setter
    def w_odour(self, v):
        self._w_odour = v

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        fb_mbon = args[2] if len(args) > 2 else kwargs.get("fb_mbon", 0.)
        return self._fprop(odour=odour, shock=shock, fb_mbon=fb_mbon)
        # return self.__update(odour=odour, shock=shock)

    def __copy__(self):
        n = MBNeuron()
        n._eta = copy.copy(self._eta)
        n._c_shock = copy.copy(self._c_shock)
        n._c_odour = copy.copy(self._c_odour)
        n._u = copy.copy(self._u)
        n._b = copy.copy(self._b)
        n._d = copy.copy(self._d)
        n._v = copy.copy(self._v)
        n._e = copy.copy(self._e)
        n._w_odour = copy.copy(self._w_odour)
        n._w_shock = copy.copy(self._w_shock)
        n.gamma = copy.copy(self.gamma)
        n.utype = copy.copy(self.utype)
        n._f = copy.copy(self._f)
        return n

    def __deepcopy__(self, memodict={}):
        n = MBNeuron()
        n._eta = copy.deepcopy(self._eta, memodict)
        n._c_shock = copy.deepcopy(self._c_shock, memodict)
        n._c_odour = copy.deepcopy(self._c_odour, memodict)
        n._u = copy.deepcopy(self._u, memodict)
        n._b = copy.deepcopy(self._b, memodict)
        n._d = copy.deepcopy(self._d, memodict)
        n._v = copy.deepcopy(self._v, memodict)
        n._e = copy.deepcopy(self._e, memodict)
        n._w_odour = copy.deepcopy(self._w_odour, memodict)
        n._w_shock = copy.deepcopy(self._w_shock, memodict)
        n.gamma = copy.deepcopy(self.gamma, memodict)
        n.utype = copy.deepcopy(self.utype, memodict)
        n._f = copy.deepcopy(self._f, memodict)
        return n

    def __str__(self):
        types = {'mc': 'Monte Carlo', 'td': 'Time Difference', 'dp': 'Dynamic Programming'}
        return unicode("Mushroom Body neuron '%s'\n"
                       "\t\tValue:\t\t\t\t\t\t\t% 2.2f, % 2.2f"
                       "\t\tRest:\t% 2.2f,\t% 2.2f\n"
                       "\t\tW_odour:\t\t\t\t\t\t% 2.2f, % 2.2f\n"
                       "\t\tW_shock:\t\t\t\t\t\t% 2.2f, % 2.2f\n"
                       "\t\tLearning rate (unpair, pair):\t% 2.2f, % 2.2f\t\ttype: %s\n"
                       "\t\tAdaptation (unpair, pair):\t\t% 2.2f, % 2.2f" % (
            self.name, self.v[0], self.v[1], self._b[0], self._b[1],
            self._c_odour[0], self._c_odour[1],
            self._c_shock[0], self._c_shock[1],
            self._eta[0], self._eta[1], types[self.utype], self._u[0], self._u[1]))


class MANetwork(Network):
    """
    Network connecting the MB (A) Neurons. Every trial here is based on 3 timesteps: 'pre-odour', 'odour' and 'shock'.

    """
    def __init__(self, dan, mbon, verbose=False):
        """

        :type mbon: list, Neuron
        :type dan: list, Neuron
        """
        super(MANetwork, self).__init__(verbose=verbose)
        self._dan = list(dan)
        self._mbon = list(mbon)
        self.neurons += self._dan + self._mbon
        for neuron in self.neurons:
            neuron.set_verbose(self._verbose)

        self.w_dan2kc = np.zeros((len(dan), len(mbon)), dtype=float)
        self.w_mbon2dan = np.zeros((len(mbon), len(dan)), dtype=float)
        self.w_mbon2mbon = np.zeros((len(mbon), len(mbon)), dtype=float)
        self.short_hist = {'m1': [], 'm2': [], 'm3': [], 'd1': [], 'd2': [], 'd3': [],
                           'cm1': [], 'cm2': [], 'cm3': [], 'cd1': [], 'cd2': [], 'cd3': []}

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        self.__update(odour=odour, shock=shock)
        return super(MANetwork, self).__call__(*args, **kwargs)

    def __update(self, odour, shock):
        i = odour > 0
        self.short_hist = {'m1': [], 'm2': [], 'm3': [], 'd1': [], 'd2': [], 'd3': [],
                           'cm1': [], 'cm2': [], 'cm3': [], 'cd1': [], 'cd2': [], 'cd3': []}
        for odr, sck in zip([0. * odour, odour, odour], [0., 0., shock]):
            if self._verbose > 2:
                print "o", odr[i][0], "s", sck,

            # odour activates KCs which drive activity in each MBON
            bm = np.array([m._b for m in self._mbon])
            # fm = (np.array([m.v for m in self._mbon]).T - bm).dot(self.w_mbon2mbon).T  # MBON2MBON reaction
            fm = [None] * 3
            m = np.array([m(odr, sck, fb)[i] for m, fb in zip(self._mbon, fm)]).flatten()

            if self._verbose > 2:
                print "M | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f || " % (m[0], m[1], m[2]),

            # MBON2MBON reaction
            m = m + (m - bm).T.dot(self.w_mbon2mbon).T[:]

            # initial DAN activity is zero
            bd = np.array([d._b for d in self._dan])
            # fd = m.T.dot(self.w_mbon2dan).T  # MBON2DAN reaction
            fd = [None] * 3
            d = np.array([d(odr, sck, fb)[i] for d, fb in zip(self._dan, fd)]).flatten()

            if self._verbose > 2:
                print "D | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f" % (d[0], d[1], d[2])

            # MBON2DAN reaction
            d = d + (m - bm).T.dot(self.w_mbon2dan).T
            dk = (d - bd).T.dot(self.w_dan2kc).T

            for mm, mv, dd, dv, dkk in zip(self._mbon, m - bm, self._dan, d - bd, dk):
                c = mm.c_odour
                if self._verbose > 2:
                    print "FM", fm
                c[i] += dkk
                # mm.c_odour = c
                mm[i], dd[i] = mv, dv

            self.short_hist['m1'].append(self._mbon[0].v.copy())
            self.short_hist['cm1'].append(self._mbon[0]._c_odour.copy())
            self.short_hist['m2'].append(self._mbon[1].v.copy())
            self.short_hist['cm2'].append(self._mbon[1]._c_odour.copy())
            self.short_hist['m3'].append(self._mbon[2].v.copy())
            self.short_hist['cm3'].append(self._mbon[2]._c_odour.copy())
            self.short_hist['d1'].append(self._dan[0].v.copy())
            self.short_hist['cd1'].append(self._dan[0]._c_shock.copy())
            self.short_hist['d2'].append(self._dan[1].v.copy())
            self.short_hist['cd2'].append(self._dan[1]._c_shock.copy())
            self.short_hist['d3'].append(self._dan[2].v.copy())
            self.short_hist['cd3'].append(self._dan[2]._c_shock.copy())

            if self._verbose > 1:
                print "o", odr[i][0], "s", sck,
            if self._verbose > 2:
                print "M | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f || " % (m[0], m[1], m[2]),
                print "D | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f" % (d[0], d[1], d[2])
                print "           ",
            if self._verbose > 1:
                print "M | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f || " % tuple([mm.v[i][0] for mm in self._mbon]),
                print "D | 1: v = %.2f; 2: v = %.2f; 3: v = %.2f" % tuple([dd.v[i][0] for dd in self._dan])
        if self._verbose > 1:
            print ""

    def __str__(self):
        return unicode("Mushroom Body network\nNeurons:\n" +
                       ("\t%s\n" * len(self.neurons) % tuple(unicode(n) for n in self.neurons)) + "\n"
               "Synaptic weights:\n"
               "\tMBON2MBON:\t|% 2.2f, % 2.2f, % 2.2f|\t\tMBON2DAN:\t|% 2.2f, % 2.2f, % 2.2f|\n"
               "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n"
               "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n"
               "\tDAN2MBON: \t|% 2.2f, % 2.2f, % 2.2f|\n"
               "\t          \t|% 2.2f, % 2.2f, % 2.2f|\n"
               "\t          \t|% 2.2f, % 2.2f, % 2.2f|\n" % tuple(
            self.w_mbon2mbon[0].tolist() + self.w_mbon2dan[0].tolist() +
            self.w_mbon2mbon[1].tolist() + self.w_mbon2dan[1].tolist() +
            self.w_mbon2mbon[2].tolist() + self.w_mbon2dan[2].tolist() +
            self.w_dan2kc.flatten().tolist()))


class MBNeuron(Neuron):
    """
    Mushroom body (B) neuron model. This model is well described with an adaptation rule similar to the anti-Hebbian
    one, i.e. decreasing the weights when the odour and MB response occur together and increasing it when they do not
    occur together.
    """
    def __init__(self, c_odour=1., c_shock=1., c_fback=1., **kwargs):
        """

        :param c_shock: shock affection constant
        :type c_shock: np.ndarray, float
        :param c_odour: odour affection constant
        :type c_odour: np.ndarray, float
        :param c_fback: feedback affection constant
        :type c_fback: np.ndarray, float
        """
        kwargs['d'] = kwargs.get('d', (2, 3))
        kwargs['name'] = kwargs.get('name', 'MB-neuron')
        kwargs['utype'] = kwargs.get('utype', 'mc')
        super(MBNeuron, self).__init__(**kwargs)
        self._c = np.array([
            np.array(c_odour) if type(c_odour) is not float and len(c_odour) == self._d else np.full(2, c_odour),
            np.array(c_shock) if type(c_shock) is not float and len(c_shock) == self._d else np.full(2, c_shock),
            np.array(c_fback) if type(c_fback) is not float and len(c_fback) == self._d else np.full(2, c_fback)
            ])
        self._w = np.zeros(self._d + tuple([2]), dtype=float)
        self.w_odour = np.eye(2, self._d[0], dtype=float)
        self.w_shock = np.eye(2, self._d[0], dtype=float)
        self.w_fback = np.eye(2, self._d[0], dtype=float)
        self._f = lambda x: np.maximum(x, -1.)
        self._t = 0.  # time in milliseconds
        self._dts = [5000., 4000., 1000., 5000.]  # the different times of the neuron's timesteps
        self._iterations = 0

    def _fprop(self, odour, shock, fb_mbon=None):
        v = self._v.copy()
        c = self._c.copy()

        if self._verbose > 2:
            print unicode(self.name), self._t, odour, shock, fb_mbon + self._b

        # observations
        s = (shock * odour).dot(self.w_shock)  # allocate the shock obs to the corresponding odour identity
        o = odour.dot(self.w_odour)  # scale the odour obs
        m = ((np.array(fb_mbon)
              if type(fb_mbon) is not float and len(fb_mbon) == self._d[0]
              else np.full(self._d[0], fb_mbon)
              ) if fb_mbon is not None else np.zeros(2)).dot(self.w_fback) + self._b * odour
        # s[np.isnan(s)] = 0.
        # o[np.isnan(o)] = 0.
        # m[np.isnan(m)] = 0.

        # targets
        t = np.zeros_like(v)
        t[:, 0] = o
        t[:, 1] = s
        t[:, 2] = m

        # errors
        e = (t - v) if len(t) > 0 else 0.

        if self.utype == "mc":  # Monte Carlo
            v += self._eta * e
        else:
            raise ValueError("This type of neuron is not supported. Please use 'Monte Carlo' type.")

        z = np.zeros_like(c)
        z[0] = odour
        z[1] = shock * odour
        z[2] = [1., 1.]
        z *= np.sqrt(self._u.T) / np.power(np.maximum(self._t, 1.), 1. - np.sqrt(self._u.T))
        # weaken the weights that cause high error
        if np.isnan(e.sum()):
            print unicode(self.name), "\tT:", self._t, "\tv:", v.T.flatten(), "\tt:", t.T.flatten()
        if np.isnan(z.sum()):
            print unicode(self.name), "\tT:", self._t, "\tz:", z.flatten()
        i = np.absolute(e.T) < 1e-03
        upt = -z * e.T
        # strengthen the weights that cause low error
        upt[i] += z[i]
        c += upt
        if np.isnan(c.sum()):
            print unicode(self.name), "\tT:", self._t, "\tc:", c.flatten(),
            print "\tz:", z.flatten(), "\tu:", self._u.T.flatten(), "\te:", e.T.flatten()
            print unicode(self.name), "\tT:", self._t, "\tupt:", upt.flatten()
        # TODO: make sure that you understand what is going on here

        if self._verbose > 2:
            print "\t\todour\t\t\tshock\t\t\tfback"
            print "v\t(%.2f, %.2f)\t(%.2f, %.2f)\t(%.2f, %.2f)" % (
                self._v[0, 0], self._v[1, 0], self._v[0, 1], self._v[1, 1], self._v[0, 2], self._v[1, 2])
            print "c\t(%.2f, %.2f)\t(%.2f, %.2f)\t(%.2f, %.2f)" % (
                self.c_odour[0], self.c_odour[1], self.c_shock[0], self.c_shock[1], self.c_fback[0], self.c_fback[1])
        return v, c, e

    @property
    def v(self):
        return self._f(self._v.sum(axis=1))

    @property
    def c_odour(self):
        return self._c[0]

    @c_odour.setter
    def c_odour(self, v):
        self._c[0] = v

    @property
    def c_shock(self):
        return self._c[1]

    @c_shock.setter
    def c_shock(self, v):
        self._c[1] = v

    @property
    def c_fback(self):
        return self._c[2]

    @c_fback.setter
    def c_fback(self, v):
        self._c[2] = v

    @property
    def w_odour(self):
        return self.c_odour * self._w[:, 0]

    @w_odour.setter
    def w_odour(self, v):
        self._w[:, 0] = v

    @property
    def w_shock(self):
        return self.c_shock * self._w[:, 1]

    @w_shock.setter
    def w_shock(self, v):
        self._w[:, 1] = v

    @property
    def w_fback(self):
        return self.c_fback * self._w[:, 2]

    @w_fback.setter
    def w_fback(self, v):
        self._w[:, 2] = v

    def __next_dt(self):
        dt = self._dts[self._iterations % len(self._dts)]
        self._iterations += 1
        return dt

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        fb_mbon = args[2] if len(args) > 2 else kwargs.get("fb_mbon", 0.)
        update = args[3] if len(args) > 3 else kwargs.get("update", True)
        dt = self.__next_dt()
        self._t += dt
        v, c, e = self._fprop(odour=odour, shock=shock, fb_mbon=fb_mbon)
        if update:
            self._v = v
            self._c = c
            self._e = e
        else:
            self._iterations -= 1
            self._t -= dt
        return self._f(v.sum(axis=1))

    def __copy__(self):
        n = MBNeuron()
        n._eta = copy.copy(self._eta)
        n._c = copy.copy(self._c)
        n._u = copy.copy(self._u)
        n._b = copy.copy(self._b)
        n._d = copy.copy(self._d)
        n._v = copy.copy(self._v)
        n._e = copy.copy(self._e)
        n._w = copy.copy(self._w)
        n.gamma = copy.copy(self.gamma)
        n.utype = copy.copy(self.utype)
        n._f = copy.copy(self._f)
        return n

    def __deepcopy__(self, memodict={}):
        n = MBNeuron()
        n._eta = copy.deepcopy(self._eta, memodict)
        n._c_shock = copy.deepcopy(self._c_shock, memodict)
        n._c_odour = copy.deepcopy(self._c_odour, memodict)
        n._u = copy.deepcopy(self._u, memodict)
        n._b = copy.deepcopy(self._b, memodict)
        n._d = copy.deepcopy(self._d, memodict)
        n._v = copy.deepcopy(self._v, memodict)
        n._e = copy.deepcopy(self._e, memodict)
        n._w_odour = copy.deepcopy(self._w_odour, memodict)
        n._w_shock = copy.deepcopy(self._w_shock, memodict)
        n.gamma = copy.deepcopy(self.gamma, memodict)
        n.utype = copy.deepcopy(self.utype, memodict)
        n._f = copy.deepcopy(self._f, memodict)
        return n

    def __str__(self):
        types = {'mc': 'Monte Carlo', 'td': 'Time Difference', 'dp': 'Dynamic Programming'}
        return unicode("Mushroom Body neuron '%s'\n"
                       "\t\t\t\t\t\t\t\todour\t\t\tshock\t\t\tfeedback\t\tbias\n"
                       "\t\tV:\t\t\t\t\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\n"
                       "\t\tW:\t\t\t\t\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\t(% 2.2f, % 2.2f)\n"
                       "\t\tUpdate rate (STM):\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\n"
                       "\t\tAdaptation (LTM): \t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)" % (
                        self.name,
                        self._v[0, 0], self._v[1, 0],
                        self._v[0, 1], self._v[1, 1],
                        self._v[0, 2], self._v[1, 2],
                        self.c_odour[0], self.c_odour[1],
                        self.c_shock[0], self.c_shock[1],
                        self.c_fback[0], self.c_fback[1], self._b[0], self._b[1],
                        self._eta[0, 0], self._eta[1, 0],
                        self._eta[0, 1], self._eta[1, 1],
                        self._eta[0, 2], self._eta[1, 2],
                        self._u[0, 0], self._u[1, 0],
                        self._u[0, 1], self._u[1, 1],
                        self._u[0, 2], self._u[1, 2]))


class MBNetwork(Network):
    """
    Network model for the MB (B) neurons. It makes updates for every trial in 4 steps: 'pre-odour', 'odour', 'shock' and
    'post-odour'.
    """
    def __init__(self, dan, mbon, verbose=False):
        """

        :type mbon: list, Neuron
        :type dan: list, Neuron
        """
        super(MBNetwork, self).__init__(verbose=verbose)
        self._dan = list(dan)
        self._mbon = list(mbon)
        self.neurons += self._dan + self._mbon
        for neuron in self.neurons:
            neuron.set_verbose(self._verbose)

        self.w_dan2kc = np.zeros((len(dan), len(mbon)), dtype=float)
        self.w_dan2mbon = np.zeros((len(dan), len(mbon)), dtype=float)
        self.w_mbon2dan = np.zeros((len(mbon), len(dan)), dtype=float)
        self.w_mbon2mbon = np.zeros((len(mbon), len(mbon)), dtype=float)
        self.short_hist = {'m1': [], 'm2': [], 'm3': [], 'd1': [], 'd2': [], 'd3': [],
                           'cm1': [], 'cm2': [], 'cm3': [], 'cd1': [], 'cd2': [], 'cd3': []}

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        self.__update(odour=odour, shock=shock)
        return super(MBNetwork, self).__call__(*args, **kwargs)

    def __update(self, odour, shock):
        i = odour > 0
        self.short_hist = {'m1': [], 'm2': [], 'm3': [], 'd1': [], 'd2': [], 'd3': [],
                           'cm1': [], 'cm2': [], 'cm3': [], 'cd1': [], 'cd2': [], 'cd3': []}

        if self._verbose > 1:
            print "\t\t\t1\t\t\t\t2\t\t\t\t3"
        # Repeat for the 4 discrete timesteps: pre-odour, odour, shock, post-odour
        for odr, sck in zip([0. * odour, odour, odour, 0. * odour], [0., 0., shock, 0.]):
            if self._verbose > 2:
                print "o", odr, "s", sck

            # STEP 1: odour activates KCs which (later) drive activity in MBONs
            k1 = odr  # KC values
            # feedback connections from MBONs to DANs are collected for the feedback section of the neurons
            m0 = np.array([mbon.v for mbon in self._mbon])  # MBON values
            fd = m0.T.dot(self.w_mbon2dan).T  # feedback to DANs
            # calculate DAN values without update
            d1 = np.array([d(k1, sck, fb, update=False) for d, fb in zip(self._dan, fd)])
            if self._verbose > 1:
                print "STEP 1:"
                print "k1:  (% 2.2f, % 2.2f)" % tuple(k1)
                print "m0:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(m0.flatten())
                print "fd:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(fd.flatten())
                print "d1:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(d1.flatten())

            # STEP 2: KC and DAN activity update MBONs' value
            fm = d1.T.dot(self.w_dan2mbon).T  # DANs' contribution to MBONs' activity
            m2 = np.array([m(k1, sck, fb, update=False) for m, fb in zip(self._mbon, fm)])
            if self._verbose > 1:
                print "STEP 2:"
                print "fm:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(fm.flatten())
                print "m2:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(m2.flatten())

            # STEP 3: MBONs' update
            fm += m2.T.dot(self.w_mbon2mbon).T  # MBONs' contribution to MBONs' activity is added to the feedback
            m3 = np.array([m(k1, sck, fb, update=True) for m, fb in zip(self._mbon, fm)])
            if self._verbose > 1:
                print "STEP 3:"
                print "fm:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(fm.flatten())
                print "m3:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(m3.flatten())

            # STEP 4: DANs' update
            fd = m3.T.dot(self.w_mbon2dan).T  # MBONs' contribution replaces the feedback of step 1
            d4 = np.array([d(k1, sck, fb, update=True) for d, fb in zip(self._dan, fd)])
            if self._verbose > 1:
                print "STEP 4:"
                print "fd:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(fd.flatten())
                print "d4:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(d4.flatten())

            # STEP 5: Modulation of KC-to-MBON weights through DAN activity
            dk = d4.T.dot(self.w_dan2kc).T
            for mm, dkk in zip(self._mbon, dk):
                mm.c_odour += dkk

            if self._verbose > 1:
                print "STEP 5:"
                print "dk:  (% 2.2f, % 2.2f), (% 2.2f, % 2.2f), (% 2.2f, % 2.2f)" % tuple(dk.flatten())

            self.short_hist['m1'].append(self._mbon[0].v.copy())
            self.short_hist['cm1'].append(self._mbon[0]._c.sum(axis=0))
            self.short_hist['m2'].append(self._mbon[1].v.copy())
            self.short_hist['cm2'].append(self._mbon[1]._c.sum(axis=0))
            self.short_hist['m3'].append(self._mbon[2].v.copy())
            self.short_hist['cm3'].append(self._mbon[2]._c.sum(axis=0))
            self.short_hist['d1'].append(self._dan[0].v.copy())
            self.short_hist['cd1'].append(self._dan[0]._c.sum(axis=0))
            self.short_hist['d2'].append(self._dan[1].v.copy())
            self.short_hist['cd2'].append(self._dan[1]._c.sum(axis=0))
            self.short_hist['d3'].append(self._dan[2].v.copy())
            self.short_hist['cd3'].append(self._dan[2]._c.sum(axis=0))

        if self._verbose > 1:
            print ""

    def __str__(self):
        return unicode("Mushroom Body network\nNeurons:\n" + (
            "\t%s\n" * len(self.neurons) % tuple(unicode(n) for n in self.neurons)) + "\n"
            "Synaptic weights:\n"
            "\tMBON2MBON:\t|% 2.2f, % 2.2f, % 2.2f|\t\tMBON2DAN:\t|% 2.2f, % 2.2f, % 2.2f|\n"
            "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n"
            "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n"
            "\tDAN2MBON: \t|% 2.2f, % 2.2f, % 2.2f|\t\tDAN2KC:  \t|% 2.2f, % 2.2f, % 2.2f|\n"
            "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n"
            "\t          \t|% 2.2f, % 2.2f, % 2.2f|\t\t         \t|% 2.2f, % 2.2f, % 2.2f|\n" % tuple(
            self.w_mbon2mbon[0].tolist() + self.w_mbon2dan[0].tolist() +
            self.w_mbon2mbon[1].tolist() + self.w_mbon2dan[1].tolist() +
            self.w_mbon2mbon[2].tolist() + self.w_mbon2dan[2].tolist() +
            self.w_dan2mbon[0].tolist() + self.w_dan2kc[0].tolist() +
            self.w_dan2mbon[1].tolist() + self.w_dan2kc[1].tolist() +
            self.w_dan2mbon[2].tolist() + self.w_dan2kc[2].tolist()
        ))


class MCNeuron(Neuron):
    def __init__(self, c_odour=1., c_shock=1., c_fback=1., **kwargs):
        """
        Mushroom body Chemical Neuron: Mushroom body neuron based on chemicals, e.g. neurotransmitter, neuromodulator,
        etc.

        :param c_shock: shock affection constant
        :type c_shock: np.ndarray, float
        :param c_odour: odour affection constant
        :type c_odour: np.ndarray, float
        :param c_fback: feedback affection constant
        :type c_fback: np.ndarray, float
        """
        kwargs['d'] = kwargs.get('d', (2, 3))
        kwargs['name'] = kwargs.get('name', 'MB-neuron')
        kwargs['utype'] = kwargs.get('utype', 'mc')
        super(MCNeuron, self).__init__(**kwargs)
        self._c = np.array([
            np.array(c_odour) if type(c_odour) is not float and len(c_odour) == self._d else np.full(2, c_odour),
            np.array(c_shock) if type(c_shock) is not float and len(c_shock) == self._d else np.full(2, c_shock),
            np.array(c_fback) if type(c_fback) is not float and len(c_fback) == self._d else np.full(2, c_fback)
            ])
        self._w = np.zeros(self._d + tuple([2]), dtype=float)
        self.w_odour = np.eye(2, self._d[0], dtype=float)
        self.w_shock = np.eye(2, self._d[0], dtype=float)
        self.w_fback = np.eye(2, self._d[0], dtype=float)
        self._f = lambda x: np.maximum(x, -self._b - 1.)
        self._t = 0.  # time in milliseconds
        self._dts = [5000., 4000., 1000., 5000.]  # the different times of the neuron's timesteps
        self._iterations = 0

    def _fprop(self, odour, shock, fb_mbon=None):
        v = self._v.copy()
        c = self._c.copy()

        if self._verbose > 2:
            print unicode(self.name), self._t, odour, shock, fb_mbon

        # observations
        s = (shock * odour).dot(self.w_shock)  # allocate the shock obs to the corresponding odour identity
        o = odour.dot(self.w_odour)  # scale the odour obs
        m = ((np.array(fb_mbon)
              if type(fb_mbon) is not float and len(fb_mbon) == self._d[0]
              else np.full(self._d[0], fb_mbon)
              ) if fb_mbon is not None else np.zeros(2)).dot(self.w_fback)
        # s[np.isnan(s)] = 0.
        # o[np.isnan(o)] = 0.
        # m[np.isnan(m)] = 0.

        # targets
        t = np.zeros_like(v)
        t[:, 0] = o
        t[:, 1] = s
        t[:, 2] = m

        # errors
        e = (t - v + self._b) if len(t) > 0 else 0.

        if self.utype == "mc":  # Monte Carlo
            v += self._eta * e
        else:
            raise ValueError("This type of neuron is not supported. Please use 'Monte Carlo' type.")

        z = np.zeros_like(c)
        z[0] = odour
        z[1] = shock * odour
        z[2] = [1., 1.]
        z *= np.sqrt(self._u.T) / np.power(np.maximum(self._t, 1.), 1. - np.sqrt(self._u.T))
        # weaken the weights that cause high error
        if np.isnan(e.sum()):
            print unicode(self.name), "\tT:", self._t, "\tv:", v.T.flatten(), "\tt:", t.T.flatten()
        if np.isnan(z.sum()):
            print unicode(self.name), "\tT:", self._t, "\tz:", z.flatten()
        i = np.absolute(e.T) < 1e-03
        upt = -z * e.T
        # strengthen the weights that cause low error
        upt[i] += z[i]
        c += upt
        if np.isnan(c.sum()):
            print unicode(self.name), "\tT:", self._t, "\tc:", c.flatten(),
            print "\tz:", z.flatten(), "\tu:", self._u.T.flatten(), "\te:", e.T.flatten()
            print unicode(self.name), "\tT:", self._t, "\tupt:", upt.flatten()
        # TODO: make sure that you understand what is going on here

        if self._verbose > 2:
            print "\t\todour\t\t\tshock\t\t\tfback"
            print "v\t(%.2f, %.2f)\t(%.2f, %.2f)\t(%.2f, %.2f)" % (
                self._v[0, 0], self._v[1, 0], self._v[0, 1], self._v[1, 1], self._v[0, 2], self._v[1, 2])
            print "c\t(%.2f, %.2f)\t(%.2f, %.2f)\t(%.2f, %.2f)" % (
                self.c_odour[0], self.c_odour[1], self.c_shock[0], self.c_shock[1], self.c_fback[0], self.c_fback[1])
        return v, c, e

    @property
    def v(self):
        return self._f(self._v.sum(axis=1))

    @property
    def c_odour(self):
        return self._c[0]

    @c_odour.setter
    def c_odour(self, v):
        self._c[0] = v

    @property
    def c_shock(self):
        return self._c[1]

    @c_shock.setter
    def c_shock(self, v):
        self._c[1] = v

    @property
    def c_fback(self):
        return self._c[2]

    @c_fback.setter
    def c_fback(self, v):
        self._c[2] = v

    @property
    def w_odour(self):
        return self.c_odour * self._w[:, 0]

    @w_odour.setter
    def w_odour(self, v):
        self._w[:, 0] = v

    @property
    def w_shock(self):
        return self.c_shock * self._w[:, 1]

    @w_shock.setter
    def w_shock(self, v):
        self._w[:, 1] = v

    @property
    def w_fback(self):
        return self.c_fback * self._w[:, 2]

    @w_fback.setter
    def w_fback(self, v):
        self._w[:, 2] = v

    def __next_dt(self):
        dt = self._dts[self._iterations % len(self._dts)]
        self._iterations += 1
        return dt

    def __call__(self, *args, **kwargs):
        odour = args[0] if len(args) > 0 else kwargs.get("odour", np.zeros(2, dtype=float))
        shock = args[1] if len(args) > 1 else kwargs.get("shock", 0.)
        fb_mbon = args[2] if len(args) > 2 else kwargs.get("fb_mbon", 0.)
        update = args[3] if len(args) > 3 else kwargs.get("update", True)
        dt = self.__next_dt()
        self._t += dt
        v, c, e = self._fprop(odour=odour, shock=shock, fb_mbon=fb_mbon)
        if update:
            self._v = v
            self._c = c
            self._e = e
        else:
            self._iterations -= 1
            self._t -= dt
        return self._f(v.sum(axis=1))

    def __copy__(self):
        n = MCNeuron()
        n._eta = copy.copy(self._eta)
        n._c = copy.copy(self._c)
        n._u = copy.copy(self._u)
        n._b = copy.copy(self._b)
        n._d = copy.copy(self._d)
        n._v = copy.copy(self._v)
        n._e = copy.copy(self._e)
        n._w = copy.copy(self._w)
        n.gamma = copy.copy(self.gamma)
        n.utype = copy.copy(self.utype)
        n._f = copy.copy(self._f)
        return n

    def __deepcopy__(self, memodict={}):
        n = MCNeuron()
        n._eta = copy.deepcopy(self._eta, memodict)
        n._c_shock = copy.deepcopy(self._c_shock, memodict)
        n._c_odour = copy.deepcopy(self._c_odour, memodict)
        n._u = copy.deepcopy(self._u, memodict)
        n._b = copy.deepcopy(self._b, memodict)
        n._d = copy.deepcopy(self._d, memodict)
        n._v = copy.deepcopy(self._v, memodict)
        n._e = copy.deepcopy(self._e, memodict)
        n._w_odour = copy.deepcopy(self._w_odour, memodict)
        n._w_shock = copy.deepcopy(self._w_shock, memodict)
        n.gamma = copy.deepcopy(self.gamma, memodict)
        n.utype = copy.deepcopy(self.utype, memodict)
        n._f = copy.deepcopy(self._f, memodict)
        return n

    def __str__(self):
        types = {'mc': 'Monte Carlo', 'td': 'Time Difference', 'dp': 'Dynamic Programming'}
        return unicode("Mushroom Body neuron '%s'\t\t\t\t\t\tRest:\t% 2.2f\n"
                       "\t\t\t\t\t\t\t\todour\t\t\tshock\t\t\tfeedback\n"
                       "\t\tV:\t\t\t\t\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\n"
                       "\t\tW:\t\t\t\t\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\n"
                       "\t\tUpdate rate (STM):\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)\n"
                       "\t\tAdaptation (LTM): \t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f),\t(% 2.2f, % 2.2f)" % (
                        self.name, self._b,
                        self._v[0, 0], self._v[1, 0],
                        self._v[0, 1], self._v[1, 1],
                        self._v[0, 2], self._v[1, 2],
                        self.c_odour[0], self.c_odour[1],
                        self.c_shock[0], self.c_shock[1],
                        self.c_fback[0], self.c_fback[1],
                        self._eta[0, 0], self._eta[1, 0],
                        self._eta[0, 1], self._eta[1, 1],
                        self._eta[0, 2], self._eta[1, 2],
                        self._u[0, 0], self._u[1, 0],
                        self._u[0, 1], self._u[1, 1],
                        self._u[0, 2], self._u[1, 2]))


def network_to_features(nn, sparse=True):
    if sparse:
        x = np.zeros(120, dtype=float)
    else:
        x = np.zeros(62, dtype=float)
    x_d1, x_d2, x_d3, x_m1, x_m2, x_m3 = np.reshape(x[:84] if sparse else x[:48], (6, -1))

    for n, x_i in zip(nn.neurons, [x_d1, x_d2, x_d3, x_m1, x_m2, x_m3]):
        x_i[0:6] = n._c.flatten()
        x_i[6:8] = n._b.flatten()
        if sparse:
            x_i[8:11] = n._eta.mean(axis=0)
            x_i[11:14] = n._u.mean(axis=0)

    if sparse:
        x[84:93] = nn.w_mbon2mbon.flatten()
        x[93:102] = nn.w_mbon2dan.flatten()
        x[102:111] = nn.w_dan2mbon.flatten()
        x[111:120] = nn.w_dan2mbon.flatten()
    else:
        x[48:51] = nn.w_mbon2mbon[[0, 0, 1], [1, 2, 2]]
        x[51:56] = nn.w_mbon2dan[[0, 1, 1, 2, 2], [0, 1, 2, 1, 2]]
        x[56:59] = nn.w_dan2mbon[[0, 1, 2], [0, 1, 2]]
        x[59:62] = nn.w_dan2kc[[0, 1, 2], [0, 1, 2]]

    return x


def network_from_features(x, sparse=True, verbose=False):
    names = [u"PPL1-\u03b31pedc", u"PAM-\u03b2'2m", u"PAM-\u03b2'2a",
             u"MBON-\u03b31pedc", u"MBON-\u03b2'2mp", u"MBON-\u03b35\u03b2'2a"]
    order = ['d1', 'd2', 'd3', 'm1', 'm2', 'm3']
    x_d1, x_d2, x_d3, x_m1, x_m2, x_m3 = np.reshape(x[:84] if sparse else x[:48], (6, -1))

    ns = {}
    for kw, name, x_i in zip(order, names, [x_d1, x_d2, x_d3, x_m1, x_m2, x_m3]):
        kwargs = {
            'name': name,
            'c_odour': x_i[0:2],
            'c_shock': x_i[2:4],
            'c_fback': x_i[4:6],
            'b': x_i[6:8],
            'a': np.maximum(x_i[8:11], 0.) if sparse else 1.,
            'u': np.maximum(x_i[11:14], 0.) if sparse else 0.
        }
        ns[kw] = MBNeuron(**kwargs)

    nn = MBNetwork(dan=[ns['d1'], ns['d2'], ns['d3']], mbon=[ns['m1'], ns['m2'], ns['m3']], verbose=verbose)
    if sparse:
        nn.w_mbon2mbon = x[84:93].reshape((3, 3))
        nn.w_mbon2dan = x[93:102].reshape((3, 3))
        nn.w_dan2mbon = x[102:111].reshape((3, 3))
        nn.w_dan2kc = x[111:120].reshape((3, 3))
    else:
        nn.w_mbon2mbon[[0, 0, 1], [1, 2, 2]] = x[48:51]
        nn.w_mbon2dan[[0, 1, 1, 2, 2], [0, 1, 2, 1, 2]] = x[51:56]
        nn.w_dan2mbon[[0, 1, 2], [0, 1, 2]] = x[56:59]
        nn.w_dan2kc[[0, 1, 2], [0, 1, 2]] = x[59:62]

    return nn


def network_to_file(nn, filename='new-network.yaml'):
    params = {'neurons': [], 'network': {'mbon2mbon': [], 'mbon2dan': [], 'dan2mbon': [], 'dan2kc': []}}

    for neuron in nn.neurons:
        params['neurons'].append({
            'name': neuron.name,
            'a': neuron._eta.tolist(),
            'u': neuron._u.tolist(),
            'c_odour': neuron.c_odour.tolist(),
            'c_shock': neuron.c_shock.tolist(),
            'c_fback': neuron.c_fback.tolist(),
            'b': neuron._b.tolist()
        })

    params['network']['mbon2mbon'] = nn.w_mbon2mbon.tolist()
    params['network']['mbon2dan'] = nn.w_mbon2dan.tolist()
    params['network']['dan2mbon'] = nn.w_dan2mbon.tolist()
    params['network']['dan2kc'] = nn.w_dan2kc.tolist()

    with open(os.path.join(__dir__, filename), 'wb') as f:
        yaml.safe_dump(params, f)

    return params


def network_from_file(filename='new-init.yaml', verbose=False):
    with open(os.path.join(__dir__, filename), 'rb') as f:
        params = yaml.safe_load(f)

    ns = {}
    order = ['d1', 'd2', 'd3', 'm1', 'm2', 'm3']
    for kw, kwargs in zip(order, params['neurons']):
        ns[kw] = MBNeuron(**kwargs)

    nn = MBNetwork(dan=[ns['d1'], ns['d2'], ns['d3']], mbon=[ns['m1'], ns['m2'], ns['m3']], verbose=verbose)
    nn.w_mbon2mbon = np.array(params['network']['mbon2mbon'])
    nn.w_mbon2dan = np.array(params['network']['mbon2dan'])
    nn.w_dan2mbon = np.array(params['network']['dan2mbon'])
    nn.w_dan2kc = np.array(params['network']['dan2kc'])

    return nn


def network_to_features_old(nn):
    x = np.zeros(99, dtype=float)
    x_d1, x_d2, x_d3, x_m1, x_m2, x_m3 = np.reshape(x[:72], (-1, 12))

    for n, x_i in zip(nn.neurons, [x_d1, x_d2, x_d3, x_m1, x_m2, x_m3]):
        x_i[0:2] = n._v.flatten()
        x_i[2:4] = n._c_odour.flatten()
        x_i[4:6] = n._c_shock.flatten()
        x_i[6:8] = n._eta.flatten()
        x_i[8:10] = n._u.flatten()
        x_i[10] = n._b
        x_i[11] = {"mc": 0, "td": 1, "dp": 2}[n.utype]

    x[72:81] = nn.w_mbon2mbon.flatten()
    x[81:90] = nn.w_mbon2dan.flatten()
    x[90:99] = nn.w_dan2mbon.flatten()

    return x


def network_from_features_old(x, verbose=False):
    names = [u"PPL1-\u03b31pedc", u"PAM-\u03b2'2m", u"PAM-\u03b2'2a",
             u"MBON-\u03b31pedc", u"MBON-\u03b2'2mp", u"MBON-\u03b35\u03b2'2a"]
    order = ['d1', 'd2', 'd3', 'm1', 'm2', 'm3']
    x_d1, x_d2, x_d3, x_m1, x_m2, x_m3 = np.reshape(x[:72], (-1, 12))

    ns = {}
    for kw, name, x_i in zip(order, names, [x_d1, x_d2, x_d3, x_m1, x_m2, x_m3]):
        kwargs = {
            'name': name,
            'c_odour': x_i[2:4],
            'c_shock': x_i[4:6],
            'a': x_i[6:8],
            'u': x_i[8:10],
            'b': x_i[10],
            'utype': ["mc", "td", "dp"][int(x_i[11])]
        }
        ns[kw] = MANeuron(**kwargs)
        ns[kw].v = x_i[0:2]

    nn = MANetwork(dan=[ns['d1'], ns['d2'], ns['d3']], mbon=[ns['m1'], ns['m2'], ns['m3']], verbose=verbose)
    nn.w_mbon2mbon = x[72:81].reshape((3, 3))
    nn.w_mbon2dan = x[81:90].reshape((3, 3))
    nn.w_dan2kc = x[90:99].reshape((3, 3))

    return nn


def network_to_file_old(nn, filename='new-network.yaml'):
    params = {'neurons': [], 'network': {'mbon2mbon': [], 'mbon2dan': [], 'dan2mbon': []}}

    for neuron in nn.neurons:
        params['neurons'].append({
            'name': neuron.name,
            'v': neuron.v.tolist(),
            'a': neuron._eta.tolist(),
            'u': neuron._u.tolist(),
            'c_odour': neuron.c_odour.tolist(),
            'c_shock': neuron._c_shock.tolist(),
            'b': neuron._b,
            'utype': neuron.utype
        })

    params['network']['mbon2mbon'] = nn.w_mbon2mbon.tolist()
    params['network']['mbon2dan'] = nn.w_mbon2dan.tolist()
    params['network']['dan2mbon'] = nn.w_dan2mbon.tolist()

    with open(os.path.join(__dir__, filename), 'wb') as f:
        yaml.safe_dump(params, f)

    return params


def network_from_file_old(filename='opt-init.yaml', verbose=False):
    with open(os.path.join(__dir__, filename), 'rb') as f:
        params = yaml.safe_load(f)

    ns = {}
    order = ['d1', 'd2', 'd3', 'm1', 'm2', 'm3']
    for kw, kwargs in zip(order, params['neurons']):
        v = kwargs.pop('v')
        n = MANeuron(**kwargs)
        n.v = np.array(v)
        ns[kw] = n

    nn = MANetwork(dan=[ns['d1'], ns['d2'], ns['d3']], mbon=[ns['m1'], ns['m2'], ns['m3']], verbose=verbose)
    nn.w_mbon2mbon = np.array(params['network']['mbon2mbon'])
    nn.w_mbon2dan = np.array(params['network']['mbon2dan'])
    nn.w_dan2kc = np.array(params['network']['dan2mbon'])

    return nn
