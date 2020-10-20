import numpy as np
from copy import copy
from numbers import Number
from world.conditions import Hybrid
from sphere.distance import angle_between
# from sphere import angle_between


class PolygonList(list):

    def __init__(self, polygons=None):
        """

        :param polygons:
        :type polygons: list
        """
        items = []
        ext = 0
        if polygons is not None:
            for p in polygons:  # type: Polygon
                if isinstance(p, Polygon):
                    items.append(p)
                else:
                    ext += 1
        if len(items) > 0:
            super(PolygonList, self).__init__(items)
        else:
            super(PolygonList, self).__init__()

        if ext > 0:
            print("Warning: %d elements extracted from the list because of wrong type." % ext)

    @property
    def x(self):
        return np.array([p.x for p in self])  # type: np.ndarray

    @property
    def y(self):
        return np.array([p.y for p in self])  # type: np.ndarray

    @property
    def z(self):
        return np.array([p.z for p in self])  # type: np.ndarray

    @property
    def c(self):
        return np.array([p.c for p in self])  # type: np.ndarray

    @property
    def c_int32(self):
        return np.array([p.c_int32 for p in self])  # type: np.ndarray

    def scale(self, xmax=1., ymax=1., zmax=1., centralise=False, decentralise=False):
        """
        Rescales the 3 dimensions to the given bounds.
        :param xmax: the positive bound of x
        :type xmax: float
        :param ymax: the positive bound of y
        :type ymax: float
        :param zmax: the positive bound of z
        :type zmax: float
        :param centralise: make zero the centre of the world
        :type centralise: bool
        :param decentralise: undo zero the centre of the world
        :type decentralise: bool
        :return: a sequence generator with the transformed positions
        """
        for p in self:
            yield p.scale(xmax, ymax, zmax, centralise, decentralise)

    def __add__(self, other):
        ps = []
        for p in self:
            ps.append(p.__add__(other))
        return PolygonList(ps)

    def __sub__(self, other):
        ps = []
        for p in self:
            ps.append(p.__sub__(other))
        return PolygonList(ps)

    def __mul__(self, other):
        ps = []
        for p in self:
            ps.append(p.__mul__(other))
        return PolygonList(ps)

    def __div__(self, other):
        ps = []
        for p in self:
            ps.append(p.__div__(other))
        return PolygonList(ps)

    def __str__(self):
        s = "["
        if len(self) < 10:
            for p in self:
                s += p.__str__() + ",\n "
        else:
            for p in self[:3]:
                s += p.__str__() + ",\n "
            s += "   . . .\n "
            s += self[-1].__str__() + ",\n "
        return s[:-3] + "]"


class Polygon(object):

    def __init__(self, xs, ys, zs, colour=(0, 0, 0)):
        """

        :param xs: x coodrinates in meters
        :type xs: np.ndarray
        :param ys: y coordinates in meters
        :type ys: np.ndarray
        :param zs: z coordinates in meters
        :type zs: np.ndarray
        :param colour: colour (R, G, B) in [0, 1] ^ 3
        """
        self.x = xs
        self.y = ys
        self.z = zs
        self._c = np.array(colour)

    @property
    def xyz(self):
        return tuple((x, y, z) for x, y, z in zip(self.x, self.y, self.z))

    @property
    def xy(self):
        return tuple((x, y) for x, y in zip(self.x, self.y))

    @property
    def xz(self):
        return tuple((x, z) for x, z in zip(self.x, self.z))

    @property
    def yx(self):
        return tuple((y, x) for y, x in zip(self.y, self.x))

    @property
    def yz(self):
        return tuple((y, z) for y, z in zip(self.y, self.z))

    @property
    def zx(self):
        return tuple((z, x) for z, x in zip(self.z, self.x))

    @property
    def zy(self):
        return tuple((z, y) for z, y in zip(self.z, self.y))

    @property
    def c(self):
        return tuple(self._c)

    @property
    def c_int32(self):
        return tuple(np.int32(self._c * 255))

    def scale(self, xmax=1., ymax=1., zmax=1., centralise=False, decentralise=False):
        """
        Rescales the 3 dimensions to the given bounds.
        :param xmax: the positive bound of x
        :type xmax: float
        :param ymax: the positive bound of y
        :type ymax: float
        :param zmax: the positive bound of z
        :type zmax: float
        :param centralise: make zero the centre of the world
        :type centralise: bool
        :param decentralise: undo zero the centre of the world
        :type decentralise: bool
        :return: a sequence generator with the transformed positions
        """

        xyzmax = np.abs(np.array([xmax, ymax, zmax]))
        p = []

        for x, y, z in zip(self.x, self.y, self.z):
            nxyz = np.array([x, y, z]) / xyzmax

            if centralise:
                # code to [-1, 1]
                nxyz -= .5
                nxyz *= 2.
            if decentralise:
                # code to [0, 1]
                nxyz += .5
                nxyz / 2.

            nx, ny, nz = nxyz
            p.append(tuple((nx, ny, nz)))
        return Polygon.from_tuples(p, self.c)

    def __add__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x += other[0]
            if len(other) > 1:
                p.y += other[1]
            elif len(other) > 0:
                p.y += other[0]
            if len(other) > 2:
                p.z += other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z += other[0]
        if isinstance(other, Number):
            p.x += other
            p.y += other
            p.z += other
        return p

    def __sub__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x -= other[0]
            if len(other) > 1:
                p.y -= other[1]
            elif len(other) > 0:
                p.y -= other[0]
            if len(other) > 2:
                p.z -= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z -= other[0]
        if isinstance(other, Number):
            p.x -= other
            p.y -= other
            p.z -= other
        return p

    def __mul__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x *= other[0]
            if len(other) > 1:
                p.y *= other[1]
            elif len(other) > 0:
                p.y *= other[0]
            if len(other) > 2:
                p.z *= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z *= other[0]
        if isinstance(other, Number):
            p.x *= other
            p.y *= other
            p.z *= other
        return p

    def __div__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x /= other[0]
            if len(other) > 1:
                p.y /= other[1]
            elif len(other) > 0:
                p.y /= other[0]
            if len(other) > 2:
                p.z /= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z /= other[0]
        if isinstance(other, Number):
            p.x /= other
            p.y /= other
            p.z /= other
        return p

    def __copy__(self):
        return Polygon(self.x.copy(), self.y.copy(), self.z.copy(), self._c.copy())

    def __str__(self):
        s = "C: (%.2f, %.2f, %.2f), P: " % self.c
        for x, y, z in self.xyz:
            s += "(%.2f, %.2f, %.2f) " % (x, y, z)
        return s[:-1]

    @classmethod
    def from_tuples(cls, tup, c=(0, 0, 0)):
        x, y, z = np.array(tup).T
        return Polygon(xs=x, ys=y, zs=z, colour=c)


class Route(object):
    """
    Route of an agent in a world.
    """

    def __init__(self, xs, ys, zs=None, phis=None, condition=Hybrid(), agent_no=None, route_no=None):
        """

        :param xs: the x position of the agent in the world (in meters) in each time-step
        :type xs: np.ndarray, list
        :param ys: the y position of the agent in the world (in meters) in each time-step
        :type ys: np.ndarray, list
        :param zs: the z position of the agent in the world (in meters) in each time-step
        :type zs: np.ndarray, list, float
        :param phis: the facing direction of the agent in the world (in meters) in each time-step
        :type phis: np.ndarray, list, float
        :param condition: the stepping condition - default it NoneCondition
        :type condition: Hybrid
        :param agent_no: the agent's ID
        :type agent_no: int
        :param route_no: the route's ID
        :type route_no: int
        """
        self.x = np.array(xs)  # type: np.ndarray
        self.y = np.array(ys)  # type: np.ndarray
        if isinstance(zs, Number):
            self.z = np.ones_like(xs) * zs  # type: np.ndarray
        elif isinstance(zs, list) or isinstance(zs, np.ndarray):
            self.z = np.array(zs)  # type: np.ndarray
        else:
            self.z = np.ones_like(xs) * .01  # type: np.ndarray

        if isinstance(phis, Number):
            self.phi = np.ones_like(xs) * phis  # type: np.ndarray
        elif isinstance(phis, list) or isinstance(phis, np.ndarray):
            self.phi = np.array(phis)  # type: np.ndarray
        else:
            self.phi = np.arctan2(ys, xs)  # type: np.ndarray
        self.agent_no = agent_no if agent_no is not None else -1  # type: int
        self.route_no = route_no if route_no is not None else -1  # type: int

        self.__condition = condition if condition is not None else Hybrid()  # type: Hybrid
        dx = np.sqrt(
            np.square(self.x[1:] - self.x[:-1]) +
            np.square(self.y[1:] - self.y[:-1]) +
            np.square(self.z[1:] - self.z[:-1])
        )
        self.__mean_dx = dx.mean() if dx.size > 0 else 0.  # type: float
        # the duration of the route is assumed to be 2s
        self.dt = 2. / self.x.size  # type: float

    @property
    def dx(self):
        """
        The step size of the agent (in meters)
        :return:
        """
        if self.condition.step > 0.:
            return self.condition.step  # type: float
        else:
            return self.__mean_dx  # type: float

    @dx.setter
    def dx(self, value):
        """

        :param value: the step size of the agent in meters
        :type value: float
        """
        if self.condition.step == 0.:
            raise AttributeError("Cannot change the distance threshold of a condition. "
                                 "You have to change the entire condition instead.")
        else:
            self.__mean_dx = value

    @property
    def condition(self):
        """
        :return: the step thresholding condition
        """
        return self.__condition  # type: Hybrid

    @condition.setter
    def condition(self, value):
        """

        :param value: the new condition
        :type value: Hybrid
        :return:
        """
        self.__condition = value if value is not None else Hybrid()  # type: Hybrid

    @property
    def xyz(self):
        """
        :return: the 3D position of the agent grouped in tuples
        """
        return tuple((x, y, z) for x, y, z, _ in self.__iter__())  # type: tuple

    @property
    def xy(self):
        """
        :return: the 2D position (top view) of the agent grouped in tuples (used for drawing the agent on the map)
        """
        return tuple((x, y) for x, y, _, _ in self.__iter__())  # type: tuple

    def scale(self, xmax=1., ymax=1., zmax=1., centralise=False, decentralise=False):
        """
        Rescales the 3 dimensions to the given bounds.
        :param xmax: the positive bound of x
        :type xmax: float
        :param ymax: the positive bound of y
        :type ymax: float
        :param zmax: the positive bound of z
        :type zmax: float
        :param centralise: make zero the centre of the world
        :type centralise: bool
        :param decentralise: undo zero the centre of the world
        :type decentralise: bool
        :return: a sequence generator with the transformed positions
        """

        xyzmax = np.abs(np.array([xmax, ymax, zmax]))
        r = []

        for x, y, z, phi in self.__iter__():
            nxyz = np.array([x, y, z]) / xyzmax

            if centralise:
                # code to [-1, 1]
                nxyz -= .5
                nxyz *= 2.
            if decentralise:
                # code to [0, 1]
                nxyz += .5
                nxyz / 2.

            nx, ny, nz = nxyz
            r.append(tuple((nx, ny, nz, phi)))
        r = np.array(r).T
        return route_like(self, xs=r[0], ys=r[1], zs=r[2], phis=r[3],
                          condition=Hybrid(tau_x=self.condition.step / xyzmax[:2].mean(),
                                           tau_phi=self.condition.phi))

    def reverse(self):
        rt = self.__copy__()
        rt.x = rt.x[::-1]
        rt.y = rt.y[::-1]
        rt.z = rt.z[::-1]
        rt.phi = np.roll((rt.phi[::-1] + np.pi) % (2 * np.pi), 1)
        return rt

    def __iter__(self):
        px, py, pz, p_phi = self.x[0], self.y[0], self.z[0], self.phi[0]
        phi = 0.

        for x, y, z in zip(self.x[1:], self.y[1:], self.z[1:]):
            dv = np.array([x - px, y - py, z - pz])
            d = np.sqrt(np.square(dv).sum())
            phi = (np.arctan2(dv[0], dv[1]) + np.pi) % (2 * np.pi) - np.pi
            if self.condition(d, angle_between(phi, p_phi, sign=False)):
                yield px, py, pz, phi  # type: tuple
                px, py, pz, pphi = x, y, z, phi

        yield px, py, pz, phi  # type: tuple

    def __add__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x += other[0]
            if len(other) > 1:
                p.y += other[1]
            elif len(other) > 0:
                p.y += other[0]
            if len(other) > 2:
                p.z += other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z += other[0]
        if isinstance(other, Number):
            p.x += other
            p.y += other
            p.z += other
        return p  # type: Route

    def __sub__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x -= other[0]
            if len(other) > 1:
                p.y -= other[1]
            elif len(other) > 0:
                p.y -= other[0]
            if len(other) > 2:
                p.z -= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z -= other[0]
        if isinstance(other, Number):
            p.x -= other
            p.y -= other
            p.z -= other
        return p  # type: Route

    def __mul__(self, other):
        p = self.__copy__()
        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x *= other[0]
            if len(other) > 1:
                p.y *= other[1]
            elif len(other) > 0:
                p.y *= other[0]
            if len(other) > 2:
                p.z *= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z *= other[0]
        if isinstance(other, Number):
            p.x *= other
            p.y *= other
            p.z *= other
        return p  # type: Route

    def __div__(self, other):
        p = self.__copy__()

        if type(other) is tuple or type(other) is list or type(other) is np.ndarray:
            if len(other) > 0:
                p.x /= other[0]
            if len(other) > 1:
                p.y /= other[1]
            elif len(other) > 0:
                p.y /= other[0]
            if len(other) > 2:
                p.z /= other[2]
            elif len(other) > 1:
                pass
            elif len(other) > 0:
                p.z /= other[0]
        if isinstance(other, Number):
            p.x /= other
            p.y /= other
            p.z /= other
        return p  # type: Route

    def __copy__(self):
        r = Route(xs=self.x.copy(), ys=self.y.copy(), zs=self.z.copy(), phis=self.phi.copy(),
                  condition=self.condition, agent_no=self.agent_no, route_no=self.route_no)

        return r  # type: Route

    def __str__(self):
        if self.agent_no > 0 and self.route_no > 0:
            s = "Ant: %02d, Route %02d," % (self.agent_no, self.route_no)
        elif self.agent_no > 0 >= self.route_no:
            s = "Ant: %02d," % self.agent_no
        elif self.agent_no <= 0 < self.route_no:
            s = "Route: %02d," % self.route_no
        else:
            s = "Route:"
        for x, y, z in self.xyz:
            s += " (%.2f, %.2f)" % (x, y)
        s += ", Step: % 2.2f" % self.dx
        return s[:-1]  # type: basestring

    def save(self, filename):
        np.savez_compressed(filename,
                            condition=self.condition.to_array(),
                            agent=self.agent_no, route=self.route_no,
                            x=self.x, y=self.y, z=self.z, phi=self.phi)

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        new_route = Route(
            xs=data['x'], ys=data['y'], zs=data['z'], phis=data['phi_z'], condition=Hybrid.from_array(data['condition']),
            agent_no=data['agent'], route_no=data['route'])

        return new_route  # type: Route


def route_like(r, xs=None, ys=None, zs=None, phis=None,
               condition=None, agent_no=None, route_no=None):
    new_route = copy(r)
    if xs is not None:
        new_route.x = np.array(xs)
    if ys is not None:
        new_route.y = np.array(ys)
    if zs is not None:
        new_route.z = np.array(zs)
    if phis is not None:
        new_route.phi_z = np.array(phis)
    if condition is not None:
        new_route.condition = condition
    if agent_no is not None:
        new_route.agent_no = agent_no
    if route_no is not None:
        new_route.route_no = route_no
    return new_route  # type: Route


if __name__ == "__main__":
    from world import load_world, load_routes
    from datetime import datetime
    import matplotlib.pyplot as plt

    step = .01  # 1 cm
    tau_phi = np.pi    # 180 deg
    date = datetime(2017, 12, 21, 12, 0, 0)

    world = load_world()
    routes = load_routes()
    route = routes[0]
    route.agent_no = 1
    route.route_no = 2
    route.condition = Hybrid(tau_x=step, tau_phi=tau_phi)
    world.add_route(route)
    world.sky.obs.date = date
    world.sky.generate()

    img, _ = world.draw_top_view(1000, 1000)
    img.show(title="Route")

    plt.figure("Statistics")

    x = np.array(route.x)
    y = np.array(route.y)
    plt.subplot(211)
    plt.plot(x, label="x")
    plt.plot(y, label="y")
    plt.ylim([0, 10])
    plt.legend()

    plt.subplot(212)
    plt.plot(np.arctan2(np.diff(x), np.diff(y)), label="heading")
    plt.plot(np.ones(len(x)) * world.sky.lon, label="sun")
    plt.ylim([-np.pi, np.pi])
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["-pi", "-pi/2", "0", "pi/2", "pi"])
    plt.legend()

    plt.show()

