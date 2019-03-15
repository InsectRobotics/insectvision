#!/usr/bin/env python

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


import os
from scipy.io import loadmat

import numpy as np
import numpy.linalg as la
import ephem

from matplotlib import cm
from datetime import timedelta, datetime
from PIL import ImageDraw, Image
from utils import shifted_datetime
from ephem import Observer
from geometry import PolygonList, Polygon, Route


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + "/../data/"
__seville_2009__ = __data__ + "Seville2009_world/"
WORLD_FILENAME = "world5000_gray.mat"
ROUTES_FILENAME = "AntRoutes.mat"


cmap = cm.get_cmap('brg')

WIDTH = 36
HEIGHT = 10
LENGTH = 36

GRASS_COLOUR = (0, 255, 0)
GROUND_COLOUR = (229, 183, 90)
SKY_COLOUR = (13, 135, 201)


class World(object):
    """
    A representation of the world with object (described as polygons) and agents' routes
    """

    def __init__(self, observer=None, polygons=None, width=WIDTH, length=LENGTH, height=HEIGHT,
                 uniform_sky=False, enable_pol_filters=True, day_shift=153, daylight_only=True):
        """
        Creates a world.

        :param observer: a reference to an observer
        :type observer: Observer
        :param polygons: polygons of the objects in the world
        :type polygons: PolygonList
        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :param uniform_sky: flag that indicates if there is a uniform sky or not
        :type uniform_sky: bool
        :param enable_pol_filters: flag to switch on/off the POL filters of the eyes
        :type enable_pol_filters: bool
        :param day_shift: the number of days to roll back
        :type day_shift: int
        :param daylight_only: bound the time in between 7.30 am and 7.30 pm
        :type daylight_only: bool
        """
        # normalise world
        xmax = np.array([polygons.x.max(), polygons.y.max(), polygons.z.max()]).max()

        # default observer is in Seville (where the data come from)
        if observer is None:
            observer = ephem.Observer()
            observer.lat = '37.392509'
            observer.lon = '-5.983877'
        self.day_shift = day_shift
        self.daylight_only = daylight_only
        self.__shifted = False
        observer.date = self.datetime_now(init=True)

        # create and generate a sky instance
        self.sky = SkyModel(observer=observer)
        self.sky.generate()

        # create a compound eye model for the sky pixels
        self.eye = None  # type: CompoundEye
        self.__pol_filters = enable_pol_filters

        # store the polygons and initialise the parameters
        self.polygons = polygons
        self.routes = []
        self.width = width
        self.length = length
        self.height = height
        self.__normalise_factor = xmax  # type: float
        self.uniform_sky = uniform_sky

    @property
    def ratio2meters(self):
        return self.__normalise_factor  # type: float

    @property
    def date(self):
        return self.sky.obs.date.datetime()  # type: datetime

    def enable_pol_filters(self, value):
        """

        :param value:
        :type value: bool
        :return:
        """
        self.__pol_filters = value

    def add_route(self, route):
        """
        Adds an ant-route in the world

        :param route: the new route
        :type route: Route
        :return: None
        """
        self.routes.append(route)

    def draw_top_view(self, width=None, length=None, height=None):
        """
        Draws a top view of the world and all the added paths in it.

        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :return: an image of the top view
        """

        # set the default values to the dimensions of the world
        if width is None:
            width = self.width
        if length is None:
            length = self.length
        if height is None:
            height = self.height

        # create new image and drawer
        image = Image.new("RGB", (width, length), GROUND_COLOUR)
        draw = ImageDraw.Draw(image)

        # draw the polygons
        for p in self.polygons.scale(*((self.ratio2meters,) * 3)):
            pp = p * [width, length, height]
            draw.polygon(pp.xy, fill=pp.c_int32)

        # draw the routes
        nants = int(np.array([r.agent_no for r in self.routes]).max())      # the ants' ID
        nroutes = int(np.array([r.route_no for r in self.routes]).max())  # the routes' ID
        for route in self.routes:
            # code the routes similarly to the polygons
            rt = route.scale(*(self.ratio2meters,) * 3)
            rt = rt * [width, length, height]
            r, g, b, _ = cmap(float(rt.agent_no) / float(len(self.routes)))
            draw.line(rt.xy, fill=(int(r * 255), int(g * 255), int(b * 255)))

            r = 20.
            for x0, y0, _, phi in rt:
                x1 = x0 + r * np.sin(phi)
                y1 = y0 + r * np.cos(phi)
                draw.line(((x0, y0), (x1, y1)), fill=(int(r * 255), int(g * 255), int(b * 255)))

        return image

    def draw_panoramic_view(self, x=None, y=None, z=None, r=0, width=None, length=None, height=None,
                            include_ground=1., include_sky=1., update_sky=True):
        """
        Draws a panoramic view of the world

        :param x: The x coordinate of the agent in the world
        :type x: float
        :param y: The y coordinate of the agent in the world
        :type y: float
        :param z: The z coordinate of the agent in the world
        :type z: float
        :param r: The orientation of the agent in the world
        :type r: float
        :param width: the width of the world
        :type width: int
        :param length: the length of the world
        :type length: int
        :param height: the height of the world
        :type height: int
        :param include_ground: the percentage of the ground to include in the image
        :type include_ground: float
        :param include_sky: the percentage of the sky to include in the image
        :type include_sky: float
        :param update_sky: flag that specifies if we want to update the sky
        :type update_sky: bool
        :return: an image showing the 360 degrees view of the agent
        """

        # set the default values for the dimensions of the world
        if width is None:
            width = self.width
        if length is None:
            length = self.length
        if height is None:
            height = self.height
        if x is None:
            x = width / 2.
        if y is None:
            y = length / 2.
        if z is None:
            z = height / 2. + .06 * height

        # calculate the number of pixels allocated to the sky (counting from the top border)
        Z = (include_sky + include_ground) / 2
        horizon = int(height * include_sky / (2 * Z))

        # create ommatidia positions with respect to the resolution
        # (this is for the sky drawing on the panoramic images)
        thetas = np.linspace(np.pi/2 * include_sky, 0, horizon, endpoint=False)
        phis = np.linspace(-np.pi, np.pi, width, endpoint=False)
        thetas, phis = np.meshgrid(thetas, phis)
        ommatidia = np.array([thetas.flatten(), phis.flatten()]).T

        image = Image.new("RGB", (width, height), rgb2gbuv(GROUND_COLOUR))
        draw = ImageDraw.Draw(image)

        if self.uniform_sky:
            draw.rectangle((0, 0, width, horizon), fill=rgb2gbuv(SKY_COLOUR, 255))
        else:
            # create a compound eye model for the sky pixels
            # self.eye = CompoundEye(ommatidia,
            #                        central_microvili=(0., 0.),
            #                        noise_factor=.1,
            #                        activate_dop_sensitivity=True)
            self.eye = AntEye(ommatidia)
            self.eye.activate_pol_filters(self.__pol_filters)
            self.eye.sky = self.sky
            if update_sky:
                self.sky.obs.date = self.datetime_now()
                self.sky.generate()
            self.eye.rotate(yaw=-r)

            pix = image.load()
            for i, c in enumerate(self.eye.L):
                pix[i // horizon, i % horizon] = tuple(np.int32(255 * c))

        # rotation matrix of the POV orientation
        R = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ])
        thetas, phis, rhos = [], [], []

        # code position for meters to pixel-space
        pos = np.array([x, y, z]) / self.ratio2meters
        pos *= np.array([width, length, height / Z])
        for p in self.polygons.scale(*((self.ratio2meters,) * 3)):
            # code polygons' points from meters to pixels
            pp = p * [width, length, height / Z]  # type: Polygon
            # and then into spherical coordinates
            xyz = np.array(pp.xyz) - pos
            theta, phi, rho = vec2sph(xyz.dot(R).T)
            thetas.append(theta)
            phis.append(phi)
            rhos.append(rho)

        # code spherical elevation to pixel height
        thetas = (height / Z) * ((((np.pi/2 - np.array(thetas)) % np.pi) / np.pi) - (1 - include_sky) / 2.)
        phis = width * ((np.pi + np.array(phis)) % (2 * np.pi)) / (2 * np.pi)
        rhos = la.norm(np.array(rhos), axis=-1)
        ind = np.argsort(rhos)[::-1]
        for theta, phi, c in zip(thetas[ind], phis[ind], self.polygons.c_int32[ind]):
            if phi.max() - phi.min() < width/2:  # normal conditions
                p = tuple((b, a) for a, b in zip(theta, phi))
                draw.polygon(p, fill=rgb2gbuv(c))
            else:   # in case that the object is on the edge of the screen
                phi0, phi1 = phi.copy(), phi.copy()
                phi0[phi < width/2] += width
                phi1[phi >= width/2] -= width
                p = tuple((b, a) for a, b in zip(theta, phi0))
                draw.polygon(p, fill=rgb2gbuv(c))
                p = tuple((b, a) for a, b in zip(theta, phi1))
                draw.polygon(p, fill=rgb2gbuv(c))

            # draw visible polygons

        return image

    def datetime_now(self, init=False):
        date = shifted_datetime(self.day_shift, lower_limit=None, upper_limit=None)
        if init:
            date_shift = shifted_datetime(self.day_shift, lower_limit=7.5, upper_limit=19.5)
            if date_shift.day != date.day:
                self.__shifted = True

        if self.__shifted:
            date += timedelta(hours=12)

        return date


def rgb2gbuv(rgb, uv=0):
    return tuple((rgb[1], rgb[2], uv))


def load_world(world_filename=WORLD_FILENAME, width=WIDTH, length=LENGTH, height=HEIGHT):
    mat = loadmat(__seville_2009__ + world_filename)
    polygons = PolygonList()
    for xs, ys, zs, col in zip(mat["X"], mat["Y"], mat["Z"], mat["colp"]):
        col[0] = col[2] = 0
        polygons.append(Polygon(xs, ys, zs, col))
    observer = ephem.Observer()
    observer.lat = '37.392509'
    observer.lon = '-5.983877'

    return World(observer=observer, polygons=polygons, width=width, length=length, height=height)


def load_routes(routes_filename=ROUTES_FILENAME):
    mat = loadmat(__seville_2009__ + routes_filename)
    ant, route, key = 1, 1, lambda a, r: "Ant%d_Route%d" % (a, r)
    routes = []
    while key(ant, route) in mat.keys():
        while key(ant, route) in mat.keys():
            mat[key(ant, route)][:, :2] /= 100.  # convert the route data to meters
            xs, ys, phis = mat[key(ant, route)].T
            r = Route(xs, ys, .01, phis=np.deg2rad(phis), agent_no=ant, route_no=route)
            routes.append(r)
            route += 1
        ant += 1
        route = 1
    return routes


def load_route(name):
    return Route.from_file(__data__ + "routes/" + name + ".npz")


def save_route(rt, name):
    rt.save(__data__ + "routes/" + name + ".npz")


if __name__ == "__main__":
    import pygame
    from conditions import Stepper

    H = 500
    W = 1000
    # mode = "panorama"
    mode = "top"

    done = False

    world = load_world()
    # world.uniform_sky = True
    routes = load_routes()
    for route in routes:
        route.condition = Stepper(.1)
        world.add_route(route)
        break

    if mode == "top":
        img, draw = world.draw_top_view(width=W, length=W)
        img.show()
    elif mode == "panorama":
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        for x, y, z, phi in world.routes[-1]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            # transform position to meters
            x, y, z = (np.array([x, y, z]) + .5) * world.ratio2meters
            img, draw = world.draw_panoramic_view(x, y, z, phi, W, W, H)
            img = img.resize((W, H), Image.ANTIALIAS)
            screen.blit(pygame.image.fromstring(img.tobytes("raw", "RGB"), img.size, "RGB"), (0, 0))
            pygame.display.flip()

            if done:
                break
