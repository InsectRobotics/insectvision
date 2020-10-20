import numpy as np
from world import World, Route, Hybrid
from .utils import *


class Agent(object):
    FOV = (-np.pi/2, np.pi/2)
    __latest_agent_id__ = 0

    def __init__(self, init_pos=np.zeros(3), init_rot=np.zeros(3), condition=Hybrid(),
                 live_sky=True, fov=None, rgb=False, visualiser=None, name=None):
        """
        :param init_pos: the initial position
        :type init_pos: np.ndarray
        :param init_rot: the initial orientation (yaw, pitch, roll)
        :type init_rot: np.ndarray
        :param condition:
        :type condition: Hybrid
        :param live_sky: flag to update the sky with respect to the time
        :type live_sky: bool
        :param fov: vertical field of view of the agent (the widest: -pi/2 to pi/2)
        :type fov: tuple, list, np.ndarray
        :param rgb: flag to set as input to the network all the channels (otherwise use only green)
        :type rgb: bool
        :param visualiser:
        :type visualiser: Visualiser
        :param name: a name for the agent
        :type name: string
        """

        if fov is None:
            fov = Agent.FOV

        self.pos = init_pos
        self.rot = init_rot
        self.nest = np.zeros(2)
        self.feeder = np.zeros(2)
        self.live_sky = live_sky
        self.rgb = rgb
        self.visualiser = visualiser

        self.homing_routes = []
        self.world = None  # type: World
        self.__is_foraging = False
        self.__is_homing = False
        self.dx = 0.  # type: float
        self.condition = condition
        self.__per_ground = np.abs(fov[0]) / (np.pi / 2)  # type: float
        self.__per_sky = np.abs(fov[1]) / (np.pi / 2)  # type: float

        self.log = Logger()

        Agent.__latest_agent_id__ += 1
        self.id = Agent.__latest_agent_id__
        if name is None:
            self.name = "agent_%02d" % Agent.__latest_agent_id__
        else:
            self.name = name

    @property
    def d_feeder(self):
        # calculate the distance from the start position (feeder)
        return np.sqrt(np.square(self.pos[:2] - self.feeder[:2]).sum())

    @property
    def d_nest(self):
        return np.sqrt(np.square(self.pos[:2] - self.nest).sum())

    @property
    def yaw(self):
        return self.rot[0]

    @yaw.setter
    def yaw(self, value):
        self.rot[0] = value

    @property
    def pitch(self):
        return self.rot[1]

    @pitch.setter
    def pitch(self, value):
        self.rot[1] = value

    @property
    def roll(self):
        return self.rot[2]

    @roll.setter
    def roll(self, value):
        self.roll[2] = value

    def reset(self):
        """
        Resets the agent at the feeder
        :return: a boolean notifying whether the update of the position and orientation is done or not
        """
        self.__is_foraging = False
        self.__is_homing = True
        self.log.reset()

        if len(self.homing_routes) > 0:
            self.pos[:2] = self.feeder.copy()
            self.yaw = self.homing_routes[0].phi[0]
            return True
        else:
            # TODO: warn about the existence of the route
            return False

    def add_homing_route(self, rt):
        """
        Updates the homing route, home and nest points.
        :param rt: The route from the feeder to the nest
        :type rt: Route
        :return: a boolean notifying whether the update is done or not
        """
        if not isinstance(rt, Route):
            return False

        if rt not in self.homing_routes:
            rt.condition = self.condition
            self.homing_routes.append(rt)
            self.nest = np.array(rt.xy[-1])
            self.feeder = np.array(rt.xy[0])
            self.dx = rt.dx
            return True
        return False

    def set_world(self, w):
        """
        Update the world of the agent.
        :param w: the world to be placed in
        :return: a boolean notifying whether the update is done or not
        """
        if not isinstance(w, World):
            return False

        self.world = w
        for rt in self.world.routes:
            self.add_homing_route(rt)
        self.world.routes = self.homing_routes
        return True

    def start_learning_walk(self):
        raise NotImplementedError()

    def start_homing(self, reset=True):
        raise NotImplementedError()

    def world_snapshot(self, d_phi=0, width=None, height=None):
        x, y, z = self.pos
        phi = self.yaw + d_phi
        img = self.world.draw_panoramic_view(x, y, z, phi, update_sky=self.live_sky,
                                             include_ground=self.__per_ground, include_sky=self.__per_sky,
                                             width=width, length=width, height=height)
        return img

    def update_state(self, heading, rotation=0):
        phi, v = self.translate(heading, rotation, self.dx)

        # update the agent position
        self.pos[:] += np.array([v[0], -v[1], 0.])
        self.yaw = np.pi - phi
        self.log.add(self.pos[:3], self.yaw)

        return phi, v

    @staticmethod
    def translate(heading, rotation, acceleration):
        phi = Agent.rotate(heading, rotation)
        v = Agent.get_velocity(phi, acceleration)
        return phi, v

    @staticmethod
    def rotate(heading, rotation):
        return ((heading + rotation + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def get_velocity(phi, acceleration):
        return np.array([np.sin(phi), np.cos(phi)]) * acceleration


class Logger(object):

    def __init__(self):

        self.__x = [np.empty(0)] * 2
        self.__y = [np.empty(0)] * 2
        self.__z = [np.empty(0)] * 2
        self.__phi = [np.empty(0)] * 2

        self.hist = {}
        self.__stage = "training"

    @property
    def x(self):
        i = np.int(self.__stage == "training")
        return self.__x[i]

    @x.setter
    def x(self, value):
        i = np.int(self.__stage == "training")
        self.__x[i] = value

    @property
    def y(self):
        i = np.int(self.__stage == "training")
        return self.__y[i]

    @y.setter
    def y(self, value):
        i = np.int(self.__stage == "training")
        self.__y[i] = value

    @property
    def z(self):
        i = np.int(self.__stage == "training")
        return self.__z[i]

    @z.setter
    def z(self, value):
        i = np.int(self.__stage == "training")
        self.__z[i] = value

    @property
    def phi(self):
        i = np.int(self.__stage == "training")
        return self.__phi[i]

    @phi.setter
    def phi(self, value):
        i = np.int(self.__stage == "training")
        self.__phi[i] = value

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z]).T

    @xyz.setter
    def xyz(self, value):
        self.x, self.y, self.z = value.T

    @property
    def stage(self):
        return self.__stage

    @stage.setter
    def stage(self, mode):
        assert mode in ["training", "homing"]
        self.__stage = mode

    def reset(self):
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.z = np.empty(0)
        self.phi = np.empty(0)

        self.hist = {}

    def add(self, pos, rot):
        self.x = np.append(self.x, pos[0])
        self.y = np.append(self.y, pos[1])
        self.z = np.append(self.z, pos[2])
        self.phi = np.append(self.phi, rot)

    def update_hist(self, *args, **kwargs):
        for i, key in enumerate(self.hist.keys()[:len(args)]):
            self.hist[key].append(args[i])

        for key in kwargs.keys():
            if key in self.hist.keys():
                self.hist[key].append(kwargs[key])

    def distance(self, point):
        return np.sqrt(np.square(self.xyz - point).sum(axis=-1))
