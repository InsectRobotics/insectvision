import numpy as np
from PIL import Image
from world import Route, route_like, Hybrid, save_route
from world.data_manager import __data__
from net import WillshawNet as MB
from .base import Agent, Logger
from .visualiser import Visualiser
from world.utils import shifted_datetime
from .utils import *


class MBAgent(Agent):
    FOV = (-np.pi/6, 4*np.pi/9)

    def __init__(self, *args, **kwargs):
        """

        :param init_pos: the initial position
        :type init_pos: np.ndarray
        :param init_rot: the initial orientation
        :type init_rot: np.ndarray
        :param condition:
        :type condition: Hybrid
        :param live_sky: flag to update the sky with respect to the time
        :type live_sky: bool
        :param rgb: flag to set as input to the network all the channels (otherwise use only green)
        :type rgb: bool
        :param fov: vertical field of view of the agent (the widest: -pi/2 to pi/2)
        :type fov: tuple, list, np.ndarray
        :param visualiser:
        :type visualiser: Visualiser
        :param name: a name for the agent
        :type name: string
        """
        if 'fov' in kwargs.keys() and kwargs['fov'] is None:
            kwargs['fov'] = MBAgent.FOV

        super(MBAgent, self).__init__(*args, **kwargs)

        self._net = MB(nb_channels=3 if self.rgb else 1)  # learning_rate=1)
        self.log = MBLogger()

        if 'name' in kwargs.keys() and kwargs['name'] is None:
            self.name = "mb_agent_%02d" % self.id

    def reset(self):
        """
        Resets the agent at the feeder

        :return: a boolean notifying whether the update of the position and orientation is done or not
        """
        if super(MBAgent, self).reset():
            self._net.update = False

            return True
        else:
            return False

    def start_learning_walk(self):
        if self.world is None:
            # TODO: warn about not setting the world
            yield None
            return
        elif len(self.homing_routes) == 0:
            # TODO: warn about not setting the homing route
            yield None
            return

        print("Resetting...")
        self.reset()

        # let the network update its parameters (learn)
        self._net.update = True

        # initialise visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # learn all the available homing routes
        for i, r in enumerate(self.homing_routes):
            self.log.reset()
            self.log.stage = "training"

            # add a copy of the current route to the world to visualise the path
            self.log.add(self.pos[:3], self.rot[1])
            self.world.routes.append(
                route_like(r, self.log.x, self.log.y, self.log.z, self.log.phi,
                           self.condition, agent_no=self.id + 1, route_no=i + 1)
            )
            counter = 0         # count the steps
            phi_ = (np.array([np.pi - phi for _, _, _, phi in r]) + np.pi) % (2 * np.pi) - np.pi
            # phi_ = np.roll(phi_, 1)  # type: np.ndarray

            for phi in phi_:
                if not self.step(phi, counter):
                    break
                counter += 1

            # remove the copy of the route from the world
            self.world.routes.remove(r)
            yield r     # return the learned route

        # freeze the parameters in the network
        self._net.update = False

    def start_homing(self, reset=True):
        if self.world is None:
            # TODO: warn about not setting the world
            return None

        if reset:
            print("Resetting...")
            self.reset()
            self.log.reset()
            self.log.stage = "homing"

        # initialise the visualisation
        if self.visualiser is not None:
            self.visualiser.reset()

        # add a copy of the current route to the world to visualise the path
        self.log.add(self.pos[:3], self.rot[1])
        self.world.routes.append(route_like(
            self.world.routes[0], self.log.x, self.log.y, self.log.z, self.log.phi,
            agent_no=self.id, route_no=1)
        )

        phi = np.pi - self.rot[1]
        counter = 0
        start_time = datetime.now()
        while self.d_nest > 0.1:
            if not self.step(phi, counter, start_time=start_time, compute_en=True):
                break
            counter += 1
        self.world.routes.remove(self.world.routes[-1])
        np.savez(__data__ + "EN/%s.npz" % self.name, en=np.array(self.log.hist["ens"]))
        return Route(self.log.x, self.log.y, self.log.z, self.log.phi,
                     condition=self.condition, agent_no=self.id, route_no=len(self.world.routes) + 1)

    def step(self, phi, counter=0., start_time=None, heading=None, compute_en=False):
        # stop the loop when we close the visualisation window
        if self.visualiser is not None and self.visualiser.is_quit():
            return False

        if heading is None:
            heading = np.pi - self.rot[1]
        # generate the visual input and code it to the projecting neurons
        pn = self.img2pn(self.world_snapshot())
        # make a forward pass from the network (updating the parameters)
        if compute_en:
            ens, snaps = [], []
            for d_phi in np.linspace(-np.pi / 3, np.pi / 3, 61):
                if self.visualiser is not None and self.visualiser.is_quit():
                    return False

                # generate the visual input and code to the PN values
                snap = self.world_snapshot(d_phi=d_phi)
                snaps.append(snap)
                pn = self.img2pn(snap)

                # make a forward pass from the network
                ens.append(self._net(pn))

                if self.visualiser is not None:
                    now = datetime.now()
                    if start_time is not None:
                        now = now - start_time
                    min = now.seconds // 60
                    sec = now.seconds % 60
                    self.visualiser.update_thumb(snap, pn=self._net.pn, pn_mode="L",
                                                 caption="Elapsed time: %02d:%02d" % (min, sec))
            ens = np.array(ens).flatten()
            # show preference to the least turning angle
            ens += np.append(np.linspace(.01, 0., 30, endpoint=False), np.linspace(0., .01, 31))
            en = ens.min()
            d_phi = np.deg2rad(2 * (ens.argmin() - 30))
        else:
            en = self._net(pn)
            d_phi = (phi - heading + np.pi) % (2 * np.pi) - np.pi
            # d_phi = 0
            ens = None
            snaps = None
        nphi, v = self.update_state(heading, rotation=d_phi)

        self.log.update_hist(pn=self._net.pn, kc=self._net.kc, en=en, ens=ens, turn=d_phi, phi=phi)

        self.world.routes[-1] = route_like(self.world.routes[-1], self.log.x, self.log.y, self.log.z, self.log.phi)

        counter += 1

        # update view
        img_func = None
        if self.visualiser.mode == "top":
            img_func = self.world.draw_top_view
        elif self.visualiser.mode == "panorama":
            img_func = self.world_snapshot
        if self.visualiser is not None:
            names = self.name.split('_')
            names[0] = self.world.date.strftime(datestr)
            names.append(counter)
            names.append(self.d_feeder)
            names.append(self.d_nest)
            names.append(np.rad2deg(d_phi))
            n = 4
            if start_time is not None:
                now = datetime.now()
                now = now - start_time
                names.append(now.seconds // 60)
                names.append(now.seconds % 60)
                n += 2

            capt_format = "%s " * (len(names) - n) + "| C: % 2d D_f: % 2.2f D_n: % 2.2f EN: % 2.1f"
            if start_time is not None:
                capt_format += " | Elapsed time: %02d:%02d"

            self.visualiser.update_main(img_func, en=ens, thumbs=snaps, caption=capt_format % tuple(names))

        d_max = 2 * np.sqrt(np.square(self.feeder - self.nest).sum())
        if self.d_feeder > d_max and self.d_nest > d_max or counter > 20. / self.dx:
            return False

        return True

    def img2pn(self, image):
        """

        :param image:
        :type image: Image.Image
        :return:
        """
        # TODO: make this parametriseable for different pre-processing of the input
        # print np.array(image).max()
        if self.rgb:
            return np.array(image).flatten()
        else:  # keep only green channel
            return np.array(image).reshape((-1, 3))[:, 0].flatten()


class MBLogger(Logger):

    def __init__(self):
        super(MBLogger, self).__init__()

    def reset(self):
        super(MBLogger, self).reset()

        self.hist["pn"] = []
        self.hist["kc"] = []
        self.hist["en"] = []
        self.hist["ens"] = []
        self.hist["turn"] = []
        self.hist["phi_z"] = []


if __name__ == "__main__":
    from world import load_world, load_routes
    from datetime import datetime

    exps = [
        # (True, False, True, False, None),     # live
        # (True, False, True, True, None),      # live-rgb
        # (True, False, False, False, None),    # live-no-pol
        # (True, False, False, True, None),     # live-no-pol-rgb
        # (False, True, True, False, np.random.RandomState(2018)),  # uniform
        # (False, True, True, True, np.random.RandomState(2018)),  # uniform-rgb
        # (False, False, True, False, None),    # fixed
        (False, False, True, True, None),     # fixed-rgb
        (False, False, False, False, None),    # fixed-no-pol
        (False, False, False, True, None),     # fixed-no-pol-rgb
    ]

    bin = True

    for update_sky, uniform_sky, enable_pol, rgb, rng in exps:
        date = shifted_datetime()
        if rng is None:
            rng = np.random.RandomState(2018)
        RND = rng
        fov = (-np.pi/2, np.pi/2)
        # fov = (-np.pi/6, np.pi/2)
        sky_type = "uniform" if uniform_sky else "live" if update_sky else "fixed"
        if not enable_pol and "uniform" not in sky_type:
            sky_type += "-no-pol"
        if rgb:
            sky_type += "-rgb"
        step = .1         # 10 cm
        tau_phi = np.pi    # 60 deg
        condition = Hybrid(tau_x=step, tau_phi=tau_phi)
        agent_name = create_agent_name(date, sky_type, step, fov[0], fov[1])
        print(agent_name)

        world = load_world()
        world.enable_pol_filters(enable_pol)
        world.uniform_sky = uniform_sky
        routes = load_routes()
        world.add_route(routes[0])

        agent = MBAgent(condition=condition, live_sky=update_sky, visualiser=Visualiser(), rgb=rgb,
                        fov=fov, name=agent_name)
        agent.set_world(world)
        print(agent.homing_routes[0])

        agent.visualiser.set_mode("panorama")
        for route in agent.start_learning_walk():
            print("Learned route:", route)

        agent.visualiser.set_mode("top")
        route = agent.start_homing()
        print(route)
        if route is not None:
            save_route(route, agent_name)

        if not update_tests(sky_type, date, step, gfov=fov[0], sfov=fov[1], bin=bin):
            break
        agent.world.routes.append(route)
        img, _ = agent.world.draw_top_view(1000, 1000)
        img.save(__data__ + "routes-img/%s.png" % agent_name, "PNG")
        # img.show(title="Testing route")
