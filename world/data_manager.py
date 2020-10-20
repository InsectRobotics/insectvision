import os
from scipy.io import loadmat
from .model import *
from .geometry import PolygonList, Polygon, Route

__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + "/../data/"
__seville_2009__ = __data__ + "Seville2009_world/"
WORLD_FILENAME = "world5000_gray.mat"
ROUTES_FILENAME = "AntRoutes.mat"


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
    from .conditions import Stepper

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
