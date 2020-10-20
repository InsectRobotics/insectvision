import numpy as np
from world.geometry import Route


def distance_from_route(route, point):
    """

    :param route: the route
    :type route: Route
    :param point:
    :type point: np.ndarray
    :return:
    """

    xyz = np.array([route.x, route.y, route.z]).T
    if point.ndim == 1:
        point = point.reshape((1, -1))
    if point.shape[1] < 2:
        raise AttributeError()
    if point.shape[1] < 3:
        point = np.append(point, route.z.mean(), axis=1)

    d = np.sqrt(np.square(xyz - point).sum(axis=1))

    return d.min()


if __name__ == "__main__":
    from world import load_route
    import matplotlib.pyplot as plt
    from agent.utils import *

    fov = True
    bin = True

    nb_columns = 5
    nb_rows = 2

    skies = ["uniform", "fixed", "fixed-no-pol", "live", "live-no-pol",
             "uniform-rgb", "fixed-rgb", "fixed-no-pol-rgb", "live-rgb", "live-no-pol-rgb"]

    tsts = bin_tests if bin else fov_tests if fov else tests

    plt.figure(figsize=(30, 20))
    for i, sky in enumerate(skies):
        if sky not in tsts.keys():
            continue

        nb_trials = len(tsts[sky])
        plt.subplot(nb_rows, nb_columns, i + 1)
        for id in range(nb_trials):
            try:
                name = get_agent_name(sky, id, fov=fov, bin=bin)
                print("")
            except AttributeError:
                print("aboard")
                continue

            r = load_route("%s" % name)
            h_xyz = np.array([r.x, r.y, r.z]).T

            r = load_route("learned")

            dist = []
            for p in h_xyz:
                dist.append(distance_from_route(r, p))
            dist = np.array(dist)
            plt.plot(dist, label="%s %s" % (tsts[sky][id]["date"], tsts[sky][id]["time"]))
        plt.title(sky)
        plt.xlim([0, 100])
        plt.ylim([0, 5])
        if i < 5:
            plt.xticks(np.linspace(0, 100, 5), [""] * 5)
        else:
            plt.xticks(np.linspace(0, 100, 5), np.linspace(0, 2, 5))
            plt.xlabel("Time (sec)")
        if i not in [0, 5]:
            plt.yticks(np.linspace(0, 5, 6), [""] * 6)
        else:
            plt.yticks(np.linspace(0, 5, 6))
            plt.ylabel("Distance (m)")
        plt.legend()
        plt.grid()
    plt.tight_layout(pad=5)
    plt.show()
