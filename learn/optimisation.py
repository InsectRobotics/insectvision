import numpy as np
import pygmo as pg
import os
import yaml
from datetime import datetime


__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/opt/"

# Initialise the random seed
pg.set_global_rng_seed(2018)

with open(__dir__ + 'global-optimisation-algorithms.yaml', 'rb') as config:
    go_algorithms = yaml.safe_load(config)


def optimise(func, algo_name="sga", population=100, verbosity=100, plot=True, save=True, name=None):

    # Initialise the problem
    prob = pg.problem(func)

    # Initialise the algorithms
    algorithm = get_algorithm(algo_name)
    a = pg.algorithm(algorithm)
    a.set_verbosity(verbosity)

    # Initialise population
    if hasattr(func, "x_init"):
        pop = pg.population(prob, population-1)
        pop.push_back(func.x_init)
    else:
        pop = pg.population(prob, population)

    print a.get_name()

    pop = a.evolve(pop)
    log = np.array(a.extract(algorithm.__class__).get_log())

    f = pop.champion_f
    x = pop.champion_x

    if name is None:
        name = "%s-%s" % (datetime.now().strftime("%Y%m%d"), algo_name)

    if save:
        np.savez_compressed(__datadir__ + "%s.npz" % name, x=x, f=f, log=log)

    if plot:
        plot_log(log, title=name)

    return x, f, log


def plot_log(log, algo_name="sga", title="Log"):
    from matplotlib import pyplot as plt

    names = get_log(algo_name)
    x_label = names[names.index("gen")]
    x = log[:, names.index(x_label)]
    obj_label = "gbest" if algo_name == "pso" else "best"
    obj = log[:, names.index(obj_label)]

    plt.figure(title)
    plt.plot(x, obj, label="objective")
    # plt.plot(x, log[:, 3], label="convergence")
    # plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("value")
    plt.show()


def get_algorithm(name):
    algorithm = go_algorithms[name]
    algo_class = eval(algorithm["class"])
    if "params" in algorithm.keys():
        params = algorithm["params"]
    else:
        params = []
    if isinstance(params, list):
        args = params
    else:
        args = []
    if isinstance(params, dict):
        kwargs = params
    else:
        kwargs = {}

    return algo_class(*args, **kwargs)


def get_log(name):
    algorithm = go_algorithms[name]
    if "log" in algorithm.keys():
        return algorithm["log"]
    else:
        return list()


if __name__ == "__main__":
    name = "20180311-sea-060-060"

    log = np.load(__datadir__ + "%s.npz" % name)["log"]
    plot_log(log, algo_name="sea", title=name)
