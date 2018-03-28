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


def optimise(func, algo_name="sga", population=100, verbosity=100,
             plot=True, save=True, save_log_x=True, name=None,
             **kwargs):

    # Initialise the problem
    prob = pg.problem(func)

    # Initialise the algorithms
    gen = kwargs.get("gen", 10000)
    if save_log_x:
        kwargs["gen"] = verbosity
        iterations = gen / verbosity
    else:
        iterations = 1
    algorithm = get_algorithm(algo_name, **kwargs)
    a = pg.algorithm(algorithm)
    a.set_verbosity(verbosity)

    # Initialise population
    if hasattr(func, "x_init"):
        pop = pg.population(prob, population-1)
        pop.push_back(func.x_init)
    else:
        pop = pg.population(prob, population)

    print a.get_name()

    log = np.empty((0, len(get_log(algo_name))))
    log_x = np.empty((0, func.ndim))
    labels = get_log(algo_name)
    row_format = "{:>7}" + "{:>15}" * (len(labels)-1)
    # if save_log_x and verbosity is not None and verbosity > 0:
    #     print row_format.format(*[label.capitalize() + ":" for label in labels])
    for it in xrange(iterations):
        pop = a.evolve(pop)
        new_log = np.array(a.extract(algorithm.__class__).get_log())
        new_log[:, labels.index("gen")] += it * verbosity
        if save_log_x and verbosity is not None and verbosity > 0:
            print row_format.format(*["%.4f" % e for e in new_log[0]])
        log = np.vstack([log, new_log])
        log_x = np.vstack([log_x, pop.champion_x])

    f = pop.champion_f
    x = func.correct_vector(pop.champion_x)

    if name is None:
        name = "%s-%s" % (datetime.now().strftime("%Y%m%d"), algo_name)

    if save:
        np.savez_compressed(__datadir__ + "%s.npz" % name, x=x, f=f, log=log, log_x=log_x)

    if plot:
        plot_log(log, title=name)

    return x, f, log


def plot_log(log, algo_name="sga", label=None, title="Log", show=True):
    from matplotlib import pyplot as plt

    names = get_log(algo_name)
    x_label = names[names.index("gen")]
    x = log[:, names.index(x_label)]
    obj_label = "gbest" if algo_name == "pso" else "best"
    obj = log[:, names.index(obj_label)]
    if label is None:
        label = "objective"

    ax = plt.figure(title)
    plt.plot(x, obj, label=label)
    # plt.plot(x, log[:, 3], label="convergence")
    # plt.legend()
    plt.xlabel(x_label)
    plt.ylabel("value")
    plt.ylim([0, 90])
    if show:
        plt.show()

    return ax


def get_algorithm(name, **kwargs):
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
        for key, value in params.items():
            kwargs[key] = kwargs.get(key, value)

    return algo_class(*args, **kwargs)


def get_log(name):
    algorithm = go_algorithms[name]
    if "log" in algorithm.keys():
        return algorithm["log"]
    else:
        return list()
