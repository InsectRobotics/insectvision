import numpy as np
import pygmo as pg
import os
from datetime import datetime


__dir__ = os.path.dirname(os.path.realpath(__file__)) + "/"
__datadir__ = __dir__ + "../data/opt/"

# Initialise the random seed
pg.set_global_rng_seed(2018)


algorithms = {

    # HEURISTIC GLOBAL OPTIMISATION

    # Differential Evolution (DE)
    "de": pg.de(
        gen=10000,  # number of generations
        F=.8,       # weight coefficient (default value is 0.8)
        CR=.9,      # crossover probability (default value is 0.9)
        variant=2,  # mutation variant (default variant is 2: /rand/1/exp)
        ftol=1e-6,  # stopping criteria on the f tolerance (default is 1e-6)
        tol=1e-6    # stopping criteria on the x tolerance (default is 1e-6)
    ),
    # Improved Harmony Search
    # "ihs": pg.ihs(),
    # Self-adaptive DE (jDE amd iDE)
    "sade": pg.sade(
        gen=10000,        # number of generations
        variant=2,        # mutation variant (dafault variant is 2: /rand/1/exp)
        variant_adptv=1,  # F and CR parameter adaptation scheme to be used (one of 1..2)
        ftol=1e-6,        # stopping criteria on the f tolerance (default is 1e-6)
        xtol=1e-6,        # stopping criteria on the x tolerance (default is 1e-6)
        memory=False      # when true the adapted parameters CR anf F are not reset between successive calls
                          #   to the evolve method
    ),
    # Self-adaptive DE (de_1220 aka pDE)
    "de1220": pg.de1220(
        gen=10000,        # number of generations
        allowed_variants=[2, 3, 7, 10, 13, 14, 15, 16],  # allowed mutation variants, each one being a number in [1, 18]
        variant_adptv=1,  # F and CR parameter adaptation scheme to be used (one of 1..2)
        ftol=1e-6,        # stopping criteria on the f tolerance (default is 1e-6)
        xtol=1e-6,        # stopping criteria on the x tolerance (default is 1e-6)
        memory=False      # when true the adapted parameters CR anf F are not reset between successive calls
                          #   to the evolve method
    ),
    # Particle Swarm Optimisation
    "pso": pg.pso(
        gen=10000,       # number of generations
        omega=0.7298,    # inertia weight (or constriction factor)
        eta1=2.05,       # social component
        eta2=2.05,       # cognitive component
        max_vel=0.5,     # maximum allowed particle velocities (normalized with respect to the bounds width)
        variant=5,       # algoritmic variant
        neighb_type=2,   # swarm topology (defining each particle's neighbours)
        neighb_param=4,  # topology parameter (defines how many neighbours to consider)
        memory=False     # when true the velocities are not reset between successive calls to the evolve method
    ),
    # (N+1)-ES Simple Evolutionary Algorithm
    "sea": pg.sea(
        gen=10000  # number of generations to consider (each generation will compute the objective function once)
    ),
    # Simple Genetic Algorith
    "sga": pg.sga(
        gen=10000,                # number of generations
        cr=.90,                   # crossover probability
        eta_c=1.,                 # distribution index for sbx crossover.
                                  #   This parameter is inactive if other types of crossover are selected
        m=0.02,                   # mutation probability
        param_m=1.,               # distribution index (polynomial mutation), gaussian width (gaussian mutation) or
                                  #   inactive (uniform mutation)
        param_s=2,                # the number of best individuals to use in "truncated" selection or
                                  #   the size of the tournament in tournament selection.
        crossover="exponential",  # the crossover strategy. One of exponential, binomial, single or sbx
        mutation="polynomial",    # the mutation strategy. One of gaussian, polynomial or uniform
        selection="tournament"    # the selection strategy. One of tournament, truncated
    ),
    # Coranas's Simulated Annealing (SA)
    "sa": pg.simulated_annealing(
        Ts=.01,          # starting temperature
        Tf=1e-05,        # final temperature
        n_T_adj=10,      # number of temperature adjustments in the annealing schedule
        n_range_adj=10,  # number of adjustments of the search range performed at a constant temperature
        bin_size=10,     # number of mutations that are used to compute the acceptance rate
        start_range=.5   # starting range for mutating the decision vector
    ),
    # Artificial Bee Colony (ABC)
    "abc": pg.bee_colony(
        gen=10000,   # number of generations
        limit=10     # maximum number of trials for abandoning a source
    ),
    # Covariance Matrix Adaptation Evolutionary Strategy (CMA_ES)
    "cmaes": pg.cmaes(
        gen=10000,          # number of generations
        cc=-1,              # backward time horizon for the evolution path (by default is automatically assigned)
        cs=-1,              # makes partly up for the small variance loss in case the indicator is zero
                            #   (by default is automatically assigned)
        c1=-1,              # learning rate for the rank-one update of the covariance matrix
                            #   (by default is automatically assigned)
        cmu=-1,             # learning rate for the rank-mu update of the covariance matrix
                            #   (by default is automatically assigned)
        sigma0=0.5,         # initial step size
        ftol=1e-6,          # stopping criteria on the f tolerance
        xtol=1e-6,          # stopping criteria on the x tolerance
        memory=False,       # when true the adapted parameters are not reset between successive calls
                            #   to the evolve method
    # force_bounds=False  # when true the box bounds are enforced. The fitness will never be called outside the bounds
                            #   but the covariance matrix adaptation mechanism will worsen
    ),
    # Exponential Evolution Strategies (xNES)
    # "xnes": pg.xnes(),
    # Non-dominated Sorting GA (NSGA2)
    "nsga2": pg.nsga2(
        gen=10000,  # number of generations
        cr=0.95,    # crossover probability
        eta_c=10,   # distribution index for crossover
        m=0.01,     # mutation probability
        eta_m=10    # distribution index for mutation
    ),
    # Multi-objective EA with Decomposition (MOEA/D)
    "moead": pg.moead(
        gen=10000,                    # number of generations
        weight_generation="grid",     # method used to generate the weights, one of grid, low discrepancy or random
        decomposition="tchebycheff",  # method used to decompose the objectives, one of tchebycheff, weighted or bi
        neighbours=20,                # size of the weight's neighborhood
        CR=1,                         # crossover parameter in the Differential Evolution operator
        F=0.5,                        # parameter for the Differential Evolution operator
        eta_m=20,                     # distribution index used by the polynomial mutation
        realb=0.9,                    # chance that the neighbourhood is considered at each generation,
                                      #   rather than the whole population (only if preserve_diversity is True)
        limit=2,                      # maximum number of copies reinserted in the population
                                      #   (only if m_preserve_diversity is true)
        preserve_diversity=True       # when true activates diversity preservation mechanisms
    ),

    # META_ALGORITHMS

    # Monotonic Basin Hopping (MBH)
    "mbh": pg.mbh(
        algo=pg.compass_search(),  # an algorithm or a user-defined algorithm
        stop=5,                    # consecutive runs of the inner algorithm that need to result in no improvement
                                   #   for mbh to stop
        perturb=1e-2               # perturb the perturbation to be applied to each component
    ),
    # Cstrs Self-Adaptive
    "cstrs": pg.cstrs_self_adaptive(),
    # Augmented Lagrangian Algorithm
    "auglag": pg.nlopt("auglag"),
    "auglag_eq": pg.nlopt("auglag_eq"),

    # LOCAL OPTIMISATION

    # Compass Search
    "compass": pg.compass_search(
        max_fevals = 1,       # maximum number of function evaluation
        start_range = .1,     # start range (dafault value is .1)
        stop_range = .01,     # stop range (dafault value is .01)
        reduction_coeff = .5  # range reduction coefficient (dafault value is .5)
    ),
    # COBYLA
    "cobyla": pg.nlopt("cobyla"),
    # BOBYQA
    "bobyqa": pg.nlopt("bobyqa"),
    # NEWUOA
    "newuoa": pg.nlopt("newuoa"),
    # NEWUOA + bound
    "newuoa_bound": pg.nlopt("newuoa_bound"),
    # PRAXIS
    "praxis": pg.nlopt("praxis"),
    # Nelder-Mead simplex
    "neldermead": pg.nlopt("neldermead"),
    # sbplx
    "sbplx": pg.nlopt("sbplx"),
    # MMA (Method of Moving Asymptotes)
    "mma": pg.nlopt("mma"),
    # CCSA
    "ccsaq": pg.nlopt("ccsaq"),
    # SLSQP
    "slsqp": pg.nlopt("slsqp"),
    # Low-storage BFGS
    "lbfgs": pg.nlopt("lbfgs"),
    # Preconditioned Turncated Newton
    "tnewton_precond_restart": pg.nlopt("tnewton_precond_restart"),
    "tnewton_precond": pg.nlopt("tnewton_precond"),
    "tnewton_restart": pg.nlopt("tnewton_restart"),
    "tnewton": pg.nlopt("tnewton"),
    # Shifted limited-memory variable-metric
    "var2": pg.nlopt("var2"),
    "var1": pg.nlopt("var1"),
    # "ipopt": pg.ipopt()
}


def optimise(func, algo_name="sga", population=100, verbosity=100, plot=True, save=True):

    # Initialise the problem
    prob = pg.problem(func)

    # Initialise the algorithms
    a = pg.algorithm(algorithms[algo_name])
    a.set_verbosity(verbosity)

    # Initialise population
    if hasattr(func, "x_init"):
        pop = pg.population(prob, population-1)
        pop.push_back(func.x_init)
    else:
        pop = pg.population(prob, population)

    print a.get_name()

    pop = a.evolve(pop)
    log = a.extract(algorithms[algo_name].__class__).get_log()

    f = pop.champion_f

    name = "%s-%s-%.2f" % (datetime.now().strftime("%Y%m%d"), algo_name, f)

    print log
    log = np.array(log)
    print log
    if plot:
        plot_log(log, title=name)

    x = pop.champion_x

    if save:
        np.savez_compressed(__datadir__ + "%s.npz" % name, x=x, f=f, log=log)

    return x, f, log


def plot_log(log, title="Log"):
    from matplotlib import pyplot as plt

    plt.figure(title)
    plt.plot(log[:, 0], log[:, 2], label="objective")
    plt.plot(log[:, 0], log[:, 3], label="convergence")
    plt.legend()
    plt.xlabel("objevals")
    plt.ylabel("value")
    plt.show()


algo_logs = {
    "de": ["gen", "feval", "best", "dx", "df"],
    "sade": ["gen", "feval", "best", "F", "CR", "dx", "df"],
    "de1220": ["gen", "feval", "best", "F", "CR", "variant", "dx", "df"],
    "sa": ["fevals", "best", "current", "mean range", "temperature"],
    "sea": ["gen", "fevals", "best", "improvement", "mutations"],
    "sga": ["gen", "fevals", "best", "improvement"],
    "cmaes": ["gen", "fevals", "best", "dx", "df", "sigma"],
    "pso": ["gen", "feval", "gbest", "mean vel.", "mean lbest", "avg. dist."],
    "abc": ["gen", "feval", "best", "current best"]
}


if __name__ == "__main__":
    algorithm = "sea"
    date = "20180308"
    f = 0.55

    name = "%s-%s-%.2f" % (date, algorithm, f)
    data = np.load(__datadir__ + "%s.npz" % name)

    log = data["log"]
    print log

    plot_log(log, title=name)
