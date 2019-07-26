import numpy as np
import matplotlib.pyplot as plt


taus, errs = [], []
modes = ["uniform", "canopy", "corridor"]
for mode in modes:
    try:
        data = np.load("../data/noise-%s.npz" % mode)
    except Exception:
        continue

    tau = data['x']
    err = data['y']

    try:
        tau = np.append(np.concatenate(tau[:2100]).flatten(), np.concatenate(tau[2100:]).flatten())
        err = np.append(np.concatenate(err[:2100]).flatten(), np.concatenate(err[2100:]).flatten())
    except ValueError:
        tau = np.concatenate(tau).flatten()
        err = np.concatenate(err).flatten()

    taus.append(tau)
    errs.append(err)

tau = np.concatenate(taus)
err = np.concatenate(errs)

# setup the dataset
x = np.array([tau, np.ones(tau.shape)]).T
y = err * tau

# compute parameters and set transformation function
w = np.linalg.pinv(x).dot(y[np.newaxis].T)
print w[0], w[1]
# sig = lambda xx: w[0] / xx + w[1]
sig = lambda xx: 72 - xx * 45

# print error
sigg = sig(tau)
print np.square(sigg[~np.isinf(sigg)] - err[~np.isinf(sigg)]).sum()

# # plot the data
# taus = np.linspace(tau.min(), tau.max(), 100)
# sigs = sig(taus)
#
# plt.figure("noise-function", figsize=(4, 3))
# i = np.random.permutation(np.arange(0, tau.size))
# plt.plot(tau[i][:1000], err[i][:1000], 'k.')
# plt.plot(tau[i][:1000], sigg[i][:1000], 'rx')
# plt.plot(taus, sigs, 'r-', linewidth=3)
# plt.ylim([-1, 90])
# plt.xlim([0, 5])
# plt.show()

etas = np.linspace(0, 1, 21)
for mode in modes:
    try:
        data = np.load("../data/noise-%s.npz" % mode)
    except Exception:
        continue

    tau = data['x']
    err = data['y']

    try:
        tau_t = np.concatenate([t[np.newaxis] for t in tau[:2100]], axis=0).mean(axis=(1, 2))
        tau_n = np.array([t.mean() for t in tau[2100:]])
        err_t = np.concatenate([e[np.newaxis] for e in err[:2100]], axis=0).mean(axis=(1, 2))
        err_n = np.array([e.mean() for e in err[2100:]])
    except ValueError:
        tau_t = np.concatenate([t[np.newaxis] for t in tau], axis=0).mean(axis=(1, 2))
        err_t = np.concatenate([e[np.newaxis] for e in err], axis=0).mean(axis=(1, 2))
        tau_n = None
        err_n = None

    tau_t_mean = tau_t.reshape((-1, 21)).mean(axis=0)
    err_t_mean = err_t.reshape((-1, 21)).mean(axis=0)
    tau_t_serr = tau_t.reshape((-1, 21)).std(axis=0) / np.sqrt(tau_t.size / 21)
    err_t_serr = err_t.reshape((-1, 21)).std(axis=0) / np.sqrt(err_t.size / 21)

    plt.figure("noise-%s" % mode, figsize=(4, 3))

    if mode == "uniform":
        dig = sig
        sig = lambda xx: 6. / xx + 6.
    plt.fill_between(etas * 100, err_t_mean-err_t_serr, err_t_mean+err_t_serr, facecolor="grey", alpha=.5)
    plt.plot(etas * 100, err_t_mean, color="red", linestyle="-", label="tilt")
    plt.plot(etas * 100, tau_t_mean * 45, color="red", linestyle="--", label="tau-tilt")
    plt.plot(etas * 100, sig(tau_t_mean), color="red", linestyle="--", label="sigma-tilt")
    plt.ylim([0, 90])
    plt.yticks([0, 30, 60, 90], [r'%d$^\circ$' % o for o in [0, 30, 60, 90]])
    plt.xlim([0, 100])
    plt.xlabel(r'noise ($\eta$)')
    plt.ylabel("MAE ($^\circ$)")

    if tau_n is not None and err_n is not None:
        if mode == "uniform":
            sig = lambda xx: 4. / xx - 2.

        tau_n_mean = tau_n.reshape((-1, 21)).mean(axis=0)
        err_n_mean = err_n.reshape((-1, 21)).mean(axis=0)
        tau_n_serr = tau_n.reshape((-1, 21)).std(axis=0) / np.sqrt(tau_n.size / 21)
        err_n_serr = err_n.reshape((-1, 21)).std(axis=0) / np.sqrt(err_n.size / 21)

        plt.fill_between(etas * 100, err_n_mean - err_n_serr, err_n_mean + err_n_serr, facecolor="grey", alpha=.5)
        plt.plot(etas * 100, err_n_mean, color="black", linestyle="-", label="plane")
        plt.plot(etas * 100, tau_n_mean * 45, color="black", linestyle="--", label="tau-plane")
        plt.plot(etas * 100, sig(tau_n_mean), color="black", linestyle="--", label="sigma-plane")
        plt.ylim([0, 90])
        plt.yticks([0, 30, 60, 90], [r'%d$^\circ$' % o for o in [0, 30, 60, 90]])
        plt.xlim([0, 100])
        plt.xlabel(r'noise ($\eta$)')
        plt.ylabel("MAE ($^\circ$)")

    if mode == "uniform":
        sig = dig

plt.show()

