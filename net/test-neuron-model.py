from net.mb import network_from_file
from routines import run_net

verbose = 2
show = True
timesteps = 4  # t: 0 = pre-odour, 1 = odour, 2 = shock, 3 = post-odour

if __name__ == '__main__':

    net = network_from_file('new-init.yaml', verbose=verbose)
    df_model = run_net(net, verbose=verbose, show=show, interactive=False)

    # if show:
    #     fig = plt.figure("single-neuron-test", figsize=(12 if 'nc' in hist.keys() else 6, 5))
    #     plt.clf()
    #     labels = ['Odour A response', 'Odour B response']
    #     labels_c = ['Odour A weight', 'Odour B weight']
    #     if 'nc' in hist.keys():
    #         plt.subplot(121)
    #         plt.title("Response")
    #     hist["n"][1::2, 0] = (hist["n"][0:-2:2, 0] + hist["n"][2::2, 0]) / 2.
    #     hist["n"][0, 1] = hist["n"][1, 1]
    #     hist["n"][-1, 1] = hist["n"][-2, 1]
    #     hist["n"][2:-2:2, 1] = (hist["n"][1:-3:2, 1] + hist["n"][3:-1:2, 1]) / 2.
    #     for i in [0, 1]:
    #         plt.plot(np.arange(17) - 1./4., hist["n"][:, i], "C%d-" % i)
    #         if 'ns' in hist.keys():
    #             plt.plot(np.arange(-3. / 4., 16.25, 1. / 4.), hist['ns'][:, i], "C%d:" % i, label=labels[i])
    #         plt.plot(np.arange(17)[i::2] - 1./4., hist["n"][i::2, i], "C%d." % i)
    #     plt.plot(np.array([3, 5, 7, 9, 11, 14, 16]) - .25, [-1] * 7, 'r*')
    #     plt.plot(np.array([[3, 5, 7, 9, 11, 14, 16]] * 2) - .25, [[-1] * 7, [15] * 7], 'r-')
    #     plt.ylim([-1, 15])
    #     plt.xticks(np.arange(17), ["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
    #     plt.xlim([-3. / 4., 16 + 1. / 4.])
    #     plt.xlabel('trial')
    #     plt.legend()
    #     plt.grid(axis='x')
    #
    #     if 'nc' in hist.keys():
    #         plt.subplot(122)
    #         plt.title("Synaptic weights")
    #         labels_c = [['odour (CS-)', 'shock (CS-)', 'fback (CS-)'],
    #                     ['odour (CS+)', 'shock (CS+)', 'fback (CS+)']]
    #         for i in xrange(6):
    #             plt.plot(np.arange(-3. / 4., 16.25, 1. / 4.), hist['nc'][:, i % 2, i // 2],
    #                      "C%d-" % i, label=labels_c[i % 2][i // 2], lw=3)
    #         plt.plot(np.array([3, 5, 7, 9, 11, 14, 16]) - .25, [-15] * 7, 'r*')
    #         plt.plot(np.array([[3, 5, 7, 9, 11, 14, 16]] * 2) - .25, [[-15] * 7, [15] * 7], 'r-')
    #         plt.ylim([-10, 10])
    #         plt.xticks(np.arange(17), ["%d%s" % (i // 2 + 1, ["-", "+"][i % 2]) for i in np.arange(17)])
    #         plt.xlim([-3. / 4., 16 + 1. / 4.])
    #         plt.xlabel('trial')
    #         plt.legend()
    #         plt.grid()
    #     plt.tight_layout()
    #     plt.show()
