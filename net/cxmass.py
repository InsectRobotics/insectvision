import numpy as np


class CXMass(object):
    """
    Class to keep a set of parameters for a model together.
    No state is held in the class currently.
    """

    def __init__(self, **kwargs):
        super(CXMass, self).__init__(**kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    plt.ion()

    nb_steps = 300
    sleep = 0.
    w = 1.5
    feeder_novel = True
    unbounded = True

    pos_nest = 0 + 0j
    pos_feed = 0 + 3j
    mean_nest = 0 + 0j
    mean_feed = 1.125 + 1.125j
    steps_nest = 0
    steps_feed = nb_steps
    dist_t = np.abs((3+3j) - (0-3j)) + 6
    print "Distance:", dist_t

    fig = plt.figure("Path", figsize=(10, 10))
    plt.plot(0, -3, 'go')
    plt.plot(0, 0, 'go')
    line1, = plt.plot([0, 0], [-3, -3], 'k-')
    line2, = plt.plot([3, 3], [3, 3], 'k-')
    line3, = plt.plot([0, 0], [3, 3], 'k-')
    line4, = plt.plot([0, 0], [0, 0], 'g--', lw=3)
    line5, = plt.plot([0, 0], [3, 3], 'g--', lw=3)
    line6, = plt.plot([3, 3], [3, 3], 'g--', lw=3)
    mean_nest_mark, = plt.plot([mean_nest.real], [mean_nest.imag], 'r*')
    mean_feed_mark, = plt.plot([mean_feed.real], [mean_feed.imag], 'b^')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    # outbound
    for pos in np.linspace(0-3j, 3+3j, nb_steps / 2):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1 and not feeder_novel:
            mean_feed = ((mean_feed * (np.power(steps_feed) - 1, 1/w) + 1) - pos) / np.power(steps_feed - 1, 1/w)
        pos_nest = (0 - 3j) - pos
        pos_feed = (0 - 0j) - pos
        steps_nest += 1
        steps_feed -= 1
        line1.set_xdata([0, pos.real])
        line1.set_ydata([-3, pos.imag])
        mean_nest_mark.set_xdata([mean_nest.real])
        mean_nest_mark.set_ydata([mean_nest.imag])
        mean_feed_mark.set_xdata([mean_feed.real])
        mean_feed_mark.set_ydata([mean_feed.imag])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    for pos in np.linspace(3+3j, 0+3j, nb_steps / 4):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1 and not feeder_novel:
            mean_feed = ((mean_feed * (np.power(steps_feed) - 1, 1/w) + 1) - pos) / np.power(steps_feed - 1, 1/w)
        pos_nest = (0 - 3j) - pos
        pos_feed = (0 - 0j) - pos
        steps_nest += 1
        steps_feed -= 1
        line2.set_xdata([3, pos.real])
        line2.set_ydata([3, pos.imag])
        mean_nest_mark.set_xdata([mean_nest.real])
        mean_nest_mark.set_ydata([mean_nest.imag])
        mean_feed_mark.set_xdata([mean_feed.real])
        mean_feed_mark.set_ydata([mean_feed.imag])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    for pos in np.linspace(0+3j, 0+0j, nb_steps / 4):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1 and not feeder_novel:
            mean_feed = ((mean_feed * (np.power(steps_feed) - 1, 1/w) + 1) - pos) / np.power(steps_feed - 1, 1/w)
        pos_nest = (0 - 3j) - pos
        pos_feed = (0 - 0j) - pos
        steps_nest += 1
        steps_feed -= 1
        line3.set_xdata([0, pos.real])
        line3.set_ydata([3, pos.imag])
        mean_nest_mark.set_xdata([mean_nest.real])
        mean_nest_mark.set_ydata([mean_nest.imag])
        mean_feed_mark.set_xdata([mean_feed.real])
        mean_feed_mark.set_ydata([mean_feed.imag])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)

    print "Feeder -- Nest mean:", mean_nest
    print "Feeder -- Feed mean:", mean_feed

    if feeder_novel:
        steps_feed = 0
        mean_feed = 0 + 0j

    time.sleep(5)

    # inbound
    if unbounded:
        step = dist_t / float(nb_steps)
        print "Step size:", step
        pos_hist = []
        for i in xrange(2 * nb_steps):
            dist_mn = abs(mean_nest - pos)
            dist_mf = abs(mean_feed - pos)
            if steps_nest >= 0:
                mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
            if steps_feed > 1 and not feeder_novel:
                mean_feed = ((mean_feed * (np.power(steps_feed) - 1, 1/w) + 1) - pos) / np.power(steps_feed - 1, 1/w)
            steps_nest += 1
            steps_feed -= 1
            pos += step * (mean_feed - pos) / dist_mf
            pos_nest = (0 - 3j) - pos
            pos_feed = (0 - 0j) - pos
            pos_hist.append(pos)
            line4.set_xdata(np.array(pos_hist).real)
            line4.set_ydata(np.array(pos_hist).imag)
            mean_nest_mark.set_xdata([mean_nest.real])
            mean_nest_mark.set_ydata([mean_nest.imag])
            mean_feed_mark.set_xdata([mean_feed.real])
            mean_feed_mark.set_ydata([mean_feed.imag])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)
    else:
        for pos in np.linspace(0+0j, 0+3j, nb_steps / 4):
            if steps_nest > 1:
                mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
            if steps_feed >= 0:
                mean_feed = ((mean_feed * np.power(steps_feed, 1/w)) + pos) / (np.power(steps_feed, 1/w) + 1)
            pos_nest = (0 - 3j) - pos
            pos_feed = (0 - 0j) - pos
            steps_nest -= 1
            steps_feed += 1
            line4.set_xdata([0, pos.real])
            line4.set_ydata([0, pos.imag])
            mean_nest_mark.set_xdata([mean_nest.real])
            mean_nest_mark.set_ydata([mean_nest.imag])
            mean_feed_mark.set_xdata([mean_feed.real])
            mean_feed_mark.set_ydata([mean_feed.imag])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)
        for pos in np.linspace(0+3j, 3+3j, nb_steps / 4):
            if steps_nest > 1:
                mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
            if steps_feed >= 0:
                mean_feed = ((mean_feed * np.power(steps_feed, 1/w)) + pos) / (np.power(steps_feed, 1/w) + 1)
            pos_nest = (0 - 3j) - pos
            pos_feed = (0 - 0j) - pos
            steps_nest -= 1
            steps_feed += 1
            line5.set_xdata([0, pos.real])
            line5.set_ydata([3, pos.imag])
            mean_nest_mark.set_xdata([mean_nest.real])
            mean_nest_mark.set_ydata([mean_nest.imag])
            mean_feed_mark.set_xdata([mean_feed.real])
            mean_feed_mark.set_ydata([mean_feed.imag])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)
        for pos in np.linspace(3+3j, 0-3j, nb_steps / 2):
            if steps_nest > 1:
                mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
            if steps_feed >= 0:
                mean_feed = ((mean_feed * np.power(steps_feed, 1/w)) + pos) / (np.power(steps_feed, 1/w) + 1)
            pos_nest = (0 - 3j) - pos
            pos_feed = (0 - 0j) - pos
            steps_nest -= 1
            steps_feed += 1
            line6.set_xdata([3, pos.real])
            line6.set_ydata([3, pos.imag])
            mean_nest_mark.set_xdata([mean_nest.real])
            mean_nest_mark.set_ydata([mean_nest.imag])
            mean_feed_mark.set_xdata([mean_feed.real])
            mean_feed_mark.set_ydata([mean_feed.imag])
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(sleep)

    print "Nest -- Nest mean:", mean_nest
    print "Nest -- Feed mean:", mean_feed

    plt.ioff()
    plt.show()
