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

    sleep = 0.1
    mean_nest = 0 + 0j
    mean_feed = 1.125 + 1.125j
    steps_nest = 0
    steps_feed = 200
    w = 3.

    fig = plt.figure("Path", figsize=(10, 10))
    plt.plot(0, -3, 'go')
    plt.plot(0, 0, 'go')
    line1, = plt.plot([0, 0], [-3, -3], 'k-')
    line2, = plt.plot([3, 3], [3, 3], 'k-')
    line3, = plt.plot([0, 0], [3, 3], 'k-')
    mean_nest_mark, = plt.plot([mean_nest.real], [mean_nest.imag], 'r*')
    mean_feed_mark, = plt.plot([mean_feed.real], [mean_feed.imag], 'b^')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    # outbound
    for pos in np.linspace(0-3j, 3+3j, 100):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1:
            mean_feed = ((mean_feed * steps_feed) - pos) / (steps_feed - 1)
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
    for pos in np.linspace(3+3j, 0+3j, 50):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1:
            mean_feed = ((mean_feed * steps_feed) - pos) / (steps_feed - 1)
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
    for pos in np.linspace(0+3j, 0+0j, 50):
        if steps_nest >= 0:
            mean_nest = ((mean_nest * np.power(steps_nest, 1/w)) + pos) / (np.power(steps_nest, 1/w) + 1)
        if steps_feed > 1:
            mean_feed = ((mean_feed * steps_feed) - pos) / (steps_feed - 1)
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
    time.sleep(5)
    # inbound
    for pos in np.linspace(0+0j, 0+3j, 50):
        if steps_nest > 1:
            mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
        if steps_feed >= 0:
            mean_feed = ((mean_feed * steps_feed) + pos) / (steps_feed + 1)
        steps_nest -= 1
        steps_feed += 1
        line3.set_xdata([0, pos.real])
        line3.set_ydata([3, pos.imag])
        mean_nest_mark.set_xdata([mean_nest.real])
        mean_nest_mark.set_ydata([mean_nest.imag])
        mean_feed_mark.set_xdata([mean_feed.real])
        mean_feed_mark.set_ydata([mean_feed.imag])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    for pos in np.linspace(0+3j, 3+3j, 50):
        if steps_nest > 1:
            mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
        if steps_feed >= 0:
            mean_feed = ((mean_feed * steps_feed) + pos) / (steps_feed + 1)
        steps_nest -= 1
        steps_feed += 1
        line2.set_xdata([3, pos.real])
        line2.set_ydata([3, pos.imag])
        mean_nest_mark.set_xdata([mean_nest.real])
        mean_nest_mark.set_ydata([mean_nest.imag])
        mean_feed_mark.set_xdata([mean_feed.real])
        mean_feed_mark.set_ydata([mean_feed.imag])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(sleep)
    for pos in np.linspace(3+3j, 0-3j, 100):
        if steps_nest > 1:
            mean_nest = ((mean_nest * (np.power(steps_nest - 1, 1/w) + 1)) - pos) / np.power(steps_nest - 1, 1/w)
        if steps_feed >= 0:
            mean_feed = ((mean_feed * steps_feed) + pos) / (steps_feed + 1)
        steps_nest -= 1
        steps_feed += 1
        line1.set_xdata([0, pos.real])
        line1.set_ydata([-3, pos.imag])
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
