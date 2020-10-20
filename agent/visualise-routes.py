import numpy as np
import matplotlib.pyplot as plt

from world import load_world, load_route
from world.model import cmap
from utils import *


# sky_type = "uniform"
fov = True
bin = True

if 'sky_type' in locals():
    print sky_type
    w = load_world()
    r = load_route("learned")
    r.agent_no = 1
    r.route_no = 2
    w.add_route(r)
    labels = []
    test_ = bin_tests if bin else fov_tests if fov else tests

    for j in xrange(len(test_[sky_type])):
        name = get_agent_name(sky_type, j, fov=fov, bin=bin)
        labels.append(name.split("_")[0] + " " + name.split("_")[1])
        r = load_route(name)
        r.agent_no = j + 2
        r.route_no = 2
        w.add_route(r)

    plt.figure("%s-%sfov-%sbin" % (sky_type, "" if fov else "no", "" if bin else "no"), figsize=(15, 10))
    img, _ = w.draw_top_view(width=800, length=800)
    plt.imshow(img)
    plt.title(sky_type)
    plt.xticks([])
    plt.yticks([])

    for i, label in enumerate(labels):
        plt.text(810, 15 * (i + 1), label, color=cmap(float(i+2) / float(len(labels)+1)))
    plt.show()
else:
    nb_columns = 5
    nb_rows = 2

    plt.figure("%sfov-%sbin" % ("" if fov else "no", "" if bin else "no"), figsize=(30, 20))
    sky_types = ["uniform", "fixed", "fixed-no-pol", "live", "live-no-pol",
                 "uniform-rgb", "fixed-rgb", "fixed-no-pol-rgb", "live-rgb", "live-no-pol-rgb"]
    for i, sky_type in enumerate(sky_types):
        print i, sky_type,
        w = load_world()
        try:
            name = get_agent_name(sky_type, 0, fov=fov, bin=bin)
            print ""
        except AttributeError, e:
            print "aboard"
            continue
        r = load_route("learned")
        r.agent_no = 1
        r.route_no = 2
        w.add_route(r)
        labels = []
        test_ = bin_tests if bin else fov_tests if fov else tests
        for j in xrange(len(test_[sky_type])):
            name = get_agent_name(sky_type, j, fov=fov, bin=bin)
            labels.append(name.split("_")[0] + " " + name.split("_")[1])
            r = load_route(name)
            r.agent_no = j+2
            r.route_no = 2
            w.add_route(r)
        img, _ = w.draw_top_view(width=300, length=300)
        plt.subplot(nb_rows, nb_columns, i + 1)
        plt.imshow(img)
        plt.title(sky_type)
        plt.xticks([])
        plt.yticks([])

        for j, label in enumerate(labels):
            x = 300 * (j // 5) / 2 + 15
            y = 300 + 15 * (j - (5 * (j // 5)) + 1)
            plt.text(x, y, label, color=cmap(float(j+2) / float(len(labels)+1)))
    plt.tight_layout(pad=5)
    plt.show()
