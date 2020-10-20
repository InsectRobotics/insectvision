import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

from world import load_world, load_route

# get path of the script
cpath = os.path.dirname(os.path.abspath(__file__)) + '/'
logpath = cpath + "../data/tests.yaml"

# load tests
with open(logpath, 'rb') as f:
    tests = yaml.safe_load(f)

sky_type = "live-no-pol"
test_no = 2

nb_tests = len(tests[sky_type])
print "Number of tests:", nb_tests

date = tests[sky_type][test_no-1]["date"]
time = tests[sky_type][test_no-1]["time"]
step = tests[sky_type][test_no-1]["step"]  # cm

name = "%s_%s_s%02d-%s-sky" % (date, time, step, sky_type)
en = np.load("../data/EN/%s.npz" % name)["en"].T
min_en = np.argmin(en, axis=0)

w = load_world()
r = load_route("learned")
w.add_route(r)
r = load_route(name)
w.add_route(r)
img, _ = w.draw_top_view(width=500, length=500)

plt.figure("ENs activation - Step: %d cm - %s sky - %s : %s" % (step, sky_type, date, time), figsize=(15, 7))
plt.subplot(121)
plt.imshow(np.log(en), cmap="Greys", vmin=0, vmax=2)
plt.plot(min_en, 'r.-')
plt.yticks(np.linspace(0, 60, 7), np.linspace(-60, 60, 7))
plt.xticks(np.linspace(0, 150, 10), (np.linspace(0, 150 * w.routes[0].dt, 10) // .01) * .1)
plt.colorbar(orientation="horizontal", pad=.2)
plt.xlabel("time (sec)")
plt.ylabel("Turning (degrees)")
plt.subplot(122)
plt.imshow(np.array(img))
plt.show()
