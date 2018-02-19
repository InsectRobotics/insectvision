import numpy as np
from compoundeye.sensor import decode_sun, encode_sun


data = np.load("cross-seville-F060-I060-O008-M07-D3600-tilt-2.npz")
t = data["t"]
t = np.array([decode_sun(t0) for t0 in t])
t[:, 0] -= np.pi
t[:, 1] = np.pi / 2 - t[:, 1]
t = np.array([encode_sun(t0[0], t0[1]) for t0 in t])
np.savez_compressed("cross-seville-F060-I060-O008-M07-D3600-tilt.npz", x=data["x"], t=t)
