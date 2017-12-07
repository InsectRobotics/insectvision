import numpy as np


def angle_between(ang1, ang2, sign=True):
    d = (ang1 - ang2 + np.pi) % (2 * np.pi) - np.pi
    if not sign:
        d = np.abs(d)
    return d
