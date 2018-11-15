from model import DRA

import numpy as np


class POLCompassDRA(DRA):

    def __init__(self, n=60, omega=56, rho=5.4, nb_pr=2):
        super(POLCompassDRA, self).__init__(n=n, omega=omega, rho=rho, nb_pr=nb_pr, name="pol")
        self.rhabdom = np.array([[spectrum["uv"], spectrum["uv"]]] * n).T
        self.mic_l = np.array([[0., np.pi / 2]] * n).T
