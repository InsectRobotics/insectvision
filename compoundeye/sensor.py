from model import DRA, spectrum

import numpy as np


class POLCompassDRA(DRA):

    def __init__(self, n=60, omega=56, rho=5.4, nb_pr=2):
        super(POLCompassDRA, self).__init__(n=n, omega=omega, rho=rho, nb_pr=nb_pr, name="pol")
        self.rhabdom = np.array([[spectrum["uv"], spectrum["uv"]]] * n).T
        self.mic_l = np.array([[0., np.pi / 2]] * n).T


if __name__ == "__main__":
    from environment import Sky
    from model import visualise

    sky = Sky(theta_s=np.pi/3)
    dra = POLCompassDRA()
    dra.theta_t = np.pi/6
    dra.phi_t = np.pi/3
    # s = dra(sky)
    r_pol = dra(sky)
    r_po = dra.r_po
    # print s.shape

    visualise(sky, r_po)