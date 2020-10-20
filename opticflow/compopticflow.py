from .compound import *
from numpy.linalg import LinAlgError


def lucas_kanade(n_val, o_val, rdir, rsensor, w=None, kernel=np.pi):
    if w is None:
        w = gaussian_weight(kernel, rdir, rsensor[0], even=False)
    w2 = np.square(w)

    v = np.zeros((rsensor.shape[0], 2))
    for i in range(rsensor.shape[0]):
        a = build_a(n_val, rdir, rsensor[i], kernel)
        b = build_b(n_val, o_val, rdir, rsensor[i], kernel)

        try:
            # solve for v
            vpt = np.linalg.inv(a.T.dot(w2).dot(a)).dot(a.T).dot(w2).dot(b)
            v[i, :] = vpt
        except LinAlgError:
            v[i, :] = np.zeros(2)
    return v

