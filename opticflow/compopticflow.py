from compound import *


def lucas_kanade(n_val, o_val, rdir, rsensor, w, kernel):
    v = np.zeros((rsensor.shape[0], 2))
    for i in xrange(rsensor.shape[0]):
        a = build_a(n_val, rdir, rsensor[i], kernel)
        b = build_b(n_val, o_val, rdir, rsensor[i], kernel)

        # solve for v
        vpt = np.linalg.inv(a.T.dot(w**2).dot(a)).dot(a.T).dot(w**2).dot(b)
        v[i, :] = vpt
    return v

