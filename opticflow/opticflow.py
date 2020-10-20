from .simple import *


def horn_schunck(im1, im2, alpha=0.001, n_iter=8):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # Set initial value for the flow vectors
    u = np.zeros([im1.shape[0], im1.shape[1]])
    v = np.zeros([im1.shape[0], im1.shape[1]])

    # Estimate derivatives
    fx, fy, ft = part_derivatives(im1, im2)

    # Iteration to reduce error
    for _ in range(n_iter):
        # Compute local averages of the flow vectors
        u_avg = convolve(u, HS_KERN)
        v_avg = convolve(v, HS_KERN)
        # common part of update step
        der = (fx * u_avg + fy * v_avg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        # iterative step
        u = u_avg - fx * der
        v = v_avg - fy * der

    return u, v


def lucas_kanade(im1, im2, poi, w, kernel):
    # evaluate every POI
    v = np.zeros((poi.shape[0], 2))
    for i in range(poi.shape[0]):
        a = build_a(im2,      poi[i][0][1], poi[i][0][0], kernel)
        b = build_b(im2, im1, poi[i][0][1], poi[i][0][0], kernel)

        # solve for v
        vpt = np.linalg.inv(a.T.dot(w**2).dot(a)).dot(a.T).dot(w**2).dot(b)
        v[i, 0] = vpt[0]
        v[i, 1] = vpt[1]

    return v
