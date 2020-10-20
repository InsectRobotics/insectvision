import numpy as np
from scipy.ndimage.filters import convolve


HS_KERN = np.array([[1, 2, 1],
                    [2, 0, 2],
                    [1, 2, 1]], dtype=float) / 12.

kernelX = np.array([[-1, 1],  # kernel for computing d/dx
                    [-1, 1]], dtype=float) * .25

kernelY = np.array([[-1, -1],  # kernel for computing d/dy
                    [1, 1]], dtype=float) * .25

kernelT = np.ones((2, 2), dtype=float) * .25


def part_derivatives(im1, im2):

    fx = convolve(im1, kernelX) + convolve(im2, kernelX)
    fy = convolve(im1, kernelY) + convolve(im2, kernelY)

    ft = convolve(im1, kernelT) + convolve(im2, -kernelT)

    return fx, fy, ft


def build_a(img, center_x, center_y, kernel_size):
    """
    Build a kernel containing pixel intensities
    """
    mean = kernel_size // 2
    count = 0
    home = img[center_x, center_y]  # storing the intensity of the center pixel
    a = np.zeros([kernel_size * kernel_size, 2])
    for j in range(-mean, mean+1):  # advance the y
        for i in range(-mean, mean+1):  # advance the x
            if i == 0:
                ax = 0
            else:
                ax = (home - img[center_y+j, center_x+i])/i
            if j == 0:
                ay = 0
            else:
                ay = (home - img[center_y+j, center_x+i])/j
            # write to A
            a[count] = np.array([ay, ax])
            count += 1

    return a


def build_b(n_img, o_img, center_x, center_y, kernel_size):
    mean = kernel_size // 2
    count = 0

    b = np.zeros(kernel_size * kernel_size)
    for j in range(-mean, mean+1):
        for i in range(-mean, mean+1):
            bt = n_img[center_y+j, center_x+i] - o_img[center_y+j, center_x+i]
            b[count] = bt
            count += 1

    return b


def gaussian_weight(size, even=False):
    """

    :param size: the kernel size
    :param even:
    :return:
    """
    if even:
        return np.diag(np.ones(size * size))

    sigma = 1  # the standard deviation of your normal curve
    correlation = 0  # see wiki for multivariate normal distributions
    weight = np.zeros([size, size])
    cpt = size % 2 + size // 2  # gets the center point
    for i in range(len(weight)):
        ptx = i + 1
        for j in range(len(weight)):
            pty = j + 1
            weight[i, j] = 1 / (2 * np.pi * sigma ** 2) / (1 - correlation ** 2) ** .5
            weight[i, j] *= np.exp(
                -1 / (2 * (1 - correlation ** 2)) * ((ptx - cpt) ** 2 + (pty - cpt) ** 2) / (sigma ** 2)
            )
    weight = weight.reshape((size * size,))
    weight = np.diag(weight)  # convert to n**2xn**2 diagonal matrix

    return weight


def get_poi(x_size, y_size, kernel_size):
    mean = kernel_size // 2
    x_pos = mean
    y_pos = mean
    x_step = (x_size - mean) // kernel_size
    y_step = (y_size - mean) // kernel_size
    length = x_step * y_step

    poi = np.zeros((length, 1, 2), dtype=int)
    count = 0
    for i in range(y_step):
        for j in range(x_step):
            poi[count, 0, 1] = x_pos
            poi[count, 0, 0] = y_pos
            x_pos += kernel_size
            count += 1
        x_pos = mean
        y_pos += kernel_size

    return poi


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img1 = np.arange(65 * 65, dtype=float).reshape((65, 65)) / (65 * 65)
    img2 = img1.T

    plt.subplot(221)
    plt.imshow(img1)

    plt.subplot(223)
    plt.imshow(img2, vmin=0, vmax=1)

    A = build_a(img2, 32, 32, 65)
    plt.subplot(243)
    plt.imshow(A[:, 1].reshape((65, 65)), vmin=-1, vmax=1)
    plt.title("Ax")
    plt.subplot(244)
    plt.imshow(A[:, 0].reshape((65, 65)), vmin=-1, vmax=1)
    plt.title("Ay")

    B = build_b(img2, img1, 32, 32, 65)
    # plt.subplot(247)
    # plt.imshow(B.reshape((65, 65)), vmin=-1, vmax=1)
    # plt.title("B")

    W = gaussian_weight(65, even=False)
    vpt = np.linalg.inv(A.T.dot(W ** 2).dot(A)).dot(A.T).dot(W ** 2)
    print(vpt.shape)
    plt.subplot(247)
    plt.imshow(vpt[1].reshape((65, 65)), vmin=-1, vmax=1)
    plt.title("Vptx")
    plt.subplot(248)
    plt.imshow(vpt[0].reshape((65, 65)), vmin=-1, vmax=1)
    plt.title("Vpty")
    vpt = np.linalg.inv(A.T.dot(W ** 2).dot(A)).dot(A.T).dot(W ** 2).dot(B)
    print(vpt)

    plt.show()
