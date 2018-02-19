import numpy as np
import logging.config
import yaml
import os
__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = __dir__ + '/../data/'
__params__ = __data__ + 'params/'

# setup logger
with open(__data__ + 'logging.yaml', 'rb') as config:
    logging.config.dictConfig(yaml.load(config))
    logger = logging.getLogger(__name__)


def svd2pca(U, S, V, epsilon=10e-5):
    """
    Creates the PCA transformation matrix using the SVD.

    :param U:
    :type U: np.ndarray
    :param S:
    :type S: np.ndarray
    :param V:
    :type V: np.ndarray
    :param epsilon: the smoothing parameter of the data
    :type epsilon: float
    """
    return (np.diag(1. / np.sqrt(S + epsilon))).dot(U.T)


def svd2zca(U, S, V, epsilon=10e-5):
    """
    Creates the ZCA transformation matrix using the SVD.

    :param epsilon: the smoothing parameter of the data
    """
    return U.dot(np.diag(1. / np.sqrt(S + epsilon))).dot(U.T)


def build_kernel(x, svd2ker, m=None, epsilon=10e-5):
    """
    Creates the transformation matrix of a dataset x using the given kernel
    function.

    :param epsilon: the smoothing parameter of the data
    """
    shape = np.shape(x)

    # reshape the matrix in n x d, where:
    # - n: number of instances
    # - d: number of features

    x_flat = np.reshape(x, (shape[0], -1))
    n, d = np.shape(x_flat)
    logger.debug('x.shape = %s, n = %d, d = %d' % (str(shape), n, d))

    # subtract the mean value from the data
    if m is None:
        m = np.mean(x_flat, axis=0)

    x_flat = x_flat - m
    logger.debug('x.min = %0.2f, x.max = %0.2f, x.mean = %0.2f' % (np.min(x_flat), np.max(x_flat), np.mean(m)))

    # compute the correlation matrix
    logger.debug('Creating the correlation matrix...')
    C = np.dot(np.transpose(x_flat), x_flat) / n
    logger.debug('C.shape = %s' % str(np.shape(C)))

    # compute the singular value decomposition
    logger.debug('Applying SVD...')
    U, S, V = np.linalg.svd(C)

    # compute kernel weights
    logger.debug('Creating the transformation matrix...')
    w = svd2ker(U, S, V, epsilon)
    logger.debug('w.shape = %s' % str(w.shape))

    return w


def zca(x, shape=None, m=None, epsilon=10e-5):
    """
    :param epsilon: whitening constant, it prevents division by zero
    """
    if shape is not None:
        x = x.reshape(shape)
    return build_kernel(x, svd2zca, m=m, epsilon=epsilon)


def pca(x, shape=None, m=None, epsilon=10e-5):
    """
    :param epsilon: whitening constant, it prevents division by zero
    """
    if shape is not None:
        x = x.reshape(shape)
    return build_kernel(x, svd2pca, m=m, epsilon=epsilon)


def transform(x, m=None, w=None, func=zca, epsilon=10e-5, reshape='first', load_filepath=None, save_filepath=None):
    """
    Whitens the given data using the given parameters.
    By default it applies ZCA whitening.

    :param x:               the input data
    :param w:               the transformation matrix
    :param func:            the transformation we want to apply
    :param epsilon:         whitening constant (10e-5 is typical for values around [-1, 1]
    :param reshape:         the reshape option of the data
    :param load_filepath:   the filepath to load the weights
    :param save_filepath:   the filepath to save the weights
    """
    if w is None:
        if load_filepath is None:
            if 'first' in reshape:
                shape = (x.shape[0], -1)
            elif 'last' in reshape:
                shape = (-1, x.shape[-1])
            else:
                shape = None
            w = func(x, shape, epsilon)
            if save_filepath is not None:
                logger.debug('Saving weights matrix...')
                np.savez_compressed(__params__ + save_filepath, w=w)
        else:
            logger.debug('Loading weights matrix...')
            w = np.load(__params__ + load_filepath)['w']

    logger.debug('Applying whitening to x...')

    # whiten the input data
    shape = np.shape(x)
    x = np.reshape(x, (-1, np.shape(w)[0]))

    if m is None:
        m = np.mean(x, axis=0) if np.shape(x)[0] > 1 else np.zeros((1, np.shape(w)[0]))

    x = x - m
    logger.debug('Before normalization: w.sum = %0.2f' % np.sum(w))
    logger.debug('w.min = %0.2f, w.max = %0.2f' % (np.min(w), np.max(w)))

    logger.debug('w.row.sum.mean = %0.2f' % np.mean(np.sum(w, axis=0)))
    logger.debug('w.col.sum.mean = %0.2f' % np.mean(np.sum(w, axis=1)))

    return np.reshape(np.dot(x, w), shape)
