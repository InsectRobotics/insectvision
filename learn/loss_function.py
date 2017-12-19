from code import decode_sph
from sphere import angle_between, sph2vec
import numpy as np


def angular_distance_rad(y_target, y_predict):
    return angle_between(decode_sph(y_target), decode_sph(y_predict))


def angular_distance_deg(y_target, y_predict):
    return 180 * angular_distance_rad(y_target, y_predict) / np.pi


def angular_distance_per(y_target, y_predict):
    return angle_between(decode_sph(y_target), decode_sph(y_predict)) / np.pi


def angular_distance_3d(y_predict, y_target, theta=True, phi=True):
    if theta:
        thy = y_predict[:, 1]
        tht = y_target[:, 1]
    else:
        thy = np.zeros_like(y_predict[:, 1])
        tht = np.zeros_like(y_target[:, 1])
    if phi:
        phy = y_predict[:, 0]
        pht = y_target[:, 0]
    else:
        phy = np.zeros_like(y_predict[:, 0])
        pht = np.zeros_like(y_target[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).mean())


def angular_deviation_3d(y_predict, y_target, theta=True, phi=True):
    if theta:
        thy = y_predict[:, 1]
        tht = y_target[:, 1]
    else:
        thy = np.zeros_like(y_predict[:, 1])
        tht = np.zeros_like(y_target[:, 1])
    if phi:
        phy = y_predict[:, 0]
        pht = y_target[:, 0]
    else:
        phy = np.zeros_like(y_predict[:, 0])
        pht = np.zeros_like(y_target[:, 0])
    v1 = sph2vec(thy, phy)
    v2 = sph2vec(tht, pht)
    return np.rad2deg(np.arccos((v1 * v2).sum(axis=0)).std())


losses = {
    "adr": angular_distance_rad,
    "angular distance rad": angular_distance_rad,
    "add": angular_distance_deg,
    "angular distance degrees": angular_distance_deg,
    "adp": angular_distance_per,
    "angular distance percentage": angular_distance_per,
    "ad3": angular_distance_3d,
    "angular distance 3D": angular_distance_3d,
    "astd3": angular_deviation_3d,
    "angular deviation 3D": angular_deviation_3d
}


def get_loss(name):
    assert name in losses.keys(), "Name of loss function does not exist."
    return losses[name]
