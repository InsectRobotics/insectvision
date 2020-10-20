import numpy as np
from .utils import pix2sph, Width as W, Height as H
from sphere import vec2sph


def cubebox_angles(side):
    if side == "left":
        y = np.linspace(1, -1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = -np.ones(W * H)
    elif side == "front":
        x = np.linspace(-1, 1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = -np.ones(W * H)
    elif side == "right":
        y = np.linspace(-1, 1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        y, z = np.meshgrid(y, z)
        x = np.ones(W * H)
    elif side == "back":
        x = np.linspace(1, -1, W, endpoint=False)
        z = np.linspace(1, -1, H, endpoint=False)
        x, z = np.meshgrid(x, z)
        y = np.ones(W * H)
    elif side == "top":
        x = np.linspace(-1, 1, W, endpoint=False)
        y = np.linspace(1, -1, W, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = np.ones(W * W)
    elif side == "bottom":
        x = np.linspace(-1, 1, W, endpoint=False)
        y = np.linspace(-1, 1, W, endpoint=False)
        x, y = np.meshgrid(x, y)
        z = -np.ones(W * W)
    else:
        x, y, z = np.zeros((3, H * W))
    vec = np.stack([x.reshape(H * W), y.reshape(H * W), z.reshape(H * W)]).T
    theta, phi, _ = vec2sph(vec)
    return theta, phi


def cubebox(sky, side, rot=0, eye_model=None):
    theta, phi = cubebox_angles(side)
    if eye_model is not None:
        view = eye_model(np.array([np.pi/2-theta, np.pi-phi]).T)
        view.rotate(np.deg2rad(rot))
        view.set_sky(sky)
        L = view.L
        DOP = view.DOP
        AOP = view.AOP
    else:
        L, DOP, AOP = sky.get_features(theta, phi)
        L /= np.sqrt(2)
        L = ((1. - L[..., np.newaxis]).dot(np.array([[.05, .53, .79]])).T + L).T
    L_cube = np.clip(L.reshape((W, H, 3)), 0., 1)
    DOP[np.isnan(DOP)] = -1
    DOP = DOP.reshape((W, H))
    AOP = AOP.reshape((W, H))

    DOP_cube = np.zeros((W, H, 3))
    DOP_cube[..., 0] = DOP * .53 + (1. - DOP)
    DOP_cube[..., 1] = DOP * .81 + (1. - DOP)
    DOP_cube[..., 2] = DOP * 1.0 + (1. - DOP)
    DOP_cube = np.clip(DOP_cube, 0, 1)

    AOP_cube = AOP % np.pi
    AOP_cube = np.clip(AOP_cube, 0, np.pi)

    return L_cube, DOP_cube, AOP_cube


def skydome(sky, rot=0, eye_model=None):
    x, y = np.arange(W), np.arange(H)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    theta, phi = pix2sph(np.array([x, y]), H, W)
    if eye_model is not None:
        view = eye_model(np.array([np.pi/2-theta, np.pi-phi]).T)
        view.rotate(np.deg2rad(rot))
        view.set_sky(sky)
        sky_L = view.L
        sky_DOP = view.DOP
        sky_AOP = view.AOP
    else:
        sky_L, sky_DOP, sky_AOP = sky.get_features(theta, phi)
        sky_L = ((1. - sky_L[..., np.newaxis]).dot(np.array([[.05, .53, .79]])).T + sky_L).T
    sky_DOP = np.clip(sky_DOP, 0, 1)

    L = np.zeros((W, H, 3))
    L[x, y] = np.clip(sky_L, 0., 1)

    DOP = np.zeros((W, H, 3))
    DOP[x, y, 0] = sky_DOP * .53 + (1. - sky_DOP)
    DOP[x, y, 1] = sky_DOP * .81 + (1. - sky_DOP)
    DOP[x, y, 2] = sky_DOP * 1.0 + (1. - sky_DOP)

    AOP = sky_AOP % np.pi
    AOP = np.clip(AOP, 0, np.pi).reshape((W, H))

    return L, DOP, AOP