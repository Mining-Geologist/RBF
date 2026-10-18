"""Experimental 3D folded vein reference sheet reconstructed from midpoint locations.

Assumes one synthetic five-point U-shaped cross-section per Y level. This
is NOT a general fold algorithm or verified Leapfrog reference construction.
The saved Leapfrog reference mesh is not an input to this function.
"""
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator


def make_synthetic_fold_reference(midpoints, nu=81, nv=31,
                                  y_min=-121., y_max=121.):
    p = np.asarray(midpoints, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite midpoint (N,3)")
    levels = np.unique(np.round(p[:, 1], 5))
    if len(levels) < 2 or nu < 6 or nv < 2:
        raise ValueError("Requires multiple Y sections and adequate mesh sampling")
    profiles = []
    for y in levels:
        section = p[np.abs(p[:, 1] - y) < .01]
        if len(section) != 5:
            raise ValueError("Synthetic U-fold requires five input points per Y slice")
        xlevels = np.unique(np.round(section[:, 0], 4))
        if len(xlevels) != 3:
            raise ValueError("Expected three distinct X positions")
        left = section[np.abs(section[:, 0] - xlevels[0]) < .01]
        mid = section[np.abs(section[:, 0] - xlevels[1]) < .01]
        right = section[np.abs(section[:, 0] - xlevels[2]) < .01]
        if len(left) != 1 or len(mid) != 2 or len(right) != 2:
            raise ValueError("Expected one fold nose and paired upper/lower limbs")
        profile = np.vstack((
            left[0], mid[np.argmax(mid[:, 2])],
            right[np.argmax(right[:, 2])], right[np.argmin(right[:, 2])],
            mid[np.argmin(mid[:, 2])]
        ))[:, [0, 2]]
        profiles.append(profile)
    profile = np.mean(profiles, axis=0)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
    if np.min(np.diff(arc)) <= 0:
        raise ValueError("Fold profile contains zero-length segments")
    s = np.linspace(0, arc[-1], nu)
    x = PchipInterpolator(arc, profile[:, 0])(s)
    z = PchipInterpolator(arc, profile[:, 1])(s)
    ys = np.linspace(y_min, y_max, nv)
    xx, yy = np.meshgrid(x, ys, indexing="ij")
    zz = np.broadcast_to(z[:, None], xx.shape)
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b = i * nv + j, (i + 1) * nv + j
            faces.extend(((a, b, a+1), (b, b+1, a+1)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces),
                           process=False), profile


def point_to_fold_profile_distance(xz, profile):
    p = np.asarray(xz, float)[:, None, :]
    a = np.asarray(profile, float)[None, :-1, :]
    b = np.asarray(profile, float)[None, 1:, :]
    d = b-a
    t = np.clip(np.sum((p-a)*d, axis=-1) /
                np.maximum(np.sum(d*d, axis=-1), 1.e-12), 0., 1.)
    closest = a + t[..., None]*d
    return np.sqrt(np.min(np.sum((p-closest)**2, axis=2), axis=1))
