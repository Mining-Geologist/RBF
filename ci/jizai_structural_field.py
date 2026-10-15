"""Pure NumPy structural trend and indicator preprocessing for Jizai LVA port.

Ported from the user's legacy Polatory structural-LVA branch; derived formulae
were benchmarked there against Leapfrog's serialized WolfPass output. This
module does NOT implement local-domain fitting or blending.
"""
import numpy as np


def nearest_vertex_trend_matrices(query, vertices, faces, strength, range_m):
    """Return n x 3 x 3 single-input structural anisotropy matrices.

    Normal at closest mesh vertex = normalized equal-weight face-normal sum.
    The distance falloff is exp(-d/range) for d < 4 range, else zero.
    """
    q = np.asarray(query, dtype=float)
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 3 or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("query and vertices must have shape (N,3)")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("faces must have shape (M,3)")
    if range_m <= 0 or strength <= 0:
        raise ValueError("range_m and strength must be positive")

    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lens = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lens[:, None], out=np.zeros_like(normals), where=lens[:, None]>0)
    vn = np.zeros_like(v)
    for corner in range(3):
        np.add.at(vn, f[:, corner], normals)
    lengths = np.linalg.norm(vn, axis=1)
    vn = np.divide(vn, lengths[:, None], out=np.zeros_like(vn), where=lengths[:, None]>0)
    vn[lengths == 0] = [0., 0., 1.]

    nearest = np.empty(len(q), dtype=int)
    dists = np.empty(len(q))
    for start in range(0, len(q), 512):
        stop = min(start+512, len(q))
        sq = np.sum((q[start:stop, None, :] - v[None, :, :])**2, axis=2)
        closest = np.argmin(sq, axis=1)
        nearest[start:stop] = closest
        dists[start:stop] = np.sqrt(sq[np.arange(stop-start), closest])

    weight = np.exp(-dists / range_m)
    weight[dists >= 4. * range_m] = 0.0
    ratio = 1.0 + (strength - 1.0) * weight
    n = vn[nearest]
    proj = n[:,:,None] * n[:,None,:]
    eye = np.eye(3)[None,:,:]
    return ratio[:,None,None]**(-1./3.) * (eye - proj) + ratio[:,None,None]**(2./3.) * proj


def indicator_distance_values(points, indicators):
    """Compute recovered Leapfrog sign/distance/clip preprocessing.

    Indicator input must be +/-1. Return (values, fit_accuracy, clip_value,
    raw_values). No evaluation or mesh work is performed here.
    """
    p = np.asarray(points, dtype=float)
    labels = np.asarray(indicators, dtype=float).ravel()
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(labels):
        raise ValueError("Expected (N,3) positions and N indicators")
    if not np.all(np.isin(labels, [-1., 1.])):
        raise ValueError("Indicators must be +/-1")
    if not np.any(labels > 0) or not np.any(labels < 0):
        raise ValueError("Both classes are required")
    d = np.empty(len(p))
    for target in (-1., 1.):
        ids = np.flatnonzero(labels == target)
        opposite = p[labels != target]
        for start in range(0, len(ids), 128):
            i = ids[start:start+128]
            dsq = np.sum((p[i, None, :] - opposite[None,:,:])**2, axis=2)
            d[i] = np.sqrt(np.min(dsq, axis=1))
    raw = -labels * d
    diagonal = float(np.linalg.norm(p.max(axis=0) - p.min(axis=0)))
    tolerance = 1.e-5 * diagonal
    clip = 1.e-2 * diagonal
    return np.clip(raw, -clip, clip), tolerance, clip, raw
