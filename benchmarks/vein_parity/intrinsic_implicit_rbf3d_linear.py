"""Independent 3D linear/biharmonic RBF field with affine polynomial drift.

No proprietary code, coordinates or data. For a known solution,
    f(X) = sum_j weights[j] * ||X - centres[j]||
           + a0 + (X-polybase) dot (ax,ay,az).
Returns scalar values; surface extraction requires a separate mesher.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist


def evaluate_biharmonic_3d_linear(xyz, centres, coefficients, polybase):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    centres = np.asarray(centres, dtype=float).reshape(-1, 3)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    base = np.asarray(polybase, dtype=float).reshape(3)
    n = len(centres)
    if len(coefficients) != n + 4:
        raise ValueError("Expected one weight per centre plus four affine coefficients")
    return (cdist(xyz, centres) @ coefficients[:n]
            + coefficients[n]
            + (xyz-base) @ coefficients[n+1:])


def orientation_constraints(center, direction, half_distance, signed_value=1.75):
    center = np.asarray(center, dtype=float).reshape(3)
    normal = np.asarray(direction, dtype=float).reshape(3)
    norm = np.linalg.norm(normal)
    if norm <= 0:
        raise ValueError("Orientation vector must be nonzero")
    normal /= norm
    if half_distance <= 0:
        raise ValueError("Half distance must be positive")
    return (np.vstack([center+half_distance*normal,
                       center-half_distance*normal]),
            np.array([signed_value, -signed_value]))


def implicit_surface_proxy(zero_points, orientation_center, direction,
                           half_distance, signed_value=1.75, smoothing=0):
    zero_points = np.asarray(zero_points, dtype=float).reshape(-1, 3)
    anchors, vals = orientation_constraints(orientation_center, direction,
                                            half_distance, signed_value)
    centres = np.vstack([zero_points, anchors])
    values = np.r_[np.zeros(len(zero_points)), vals]
    return RBFInterpolator(centres, values, kernel="linear", degree=1,
                           smoothing=smoothing), centres
