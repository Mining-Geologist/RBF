"""Independent symmetric thickness correction for reference-relative vein walls.

This is original implementation of a recovered behavioral numerical rule.
No vendor code or binary data.

The two wall offsets must use a consistent oriented normal. Pinchout,
contact snapping, branch selection and meshing are separate operations.
"""
import numpy as np


def clamp_signed_wall_thickness(upper, lower, minimum=None, maximum=None):
    """Clamp signed thickness by moving both walls equally about their midpoint."""
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum thickness must not exceed maximum")
    thickness = upper-lower
    target = thickness
    if minimum is not None:
        target = np.maximum(target, minimum)
    if maximum is not None:
        target = np.minimum(target, maximum)
    half_excess = (thickness-target)*0.5
    return upper-half_excess, lower+half_excess
