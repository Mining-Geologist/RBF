"""Research-only robust midpoint reference surface, requiring no native runtime.

Reads owner-exported midpoint (X,Y,Z) values. Does not use Leapfrog meshes
for model fitting; separate benchmark scores against saved native OBJ.
Only single-valued, non-overturned XY charts are supported. Rejects obvious
multi-sheet folded inputs rather than silently smoothing through them.

This is an independently authored HYPOTHESIS, not a native Leapfrog recovery.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import RBFInterpolator


@dataclass
class RobustReference:
    model: object
    plane_coefficients: np.ndarray
    inlier_sites: np.ndarray
    input_count: int
    rejected: tuple
    smoothing: float

    def height(self, xy):
        q = np.asarray(xy, float)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("Expected query XY array (N,2)")
        return np.asarray(self.model(q), float)


def _plane(xy):
    q = np.asarray(xy, float)
    return np.column_stack((np.ones(len(q)), q[:,0]/100, q[:,1]/100))


def fit_robust_reference(midpoints, *, outlier_m=7.,
                         smoothings=(.01, 1., 20., 100., 500., 2000.)):
    """Robust global affine trend + cross-validated smooth inlier surface.

    Deduplicate a single multivalued XY site by its mean Z; reject >=2 such
    sites as likely multi-sheet folded geometry. Then use a robust plane
    residual threshold fixed *without* seeing any reference-mesh vertices.
    Selection of radial smoothness uses only midpoint leave-one-out error.
    """
    xyz = np.asarray(midpoints, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError("Expected >=4 finite midpoint XYZ rows")
    xy, reverse, n = np.unique(
        np.round(xyz[:,:2], 6), return_inverse=True, return_counts=True, axis=0
    )
    if np.count_nonzero(n > 1) >= 2:
        raise ValueError("Multisheet reference: more than one XY site has several Z values")
    z = np.bincount(reverse, weights=xyz[:,2]) / n
    design = _plane(xy)
    robust = least_squares(lambda p: design@p-z,
                           np.linalg.lstsq(design,z,rcond=None)[0],
                           loss="soft_l1", f_scale=2.)
    discarded = np.abs(design@robust.x-z) > outlier_m
    keep = ~discarded
    if keep.sum() < 4 or np.linalg.matrix_rank(design[keep]) < 3:
        raise ValueError("Insufficient stable, non-collinear midpoint sites")
    xx, yy = xy[keep], z[keep]
    loocv = {}
    for smoothing in smoothings:
        errors = []
        for i in range(len(xx)):
            valid = np.arange(len(xx)) != i
            if np.linalg.matrix_rank(_plane(xx[valid])) < 3:
                continue
            try:
                fit = RBFInterpolator(xx[valid],yy[valid],
                                      kernel="thin_plate_spline",
                                      degree=1,smoothing=float(smoothing))
                errors.append(float(fit(xx[i:i+1])[0]-yy[i]))
            except np.linalg.LinAlgError:
                continue
        loocv[float(smoothing)] = (float(np.sqrt(np.mean(np.square(errors))))
                                   if errors else float("inf"))
    best = min(loocv,key=loocv.get)
    if not np.isfinite(loocv[best]):
        raise ValueError("Reference training geometry is singular")
    fit = RBFInterpolator(xx,yy,kernel="thin_plate_spline",degree=1,smoothing=best)
    return RobustReference(
        model=fit,
        plane_coefficients=np.array(robust.x),
        inlier_sites=xx.copy(),
        input_count=len(xyz),
        rejected=tuple((float(xy[i,0]),float(xy[i,1]),float(z[i]))
                       for i in np.flatnonzero(discarded)),
        smoothing=best,
    )
