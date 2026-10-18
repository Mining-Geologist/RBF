"""Iteratively invert a saved native asphere wall snap field along live normals.

Independently-authored numerical reconstruction. Requires numpy/scipy/trimesh;
takes *only* owner-decoded safe arrays, not Leapfrog binaries or a live app.
Selects the iterate minimizing the equation's own fixed-point residual,
NEVER a distance to saved Leapfrog centres or mesh truth.
"""
from __future__ import annotations
import numpy as np
import trimesh
from invert_native_asphere_snap import asphere
from scipy.spatial.distance import cdist


def inverse_iterative(vertices, faces, centres, coefficients, side,
                      scale=10.0, damping=0.65, iterations=24):
    if side not in ("hw","fw"):
        raise ValueError("Side must be hw or fw")
    if not 0 < damping <= 1 or iterations < 1 or scale <= 0:
        raise ValueError("Invalid iteration controls")
    final = np.asarray(vertices, float)
    faces = np.asarray(faces, int)
    centres = np.asarray(centres, float)
    coefficients = np.asarray(coefficients, float)
    if final.ndim!=2 or final.shape[1]!=3 or centres.ndim!=2 or centres.shape[1]!=3 or coefficients.shape!=(len(centres),):
        raise ValueError("Invalid vertex or native-solution arrays")
    if faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError("Faces must be triangles")
    sign=1 if side=="hw" else -1
    x=final.copy()
    best=x.copy()
    best_residual=float("inf")
    best_iteration=-1
    history=[]
    for i in range(iterations):
        model=trimesh.Trimesh(vertices=x, faces=faces, process=False)
        value=asphere(cdist(x, centres), scale) @ coefficients
        wanted=final+sign*model.vertex_normals*value[:,None]
        residual=float(np.sqrt(np.mean((wanted-x)**2)))
        history.append(residual)
        if residual<best_residual:
            best=x.copy()
            best_residual=residual
            best_iteration=i
        x += damping*(wanted-x)
    return best, {
        "best_iteration":best_iteration,
        "min_internal_fixed_point_residual":best_residual,
        "internal_residual_history":history
    }
