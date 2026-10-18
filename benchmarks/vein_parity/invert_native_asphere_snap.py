"""Invert native saved asphere wall snapping at final mesh vertices.

Owner-provided preconverted NPZ only; never load/execute Leapfrog binaries.
Exact in small smooth displacements; folded large displacements require
original pre-snap mesh normals and surface offsets to improve accuracy.
"""
import numpy as np
from scipy.spatial.distance import cdist

def asphere(distance, range_m=10.0):
    u=np.asarray(distance,dtype=float)/range_m
    return np.where(u<.5,1.-.75*u,
                    .8734640537108553/(1.+u*u)**1.5)

def inverse_snap(vertices,normals,centres,coefficients,side,range_m=10.0):
    """Inverse of first-order mesh-normal snap with frozen saved native RBF."""
    if side not in ("hw","fw"):
        raise ValueError("side must be hw or fw")
    xyz=np.asarray(vertices,float);normals=np.asarray(normals,float)
    c=np.asarray(centres,float);w=np.asarray(coefficients,float)
    if xyz.shape!=normals.shape or xyz.ndim!=2 or xyz.shape[1]!=3:
        raise ValueError("vertex and normal shapes must be (N,3)")
    if c.ndim!=2 or c.shape[1]!=3 or w.shape!=(len(c),) or range_m<=0:
        raise ValueError("invalid saved Solution3")
    normal_lengths=np.linalg.norm(normals,axis=1)
    if np.any(normal_lengths<=0):
        raise ValueError("zero-length normal")
    field=asphere(cdist(xyz,c),range_m)@w
    polarity=+1 if side=="hw" else -1
    return xyz+polarity*(normals/normal_lengths[:,None])*field[:,None]
