"""Group reference-mesh line intersections into distinct geological sheets.

This is an original topology-aware numerical surrogate informed by behavior
observed in synthetic native cases. It is NOT copied Leapfrog code and does
NOT claim full parity with its proprietary mesh-projection algorithm.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReferenceBranch:
    z: float
    face_ids: tuple[int, ...]
    normal_z_signs: tuple[int, ...]
    ambiguous_polarity: bool


def vertical_reference_branches(vertices, triangles, x, y,
                                bary_tol=1.e-9, z_tol=1.e-7):
    """Find all unique Z crossings of a vertical trace at (x, y).

    Coplanar face hits sharing an edge/vertex are grouped into one crossing.
    Different-height folded reference sheets remain separate branches.
    Opposite-sign face normals on the same crossing are marked ambiguous.
    """
    v=np.asarray(vertices,float)
    f=np.asarray(triangles,int)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:
        raise ValueError("Expected Nx3 vertices and Mx3 triangle indices")
    t=v[f]
    a=t[:,0,:2]
    b=t[:,1,:2]-a
    c=t[:,2,:2]-a
    det=b[:,0]*c[:,1]-b[:,1]*c[:,0]
    valid=np.abs(det)>bary_tol
    q=np.array([float(x),float(y)])-a
    u=np.full(len(f),np.nan)
    w=np.full(len(f),np.nan)
    u[valid]=(q[valid,0]*c[valid,1]-q[valid,1]*c[valid,0])/det[valid]
    w[valid]=(b[valid,0]*q[valid,1]-b[valid,1]*q[valid,0])/det[valid]
    mask=valid&(u>=-bary_tol)&(w>=-bary_tol)&(u+w<=1+bary_tol)
    face_idx=np.where(mask)[0]
    if len(face_idx)==0:
        return []
    z=t[face_idx,0,2]+u[face_idx]*(t[face_idx,1,2]-t[face_idx,0,2]) + w[face_idx]*(t[face_idx,2,2]-t[face_idx,0,2])
    raw_n=np.cross(t[face_idx,1]-t[face_idx,0],t[face_idx,2]-t[face_idx,0])
    signs=np.sign(raw_n[:,2]).astype(int)
    hits=sorted(zip(z,face_idx,signs),key=lambda item:item[0])
    result=[]
    current=[]
    for hit in hits:
        if current and abs(hit[0]-current[0][0])>z_tol:
            result.append(_group(current))
            current=[]
        current.append(hit)
    if current:result.append(_group(current))
    return result


def _group(hits):
    z=float(np.mean([h[0] for h in hits]))
    indices=tuple(int(h[1]) for h in hits)
    signs=tuple(sorted(set(int(h[2]) for h in hits)))
    return ReferenceBranch(z,indices,signs,len(signs)!=1 or signs==(0,))


def pick_nearest_reference_branch(branches, query_z):
    """Select by drilled depth; report an ambiguous crossing explicitly."""
    if not branches:return None
    return min(branches,key=lambda b:(abs(b.z-float(query_z)),b.z))
