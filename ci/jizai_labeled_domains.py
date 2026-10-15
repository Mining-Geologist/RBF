"""Jizai-compatible pure NumPy oracle-label local-domain metadata builder.

Port of the recovered post-cluster rules from feature/automatic-subdomainer.
Does NOT perform automatic domain labelling or local RBF fitting/blending.
"""
import numpy as np

SPHEROIDAL3_C = 7.181510581693163


def build_labeled_domain_metadata(points, labels, anisotropies, base_range,
                                  support_multiplier=5, min_support_points=1):
    pts=np.asarray(points, dtype=float)
    labels=np.asarray(labels, dtype=np.int64).ravel()
    matrices=np.asarray(anisotropies,dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) != len(labels):
        raise ValueError("Expected N XYZ coordinates and N labels")
    if matrices.shape != (len(pts),3,3) or base_range <= 0:
        raise ValueError("Expected N 3x3 matrices and positive base range")
    c_root = np.sqrt(SPHEROIDAL3_C)
    output=[]
    for label in np.unique(labels):
        core=np.flatnonzero(labels==label)
        M=np.mean(matrices[core],axis=0)
        M=0.5*(M+M.T)
        eigen=np.linalg.eigvalsh(M)
        if np.any(eigen<=0):
            raise ValueError("Domain average must be positive definite")
        desired=float(base_range*eigen[-1]/c_root)
        transformed=pts @ M
        core_transformed=transformed[core]
        d=np.empty(len(pts), dtype=float)
        for start in range(0,len(pts),128):
            chunk=transformed[start:start+128]
            diff=chunk[:,None,:]-core_transformed[None,:,:]
            d[start:start+len(chunk)]=np.sqrt(np.min(np.sum(diff*diff,axis=2),axis=1))
        cap=min(len(pts),support_multiplier*len(core))
        if np.count_nonzero(d<desired)>cap and cap<len(pts):
            radius=float(np.partition(d,cap)[cap])
        else:
            radius=desired
        support=np.flatnonzero(d<radius)
        if len(support)<min_support_points:
            needed=min(min_support_points,len(pts))
            nearest=np.argpartition(d,needed-1)[:needed]
            support=np.sort(nearest)
            radius=float(np.nextafter(np.max(d[nearest]),np.inf))
        expansion=radius*np.linalg.norm(np.linalg.inv(M),axis=1)
        output.append({
            "label":int(label), "core_indices":core.tolist(),
            "support_indices":support.tolist(),
            "anisotropy":M.tolist(),
            "desired_internal_radius":desired,
            "internal_radius":radius,
            "local_kernel_range":float(radius*c_root),
            "bbox_min":(pts[core].min(axis=0)-expansion).tolist(),
            "bbox_max":(pts[core].max(axis=0)+expansion).tolist()
        })
    return output
