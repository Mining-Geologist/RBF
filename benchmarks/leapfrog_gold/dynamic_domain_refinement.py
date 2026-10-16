"""Experimental offline second-stage structural-domain refinement.

Only inputs are training-point XYZ, pointwise SPD anisotropy matrices, and
independently generated initial cluster IDs. The algorithm NEVER sees a
Leapfrog reference label, and never invokes Leapfrog. Requires numpy/scipy.

Parameter settings selected on SIX development WolfPass cases; keep locked
while evaluating other saved models (see research notes).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def refinement(x, v, labels, *, iterations=14, neighbours=12,
               spatial_power=1., tensor_graph_weight=.5,
               neighbor_reward=1., coordinate_reward=.1,
               orientation_reward=.5, prior_reward=0.,
               inertia=0., covariance="mahal", covariance_floor=.1):
    labels = np.asarray(labels, int).copy()
    original = labels.copy()
    n = len(labels)
    k = labels.max() + 1
    dist, near = cKDTree(x).query(x, k=min(neighbours + 1, n))
    ii = np.repeat(np.arange(n), near.shape[1] - 1)
    jj = near[:, 1:].reshape(-1)
    d = dist[:, 1:].reshape(-1)
    w = np.maximum(d, 1e-4) ** (-spatial_power)
    if tensor_graph_weight:
        dv = v[ii] - v[jj]
        w *= np.exp(-tensor_graph_weight * np.sum(dv * dv, axis=1) / v.shape[1])
    W = csr_matrix((w, (ii, jj)), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()
    normalizer = np.asarray(W.sum(axis=1)).ravel() + 1e-15
    eye = np.eye(k)
    for _ in range(iterations):
        count = np.bincount(labels, minlength=k)
        mx = np.vstack([
            x[labels == c].mean(axis=0) if count[c] else x.mean(axis=0)
            for c in range(k)
        ])
        mv = np.vstack([
            v[labels == c].mean(axis=0) if count[c] else v.mean(axis=0)
            for c in range(k)
        ])
        delta = x[:, None, :] - mx[None, :, :]
        if covariance == "mahal":
            inverse = np.zeros((k, 3, 3))
            for c in range(k):
                if count[c] < 4:
                    inverse[c] = np.eye(3)
                else:
                    local = x[labels == c] - mx[c]
                    cov = local.T @ local / len(local)
                    inverse[c] = np.linalg.inv(cov + np.eye(3) * covariance_floor)
            coord = np.einsum("nki,kij,nkj->nk", delta, inverse, delta)
        else:
            coord = np.einsum("nki,nki->nk", delta, delta)
        orient = np.sum((v[:, None, :] - mv[None, :, :]) ** 2, axis=-1) / v.shape[1]
        graph = np.asarray(W @ eye[labels]) / normalizer[:, None]
        feasible = graph > 1e-12
        feasible[np.arange(n), labels] = True
        cost = (coordinate_reward * coord + orientation_reward * orient
                - neighbor_reward * graph
                - prior_reward * np.log(np.maximum(count, 1))[None, :]
                + inertia * (eye[original] == 0))
        cost[~feasible] = 1e8
        proposal = np.argmin(cost, axis=1)
        for c in range(k):
            if np.sum(proposal == c) == 0:
                ix = np.flatnonzero(labels == c)
                if len(ix):
                    protected = ix[np.argmin(cost[ix, c] - cost[ix, proposal[ix]])]
                    proposal[protected] = c
        if np.array_equal(labels, proposal):
            break
        labels = proposal
    return labels


def refine_domains(points, tensors, initial_labels, *, max_changed_fraction=.10):
    """Return new labels and safeguard diagnostics, with no oracle dependence."""
    x = np.asarray(points, float)
    m = np.asarray(tensors, float)
    original = np.asarray(initial_labels)
    if x.ndim != 2 or x.shape[1] != 3 or m.shape != (len(x), 3, 3) or original.shape != (len(x),):
        raise ValueError("Expected points (N,3), tensors (N,3,3), labels (N,)")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("Expected at least three finite points/tensors")
    if not 0 <= max_changed_fraction <= 1:
        raise ValueError("max_changed_fraction must lie between 0 and 1")
    identifiers, index = np.unique(original, return_inverse=True)
    if len(identifiers) == 1:
        return original.copy(), {
            "accepted": True, "changed_fraction": 0., "changed_points": 0,
            "reason": "single initial domain",
        }
    z = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-10)
    u = m[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    u = (u - u.mean(axis=0)) / (u.std(axis=0) + 1e-10)
    proposal = refinement(z, u, index)
    count = int(np.count_nonzero(index != proposal))
    fraction = count / len(original)
    accepted = bool(fraction <= max_changed_fraction)
    out = identifiers[proposal] if accepted else original.copy()
    return out, {
        "accepted": accepted, "changed_fraction": round(fraction, 8),
        "changed_points": count, "threshold": max_changed_fraction,
        "reason": "bounded reassignment" if accepted else "rejected: excessive reassignment",
    }
