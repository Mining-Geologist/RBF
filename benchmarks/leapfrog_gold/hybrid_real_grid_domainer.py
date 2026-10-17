"""Offline hybrid: grid-initialized, real-sample structural domain growth.

Pure NumPy/SciPy, no Leapfrog runtime. Reference memberships are NEVER passed
into prediction. Existing grid labels may come from any unsupervised builder.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from dynamic_domain_refinement import refine_domains


@dataclass(frozen=True)
class Settings:
    refine: bool = True
    changed_gate: float = .10
    knn: int = 8
    merge_min_similarity: float = .99
    merge_min_contacts: float = .10
    merge_small_fraction: float = .05
    merge_max_fraction: float = .35
    split: bool = False
    split_min_fraction: float = .025
    split_max_similarity: float = .93


def get_edges(x: np.ndarray, knn: int):
    """Unique, symmetric real-sample kNN edges."""
    if knn < 1:
        raise ValueError("knn must be positive")
    _, ix = cKDTree(x).query(x, k=min(knn + 1, len(x)))
    u = np.repeat(np.arange(len(x)), ix.shape[1] - 1)
    v = ix[:, 1:].reshape(-1)
    u, v = np.minimum(u, v), np.maximum(u, v)
    valid = u != v
    keys = np.unique(u[valid] * len(x) + v[valid])
    return keys // len(x), keys % len(x)


def consistency(a, size_a, b, size_b):
    m = (a * size_a + b * size_b) / (size_a + size_b)
    d = float(np.linalg.det(m))
    return 1. / d if np.isfinite(d) and d > 0 else -np.inf


def dynamic_adjacent_merges(labels, matrices, u, v, *, min_similarity,
                            min_contacts, small_fraction, max_fraction, knn):
    """Rebuild real-point boundary graph and group matrices after every merge."""
    n = len(labels)
    output = labels.copy()
    details = []
    while True:
        domains, ids = np.unique(output, return_inverse=True)
        k = len(domains)
        if k <= 1:
            break
        sizes = np.bincount(ids, minlength=k)
        sums = np.zeros((k, 3, 3))
        np.add.at(sums, ids, matrices)
        means = sums / sizes[:, None, None]
        boundary = ids[u] != ids[v]
        i = np.minimum(ids[u[boundary]], ids[v[boundary]])
        j = np.maximum(ids[u[boundary]], ids[v[boundary]])
        if len(i) == 0:
            break
        pairs, counts = np.unique(i*k + j, return_counts=True)
        proposals = []
        for pair, contacts in zip(pairs, counts):
            a, b = int(pair // k), int(pair % k)
            sa, sb = int(sizes[a]), int(sizes[b])
            if sa + sb > max_fraction*n or min(sa, sb) > small_fraction*n:
                continue
            score = consistency(means[a], sa, means[b], sb)
            support = contacts / max(1., min(sa, sb)*knn)
            if score < min_similarity or support < min_contacts:
                continue
            proposals.append((support*score, score, a, b, contacts))
        if not proposals:
            break
        _, score, a, b, contacts = max(proposals)
        output[ids == b] = domains[a]
        details.append({"before_a": int(domains[a]), "before_b": int(domains[b]),
                        "similarity": round(float(score), 6),
                        "contacts": int(contacts)})
    return output, details


def birth_connected_components(labels, matrices, u, v,
                               min_fraction, max_similarity):
    """Birth a region only if a large disconnected component is tensor-distinct.

    This is optional and deliberately off in the reported benchmark candidate.
    """
    out = labels.copy()
    n = len(labels)
    smallest = max(3, int(np.ceil(min_fraction*n)))
    nextid = int(out.max()) + 1
    born = []
    for label in np.unique(labels):
        idx = np.flatnonzero(out == label)
        if len(idx) < 2*smallest:
            continue
        local = np.full(n, -1, dtype=int)
        local[idx] = np.arange(len(idx))
        keep = (out[u] == label) & (out[v] == label)
        a, b = local[u[keep]], local[v[keep]]
        graph = coo_matrix((np.ones(2*len(a)), (np.r_[a,b], np.r_[b,a])),
                           shape=(len(idx),len(idx))).tocsr()
        component_count, membership = connected_components(graph, directed=False)
        if component_count <= 1:
            continue
        sizes = np.bincount(membership)
        large = np.where(sizes >= smallest)[0]
        if len(large) <= 1:
            continue
        mainid = large[np.argmax(sizes[large])]
        main = matrices[idx[membership == mainid]].mean(axis=0)
        for part in large:
            if part == mainid:
                continue
            subset = idx[membership == part]
            other = matrices[subset].mean(axis=0)
            score = consistency(main, int(sizes[mainid]), other, len(subset))
            if score <= max_similarity:
                out[subset] = nextid
                born.append({"from":int(label), "new":nextid,
                             "size":int(len(subset)),
                             "similarity":round(float(score),6)})
                nextid += 1
    return out, born


def fit_hybrid(points, tensors, initial_labels, settings=Settings()):
    """Predict hybrid real-point domains without reference labels."""
    x = np.asarray(points, float)
    m = np.asarray(tensors, float)
    initial = np.asarray(initial_labels)
    n = len(x)
    if x.shape != (n, 3) or m.shape != (n, 3, 3) or initial.shape != (n,) or n < 5:
        raise ValueError("Expected N x 3 points, N x 3 x 3 SPD tensors, N labels")
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("Expected finite inputs")
    if not np.allclose(m, m.transpose(0,2,1), atol=1e-8, rtol=1e-8):
        raise ValueError("Expected symmetric anisotropy matrices")
    if np.any(np.linalg.eigvalsh(m) <= 0):
        raise ValueError("Expected positive-definite anisotropy tensors")

    _, compact_initial = np.unique(initial, return_inverse=True)
    working = compact_initial.copy()
    if settings.refine:
        working, diag = refine_domains(
            x, m, working, max_changed_fraction=settings.changed_gate)
    else:
        diag = {"accepted": False, "changed_fraction": 0.,
                "changed_points": 0, "reason": "disabled"}
    det = np.linalg.det(m)
    normalized = m / (det[:,None,None]**(1/3))
    u, v = get_edges(x, settings.knn)

    merge_records = []
    if settings.merge_small_fraction > 0:
        working, merge_records = dynamic_adjacent_merges(
            working, normalized, u, v,
            min_similarity=settings.merge_min_similarity,
            min_contacts=settings.merge_min_contacts,
            small_fraction=settings.merge_small_fraction,
            max_fraction=settings.merge_max_fraction,
            knn=settings.knn)
    births = []
    if settings.split:
        working, births = birth_connected_components(
            working, normalized, u, v,
            settings.split_min_fraction, settings.split_max_similarity)
    _, output = np.unique(working, return_inverse=True)
    return output, {
        "stage2": diag, "initial_domains": int(len(np.unique(compact_initial))),
        "output_domains": int(len(np.unique(output))),
        "merges": len(merge_records), "merge_events": merge_records,
        "births": len(births), "birth_events": births,
        "real_knn_edges": int(len(u)),
    }
