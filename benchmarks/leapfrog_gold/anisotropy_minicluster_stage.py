"""Offline two-stage anisotropy-aware grid premerger (experimental).

Pure NumPy. No Leapfrog runtime; no reference domain labels enter prediction.
The baseline when micro_size=1 matches the corrected fixed-grid prototype
on the ten previously exported research benchmarks.
"""
from __future__ import annotations

import heapq
import numpy as np


def edges_6(shape):
    """Undirected six-face voxel edges with flat C-order indexing."""
    grid = np.arange(np.prod(shape)).reshape(shape)
    parts = []
    for axis in range(3):
        if shape[axis] < 2:
            continue
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        parts.append(np.column_stack(
            [grid[tuple(left)].ravel(), grid[tuple(right)].ravel()]
        ))
    return np.vstack(parts) if parts else np.empty((0, 2), dtype=int)


def det3(m):
    """Determinant of a symmetric 3x3 matrix."""
    a, b, c = m[0]
    _, e, f = m[1]
    _, _, i = m[2]
    return a*e*i + 2*b*c*f - a*f*f - e*c*c - i*b*b


def comb(a, sa, b, sb):
    """Return count-weighted average matrix and reciprocal determinant."""
    v = (a*sa+b*sb)/(sa+sb)
    d = det3(v)
    return v, (1./d if d > 0 else -np.inf)


def initial_miniclusters(matrices, shape, micro_size=4,
                         micro_consistency=.995, tile_size=0):
    """Premerge structurally compatible neighboring grid cells, with a size cap.

    tile_size=0: unrestricted adjacency; 2: within aligned 2x2x2 tiles.
    This is not a claim about proprietary implementation internals.
    """
    mats = np.asarray(matrices, float)
    n = len(mats)
    ed = edges_6(shape)
    if tile_size:
        coords = np.array(np.unravel_index(np.arange(n), shape)).T
        mask = np.all(coords[ed[:, 0]] // tile_size ==
                      coords[ed[:, 1]] // tile_size, axis=1)
        ed = ed[mask]
    parent = np.arange(n)
    weights = np.ones(n, int)
    means = mats.copy()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    heap = []
    for a, b in ed:
        _, s = comb(means[a], 1, means[b], 1)
        if s >= micro_consistency:
            heapq.heappush(heap, (-s, int(a), int(b)))

    count = 0
    while heap:
        _, ia, ib = heapq.heappop(heap)
        a, b = find(ia), find(ib)
        if a == b or weights[a]+weights[b] > micro_size:
            continue
        new_m, s = comb(means[a], weights[a], means[b], weights[b])
        if s < micro_consistency:
            continue
        if weights[a] < weights[b]:
            a, b = b, a
        parent[b] = a
        means[a] = new_m
        weights[a] += weights[b]
        count += 1
    labels = np.array([find(i) for i in range(n)])
    return labels, count


def two_stage_merge(points, matrices, shape, minimum, maximum,
                    micro_size=4, micro_consistency=.995, tile_size=0,
                    maximum_fraction=.10, consistency_threshold=.6):
    """Predict real-point memberships without seeing reference labels."""
    n = len(matrices)
    matrices = np.asarray(matrices, float)
    dets = np.linalg.det(matrices)
    if np.any(dets <= 0):
        raise ValueError("anisotropy matrices must be SPD")
    mats = matrices/(dets[:, None, None]**(1/3))
    parent, first_merges = initial_miniclusters(
        mats, shape, micro_size, micro_consistency, tile_size)
    _, own = np.unique(parent, return_inverse=True)
    k = len(np.unique(parent))
    means = np.zeros((n*2, 3, 3))
    sizes = np.zeros(n*2, np.int64)
    active = np.zeros(n*2, bool)
    version = np.zeros(n*2, np.int64)
    for c in range(k):
        selected = own == c
        sizes[c] = selected.sum()
        means[c] = mats[selected].mean(axis=0)
        active[c] = True

    neighbours = [set() for _ in range(2*n)]
    for a, b in edges_6(shape):
        a, b = int(own[a]), int(own[b])
        if a != b:
            neighbours[a].add(b)
            neighbours[b].add(a)

    maxsize = max(1, int(maximum_fraction*n))
    heap = []

    def push(a, b):
        if not active[a] or not active[b] or a == b:
            return
        if sizes[a] + sizes[b] > maxsize:
            return
        _, consistency = comb(means[a], sizes[a], means[b], sizes[b])
        if np.isfinite(consistency):
            a, b = sorted((a, b))
            heapq.heappush(heap, (-consistency, a, b,
                                  int(version[a]), int(version[b])))

    for a in range(k):
        for b in neighbours[a]:
            if a < b:
                push(a, b)

    next_id = k
    second_merges = 0
    source_parent = np.arange(2*n)
    while heap:
        neg, a, b, va, vb = heapq.heappop(heap)
        if not active[a] or not active[b] or version[a] != va or version[b] != vb:
            continue
        if b not in neighbours[a]:
            continue
        if -neg < consistency_threshold:
            break
        if sizes[a] + sizes[b] > maxsize:
            continue

        new_m, _ = comb(means[a], sizes[a], means[b], sizes[b])
        adjacent = (neighbours[a] | neighbours[b]) - {a, b}
        active[a] = active[b] = False
        version[a] += 1
        version[b] += 1
        c = next_id
        next_id += 1
        active[c] = True
        source_parent[a] = source_parent[b] = c
        sizes[c] = sizes[a] + sizes[b]
        means[c] = new_m
        for other in sorted(adjacent):
            if not active[other]:
                continue
            neighbours[other].discard(a)
            neighbours[other].discard(b)
            neighbours[other].add(c)
            neighbours[c].add(other)
        for other in sorted(neighbours[c]):
            push(c, other)
        second_merges += 1

    def root(x):
        while source_parent[x] != x:
            x = source_parent[x]
        return x

    final_seeds = np.array([root(c) for c in own])
    span = maximum-minimum
    voxel = np.clip(
        np.floor((points-minimum)/np.where(span > 0, span, 1)*shape).astype(int),
        0, np.array(shape)-1)
    pidx = (voxel[:, 0]*shape[1]+voxel[:, 1])*shape[2]+voxel[:, 2]
    _, predicted = np.unique(final_seeds[pidx], return_inverse=True)
    return predicted, {
        "initial_miniclusters": k, "initial_merges": first_merges,
        "final_grid_domains": np.unique(final_seeds).size,
        "final_point_domains": np.unique(predicted).size,
        "stage2_merges": second_merges,
    }
