"""Infer ONLY identifiable nearest-vertex ties from saved domain mean matrices.

This is an oracle diagnostic, not a deployable native tie-selection algorithm.
User-owned input file is local and must never be committed. See the research
audit for provenance, tie tolerance and identifiability limits.

Expected .npz keys: points, vertices, gradients, domain_ids, domain_labels,
reference, strength, trend_range. Requires NumPy and SciPy.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def field(distance, normals, strength, radius):
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    q = np.exp(-distance / radius)
    q[distance >= 4 * radius] = 0
    r = 1 + (strength - 1) * q
    tangent = r ** (-1 / 3)
    return (tangent[:, None, None] * np.eye(3)
            + (r ** (2 / 3) - tangent)[:, None, None] * n[:, :, None] * n[:, None, :])


def matching_subsets(deltas, target, tolerance=1e-8):
    n = len(deltas)
    if n > 20:
        raise ValueError("More than 20 choices cannot be exhaustively searched")
    if n == 0:
        return np.zeros((1, 0), dtype=np.uint8)
    out = []
    basis = deltas.reshape(n, 9)
    for start in range(0, 1 << n, 4096):
        index = np.arange(start, min(start + 4096, 1 << n), dtype=np.uint32)
        bits = ((index[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.uint8)
        errors = np.linalg.norm(bits.astype(float) @ basis - target.reshape(9), axis=1)
        out.extend(bits[errors < tolerance])
    return np.asarray(out, dtype=np.uint8).reshape(-1, n)


def solve(points, vertices, gradients, domain_ids, labels, references, strength, radius,
          max_ties=18):
    distances, nearest = cKDTree(vertices).query(points, k=2)
    ties = np.abs(distances[:, 1] - distances[:, 0]) < 1e-8
    A = field(distances[:, 0], gradients[nearest[:, 0]], strength, radius)
    B = field(distances[:, 0], gradients[nearest[:, 1]], strength, radius)
    rows, stats = [], []
    for k, domain_id in enumerate(domain_ids):
        domain_indices = np.flatnonzero(labels == domain_id)
        ambiguous = domain_indices[ties[domain_indices]]
        n = len(ambiguous)
        if n > max_ties:
            stats.append([int(domain_id), n, "not enumerated", 0])
            continue
        goal = references[k] * len(domain_indices) - A[domain_indices].sum(axis=0)
        solutions = matching_subsets(B[ambiguous] - A[ambiguous], goal)
        identifiable = int(np.all(solutions == solutions[0], axis=0).sum()) if len(solutions) else 0
        stats.append([int(domain_id), n, len(solutions), identifiable])
        for j, point_id in enumerate(ambiguous):
            choice = int(solutions[0, j]) if len(solutions) and np.all(solutions[:, j] == solutions[0, j]) else -1
            rows.append([int(domain_id), int(point_id), int(nearest[point_id, 0]),
                         int(nearest[point_id, 1]), choice, len(solutions)])
    return rows, stats, int(ties.sum())


def self_test():
    matrices = np.array([np.eye(3), np.diag([0.2, -0.1, 0.6]), np.array([[0,0,1],[0,0,0],[1,0,0]])])
    target = matrices[0] + matrices[2]
    found = matching_subsets(matrices, target, 1e-12)
    assert found.shape == (1, 3) and list(found[0]) == [1,0,1]
    duplicate = np.stack([np.eye(3), np.eye(3)])
    found = matching_subsets(duplicate, np.eye(3), 1e-12)
    assert len(found) == 2 and not np.all(found == found[0])
    print("PASS: unique and non-identifiable tie-selection controls")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, default=Path("tie-oracle"))
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    if args.input is None:
        return
    with np.load(args.input, allow_pickle=False) as data:
        rows, summary, ties = solve(*(data[name] for name in
            ["points", "vertices", "gradients", "domain_ids", "domain_labels", "reference"]),
            float(data["strength"]), float(data["trend_range"]))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, header, values in [
        ("domain_summary.csv", ["domain", "ties", "solutions", "identifiable"], summary),
        ("identifiable_rows.csv", ["domain","row","vertex_0","vertex_1","chosen_candidate","solutions"], rows),
    ]:
        with (args.output / filename).open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerows(values)
    print(f"Tied rows: {ties}; uniquely constrained choices: {sum(row[3] for row in summary)}")


if __name__ == "__main__":
    main()
