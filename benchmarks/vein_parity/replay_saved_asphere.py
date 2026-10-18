"""Replay the saved native HW/FW FastRBF Solution3 'asphere' basis offline.

This is an original implementation based on the independently recovered
piecewise kernel; native Leapfrog code or binaries are NOT required.

The vein project stores, for some cases:
  centres (N,3), coeffs (N,), basicfunc='asphere',
  basicfuncparam=[10.0,-3], degree=-1, rho=0, fittype='gmres'.
Use --centres/--coeffs preconverted from owner's own saved file (no unpickling).

Replaying this residual field is not the same as reconstructing the entire
reference surface, wall base mesh, mesh normals, or final volumetric geometry.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

ASPHERE_TAIL = 0.8734640537108553


def asphere_from_distances(distances, range_param=10.0):
    if range_param <= 0:
        raise ValueError("range must be positive")
    u = np.asarray(distances, dtype=float) / range_param
    return np.where(
        u < 0.5, 1.0 - 0.75 * u,
        ASPHERE_TAIL / np.power(1.0 + u * u, 1.5)
    )


def evaluate_saved_solution(centres, coefficients, evaluation_points, range_param=10.0):
    c = np.asarray(centres, dtype=float)
    co = np.asarray(coefficients, dtype=float)
    xyz = np.asarray(evaluation_points, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("centres and evaluation_points must be N x 3")
    if co.shape != (len(c),):
        raise ValueError("coefficient count must equal number of centres")
    return asphere_from_distances(cdist(xyz, c), range_param) @ co


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centres", type=Path, required=True)
    ap.add_argument("--coefficients", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--range", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("saved_rbf_predictions.csv"))
    a = ap.parse_args()
    outputs = evaluate_saved_solution(
        np.loadtxt(a.centres, delimiter=",", ndmin=2),
        np.loadtxt(a.coefficients, delimiter=",", ndmin=1),
        np.loadtxt(a.queries, delimiter=",", ndmin=2),
        range_param=a.range
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(a.out, outputs, delimiter=",")
    print("Saved FastRBF solution evaluated at", len(outputs), "points")


if __name__ == "__main__":
    main()
