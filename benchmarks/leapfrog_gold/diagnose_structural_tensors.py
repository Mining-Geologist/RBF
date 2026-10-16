"""Diagnose Leapfrog-style LVA tensors independently of the native engine.

Tests whether exported pointwise 3x3 SPD matrices are uniaxial (single structural
normal) or biaxial/triaxial (evidence for multi-axis tensor mixing). Provides
candidate multi-input mixing predictions, NOT a claim that any candidate is
Leapfrog's proprietary algorithm.

Example:
 python benchmarks/leapfrog_gold/diagnose_structural_tensors.py --self-test
 python benchmarks/leapfrog_gold/diagnose_structural_tensors.py --input matrices.csv --output tensor_diagnostics.csv --strength 5 --range 100

Expected CSV names: matrix_00, matrix_01,...,matrix_22.
Averaged *domain* tensors need not be uniaxial even when each pointwise tensor is.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

COLUMNS = [f"matrix_{i}{j}" for i in range(3) for j in range(3)]


def axis_tensor(axis: np.ndarray, ratio: float) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    n /= np.linalg.norm(n)
    projector = np.outer(n, n)
    return (ratio ** (-1.0 / 3.0) * (np.eye(3) - projector)
            + ratio ** (2.0 / 3.0) * projector)


def point_tensor(axis: np.ndarray, strength: float, distance: float, trend_range: float) -> np.ndarray:
    if strength <= 0 or trend_range <= 0 or distance < 0:
        raise ValueError("Invalid strength/range/distance")
    influence = np.exp(-distance / trend_range) if distance < 4.0 * trend_range else 0.0
    return axis_tensor(axis, 1.0 + (strength - 1.0) * influence)


def diagnose(matrix: np.ndarray, strength: float | None = None,
             trend_range: float | None = None) -> dict:
    a = np.asarray(matrix, dtype=float)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("Expected finite 3x3 tensor")
    symmetric = (a + a.T) * 0.5
    eigen, axes = np.linalg.eigh(symmetric)
    if eigen[0] <= 0:
        raise ValueError("Tensor must be positive definite")
    gap = np.diff(eigen)
    isotropic = float((eigen[-1] - eigen[0]) / eigen[-1]) < 1e-10
    if gap[0] <= gap[1]:
        axis_index, common = 2, float((eigen[0] + eigen[1]) * 0.5)
        uniaxial_error = float(gap[0] / eigen[-1])
    else:
        axis_index, common = 0, float((eigen[1] + eigen[2]) * 0.5)
        uniaxial_error = float(gap[1] / eigen[-1])
    normal = axes[:, axis_index].copy()
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    ratio = 1.0 if isotropic else float(eigen[axis_index] / common)
    result = {
        "det": float(np.linalg.det(symmetric)),
        "uniaxial_error": uniaxial_error,
        "isotropic": bool(isotropic),
        "axis_x": float(normal[0]) if not isotropic else float("nan"),
        "axis_y": float(normal[1]) if not isotropic else float("nan"),
        "axis_z": float(normal[2]) if not isotropic else float("nan"),
        "effective_ratio": ratio,
        "eigen_min": float(eigen[0]), "eigen_mid": float(eigen[1]),
        "eigen_max": float(eigen[2]),
    }
    if strength is not None:
        if strength <= 0 or strength == 1:
            raise ValueError("Known strength must be >0 and !=1")
        # An inversion is meaningful ONLY for a determinant-one uniaxial tensor.
        valid_single = uniaxial_error < 1e-7 and abs(result["det"] - 1.0) < 1e-6
        q = (ratio - 1.0) / (strength - 1.0) if valid_single else float("nan")
        result["inferred_influence_q"] = float(q)
        if trend_range is not None:
            if trend_range <= 0:
                raise ValueError("Known range must be positive")
            # Outside 4R, the influence is zero and only a distance bound is known.
            result["inferred_distance"] = (
                float(-trend_range * np.log(q)) if q > 0 and np.isfinite(q)
                else float("nan")
            )
    return result


def _matrix_log(a: np.ndarray) -> np.ndarray:
    eig, axes = np.linalg.eigh(a)
    if eig[0] <= 0:
        raise ValueError("Expected SPD matrix")
    return (axes * np.log(eig)) @ axes.T


def mixing_candidates(first: np.ndarray, second: np.ndarray, weight: float) -> dict[str, np.ndarray]:
    """Falsifiable candidate models for multi-mesh inputs; none is yet proven."""
    if not 0 <= weight <= 1:
        raise ValueError("Invalid weight")
    arithmetic = weight * first + (1.0 - weight) * second
    log_mean = weight * _matrix_log(first) + (1.0 - weight) * _matrix_log(second)
    eig, axes = np.linalg.eigh(log_mean)
    log_euclidean = (axes * np.exp(eig)) @ axes.T
    return {
        "arithmetic": arithmetic,
        "det_normalized_arithmetic": arithmetic / np.linalg.det(arithmetic) ** (1 / 3),
        "log_euclidean": log_euclidean,
    }


def inspect_csv(input_path: Path, output_path: Path,
                strength: float | None, trend_range: float | None) -> int:
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing matrix columns: {missing}")
        records = list(reader)
    if not records:
        raise ValueError("Empty input CSV")
    results = []
    for record in records:
        matrix = np.array([float(record[name]) for name in COLUMNS]).reshape(3, 3)
        results.append(diagnose(matrix, strength, trend_range))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row"] + list(results[0]))
        writer.writeheader()
        for index, result in enumerate(results):
            writer.writerow({"row": index, **result})
    return len(results)


def self_test() -> None:
    n = np.array([0.3, -0.6, 0.7416198487])
    n /= np.linalg.norm(n)
    for strength in (2.0, 3.0, 5.0):
        for distance in (0.0, 0.25, 50.0, 99.9, 399.9, 400.0, 500.0):
            result = diagnose(point_tensor(n, strength, distance, 100.0), strength, 100.0)
            assert abs(result["det"] - 1.0) < 1e-12
            assert result["uniaxial_error"] < 1e-12
            if distance < 400.0:
                axis = np.array([result["axis_x"], result["axis_y"], result["axis_z"]])
                assert abs(abs(axis @ n) - 1) < 1e-10
                assert abs(result["inferred_distance"] - distance) < 1e-5
            else:
                assert result["isotropic"] and np.isnan(result["inferred_distance"])
    a = point_tensor(np.array([1., 0., 0.]), 5.0, 0.0, 100.0)
    b = point_tensor(np.array([0.5, np.sqrt(3) / 2, 0.]), 3.0, 0.0, 100.0)
    for name, matrix in mixing_candidates(a, b, 0.4).items():
        result = diagnose(matrix)
        assert result["uniaxial_error"] > 0.01
        print(f"{name}: determinant={result['det']:.7f}, biaxial_signature={result['uniaxial_error']:.7f}")
    print("PASS: 21 pointwise tensors and 3 blend candidates, cutoff + inversion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--range", dest="trend_range", type=float)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.input is None:
        self_test()
    if args.input is not None:
        if args.output is None:
            parser.error("--output is required with --input")
        print(f"Rows: {inspect_csv(args.input, args.output, args.strength, args.trend_range)}")


if __name__ == "__main__":
    main()
