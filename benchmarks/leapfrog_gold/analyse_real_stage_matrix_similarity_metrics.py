"""Compare candidate SPD-matrix similarity metrics for the real SubDomainer stage.

The recovered determinant consistency has useful ordering but values are compressed
near one, so Leapfrog's 0.60 threshold cannot act directly on point-pair scores. This
diagnostic evaluates several affine-invariant, log-Euclidean, spectral, and powered
determinant similarities on the same within-coarse-domain Delaunay graph. It reports
ROC AUC, behaviour at the literal 0.60 threshold, and the best threshold that retains
at least 90% decoded-fragment connectivity. Production code is unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse_real_stage_edge_consistency as edge_base  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _spd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, np.finfo(float).eps)
    return values, vectors


def _matrix_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = _spd_eigh(matrix)
    return (vectors * np.log(values)) @ vectors.T


def _pair_scores(
    first: np.ndarray,
    second: np.ndarray,
    current_consistency: float,
) -> dict[str, float]:
    first_values, first_vectors = _spd_eigh(first)
    first_inverse_sqrt = (first_vectors * (first_values ** -0.5)) @ first_vectors.T
    relative = first_inverse_sqrt @ second @ first_inverse_sqrt
    relative_values, _ = _spd_eigh(relative)
    relative_logs = np.log(relative_values)

    affine_distance = float(np.linalg.norm(relative_logs))
    log_distance = float(np.linalg.norm(_matrix_log(first) - _matrix_log(second)))
    spectral_spread = float(np.max(relative_logs) - np.min(relative_logs))
    current = float(np.clip(current_consistency, 0.0, 1.0))

    return {
        "det_current": current,
        "det_sqrt": current ** 0.5,
        "det_pow16": current ** 16.0,
        "det_pow32": current ** 32.0,
        "det_pow64": current ** 64.0,
        "affine_exp": float(np.exp(-affine_distance)),
        "affine_exp_sq": float(np.exp(-0.5 * affine_distance * affine_distance)),
        "affine_inverse": 1.0 / (1.0 + affine_distance),
        "log_exp": float(np.exp(-log_distance)),
        "log_exp_sq": float(np.exp(-0.5 * log_distance * log_distance)),
        "log_inverse": 1.0 / (1.0 + log_distance),
        "spectral_ratio": float(np.exp(-spectral_spread)),
    }


def _evaluate_threshold(
    groups: list[dict[str, object]],
    metric: str,
    threshold: float,
    total_same: int,
) -> dict[str, float | int]:
    kept_total = 0
    kept_same = 0
    kept_cross = 0
    fragment_total = 0
    fragment_connected = 0
    split_excess = 0
    component_total = 0

    for group in groups:
        local_oracle = np.asarray(group["oracle"], dtype=np.int64)
        kept_edges: list[tuple[int, int]] = []
        for first, second, scores, same in group["edges"]:
            if float(scores[metric]) >= threshold:
                kept_edges.append((int(first), int(second)))
                kept_total += 1
                if bool(same):
                    kept_same += 1
                else:
                    kept_cross += 1

        all_indices = np.arange(len(local_oracle), dtype=np.int64)
        component_total += edge_base._component_count(all_indices, kept_edges)
        for oracle_value in np.unique(local_oracle):
            fragment_indices = np.flatnonzero(local_oracle == oracle_value)
            fragment_edges = [
                (first, second)
                for first, second in kept_edges
                if local_oracle[first] == oracle_value and local_oracle[second] == oracle_value
            ]
            components = edge_base._component_count(fragment_indices, fragment_edges)
            fragment_total += 1
            fragment_connected += int(components == 1)
            split_excess += max(components - 1, 0)

    return {
        "threshold": float(threshold),
        "kept": kept_total,
        "purity": kept_same / max(kept_total, 1),
        "recall": kept_same / max(total_same, 1),
        "connected": fragment_connected / max(fragment_total, 1),
        "split": split_excess,
        "components": component_total,
        "cross": kept_cross,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark")
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
        _normalise_determinant,
    )
    from sklearn.metrics import roc_auc_score

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(
        args.decoded_root, case_name
    )
    points = np.asarray(points, dtype=np.float64)
    oracle_labels = np.asarray(oracle_labels, dtype=np.int64)

    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = comparison._parse_case_parameters(case_name)
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        strength,
        trend_range,
    )

    builder = polatory.AutomaticStructuralDomainBuilder3(
        centroid_count=args.centroid_count,
        minimum_cluster_fraction=args.minimum_fraction,
        maximum_cluster_fraction=args.maximum_fraction,
        consistency_threshold=args.threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, _, shape, centroids = builder._prepare_grid(points)
    centroid_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    point_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        points, trend_input, non_decaying=False
    )
    point_matrices = np.asarray(
        [_normalise_determinant(matrix) for matrix in point_matrices], dtype=float
    )
    coarse_labels, _, _, _, coarse_merges = builder._automatic_labels(
        points,
        np.asarray(centroid_matrices, dtype=np.float64),
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        tuple(int(value) for value in shape),
    )
    coarse_labels = np.asarray(coarse_labels, dtype=np.int64)

    groups: list[dict[str, object]] = []
    flat_scores: dict[str, list[float]] = {}
    flat_same: list[bool] = []
    for coarse_value in np.unique(coarse_labels):
        global_indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[global_indices]
        local_oracle = oracle_labels[global_indices]
        local_edges = edge_base._delaunay_edges(local_points)
        scored_edges: list[tuple[int, int, dict[str, float], bool]] = []
        for first, second in local_edges:
            first_matrix = point_matrices[global_indices[first]]
            second_matrix = point_matrices[global_indices[second]]
            _, current = _merged_matrix_and_consistency(
                first_matrix, 1, second_matrix, 1
            )
            scores = _pair_scores(first_matrix, second_matrix, float(current))
            same = bool(local_oracle[first] == local_oracle[second])
            scored_edges.append((first, second, scores, same))
            for name, value in scores.items():
                flat_scores.setdefault(name, []).append(float(value))
            flat_same.append(same)
        groups.append(
            {
                "coarse": int(coarse_value),
                "oracle": local_oracle,
                "edges": scored_edges,
            }
        )

    same_flags = np.asarray(flat_same, dtype=bool)
    total_same = int(np.count_nonzero(same_flags))
    total_cross = int(len(same_flags) - total_same)
    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    print(
        f"Delaunay edges={len(same_flags)} same={total_same} cross={total_cross} "
        f"same_fraction={total_same / max(len(same_flags), 1):.5f}"
    )
    print("\nMetric comparison:")
    print(
        " metric              AUC   score_same50 score_cross50 | "
        "at0.60[purity recall connected split cross] | "
        "best90[threshold purity recall connected split cross]"
    )

    rows: list[tuple[float, str, str]] = []
    for metric, values_list in flat_scores.items():
        values = np.asarray(values_list, dtype=float)
        auc = float(roc_auc_score(same_flags.astype(np.int8), values))
        same_values = values[same_flags]
        cross_values = values[~same_flags]
        literal = _evaluate_threshold(groups, metric, 0.60, total_same)

        quantiles = np.linspace(0.0, 1.0, 101)
        thresholds = sorted(
            set([0.60] + [float(np.quantile(values, q)) for q in quantiles])
        )
        candidates = [
            _evaluate_threshold(groups, metric, threshold, total_same)
            for threshold in thresholds
        ]
        feasible = [row for row in candidates if float(row["connected"]) >= 0.90]
        if feasible:
            best = max(
                feasible,
                key=lambda row: (
                    float(row["purity"]),
                    -int(row["cross"]),
                    float(row["recall"]),
                ),
            )
        else:
            best = max(candidates, key=lambda row: float(row["connected"]))

        text = (
            f" {metric:<18s} {auc:>5.3f} "
            f"{float(np.median(same_values)):>12.6f} {float(np.median(cross_values)):>13.6f} | "
            f"{float(literal['purity']):.3f} {float(literal['recall']):.3f} "
            f"{float(literal['connected']):.3f} {int(literal['split']):>3d} {int(literal['cross']):>4d} | "
            f"{float(best['threshold']):.6f} {float(best['purity']):.3f} "
            f"{float(best['recall']):.3f} {float(best['connected']):.3f} "
            f"{int(best['split']):>3d} {int(best['cross']):>4d}"
        )
        rows.append((auc, metric, text))

    for _, _, text in sorted(rows, key=lambda item: item[0], reverse=True):
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
