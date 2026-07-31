"""Test global weighted-graph partitioning at the real SubDomainer stage.

The pairwise and dynamically accumulated determinant-consistency values remain too
close to one for a literal threshold to split the real-location graph. This diagnostic
tests whether the same signal becomes useful when the whole within-coarse-domain
Delaunay graph is partitioned jointly.

Tests suffixed ``oracle_k`` use the decoded number of oracle intersections only as a
recoverability upper bound. ``spectral_eigengap`` chooses its own cluster count.
Production code is unchanged.
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

import analyse_real_stage_dynamic_cluster_merging as dynamic  # noqa: E402
import analyse_real_stage_edge_consistency as edge_base  # noqa: E402
import analyse_real_stage_structural_edge_ranking as ranking  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.quantile(values, [0.05, 0.95])
    if high - low <= 1e-15:
        low, high = float(np.min(values)), float(np.max(values))
    if high - low <= 1e-15:
        return np.ones_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _weights(rows: list[dict[str, float | int]]) -> dict[str, np.ndarray]:
    similarity = np.asarray([float(row["similarity"]) for row in rows], dtype=float)
    first = np.asarray(
        [float(row["similarity_pct_first"]) for row in rows], dtype=float
    )
    second = np.asarray(
        [float(row["similarity_pct_second"]) for row in rows], dtype=float
    )
    both = np.minimum(first, second)
    either = np.maximum(first, second)
    mean = 0.5 * (first + second)

    incident: dict[int, list[int]] = {}
    for edge_index, row in enumerate(rows):
        incident.setdefault(int(row["first"]), []).append(edge_index)
        incident.setdefault(int(row["second"]), []).append(edge_index)
    chosen: dict[int, set[int]] = {}
    for node, edge_indices in incident.items():
        ordered = sorted(
            edge_indices,
            key=lambda index: float(rows[index]["similarity"]),
            reverse=True,
        )
        chosen[node] = set(ordered[: min(6, len(ordered))])
    top6 = np.asarray(
        [
            float(
                edge_index in chosen[int(row["first"])]
                or edge_index in chosen[int(row["second"])]
            )
            for edge_index, row in enumerate(rows)
        ],
        dtype=float,
    )

    relative = _robust_scale(similarity)
    return {
        "raw_similarity": similarity,
        "relative_similarity": relative,
        "endpoint_both": both,
        "endpoint_mean": mean,
        "endpoint_either": either,
        "top6_binary": top6,
        "top6_relative": top6 * relative,
    }


def _affinity(
    count: int,
    rows: list[dict[str, float | int]],
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros((count, count), dtype=float)
    for row, weight in zip(rows, np.asarray(weights, dtype=float)):
        first, second = int(row["first"]), int(row["second"])
        value = max(float(weight), 1e-9)
        result[first, second] = max(result[first, second], value)
        result[second, first] = result[first, second]
    return result


def _eigensystem(affinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(affinity)
    degrees = np.sum(affinity, axis=1)
    inverse = np.zeros_like(degrees)
    valid = degrees > np.finfo(float).eps
    inverse[valid] = degrees[valid] ** -0.5
    normalised = inverse[:, None] * affinity * inverse[None, :]
    laplacian = np.eye(count) - normalised
    values, vectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _spectral_labels(
    affinity: np.ndarray,
    clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    count = len(affinity)
    clusters = min(max(int(clusters), 1), count)
    values, vectors = _eigensystem(affinity)
    if clusters == 1:
        return np.zeros(count, dtype=np.int64), values

    embedding = np.asarray(vectors[:, :clusters], dtype=float)
    norms = np.linalg.norm(embedding, axis=1)
    valid = norms > np.finfo(float).eps
    embedding[valid] /= norms[valid, None]
    labels = KMeans(
        n_clusters=clusters,
        n_init=30,
        random_state=0,
    ).fit_predict(embedding)
    return np.asarray(labels, dtype=np.int64), values


def _auto_k(values: np.ndarray, maximum: int) -> tuple[int, float]:
    if len(values) <= 1:
        return 1, 0.0
    maximum = min(max(int(maximum), 1), len(values) - 1)
    gaps = values[1 : maximum + 1] - values[:maximum]
    best = int(np.argmax(gaps))
    return best + 1, float(gaps[best])


def _matrix_features(matrices: np.ndarray) -> np.ndarray:
    features: list[np.ndarray] = []
    indices = np.triu_indices(3)
    for matrix in matrices:
        values, vectors = np.linalg.eigh(
            0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        )
        values = np.maximum(values, np.finfo(float).eps)
        log_matrix = (vectors * np.log(values)) @ vectors.T
        features.append(log_matrix[indices])
    result = np.asarray(features, dtype=float)
    std = np.std(result, axis=0)
    std[std <= 1e-12] = 1.0
    return (result - np.mean(result, axis=0)) / std


def _format(row: dict[str, object]) -> str:
    return (
        f"{str(row['method']):<22s} {str(row['weight']):<20s} "
        f"{int(row['predicted_domains']):>7d} "
        f"{float(row['weighted_purity']):>6.3f} "
        f"{float(row['fragments_connected']):>9.3f} "
        f"{float(row['largest_fraction']):>7.3f} "
        f"{int(row['split_excess']):>5d} "
        f"{int(row['impure_domains']):>6d} "
        f"{float(row['pair_precision']):>5.3f} "
        f"{float(row['pair_recall']):>5.3f} "
        f"{float(row['pair_f1']):>6.3f} "
        f"{float(row['ari']):>5.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark")
    )
    parser.add_argument("--coarse-threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-auto-clusters", type=int, default=8)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import _normalise_determinant
    from sklearn.cluster import KMeans

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
        consistency_threshold=args.coarse_threshold,
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
    for coarse_value in np.unique(coarse_labels):
        indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[indices]
        local_matrices = point_matrices[indices]
        truth = oracle_labels[indices]
        edges = edge_base._delaunay_edges(local_points)
        rows = ranking._edge_features(local_points, local_matrices, edges)
        groups.append(
            {
                "coarse": int(coarse_value),
                "truth": truth,
                "matrices": local_matrices,
                "rows": rows,
                "weights": _weights(rows),
                "oracle_k": int(len(np.unique(truth))),
            }
        )

    truths = [np.asarray(group["truth"], dtype=np.int64) for group in groups]
    weight_names = tuple(groups[0]["weights"].keys()) if groups else ()
    upper: list[dict[str, object]] = []
    automatic: list[dict[str, object]] = []
    auto_details: dict[str, list[tuple[int, int, int, float]]] = {}

    for weight_name in weight_names:
        oracle_predictions: list[np.ndarray] = []
        auto_predictions: list[np.ndarray] = []
        details: list[tuple[int, int, int, float]] = []

        for group in groups:
            count = len(group["truth"])
            affinity = _affinity(
                count,
                list(group["rows"]),
                np.asarray(group["weights"][weight_name], dtype=float),
            )
            oracle_k = int(group["oracle_k"])
            oracle_labels_local, eigenvalues = _spectral_labels(affinity, oracle_k)
            selected_k, gap = _auto_k(eigenvalues, args.maximum_auto_clusters)
            automatic_labels_local, _ = _spectral_labels(affinity, selected_k)
            oracle_predictions.append(oracle_labels_local)
            auto_predictions.append(automatic_labels_local)
            details.append(
                (int(group["coarse"]), oracle_k, selected_k, gap)
            )

        metrics = dynamic._evaluate(truths, oracle_predictions)
        metrics.update({"method": "spectral_oracle_k", "weight": weight_name})
        upper.append(metrics)

        metrics = dynamic._evaluate(truths, auto_predictions)
        metrics.update({"method": "spectral_eigengap", "weight": weight_name})
        automatic.append(metrics)
        auto_details[weight_name] = details

    matrix_predictions: list[np.ndarray] = []
    for group in groups:
        features = _matrix_features(np.asarray(group["matrices"], dtype=float))
        oracle_k = int(group["oracle_k"])
        if oracle_k == 1:
            labels = np.zeros(len(features), dtype=np.int64)
        else:
            labels = KMeans(
                n_clusters=oracle_k,
                n_init=30,
                random_state=0,
            ).fit_predict(features)
        matrix_predictions.append(np.asarray(labels, dtype=np.int64))
    matrix_metrics = dynamic._evaluate(truths, matrix_predictions)
    matrix_metrics.update({"method": "matrix_kmeans_oracle_k", "weight": "log_SPD"})
    upper.append(matrix_metrics)

    baseline = dynamic._evaluate(
        truths,
        [np.zeros(len(truth), dtype=np.int64) for truth in truths],
    )

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    target = int(upper[0]["oracle_fragments"]) if upper else 0
    print(f"Evaluation target: {target} oracle intersections inside coarse domains.")
    print(
        "Coarse baseline: "
        f"domains={int(baseline['predicted_domains'])} "
        f"purity={float(baseline['weighted_purity']):.4f} "
        f"pairF1={float(baseline['pair_f1']):.4f} "
        f"ARI={float(baseline['ari']):.4f}"
    )

    header = (
        " method                 weight               domains purity connected largest split "
        "impure pairP pairR pairF1   ARI"
    )
    print("\nOracle-k recoverability upper bounds:")
    print(header)
    for row in sorted(
        upper,
        key=lambda item: (
            float(item["ari"]),
            float(item["pair_f1"]),
            float(item["weighted_purity"]),
        ),
        reverse=True,
    ):
        print(" " + _format(row))

    print("\nAutomatic spectral eigengap partitions:")
    print(header)
    for row in sorted(
        automatic,
        key=lambda item: (
            float(item["ari"]),
            float(item["pair_f1"]),
            float(item["weighted_purity"]),
        ),
        reverse=True,
    ):
        print(" " + _format(row))

    if automatic:
        best = max(
            automatic,
            key=lambda item: (
                float(item["ari"]),
                float(item["pair_f1"]),
            ),
        )
        weight_name = str(best["weight"])
        print("\nBest automatic candidate per coarse domain:")
        print(" coarse oracle_k predicted_k eigengap")
        for coarse, oracle_k, predicted_k, gap in auto_details[weight_name]:
            print(
                f" {coarse:>6d} {oracle_k:>8d} {predicted_k:>11d} {gap:>9.6f}"
            )

    best_upper = max(
        upper,
        key=lambda item: (
            float(item["ari"]),
            float(item["pair_f1"]),
        ),
    )
    print("\nInterpretation gate:")
    print(
        f" best_upper={best_upper['method']}/{best_upper['weight']} "
        f"ARI={float(best_upper['ari']):.4f} "
        f"pairF1={float(best_upper['pair_f1']):.4f} "
        f"purity={float(best_upper['weighted_purity']):.4f}"
    )
    if float(best_upper["ari"]) >= 0.85:
        print(
            " graph or matrix structure is sufficient when k is known; automatic "
            "model selection is the main missing mechanism."
        )
    elif float(best_upper["ari"]) >= 0.70:
        print(
            " the recovered signal is partially sufficient, but the partition "
            "objective or edge weighting still differs from Leapfrog."
        )
    else:
        print(
            " even oracle-k partitioning is weak; the recovered point matrices or "
            "candidate topology are missing a stronger Leapfrog signal."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
