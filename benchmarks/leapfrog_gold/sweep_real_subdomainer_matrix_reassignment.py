"""Test whether Leapfrog redraws coarse-domain boundaries by matrix reassignment.

The decoded final partitions cross-cut the transferred GridSeededDomainer labels.  A
possible explanation is that Leapfrog uses the coarse domains only as initial matrix
seeds, then reassigns real locations to the most compatible nearby domain matrix.
This diagnostic keeps the production builder unchanged and sweeps iterative
matrix-consistency reassignment with several spatial neighbourhoods and penalties.
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

# Importing this module patches the strict coordinate/label loader in comparison.
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _relabel(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    mapping = {int(value): index for index, value in enumerate(values)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def _domain_statistics(
    points: np.ndarray,
    point_matrices: np.ndarray,
    labels: np.ndarray,
    normalise,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.unique(labels)
    matrices = np.empty((len(values), 3, 3), dtype=float)
    centroids = np.empty((len(values), 3), dtype=float)
    sizes = np.empty(len(values), dtype=np.int64)
    for output_index, value in enumerate(values):
        indices = np.flatnonzero(labels == value)
        sizes[output_index] = len(indices)
        centroids[output_index] = points[indices].mean(axis=0)
        matrices[output_index] = normalise(point_matrices[indices].mean(axis=0))
    return values, matrices, centroids, sizes


def _run_reassignment(
    points: np.ndarray,
    point_matrices: np.ndarray,
    initial_labels: np.ndarray,
    *,
    nearest_domains: int,
    spatial_penalty: float,
    iterations: int,
    threshold: float,
    normalise,
    merged_score,
) -> tuple[np.ndarray, int, list[int]]:
    labels = _relabel(np.asarray(initial_labels, dtype=np.int64))
    total_changes = 0
    domain_history = [int(len(np.unique(labels)))]

    for _ in range(max(1, int(iterations))):
        values, matrices, centroids, _ = _domain_statistics(
            points, point_matrices, labels, normalise
        )
        label_to_position = {int(value): index for index, value in enumerate(values)}
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        own_positions = np.asarray(
            [label_to_position[int(value)] for value in labels], dtype=np.int64
        )
        own_distances = distances[np.arange(len(points)), own_positions]
        positive = own_distances[own_distances > 0.0]
        if len(positive):
            spatial_scale = float(np.median(positive))
        else:
            spans = np.ptp(points, axis=0)
            spatial_scale = float(np.linalg.norm(spans)) / max(len(values), 1)
        spatial_scale = max(spatial_scale, np.finfo(float).eps)

        new_labels = labels.copy()
        for point_index in range(len(points)):
            if nearest_domains <= 0 or nearest_domains >= len(values):
                candidates = np.arange(len(values), dtype=np.int64)
            else:
                candidates = np.argpartition(
                    distances[point_index], nearest_domains - 1
                )[:nearest_domains]
                own = own_positions[point_index]
                if own not in candidates:
                    candidates = np.append(candidates, own)

            best_position = own_positions[point_index]
            _, current_consistency = merged_score(
                matrices[best_position], 1, point_matrices[point_index], 1
            )
            best_score = float(current_consistency) - spatial_penalty * (
                distances[point_index, best_position] / spatial_scale
            ) ** 2

            for candidate in candidates:
                candidate = int(candidate)
                _, consistency = merged_score(
                    matrices[candidate], 1, point_matrices[point_index], 1
                )
                consistency = float(consistency)
                if not np.isfinite(consistency) or consistency < threshold:
                    continue
                score = consistency - spatial_penalty * (
                    distances[point_index, candidate] / spatial_scale
                ) ** 2
                if score > best_score + 1.0e-14 or (
                    abs(score - best_score) <= 1.0e-14
                    and int(values[candidate]) < int(values[best_position])
                ):
                    best_score = score
                    best_position = candidate

            new_labels[point_index] = int(values[best_position])

        new_labels = _relabel(new_labels)
        changes = int(np.count_nonzero(new_labels != labels))
        total_changes += changes
        labels = new_labels
        domain_history.append(int(len(np.unique(labels))))
        if changes == 0:
            break

    return labels, total_changes, domain_history


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
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
        _normalise_determinant,
    )
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(
        args.decoded_root, case_name
    )
    points = np.asarray(points, dtype=np.float64)
    oracle_labels = np.asarray(oracle_labels, dtype=np.int64)
    oracle_domain_count = int(len(np.unique(oracle_labels)))

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

    def metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(accuracy),
            int(matched),
        )

    coarse_ari, coarse_ami, coarse_match, coarse_matched = metrics(coarse_labels)
    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={oracle_domain_count} coarse_domains={len(np.unique(coarse_labels))}"
    )
    print(
        f"coarse: merges={coarse_merges} ARI={coarse_ari:.5f} AMI={coarse_ami:.5f} "
        f"match={coarse_match:.5f} ({coarse_matched}/{len(points)})"
    )

    neighbourhoods = (0, 2, 3, 4, 6, 8)
    penalties = (0.0, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0)
    iteration_counts = (1, 2, 4, 8)
    rows: list[dict[str, object]] = []

    for nearest_domains in neighbourhoods:
        for spatial_penalty in penalties:
            for iterations in iteration_counts:
                labels, changes, history = _run_reassignment(
                    points,
                    point_matrices,
                    coarse_labels,
                    nearest_domains=nearest_domains,
                    spatial_penalty=spatial_penalty,
                    iterations=iterations,
                    threshold=args.threshold,
                    normalise=_normalise_determinant,
                    merged_score=_merged_matrix_and_consistency,
                )
                ari, ami, match, matched = metrics(labels)
                rows.append(
                    {
                        "nearest": nearest_domains,
                        "penalty": spatial_penalty,
                        "iterations": iterations,
                        "domains": int(len(np.unique(labels))),
                        "changes": changes,
                        "history": history,
                        "ari": ari,
                        "ami": ami,
                        "match": match,
                        "matched": matched,
                    }
                )

    def format_row(row: dict[str, object]) -> str:
        neighbourhood = "all" if int(row["nearest"]) == 0 else str(int(row["nearest"]))
        return (
            f"near={neighbourhood:>3s} penalty={float(row['penalty']):>4.2f} "
            f"iter={int(row['iterations']):>2d} domains={int(row['domains']):>3d} "
            f"changes={int(row['changes']):>5d} history={row['history']} "
            f"ARI={float(row['ari']):.5f} AMI={float(row['ami']):.5f} "
            f"match={float(row['match']):.5f} ({int(row['matched'])}/{len(points)})"
        )

    exact_count = [row for row in rows if int(row["domains"]) == oracle_domain_count]
    print(f"\nCandidates with exactly {oracle_domain_count} domains, ranked by ARI:")
    if exact_count:
        for row in sorted(exact_count, key=lambda item: float(item["ari"]), reverse=True)[
            : max(args.top, 1)
        ]:
            print(format_row(row))
    else:
        print("none")

    print(f"\nTop {max(args.top, 1)} candidates by ARI:")
    for row in sorted(rows, key=lambda item: float(item["ari"]), reverse=True)[
        : max(args.top, 1)
    ]:
        print(format_row(row))

    print(f"\nTop {max(args.top, 1)} candidates by optimal label match:")
    for row in sorted(rows, key=lambda item: float(item["match"]), reverse=True)[
        : max(args.top, 1)
    ]:
        print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
