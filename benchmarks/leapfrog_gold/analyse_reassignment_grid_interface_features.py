"""Analyse coarse-grid interface features for the post-reassignment merge.

Real-point Delaunay boundary statistics do not uniquely identify the oracle-best merge.
The recovered Leapfrog architecture, however, retains a six-connected centroid grid
from the first GridSeededDomainer stage.  This diagnostic tests whether the final
merge is ranked more naturally by the shared interface on that original grid.
Production code is unchanged and oracle metrics are used only to identify the pair
whose merge best matches the decoded partition.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_matrix_reassignment as reassignment  # noqa: E402


def _merge_pair(labels: np.ndarray, first: int, second: int) -> np.ndarray:
    merged = np.asarray(labels, dtype=np.int64).copy()
    merged[merged == second] = first
    return reassignment._relabel(merged)


def _grid_edges_with_axis(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    edge_blocks: list[np.ndarray] = []
    axis_blocks: list[np.ndarray] = []
    for axis, size in enumerate(shape):
        if size <= 1:
            continue
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[axis] = slice(0, size - 1)
        right[axis] = slice(1, size)
        block = np.column_stack([grid[tuple(left)].ravel(), grid[tuple(right)].ravel()])
        edge_blocks.append(block)
        axis_blocks.append(np.full(len(block), axis, dtype=np.int64))
    if not edge_blocks:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(edge_blocks), np.concatenate(axis_blocks)


def _rank(rows: list[dict[str, object]], target: dict[str, object], key: str, *, reverse: bool) -> int:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    target_pair = tuple(target["pair"])
    return next(index for index, row in enumerate(ordered, start=1) if tuple(row["pair"]) == target_pair)


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
    parser.add_argument("--nearest-domains", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--spatial-penalty", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=15)
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
    shape = tuple(int(value) for value in shape)
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    centroid_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    point_matrices = exact.exact_leapfrog_single_input_anisotropies3(
        points, trend_input, non_decaying=False
    )
    centroid_matrices = np.asarray(
        [_normalise_determinant(matrix) for matrix in centroid_matrices], dtype=float
    )
    point_matrices = np.asarray(
        [_normalise_determinant(matrix) for matrix in point_matrices], dtype=float
    )

    coarse_labels, centroid_labels, _, _, coarse_merges = builder._automatic_labels(
        points,
        centroid_matrices,
        minimum,
        maximum,
        shape,
    )
    coarse_labels = reassignment._relabel(np.asarray(coarse_labels, dtype=np.int64))
    centroid_labels = reassignment._relabel(np.asarray(centroid_labels, dtype=np.int64))

    reassigned_labels, changes, history = reassignment._run_reassignment(
        points,
        point_matrices,
        coarse_labels,
        nearest_domains=args.nearest_domains,
        spatial_penalty=args.spatial_penalty,
        iterations=args.iterations,
        threshold=args.threshold,
        normalise=_normalise_determinant,
        merged_score=_merged_matrix_and_consistency,
    )
    reassigned_labels = reassignment._relabel(reassigned_labels)

    def metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
        accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, labels)
        return (
            float(adjusted_rand_score(oracle_labels, labels)),
            float(adjusted_mutual_info_score(oracle_labels, labels)),
            float(accuracy),
            int(matched),
        )

    coarse_metrics = metrics(coarse_labels)
    reassigned_metrics = metrics(reassigned_labels)

    values, domain_matrices, domain_centroids, point_sizes = reassignment._domain_statistics(
        points, point_matrices, reassigned_labels, _normalise_determinant
    )
    values = np.asarray(values, dtype=np.int64)
    value_to_position = {int(value): index for index, value in enumerate(values)}
    centroid_sizes = np.asarray(
        [np.count_nonzero(centroid_labels == value) for value in values], dtype=np.int64
    )

    edges, edge_axes = _grid_edges_with_axis(shape)
    spans = maximum - minimum
    cell_sizes = np.divide(spans, np.asarray(shape, dtype=float), out=np.ones(3), where=np.asarray(shape) > 0)
    face_areas = np.asarray(
        [cell_sizes[1] * cell_sizes[2], cell_sizes[0] * cell_sizes[2], cell_sizes[0] * cell_sizes[1]],
        dtype=float,
    )

    pair_edges: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for (first_index, second_index), axis in zip(edges, edge_axes, strict=True):
        first_label = int(centroid_labels[int(first_index)])
        second_label = int(centroid_labels[int(second_index)])
        if first_label == second_label:
            continue
        pair = tuple(sorted((first_label, second_label)))
        pair_edges[pair].append((int(first_index), int(second_index), int(axis)))

    rows: list[dict[str, object]] = []
    for first_value, second_value in combinations((int(value) for value in values), 2):
        first_position = value_to_position[first_value]
        second_position = value_to_position[second_value]
        pair = tuple(sorted((first_value, second_value)))
        interface = pair_edges.get(pair, [])

        _, whole_consistency = _merged_matrix_and_consistency(
            domain_matrices[first_position],
            int(point_sizes[first_position]),
            domain_matrices[second_position],
            int(point_sizes[second_position]),
        )
        distance = float(
            np.linalg.norm(domain_centroids[first_position] - domain_centroids[second_position])
        )

        if interface:
            first_boundary: list[int] = []
            second_boundary: list[int] = []
            edge_consistencies: list[float] = []
            axis_counts = np.zeros(3, dtype=np.int64)
            shared_area = 0.0
            for first_index, second_index, axis in interface:
                if int(centroid_labels[first_index]) == first_value:
                    first_boundary.append(first_index)
                    second_boundary.append(second_index)
                else:
                    first_boundary.append(second_index)
                    second_boundary.append(first_index)
                axis_counts[axis] += 1
                shared_area += float(face_areas[axis])
                _, consistency = _merged_matrix_and_consistency(
                    centroid_matrices[first_index], 1, centroid_matrices[second_index], 1
                )
                edge_consistencies.append(float(consistency))

            first_unique = np.unique(np.asarray(first_boundary, dtype=np.int64))
            second_unique = np.unique(np.asarray(second_boundary, dtype=np.int64))
            first_matrix = _normalise_determinant(centroid_matrices[first_unique].mean(axis=0))
            second_matrix = _normalise_determinant(centroid_matrices[second_unique].mean(axis=0))
            _, boundary_consistency = _merged_matrix_and_consistency(
                first_matrix, len(first_unique), second_matrix, len(second_unique)
            )
            first_fraction = len(first_unique) / max(int(centroid_sizes[first_position]), 1)
            second_fraction = len(second_unique) / max(int(centroid_sizes[second_position]), 1)
            boundary_fraction_min = min(first_fraction, second_fraction)
            boundary_fraction_hmean = (
                0.0
                if first_fraction <= 0.0 or second_fraction <= 0.0
                else 2.0 * first_fraction * second_fraction / (first_fraction + second_fraction)
            )
            edge_mean = float(np.mean(edge_consistencies))
            edge_min = float(np.min(edge_consistencies))
            edge_median = float(np.median(edge_consistencies))
        else:
            axis_counts = np.zeros(3, dtype=np.int64)
            shared_area = 0.0
            boundary_consistency = float("-inf")
            boundary_fraction_min = 0.0
            boundary_fraction_hmean = 0.0
            edge_mean = float("-inf")
            edge_min = float("-inf")
            edge_median = float("-inf")

        merged_labels = _merge_pair(reassigned_labels, first_value, second_value)
        ari, ami, match, matched = metrics(merged_labels)
        rows.append(
            {
                "pair": pair,
                "point_sizes": (int(point_sizes[first_position]), int(point_sizes[second_position])),
                "centroid_sizes": (int(centroid_sizes[first_position]), int(centroid_sizes[second_position])),
                "whole_consistency": float(whole_consistency),
                "centroid_distance": distance,
                "grid_faces": int(len(interface)),
                "grid_area": float(shared_area),
                "axis_x": int(axis_counts[0]),
                "axis_y": int(axis_counts[1]),
                "axis_z": int(axis_counts[2]),
                "grid_boundary_consistency": float(boundary_consistency),
                "grid_edge_consistency_mean": edge_mean,
                "grid_edge_consistency_median": edge_median,
                "grid_edge_consistency_min": edge_min,
                "boundary_fraction_min": float(boundary_fraction_min),
                "boundary_fraction_hmean": float(boundary_fraction_hmean),
                "area_consistency": float(shared_area) * max(float(boundary_consistency), 0.0),
                "faces_consistency": float(len(interface)) * max(edge_mean, 0.0),
                "ari": ari,
                "ami": ami,
                "match": match,
                "matched": matched,
            }
        )

    adjacent_rows = [row for row in rows if int(row["grid_faces"]) > 0]
    oracle_best = max(rows, key=lambda row: float(row["ari"]))

    def format_row(row: dict[str, object]) -> str:
        first_size, second_size = row["point_sizes"]
        first_grid, second_grid = row["centroid_sizes"]
        return (
            f"pair={row['pair']} points=({first_size},{second_size}) grid=({first_grid},{second_grid}) "
            f"whole={float(row['whole_consistency']):.6f} distance={float(row['centroid_distance']):.2f} "
            f"faces={int(row['grid_faces']):4d} area={float(row['grid_area']):10.2f} "
            f"axes=({int(row['axis_x'])},{int(row['axis_y'])},{int(row['axis_z'])}) "
            f"bfmin={float(row['boundary_fraction_min']):.4f} "
            f"bfhm={float(row['boundary_fraction_hmean']):.4f} "
            f"bc={float(row['grid_boundary_consistency']):.6f} "
            f"ecmean={float(row['grid_edge_consistency_mean']):.6f} "
            f"ecmin={float(row['grid_edge_consistency_min']):.6f} "
            f"ARI={float(row['ari']):.5f} match={float(row['match']):.5f}"
        )

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} coarse_domains={len(np.unique(coarse_labels))}"
    )
    print(
        f"coarse: merges={coarse_merges} ARI={coarse_metrics[0]:.5f} "
        f"AMI={coarse_metrics[1]:.5f} match={coarse_metrics[2]:.5f} "
        f"({coarse_metrics[3]}/{len(points)})"
    )
    print(
        f"reassigned: changes={changes} history={history} domains={len(values)} "
        f"ARI={reassigned_metrics[0]:.5f} AMI={reassigned_metrics[1]:.5f} "
        f"match={reassigned_metrics[2]:.5f} ({reassigned_metrics[3]}/{len(points)})"
    )

    print("\nOracle-best pair and coarse-grid feature ranks:")
    print(format_row(oracle_best))
    if int(oracle_best["grid_faces"]) == 0:
        print("  not adjacent on the six-connected coarse centroid grid")
    else:
        rank_specs = (
            ("whole_consistency", True),
            ("grid_faces", True),
            ("grid_area", True),
            ("boundary_fraction_min", True),
            ("boundary_fraction_hmean", True),
            ("grid_boundary_consistency", True),
            ("grid_edge_consistency_mean", True),
            ("grid_edge_consistency_min", True),
            ("area_consistency", True),
            ("faces_consistency", True),
            ("centroid_distance", False),
        )
        for key, reverse in rank_specs:
            print(
                f"  {key:31s} value={float(oracle_best[key]):12.6f} "
                f"rank={_rank(adjacent_rows, oracle_best, key, reverse=reverse)}/{len(adjacent_rows)}"
            )

    top = max(args.top, 1)
    reports = (
        ("grid shared-face count", "grid_faces", True),
        ("grid shared area", "grid_area", True),
        ("grid boundary consistency", "grid_boundary_consistency", True),
        ("grid edge-consistency mean", "grid_edge_consistency_mean", True),
        ("grid boundary-fraction harmonic mean", "boundary_fraction_hmean", True),
        ("oracle ARI", "ari", True),
    )
    for title, key, reverse in reports:
        print(f"\nTop {top} grid-adjacent pairs by {title}:")
        for row in sorted(adjacent_rows, key=lambda item: float(item[key]), reverse=reverse)[:top]:
            print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
