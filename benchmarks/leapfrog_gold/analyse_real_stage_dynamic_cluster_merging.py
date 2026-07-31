"""Test cluster-level consistency merging for Leapfrog's real SubDomainer stage.

Pairwise point-matrix scores are compressed near one, but Leapfrog's consistency
threshold may be evaluated after neighbouring point matrices have been accumulated
inside growing clusters.  This diagnostic keeps the within-coarse-domain Delaunay
topology and dynamically recomputes consistency after every cluster merge.  It also
tests endpoint-relative similarity backbones.  Oracle labels are used only for
evaluation; production code is unchanged.
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse_real_stage_edge_consistency as edge_base  # noqa: E402
import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402


def _pair_consistency(
    first_matrix: np.ndarray,
    first_count: int,
    second_matrix: np.ndarray,
    second_count: int,
) -> tuple[np.ndarray, float]:
    from polatory.leapfrog_automatic_domain_builder import (
        _merged_matrix_and_consistency,
    )

    merged, consistency = _merged_matrix_and_consistency(
        np.asarray(first_matrix, dtype=float),
        int(first_count),
        np.asarray(second_matrix, dtype=float),
        int(second_count),
    )
    return np.asarray(merged, dtype=float), float(consistency)


def _endpoint_topk_edges(
    matrices: np.ndarray,
    edges: list[tuple[int, int]],
    k: int,
) -> list[tuple[int, int]]:
    incident: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for edge_index, (first, second) in enumerate(edges):
        _, score = _pair_consistency(
            matrices[int(first)], 1, matrices[int(second)], 1
        )
        incident[int(first)].append((score, edge_index))
        incident[int(second)].append((score, edge_index))

    chosen: dict[int, set[int]] = {}
    for node, values in incident.items():
        ordered = sorted(values, key=lambda item: item[0], reverse=True)
        chosen[node] = {
            edge_index for _, edge_index in ordered[: min(max(int(k), 1), len(ordered))]
        }

    kept: list[tuple[int, int]] = []
    for edge_index, (first, second) in enumerate(edges):
        if edge_index in chosen[int(first)] or edge_index in chosen[int(second)]:
            kept.append((int(first), int(second)))
    return kept


class _DynamicAgglomerator:
    def __init__(
        self,
        matrices: np.ndarray,
        edges: list[tuple[int, int]],
    ) -> None:
        self.count = len(matrices)
        self.parent = np.arange(self.count, dtype=np.int64)
        self.active = np.ones(self.count, dtype=bool)
        self.versions = np.zeros(self.count, dtype=np.int64)
        self.cluster_counts = np.ones(self.count, dtype=np.int64)
        self.cluster_matrices = [
            np.asarray(matrix, dtype=float).copy() for matrix in matrices
        ]
        self.neighbours: list[set[int]] = [set() for _ in range(self.count)]
        for first, second in edges:
            first_i, second_i = int(first), int(second)
            if first_i == second_i:
                continue
            self.neighbours[first_i].add(second_i)
            self.neighbours[second_i].add(first_i)

    def find(self, value: int) -> int:
        value_i = int(value)
        while int(self.parent[value_i]) != value_i:
            self.parent[value_i] = self.parent[int(self.parent[value_i])]
            value_i = int(self.parent[value_i])
        return value_i

    def score(self, first: int, second: int) -> tuple[np.ndarray, float]:
        return _pair_consistency(
            self.cluster_matrices[first],
            int(self.cluster_counts[first]),
            self.cluster_matrices[second],
            int(self.cluster_counts[second]),
        )

    def merge(self, first: int, second: int, merged_matrix: np.ndarray) -> int:
        first_i, second_i = self.find(first), self.find(second)
        if first_i == second_i:
            return first_i
        if int(self.cluster_counts[first_i]) < int(self.cluster_counts[second_i]):
            first_i, second_i = second_i, first_i

        merged_neighbours = (
            self.neighbours[first_i].union(self.neighbours[second_i])
            - {first_i, second_i}
        )
        self.parent[second_i] = first_i
        self.active[second_i] = False
        self.cluster_counts[first_i] += self.cluster_counts[second_i]
        self.cluster_matrices[first_i] = np.asarray(merged_matrix, dtype=float)
        self.versions[first_i] += 1

        self.neighbours[first_i] = set()
        self.neighbours[second_i].clear()
        for neighbour in merged_neighbours:
            neighbour_root = self.find(neighbour)
            if neighbour_root == first_i or not self.active[neighbour_root]:
                continue
            self.neighbours[neighbour_root].discard(first_i)
            self.neighbours[neighbour_root].discard(second_i)
            self.neighbours[neighbour_root].add(first_i)
            self.neighbours[first_i].add(neighbour_root)
        return first_i

    def labels(self) -> np.ndarray:
        roots = np.asarray([self.find(index) for index in range(self.count)], dtype=np.int64)
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int64)


def _global_best_merge(
    matrices: np.ndarray,
    edges: list[tuple[int, int]],
    threshold: float,
) -> tuple[np.ndarray, list[float], int]:
    state = _DynamicAgglomerator(matrices, edges)
    heap: list[tuple[float, int, int, int, int]] = []

    def push(first: int, second: int) -> None:
        first_i, second_i = state.find(first), state.find(second)
        if first_i == second_i or not state.active[first_i] or not state.active[second_i]:
            return
        if second_i not in state.neighbours[first_i]:
            return
        _, score = state.score(first_i, second_i)
        low, high = sorted((first_i, second_i))
        heapq.heappush(
            heap,
            (
                -float(score),
                low,
                high,
                int(state.versions[low]),
                int(state.versions[high]),
            ),
        )

    for first, second in edges:
        push(int(first), int(second))

    accepted: list[float] = []
    stale_pops = 0
    while heap:
        negative_score, first, second, first_version, second_version = heapq.heappop(heap)
        first_i, second_i = state.find(first), state.find(second)
        if first_i != first or second_i != second:
            stale_pops += 1
            continue
        if not state.active[first_i] or not state.active[second_i]:
            stale_pops += 1
            continue
        if (
            int(state.versions[first_i]) != first_version
            or int(state.versions[second_i]) != second_version
            or second_i not in state.neighbours[first_i]
        ):
            stale_pops += 1
            continue

        merged_matrix, current_score = state.score(first_i, second_i)
        if abs(current_score + negative_score) > 1e-12:
            push(first_i, second_i)
            stale_pops += 1
            continue
        if current_score < threshold:
            break

        accepted.append(float(current_score))
        root = state.merge(first_i, second_i, merged_matrix)
        for neighbour in list(state.neighbours[root]):
            push(root, neighbour)

    return state.labels(), accepted, stale_pops


def _mutual_best_merge(
    matrices: np.ndarray,
    edges: list[tuple[int, int]],
    threshold: float,
) -> tuple[np.ndarray, list[float], int]:
    state = _DynamicAgglomerator(matrices, edges)
    accepted: list[float] = []
    rounds = 0

    while True:
        rounds += 1
        best: dict[int, tuple[float, int, np.ndarray]] = {}
        for first in range(state.count):
            if not state.active[first]:
                continue
            for second in state.neighbours[first]:
                if first >= second or not state.active[second]:
                    continue
                merged, score = state.score(first, second)
                if score > best.get(first, (-np.inf, -1, merged))[0]:
                    best[first] = (score, second, merged)
                if score > best.get(second, (-np.inf, -1, merged))[0]:
                    best[second] = (score, first, merged)

        pairs: list[tuple[float, int, int, np.ndarray]] = []
        for first, (score, second, merged) in best.items():
            if first >= second or score < threshold:
                continue
            reverse = best.get(second)
            if reverse is not None and int(reverse[1]) == first:
                pairs.append((float(score), int(first), int(second), merged))

        if not pairs:
            break

        used: set[int] = set()
        merged_this_round = 0
        for score, first, second, _ in sorted(pairs, key=lambda item: item[0], reverse=True):
            first_i, second_i = state.find(first), state.find(second)
            if first_i == second_i or first_i in used or second_i in used:
                continue
            if second_i not in state.neighbours[first_i]:
                continue
            merged_matrix, current_score = state.score(first_i, second_i)
            if current_score < threshold:
                continue
            root = state.merge(first_i, second_i, merged_matrix)
            used.add(root)
            used.add(second_i if root == first_i else first_i)
            accepted.append(float(current_score))
            merged_this_round += 1
        if merged_this_round == 0:
            break

    return state.labels(), accepted, rounds


def _comb2(values: np.ndarray) -> int:
    values_i = np.asarray(values, dtype=np.int64)
    return int(np.sum(values_i * (values_i - 1) // 2))


def _evaluate(
    truth_groups: list[np.ndarray],
    predicted_groups: list[np.ndarray],
) -> dict[str, float | int]:
    from sklearn.metrics import adjusted_rand_score

    truth_all: list[np.ndarray] = []
    predicted_all: list[np.ndarray] = []
    truth_offset = 0
    predicted_offset = 0
    fragment_total = 0
    fragment_connected = 0
    split_excess = 0
    largest_numerator = 0
    total_points = 0
    predicted_domains = 0
    impure_domains = 0
    weighted_pure_points = 0
    true_pairs = 0
    predicted_pairs = 0
    true_positive_pairs = 0

    for truth_raw, predicted_raw in zip(truth_groups, predicted_groups):
        _, truth = np.unique(np.asarray(truth_raw, dtype=np.int64), return_inverse=True)
        _, predicted = np.unique(
            np.asarray(predicted_raw, dtype=np.int64), return_inverse=True
        )

        truth_all.append(truth + truth_offset)
        predicted_all.append(predicted + predicted_offset)
        truth_offset += int(truth.max()) + 1 if len(truth) else 0
        predicted_offset += int(predicted.max()) + 1 if len(predicted) else 0

        truth_values = np.unique(truth)
        predicted_values = np.unique(predicted)
        fragment_total += len(truth_values)
        predicted_domains += len(predicted_values)
        total_points += len(truth)

        for truth_value in truth_values:
            indices = np.flatnonzero(truth == truth_value)
            parts, counts = np.unique(predicted[indices], return_counts=True)
            fragment_connected += int(len(parts) == 1)
            split_excess += max(len(parts) - 1, 0)
            largest_numerator += int(counts.max()) if len(counts) else 0
            true_pairs += _comb2(np.asarray([len(indices)]))

        for predicted_value in predicted_values:
            indices = np.flatnonzero(predicted == predicted_value)
            labels, counts = np.unique(truth[indices], return_counts=True)
            weighted_pure_points += int(counts.max()) if len(counts) else 0
            impure_domains += int(len(labels) > 1)
            predicted_pairs += _comb2(np.asarray([len(indices)]))
            true_positive_pairs += _comb2(counts)

    truth_vector = np.concatenate(truth_all) if truth_all else np.empty(0, dtype=np.int64)
    predicted_vector = (
        np.concatenate(predicted_all) if predicted_all else np.empty(0, dtype=np.int64)
    )
    pair_precision = true_positive_pairs / max(predicted_pairs, 1)
    pair_recall = true_positive_pairs / max(true_pairs, 1)
    pair_f1 = (
        2.0 * pair_precision * pair_recall / max(pair_precision + pair_recall, 1e-15)
    )
    return {
        "ari": float(adjusted_rand_score(truth_vector, predicted_vector)),
        "predicted_domains": predicted_domains,
        "oracle_fragments": fragment_total,
        "weighted_purity": weighted_pure_points / max(total_points, 1),
        "fragments_connected": fragment_connected / max(fragment_total, 1),
        "largest_fraction": largest_numerator / max(total_points, 1),
        "split_excess": split_excess,
        "impure_domains": impure_domains,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,
    }


def _parse_thresholds(text: str) -> list[float]:
    values = sorted(
        {
            float(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )
    if not values:
        raise ValueError("At least one threshold is required.")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Thresholds must lie in [0, 1].")
    return values


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
    parser.add_argument(
        "--thresholds",
        default="0.60,0.70,0.80,0.90,0.95,0.97,0.99,0.995",
        help="Comma-separated real-stage cluster-consistency thresholds.",
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from polatory.leapfrog_automatic_domain_builder import _normalise_determinant

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
        global_indices = np.flatnonzero(coarse_labels == coarse_value)
        local_points = points[global_indices]
        local_matrices = point_matrices[global_indices]
        delaunay = edge_base._delaunay_edges(local_points)
        groups.append(
            {
                "coarse": int(coarse_value),
                "truth": oracle_labels[global_indices],
                "matrices": local_matrices,
                "graphs": {
                    "delaunay": delaunay,
                    "similarity_top6_either": _endpoint_topk_edges(
                        local_matrices, delaunay, 6
                    ),
                    "similarity_top8_either": _endpoint_topk_edges(
                        local_matrices, delaunay, 8
                    ),
                    "similarity_top10_either": _endpoint_topk_edges(
                        local_matrices, delaunay, 10
                    ),
                },
            }
        )

    thresholds = _parse_thresholds(args.thresholds)
    rows: list[dict[str, object]] = []
    for graph_name in (
        "delaunay",
        "similarity_top6_either",
        "similarity_top8_either",
        "similarity_top10_either",
    ):
        for policy in ("global_best", "mutual_best"):
            for threshold in thresholds:
                predicted_groups: list[np.ndarray] = []
                accepted_scores: list[float] = []
                work_units = 0
                for group in groups:
                    matrices = np.asarray(group["matrices"], dtype=float)
                    edges = list(group["graphs"][graph_name])
                    if policy == "global_best":
                        labels, accepted, stale = _global_best_merge(
                            matrices, edges, threshold
                        )
                        work_units += stale
                    else:
                        labels, accepted, rounds = _mutual_best_merge(
                            matrices, edges, threshold
                        )
                        work_units += rounds
                    predicted_groups.append(labels)
                    accepted_scores.extend(accepted)

                metrics = _evaluate(
                    [np.asarray(group["truth"], dtype=np.int64) for group in groups],
                    predicted_groups,
                )
                metrics.update(
                    {
                        "graph": graph_name,
                        "policy": policy,
                        "threshold": threshold,
                        "accepted_merges": len(accepted_scores),
                        "accepted_min": min(accepted_scores) if accepted_scores else np.nan,
                        "accepted_median": (
                            float(np.median(accepted_scores))
                            if accepted_scores
                            else np.nan
                        ),
                        "work_units": work_units,
                    }
                )
                rows.append(metrics)

    print(
        f"case={case_name} points={len(points)} grid={shape} "
        f"oracle_domains={len(np.unique(oracle_labels))} "
        f"coarse_domains={len(np.unique(coarse_labels))} coarse_merges={coarse_merges}"
    )
    oracle_fragments = int(rows[0]["oracle_fragments"]) if rows else 0
    print(
        f"Evaluation target: {oracle_fragments} oracle fragments inside coarse domains. "
        "Unlike the previous edge test, this evaluates final dynamically merged labels."
    )

    print("\nLiteral 0.60 cluster-level threshold:")
    print(
        " graph                    policy       domains purity connected largest split "
        "impure pairP pairR pairF1   ARI merges accepted[min/median]"
    )
    literal = [row for row in rows if abs(float(row["threshold"]) - 0.60) < 1e-12]
    for row in sorted(
        literal,
        key=lambda item: (
            float(item["ari"]),
            float(item["pair_f1"]),
            float(item["weighted_purity"]),
        ),
        reverse=True,
    ):
        print(
            f" {str(row['graph']):<24s} {str(row['policy']):<11s} "
            f"{int(row['predicted_domains']):>7d} "
            f"{float(row['weighted_purity']):>6.3f} "
            f"{float(row['fragments_connected']):>9.3f} "
            f"{float(row['largest_fraction']):>7.3f} "
            f"{int(row['split_excess']):>5d} "
            f"{int(row['impure_domains']):>6d} "
            f"{float(row['pair_precision']):>5.3f} "
            f"{float(row['pair_recall']):>5.3f} "
            f"{float(row['pair_f1']):>6.3f} "
            f"{float(row['ari']):>5.3f} "
            f"{int(row['accepted_merges']):>6d} "
            f"{float(row['accepted_min']):.3f}/{float(row['accepted_median']):.3f}"
        )

    print("\nTop dynamic-merging candidates:")
    print(
        " graph                    policy      thr domains purity connected largest split "
        "impure pairP pairR pairF1   ARI"
    )
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["ari"]),
            float(item["pair_f1"]),
            float(item["weighted_purity"]),
            float(item["fragments_connected"]),
            -abs(int(item["predicted_domains"]) - int(item["oracle_fragments"])),
        ),
        reverse=True,
    )[:20]:
        print(
            f" {str(row['graph']):<24s} {str(row['policy']):<11s} "
            f"{float(row['threshold']):>4.2f} "
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

    print("\nBest candidate with high fragment preservation:")
    feasible = [
        row
        for row in rows
        if float(row["fragments_connected"]) >= 0.90
        and float(row["largest_fraction"]) >= 0.99
    ]
    if not feasible:
        print(" none")
    else:
        best = max(
            feasible,
            key=lambda item: (
                float(item["pair_f1"]),
                float(item["weighted_purity"]),
                float(item["ari"]),
                -int(item["impure_domains"]),
            ),
        )
        print(
            f" graph={best['graph']} policy={best['policy']} "
            f"threshold={float(best['threshold']):.3f} "
            f"domains={int(best['predicted_domains'])}/{int(best['oracle_fragments'])} "
            f"purity={float(best['weighted_purity']):.4f} "
            f"connected={float(best['fragments_connected']):.4f} "
            f"largest={float(best['largest_fraction']):.4f} "
            f"split={int(best['split_excess'])} impure={int(best['impure_domains'])} "
            f"pair_precision={float(best['pair_precision']):.4f} "
            f"pair_recall={float(best['pair_recall']):.4f} "
            f"pair_f1={float(best['pair_f1']):.4f} ARI={float(best['ari']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
