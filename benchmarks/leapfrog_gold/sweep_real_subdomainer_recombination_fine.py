"""Fine sweep of the most promising real-SubDomainer recombination model.

The broad sweep showed that local Delaunay subdivision followed by global k-nearest-
component recombination is plausible, but the tested 10% and 20% caps bracket the
nine decoded Leapfrog domains.  This script searches every k from 6 through 20 and
global point caps from 10% through 20% in 1% increments.  It ranks all candidates,
reports candidates that produce the decoded domain count, and leaves production code
unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys
from math import floor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_automatic_domains_to_oracle_robust  # noqa: F401,E402
import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import run_selected_exact_leapfrog_lva as exact  # noqa: E402
import sweep_real_subdomainer_recombination as recombination  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument(
        "--decoded-root", type=Path, default=Path("Leapfrog_LVA_decoded_benchmark")
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--local-maximum-fraction", type=float, default=0.10)
    parser.add_argument("--k-min", type=int, default=6)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--cap-min", type=float, default=0.10)
    parser.add_argument("--cap-max", type=float, default=0.20)
    parser.add_argument("--cap-step", type=float, default=0.01)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if args.k_min < 1 or args.k_max < args.k_min:
        raise SystemExit("k range must satisfy 1 <= k-min <= k-max")
    if not 0.0 < args.cap_min <= args.cap_max <= 1.0:
        raise SystemExit("cap range must satisfy 0 < cap-min <= cap-max <= 1")
    if args.cap_step <= 0.0:
        raise SystemExit("cap-step must be positive")

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    points, oracle_labels, mesh_path = comparison._load_oracle_inputs(
        args.decoded_root, case_name
    )
    points = np.asarray(points, dtype=np.float64)
    oracle_labels = np.asarray(oracle_labels, dtype=np.int64)
    oracle_domain_count = len(np.unique(oracle_labels))

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
        maximum_cluster_fraction=args.local_maximum_fraction,
        consistency_threshold=args.threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, _, shape, centroids = builder._prepare_grid(points)
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids, trend_input, non_decaying=False
    )
    point_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        points, trend_input, non_decaying=False
    )
    coarse_labels, centroid_labels, _, _, coarse_merges = builder._automatic_labels(
        points,
        np.asarray(centroid_anisotropies, dtype=np.float64),
        np.asarray(minimum, dtype=np.float64),
        np.asarray(maximum, dtype=np.float64),
        tuple(int(value) for value in shape),
    )

    (
        members,
        component_matrices,
        component_sizes,
        component_centroids,
        component_parents,
        local_edges,
        local_merges,
    ) = recombination._build_local_components(
        points,
        np.asarray(point_anisotropies, dtype=np.float64),
        np.asarray(coarse_labels, dtype=np.int64),
        threshold=args.threshold,
        maximum_fraction=args.local_maximum_fraction,
    )
    parent_pairs = recombination._parent_adjacency(
        np.asarray(centroid_labels, dtype=np.int64),
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
    print(
        f"local: components={len(members)} edges={local_edges} merges={local_merges} "
        f"cap={args.local_maximum_fraction:.3f}"
    )

    cap_values: list[float] = []
    value = args.cap_min
    while value <= args.cap_max + args.cap_step * 1.0e-6:
        cap_values.append(round(value, 10))
        value += args.cap_step

    rows: list[dict[str, object]] = []
    edge_cache: dict[int, np.ndarray] = {}
    for k in range(args.k_min, args.k_max + 1):
        edges = recombination._global_edges(
            component_centroids,
            component_parents,
            parent_pairs,
            "knn",
            k,
        )
        edge_cache[k] = edges
        for cap_fraction in cap_values:
            maximum_points = max(1, int(floor(cap_fraction * len(points))))
            labels, merges = recombination._merge_components(
                len(points),
                members,
                component_matrices,
                component_sizes,
                edges,
                threshold=args.threshold,
                maximum_points=maximum_points,
            )
            ari, ami, match, matched = metrics(labels)
            rows.append(
                {
                    "k": k,
                    "cap": cap_fraction,
                    "maximum_points": maximum_points,
                    "domains": int(len(np.unique(labels))),
                    "edges": int(len(edges)),
                    "merges": int(merges),
                    "ari": ari,
                    "ami": ami,
                    "match": match,
                    "matched": matched,
                }
            )

    def format_row(row: dict[str, object]) -> str:
        return (
            f"k={int(row['k']):2d} cap={float(row['cap']):.2f} "
            f"max={int(row['maximum_points']):3d} domains={int(row['domains']):3d} "
            f"edges={int(row['edges']):4d} merges={int(row['merges']):3d} "
            f"ARI={float(row['ari']):.5f} AMI={float(row['ami']):.5f} "
            f"match={float(row['match']):.5f} ({int(row['matched'])}/{len(points)})"
        )

    exact_count = [row for row in rows if int(row["domains"]) == oracle_domain_count]
    print(f"\nCandidates with exactly {oracle_domain_count} domains, ranked by ARI:")
    if exact_count:
        for row in sorted(exact_count, key=lambda item: float(item["ari"]), reverse=True):
            print(format_row(row))
    else:
        print("none")

    print(f"\nTop {max(args.top, 1)} candidates by ARI:")
    for row in sorted(rows, key=lambda item: float(item["ari"]), reverse=True)[: max(args.top, 1)]:
        print(format_row(row))

    print(f"\nTop {max(args.top, 1)} candidates by optimal label match:")
    for row in sorted(rows, key=lambda item: float(item["match"]), reverse=True)[: max(args.top, 1)]:
        print(format_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
