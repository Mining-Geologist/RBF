"""Compare reconstructed automatic domains directly with decoded Leapfrog labels.

This is the clustering benchmark that should drive SubDomainer recovery.  It uses the
real WolfPass input points, the real structural trend mesh, and the exact decoded
``point_clusters.csv`` labels extracted from Leapfrog's serialized SubDomainer.
No RBF fitting or surface meshing is performed.

For each determinant-to-consistency candidate, the script reports:

* predicted and Leapfrog domain counts;
* adjusted Rand index (ARI);
* adjusted mutual information (AMI);
* optimal one-to-one label-matching accuracy (Hungarian assignment);
* merge count and runtime.

The default decoded benchmark root is ``Leapfrog_LVA_decoded_benchmark``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
except ImportError as exc:  # pragma: no cover - environment guidance
    raise ImportError(
        "This benchmark requires scikit-learn. Install it with: "
        "python -m pip install scikit-learn"
    ) from exc

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sweep_exact_domainer_consistency as exact_sweep  # noqa: E402


@dataclass(frozen=True)
class OracleComparison:
    case: str
    formula: str
    threshold: float
    point_count: int
    centroid_count: int
    grid_shape: tuple[int, int, int]
    oracle_domains: int
    predicted_domains: int
    merge_count: int
    adjusted_rand_index: float
    adjusted_mutual_information: float
    optimal_label_accuracy: float
    matched_points: int
    elapsed_seconds: float


def _optimal_label_accuracy(
    truth: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, int]:
    """Return accuracy after optimal one-to-one relabelling of predicted clusters."""
    truth_values, truth_inverse = np.unique(truth, return_inverse=True)
    predicted_values, predicted_inverse = np.unique(predicted, return_inverse=True)
    contingency = np.zeros(
        (len(truth_values), len(predicted_values)),
        dtype=np.int64,
    )
    np.add.at(contingency, (truth_inverse, predicted_inverse), 1)
    rows, columns = linear_sum_assignment(-contingency)
    matched = int(contingency[rows, columns].sum())
    return matched / float(len(truth)), matched


def _parse_case_parameters(case_name: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"S(?P<strength>[0-9]+(?:\.[0-9]+)?)_R(?P<range>[0-9]+(?:\.[0-9]+)?)",
        case_name,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Case {case_name!r} must follow the S<strength>_R<range> convention."
        )
    return float(match.group("strength")), float(match.group("range"))


def _available_cases(decoded_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in decoded_root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"S[0-9]+(?:\.[0-9]+)?_R[0-9]+(?:\.[0-9]+)?", path.name)
        and (path / "point_clusters.csv").is_file()
    )


def _load_oracle_inputs(
    decoded_root: Path,
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, Path]:
    points_path = decoded_root / "Used Data.csv"
    mesh_path = decoded_root / "Reference Mesh.obj"
    labels_path = decoded_root / case_name / "point_clusters.csv"
    missing = [
        path
        for path in (points_path, mesh_path, labels_path)
        if not path.is_file()
    ]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required decoded benchmark files are missing:\n{details}")

    points_frame = pd.read_csv(points_path)
    coordinate_columns = ["xe", "ye", "ze"]
    missing_columns = [name for name in coordinate_columns if name not in points_frame]
    if missing_columns:
        raise ValueError(
            f"{points_path} is missing coordinate columns {missing_columns}."
        )
    points = points_frame[coordinate_columns].to_numpy(dtype=np.float64)

    labels_frame = pd.read_csv(labels_path)
    if "cluster" not in labels_frame:
        raise ValueError(f"{labels_path} does not contain a 'cluster' column.")
    oracle_labels = labels_frame["cluster"].to_numpy(dtype=np.int64)
    if len(oracle_labels) != len(points):
        raise ValueError(
            f"Point/label length mismatch: {len(points)} points versus "
            f"{len(oracle_labels)} labels."
        )
    return points, oracle_labels, mesh_path


def _compare_formula(
    *,
    case_name: str,
    formula_name: str,
    transform: Callable[[float], float] | None,
    builder_class: type,
    builder_module: object,
    original_merge_function: Callable[..., tuple[np.ndarray, float]],
    determinant_function: Callable[[np.ndarray], float],
    points: np.ndarray,
    oracle_labels: np.ndarray,
    centroid_anisotropies: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
    threshold: float,
    centroid_count: int,
    minimum_fraction: float,
    maximum_fraction: float,
) -> OracleComparison:
    started = time.perf_counter()
    replacement = (
        original_merge_function
        if transform is None
        else exact_sweep._candidate_merge_function(  # noqa: SLF001
            determinant_function,
            transform,
        )
    )
    setattr(builder_module, "_merged_matrix_and_consistency", replacement)

    builder = builder_class(
        centroid_count=centroid_count,
        minimum_cluster_fraction=minimum_fraction,
        maximum_cluster_fraction=maximum_fraction,
        consistency_threshold=threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    predicted, _, _, _, merge_count = builder._automatic_labels(  # noqa: SLF001
        points,
        centroid_anisotropies,
        minimum,
        maximum,
        shape,
    )
    predicted = np.asarray(predicted, dtype=np.int64)
    accuracy, matched = _optimal_label_accuracy(oracle_labels, predicted)

    return OracleComparison(
        case=case_name,
        formula=formula_name,
        threshold=float(threshold),
        point_count=int(len(points)),
        centroid_count=int(np.prod(shape)),
        grid_shape=shape,
        oracle_domains=int(len(np.unique(oracle_labels))),
        predicted_domains=int(len(np.unique(predicted))),
        merge_count=int(merge_count),
        adjusted_rand_index=float(adjusted_rand_score(oracle_labels, predicted)),
        adjusted_mutual_information=float(
            adjusted_mutual_info_score(oracle_labels, predicted)
        ),
        optimal_label_accuracy=float(accuracy),
        matched_points=int(matched),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def _run_case(
    *,
    case_name: str,
    decoded_root: Path,
    threshold: float,
    centroid_count: int,
    minimum_fraction: float,
    maximum_fraction: float,
) -> list[OracleComparison]:
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory
    import run_selected_exact_leapfrog_lva as exact

    points, oracle_labels, mesh_path = _load_oracle_inputs(decoded_root, case_name)
    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = _parse_case_parameters(case_name)
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        strength,
        trend_range,
    )

    builder_class = polatory.AutomaticStructuralDomainBuilder3
    required_methods = ("_prepare_grid", "_automatic_labels")
    missing = [name for name in required_methods if not hasattr(builder_class, name)]
    if missing:
        raise RuntimeError(
            "The installed top-level AutomaticStructuralDomainBuilder3 is not the "
            f"reconstructed Leapfrog builder; missing {missing}."
        )

    builder_module = importlib.import_module(builder_class.__module__)
    original_merge_function = getattr(
        builder_module,
        "_merged_matrix_and_consistency",
        None,
    )
    determinant_function = getattr(builder_module, "_symmetric_determinant", None)
    if original_merge_function is None or determinant_function is None:
        raise RuntimeError(
            f"{builder_class.__module__} does not expose the recovered merge helpers."
        )

    preparation_builder = builder_class(
        centroid_count=centroid_count,
        minimum_cluster_fraction=minimum_fraction,
        maximum_cluster_fraction=maximum_fraction,
        consistency_threshold=threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, _, shape, centroids = preparation_builder._prepare_grid(points)  # noqa: SLF001
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids,
        trend_input,
        non_decaying=False,
    )

    print(
        f"case={case_name} points={len(points)} oracle_domains="
        f"{len(np.unique(oracle_labels))} centroids={len(centroids)} grid={shape}",
        flush=True,
    )

    results: list[OracleComparison] = []
    try:
        for formula_name, transform in exact_sweep._formulae().items():  # noqa: SLF001
            results.append(
                _compare_formula(
                    case_name=case_name,
                    formula_name=formula_name,
                    transform=transform,
                    builder_class=builder_class,
                    builder_module=builder_module,
                    original_merge_function=original_merge_function,
                    determinant_function=determinant_function,
                    points=points,
                    oracle_labels=oracle_labels,
                    centroid_anisotropies=np.asarray(
                        centroid_anisotropies,
                        dtype=np.float64,
                    ),
                    minimum=np.asarray(minimum, dtype=np.float64),
                    maximum=np.asarray(maximum, dtype=np.float64),
                    shape=tuple(int(value) for value in shape),
                    threshold=threshold,
                    centroid_count=centroid_count,
                    minimum_fraction=minimum_fraction,
                    maximum_fraction=maximum_fraction,
                )
            )
    finally:
        setattr(
            builder_module,
            "_merged_matrix_and_consistency",
            original_merge_function,
        )

    results.sort(
        key=lambda item: (
            -item.adjusted_rand_index,
            -item.optimal_label_accuracy,
            abs(item.predicted_domains - item.oracle_domains),
            item.formula,
        )
    )
    print(
        f"{'formula':>29} {'oracle':>7} {'pred':>7} {'ARI':>9} "
        f"{'AMI':>9} {'match_acc':>10} {'merges':>8} {'seconds':>9}",
        flush=True,
    )
    for item in results:
        print(
            f"{item.formula:>29} {item.oracle_domains:>7d} "
            f"{item.predicted_domains:>7d} {item.adjusted_rand_index:>9.5f} "
            f"{item.adjusted_mutual_information:>9.5f} "
            f"{item.optimal_label_accuracy:>10.5f} "
            f"{item.merge_count:>8d} {item.elapsed_seconds:>9.3f}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="S3_R100",
        help="Decoded case name, or ALL to compare every recovered case.",
    )
    parser.add_argument(
        "--decoded-root",
        type=Path,
        default=Path("Leapfrog_LVA_decoded_benchmark"),
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/automatic-domain-oracle-comparison.json"),
    )
    args = parser.parse_args()

    if not args.decoded_root.is_dir():
        parser.error(f"Decoded benchmark root was not found: {args.decoded_root}")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if args.centroid_count <= 0:
        parser.error("--centroid-count must be positive")
    if not 0.0 < args.minimum_fraction <= args.maximum_fraction <= 1.0:
        parser.error("cluster fractions must satisfy 0 < minimum <= maximum <= 1")

    requested = args.case.strip().upper()
    available = _available_cases(args.decoded_root)
    if requested == "ALL":
        cases = available
    else:
        matches = [name for name in available if name.upper() == requested]
        if len(matches) != 1:
            parser.error(
                f"Case {requested!r} was not found. Available cases: {available}"
            )
        cases = matches

    all_results: list[OracleComparison] = []
    for case_name in cases:
        all_results.extend(
            _run_case(
                case_name=case_name,
                decoded_root=args.decoded_root,
                threshold=args.threshold,
                centroid_count=args.centroid_count,
                minimum_fraction=args.minimum_fraction,
                maximum_fraction=args.maximum_fraction,
            )
        )

    payload = {
        "decoded_root": str(args.decoded_root),
        "threshold": float(args.threshold),
        "centroid_count_requested": int(args.centroid_count),
        "minimum_fraction": float(args.minimum_fraction),
        "maximum_fraction": float(args.maximum_fraction),
        "cases": cases,
        "results": [asdict(item) for item in all_results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
