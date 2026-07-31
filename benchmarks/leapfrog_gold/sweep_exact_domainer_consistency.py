"""Exact in-process sweep of candidate Leapfrog SubDomainer consistency transforms.

Unlike ``sweep_domainer_consistency.py``, this runner does not transfer matrices from
the exported 25^3 diagnostic field. It loads the benchmark's real structural trend
mesh, creates the production automatic builder's centroid grid, samples the recovered
single-input LVA field directly at every centroid, and then changes only the
merged-matrix consistency transform.

The reciprocal-determinant control must reproduce the production clustering result
before any alternative is considered meaningful. For S3_R100 the current verified
control is 32 populated domains on a 16 x 25 x 15 centroid grid.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


ConsistencyTransform = Callable[[float], float]


@dataclass(frozen=True)
class SweepResult:
    formula: str
    threshold: float
    centroid_count: int
    grid_shape: tuple[int, int, int]
    merge_count: int
    surviving_grid_domains: int
    populated_domains: int
    minimum_domain_size: int
    median_domain_size: float
    maximum_domain_size: int
    elapsed_seconds: float


def _formulae() -> dict[str, ConsistencyTransform | None]:
    """Return determinant-only candidates plus the unmodified production control."""
    tiny = np.finfo(np.float64).tiny
    return {
        # None means call the currently installed production implementation unchanged.
        "reciprocal_det_control": None,
        "inverse_sqrt_det": lambda determinant: 1.0
        / math.sqrt(max(determinant, tiny)),
        "inverse_cuberoot_det": lambda determinant: max(determinant, tiny)
        ** (-1.0 / 3.0),
        "inverse_det_squared": lambda determinant: 1.0
        / (max(determinant, tiny) ** 2.0),
        "inverse_one_plus_log_det": lambda determinant: 1.0
        / (1.0 + abs(math.log(max(determinant, tiny)))),
        "exp_neg_sqrt_abs_log_det": lambda determinant: math.exp(
            -math.sqrt(abs(math.log(max(determinant, tiny))))
        ),
    }


def _candidate_merge_function(
    determinant_function: Callable[[np.ndarray], float],
    transform: ConsistencyTransform,
):
    """Build a drop-in replacement changing only determinant-to-consistency mapping."""

    def merged_matrix_and_consistency(
        first_matrix: np.ndarray,
        first_size: int,
        second_matrix: np.ndarray,
        second_size: int,
    ) -> tuple[np.ndarray, float]:
        total = int(first_size) + int(second_size)
        if total <= 0:
            raise ValueError("Merged domain population must be positive.")
        weight = int(first_size) / float(total)
        merged = weight * first_matrix + (1.0 - weight) * second_matrix
        merged = 0.5 * (merged + merged.T)
        determinant = float(determinant_function(merged))
        consistency = (
            float("-inf")
            if not np.isfinite(determinant) or determinant <= 0.0
            else float(transform(determinant))
        )
        return merged, consistency

    return merged_matrix_and_consistency


def _run_formula(
    *,
    builder_class: type,
    builder_module: object,
    original_merge_function: Callable[..., tuple[np.ndarray, float]],
    determinant_function: Callable[[np.ndarray], float],
    formula_name: str,
    transform: ConsistencyTransform | None,
    points: np.ndarray,
    centroid_anisotropies: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
    threshold: float,
    centroid_count: int,
    minimum_fraction: float,
    maximum_fraction: float,
) -> SweepResult:
    started = time.perf_counter()
    replacement = (
        original_merge_function
        if transform is None
        else _candidate_merge_function(determinant_function, transform)
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
    labels, centroid_labels, _, _, merge_count = builder._automatic_labels(
        points,
        centroid_anisotropies,
        minimum,
        maximum,
        shape,
    )

    labels = np.asarray(labels, dtype=np.int64)
    centroid_labels = np.asarray(centroid_labels, dtype=np.int64)
    _, populations = np.unique(labels, return_counts=True)
    populated = int(len(populations))
    surviving_grid = int(len(np.unique(centroid_labels[centroid_labels >= 0])))

    return SweepResult(
        formula=formula_name,
        threshold=float(threshold),
        centroid_count=int(np.prod(shape)),
        grid_shape=shape,
        merge_count=int(merge_count),
        surviving_grid_domains=surviving_grid,
        populated_domains=populated,
        minimum_domain_size=int(populations.min()) if len(populations) else 0,
        median_domain_size=float(np.median(populations)) if len(populations) else 0.0,
        maximum_domain_size=int(populations.max()) if len(populations) else 0,
        elapsed_seconds=float(time.perf_counter() - started),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--centroid-count", type=int, default=6000)
    parser.add_argument("--minimum-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-fraction", type=float, default=0.10)
    parser.add_argument(
        "--expected-control-domains",
        type=int,
        default=None,
        help=(
            "Required populated-domain count for the unmodified reciprocal-determinant "
            "control. Defaults to 32 for S3_R100 and is otherwise not enforced."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/exact-consistency-sweep.json"),
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    if not case_name:
        parser.error("--case must not be empty")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if args.centroid_count <= 0:
        parser.error("--centroid-count must be positive")
    if not 0.0 < args.minimum_fraction <= args.maximum_fraction <= 1.0:
        parser.error("cluster fractions must satisfy 0 < minimum <= maximum <= 1")

    # Set selection before importing the benchmark stack. Importing the exact runner
    # installs the recovered single-input sampler but does not execute the mesh suite.
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory  # noqa: E402
    import run_selected_exact_leapfrog_lva as exact  # noqa: E402

    suite = exact.suite
    points, _ = suite.read_dataset()
    trend_vertices, trend_faces = suite.read_obj(
        suite.DATA_DIR / "Reference Mesh(2).obj"
    )
    cases = suite.available_cases()
    matching = [case for case in cases if case.name.upper() == case_name]
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected one benchmark case named {case_name}, found {len(matching)}."
        )
    case = matching[0]

    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        float(case.strength),
        float(case.trend_range),
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
        builder_module, "_merged_matrix_and_consistency", None
    )
    determinant_function = getattr(builder_module, "_symmetric_determinant", None)
    if original_merge_function is None or determinant_function is None:
        raise RuntimeError(
            f"{builder_class.__module__} does not expose the recovered merge helpers."
        )

    preparation_builder = builder_class(
        centroid_count=args.centroid_count,
        minimum_cluster_fraction=args.minimum_fraction,
        maximum_cluster_fraction=args.maximum_fraction,
        consistency_threshold=args.threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    minimum, maximum, active_axes, shape, centroids = (
        preparation_builder._prepare_grid(np.asarray(points, dtype=np.float64))
    )
    centroid_anisotropies = exact.exact_leapfrog_single_input_anisotropies3(
        centroids,
        trend_input,
        non_decaying=False,
    )

    print(
        f"case={case.name} points={len(points)} centroids={len(centroids)} "
        f"grid={shape} threshold={args.threshold:g}",
        flush=True,
    )
    print(
        "Sampling source: real Reference Mesh(2).obj evaluated directly at every "
        "production centroid.",
        flush=True,
    )

    results: list[SweepResult] = []
    try:
        for formula_name, transform in _formulae().items():
            result = _run_formula(
                builder_class=builder_class,
                builder_module=builder_module,
                original_merge_function=original_merge_function,
                determinant_function=determinant_function,
                formula_name=formula_name,
                transform=transform,
                points=np.asarray(points, dtype=np.float64),
                centroid_anisotropies=np.asarray(
                    centroid_anisotropies, dtype=np.float64
                ),
                minimum=np.asarray(minimum, dtype=np.float64),
                maximum=np.asarray(maximum, dtype=np.float64),
                shape=tuple(int(value) for value in shape),
                threshold=args.threshold,
                centroid_count=args.centroid_count,
                minimum_fraction=args.minimum_fraction,
                maximum_fraction=args.maximum_fraction,
            )
            results.append(result)
    finally:
        setattr(
            builder_module,
            "_merged_matrix_and_consistency",
            original_merge_function,
        )

    expected_control = args.expected_control_domains
    if expected_control is None and case_name == "S3_R100":
        expected_control = 32
    control = next(
        item for item in results if item.formula == "reciprocal_det_control"
    )
    if (
        expected_control is not None
        and control.populated_domains != expected_control
    ):
        raise RuntimeError(
            "Exact sweep validation failed: the unmodified production control "
            f"returned {control.populated_domains} populated domains, expected "
            f"{expected_control}. Do not interpret the alternative formulas."
        )

    headers = (
        "formula",
        "populated",
        "grid_domains",
        "merges",
        "min_points",
        "median_points",
        "max_points",
        "seconds",
    )
    print(
        " ".join(
            (
                f"{headers[0]:>29}",
                f"{headers[1]:>10}",
                f"{headers[2]:>14}",
                f"{headers[3]:>10}",
                f"{headers[4]:>11}",
                f"{headers[5]:>13}",
                f"{headers[6]:>10}",
                f"{headers[7]:>9}",
            )
        )
    )
    for item in results:
        print(
            f"{item.formula:>29} "
            f"{item.populated_domains:>10d} "
            f"{item.surviving_grid_domains:>14d} "
            f"{item.merge_count:>10d} "
            f"{item.minimum_domain_size:>11d} "
            f"{item.median_domain_size:>13.1f} "
            f"{item.maximum_domain_size:>10d} "
            f"{item.elapsed_seconds:>9.3f}"
        )

    payload = {
        "case": case.name,
        "strength": float(case.strength),
        "trend_range": float(case.trend_range),
        "input_points": int(len(points)),
        "centroid_count_requested": int(args.centroid_count),
        "centroid_count_actual": int(len(centroids)),
        "centroid_grid_shape": [int(value) for value in shape],
        "active_axes": np.asarray(active_axes, dtype=bool).tolist(),
        "threshold": float(args.threshold),
        "minimum_fraction": float(args.minimum_fraction),
        "maximum_fraction": float(args.maximum_fraction),
        "sampling": "exact structural mesh at production centroids",
        "builder_module": builder_class.__module__,
        "control_expected_populated_domains": expected_control,
        "control_validated": (
            expected_control is None
            or control.populated_domains == expected_control
        ),
        "results": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
