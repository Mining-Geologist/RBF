"""Compare candidate Leapfrog grid geometries directly against decoded oracle labels.

The consistency-formula and threshold sweeps showed that the S3_R100 partition is
controlled by grid topology/size constraints rather than the 0.6 threshold.  This
runner therefore keeps the recovered merge implementation unchanged and varies only:

* data bounding box versus the benchmark/model bounding box;
* the current automatically factored shape versus the captured 15 x 23 x 19 shape;
* cell-centre locations with floor assignment versus endpoint nodes with nearest-node
  assignment.

No RBF fitting or surface meshing is performed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare_automatic_domains_to_oracle as comparison  # noqa: E402
import compare_automatic_domains_to_oracle_robust as robust  # noqa: E402


@dataclass(frozen=True)
class GeometryResult:
    geometry: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    centroid_count: int
    oracle_domains: int
    predicted_domains: int
    adjusted_rand_index: float
    adjusted_mutual_information: float
    optimal_label_accuracy: float
    matched_points: int
    merge_count: int
    elapsed_seconds: float


def _endpoint_nodes(
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    axes = []
    for axis, size in enumerate(shape):
        if size <= 1 or not maximum[axis] > minimum[axis]:
            axes.append(np.asarray([(minimum[axis] + maximum[axis]) * 0.5]))
        else:
            axes.append(np.linspace(minimum[axis], maximum[axis], size, dtype=float))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _nearest_node_indices(
    points: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    indices = np.zeros((len(points), 3), dtype=np.int64)
    for axis, size in enumerate(shape):
        span = float(maximum[axis] - minimum[axis])
        if size <= 1 or not span > 0.0:
            continue
        scaled = (points[:, axis] - minimum[axis]) / span * float(size - 1)
        indices[:, axis] = np.clip(np.rint(scaled).astype(np.int64), 0, size - 1)
    return (indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]


def _run_geometry(
    *,
    name: str,
    points: np.ndarray,
    oracle_labels: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    shape: tuple[int, int, int],
    centroids: np.ndarray,
    point_indexer: Callable[[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]], np.ndarray],
    builder_class: type,
    builder_module: object,
    trend_input: object,
    exact_module: object,
    threshold: float,
    centroid_count: int,
    minimum_fraction: float,
    maximum_fraction: float,
) -> GeometryResult:
    started = time.perf_counter()
    centroid_anisotropies = exact_module.exact_leapfrog_single_input_anisotropies3(
        centroids,
        trend_input,
        non_decaying=False,
    )
    original_indexer = getattr(builder_module, "_point_cell_indices")
    setattr(builder_module, "_point_cell_indices", point_indexer)
    try:
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
            np.asarray(centroid_anisotropies, dtype=np.float64),
            minimum,
            maximum,
            shape,
        )
    finally:
        setattr(builder_module, "_point_cell_indices", original_indexer)

    predicted = np.asarray(predicted, dtype=np.int64)
    accuracy, matched = comparison._optimal_label_accuracy(oracle_labels, predicted)  # noqa: SLF001
    return GeometryResult(
        geometry=name,
        bbox_min=tuple(float(value) for value in minimum),
        bbox_max=tuple(float(value) for value in maximum),
        grid_shape=tuple(int(value) for value in shape),
        centroid_count=int(len(centroids)),
        oracle_domains=int(len(np.unique(oracle_labels))),
        predicted_domains=int(len(np.unique(predicted))),
        adjusted_rand_index=float(adjusted_rand_score(oracle_labels, predicted)),
        adjusted_mutual_information=float(
            adjusted_mutual_info_score(oracle_labels, predicted)
        ),
        optimal_label_accuracy=float(accuracy),
        matched_points=int(matched),
        merge_count=int(merge_count),
        elapsed_seconds=float(time.perf_counter() - started),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="S3_R100")
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
        "--captured-shape",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(15, 23, 19),
    )
    parser.add_argument(
        "--model-min",
        type=float,
        nargs=3,
        metavar=("XMIN", "YMIN", "ZMIN"),
        default=(444600.0, 492600.0, 2000.0),
    )
    parser.add_argument(
        "--model-max",
        type=float,
        nargs=3,
        metavar=("XMAX", "YMAX", "ZMAX"),
        default=(445900.0, 494600.0, 3600.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/oracle-grid-geometry-sweep.json"),
    )
    args = parser.parse_args()

    case_name = args.case.strip().upper()
    os.environ["POLATORY_BENCHMARK_CASE"] = case_name
    os.environ["POLATORY_BASAL_CASES"] = case_name

    import polatory  # noqa: E402
    import run_selected_exact_leapfrog_lva as exact  # noqa: E402

    points, oracle_labels, mesh_path = robust._load_oracle_inputs(  # noqa: SLF001
        args.decoded_root,
        case_name,
    )
    trend_vertices, trend_faces = exact.suite.read_obj(mesh_path)
    strength, trend_range = comparison._parse_case_parameters(case_name)  # noqa: SLF001
    trend_input = polatory.StructuralTrendInput3(
        np.asarray(trend_vertices, dtype=np.float64),
        np.asarray(trend_faces, dtype=np.int64),
        strength,
        trend_range,
    )

    builder_class = polatory.AutomaticStructuralDomainBuilder3
    builder_module = importlib.import_module(builder_class.__module__)
    grid_shape_function = getattr(builder_module, "_leapfrog_grid_shape")
    centre_function = getattr(builder_module, "_grid_centroids")
    floor_indexer = getattr(builder_module, "_point_cell_indices")

    preparation = builder_class(
        centroid_count=args.centroid_count,
        minimum_cluster_fraction=args.minimum_fraction,
        maximum_cluster_fraction=args.maximum_fraction,
        consistency_threshold=args.threshold,
        base_range=0.0,
        support_multiplier=5,
        minimum_support_points=1,
    )
    data_min, data_max, data_active, data_auto_shape, data_auto_centres = (
        preparation._prepare_grid(points)  # noqa: SLF001
    )
    model_min = np.asarray(args.model_min, dtype=np.float64)
    model_max = np.asarray(args.model_max, dtype=np.float64)
    if np.any(model_max <= model_min):
        parser.error("--model-max must be greater than --model-min on every axis")
    model_spans = model_max - model_min
    model_active = model_spans > max(float(model_spans.max()), 1.0) * 1.0e-12
    model_auto_shape = grid_shape_function(
        args.centroid_count,
        model_spans,
        model_active,
    )
    captured_shape = tuple(int(value) for value in args.captured_shape)
    if any(value <= 0 for value in captured_shape):
        parser.error("--captured-shape values must be positive")

    variants = [
        (
            "data_bbox_auto_centres",
            data_min,
            data_max,
            tuple(int(value) for value in data_auto_shape),
            np.asarray(data_auto_centres, dtype=np.float64),
            floor_indexer,
        ),
        (
            "data_bbox_captured_centres",
            data_min,
            data_max,
            captured_shape,
            centre_function(data_min, data_max, captured_shape),
            floor_indexer,
        ),
        (
            "data_bbox_captured_nodes",
            data_min,
            data_max,
            captured_shape,
            _endpoint_nodes(data_min, data_max, captured_shape),
            _nearest_node_indices,
        ),
        (
            "model_bbox_auto_centres",
            model_min,
            model_max,
            tuple(int(value) for value in model_auto_shape),
            centre_function(model_min, model_max, model_auto_shape),
            floor_indexer,
        ),
        (
            "model_bbox_captured_centres",
            model_min,
            model_max,
            captured_shape,
            centre_function(model_min, model_max, captured_shape),
            floor_indexer,
        ),
        (
            "model_bbox_captured_nodes",
            model_min,
            model_max,
            captured_shape,
            _endpoint_nodes(model_min, model_max, captured_shape),
            _nearest_node_indices,
        ),
    ]

    print(
        f"case={case_name} points={len(points)} oracle_domains={len(np.unique(oracle_labels))}",
        flush=True,
    )
    print(
        f"data_bbox={tuple(data_min)} -> {tuple(data_max)} auto_shape={tuple(data_auto_shape)}",
        flush=True,
    )
    print(
        f"model_bbox={tuple(model_min)} -> {tuple(model_max)} auto_shape={tuple(model_auto_shape)} "
        f"captured_shape={captured_shape}",
        flush=True,
    )

    results = []
    for name, minimum, maximum, shape, centroids, point_indexer in variants:
        result = _run_geometry(
            name=name,
            points=np.asarray(points, dtype=np.float64),
            oracle_labels=np.asarray(oracle_labels, dtype=np.int64),
            minimum=np.asarray(minimum, dtype=np.float64),
            maximum=np.asarray(maximum, dtype=np.float64),
            shape=tuple(int(value) for value in shape),
            centroids=np.asarray(centroids, dtype=np.float64),
            point_indexer=point_indexer,
            builder_class=builder_class,
            builder_module=builder_module,
            trend_input=trend_input,
            exact_module=exact,
            threshold=args.threshold,
            centroid_count=args.centroid_count,
            minimum_fraction=args.minimum_fraction,
            maximum_fraction=args.maximum_fraction,
        )
        results.append(result)
        print(
            f"{result.geometry:>31} grid={result.grid_shape!s:>13} "
            f"pred={result.predicted_domains:>3d} ARI={result.adjusted_rand_index:>8.5f} "
            f"AMI={result.adjusted_mutual_information:>8.5f} "
            f"match={result.optimal_label_accuracy:>8.5f} merges={result.merge_count:>5d}",
            flush=True,
        )

    results.sort(
        key=lambda item: (
            -item.adjusted_rand_index,
            -item.optimal_label_accuracy,
            abs(item.predicted_domains - item.oracle_domains),
            item.geometry,
        )
    )
    print("\nBest geometries:", flush=True)
    for item in results:
        print(
            f"{item.geometry:>31} pred={item.predicted_domains:>3d} "
            f"ARI={item.adjusted_rand_index:>8.5f} match={item.optimal_label_accuracy:>8.5f} "
            f"grid={item.grid_shape}",
            flush=True,
        )

    payload = {
        "case": case_name,
        "input_points": int(len(points)),
        "oracle_domains": int(len(np.unique(oracle_labels))),
        "threshold": float(args.threshold),
        "minimum_fraction": float(args.minimum_fraction),
        "maximum_fraction": float(args.maximum_fraction),
        "captured_shape": list(captured_shape),
        "model_bbox_min": model_min.tolist(),
        "model_bbox_max": model_max.tolist(),
        "results": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
