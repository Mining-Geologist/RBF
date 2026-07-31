"""Diagnose the range-driven flat basal closure in the Leapfrog LVA benchmark.

This runner keeps the current topology-local automatic-support V2 model unchanged and
adds measurements only. It runs one or more requested benchmark cases, captures the
fitted structural interpolant and automatic domains, samples vertical scalar profiles,
and records where the lowest raw/final zero crossings occur relative to finite-domain
coverage.

The default comparison is S3_R100 versus S3_R500. Override it with either
``POLATORY_BASAL_CASES`` (comma separated) or the existing
``POLATORY_BENCHMARK_CASE`` variable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import polatory

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Installs the current V2 topology-local correction and exposes the benchmark suite.
import run_selected_no_background_blending_auto_support_decay_v2 as current  # noqa: E402

suite = current.suite
_ORIGINAL_AVAILABLE_CASES = current.v1.selected._ORIGINAL_AVAILABLE_CASES
_ORIGINAL_BUILD_CASE = suite.build_case
_ORIGINAL_BUILDER = suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder
_ORIGINAL_STRUCTURAL_FACTORY = polatory.StructuralInterpolant3

_CAPTURE: dict[str, Any] = {
    "structural": None,
    "domains": None,
    "builder": None,
}

suite.OUTPUT_DIR = suite.ROOT / "benchmark-results" / "diagnostic-basal-range-v3"
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"
DIAGNOSTIC_DIR = suite.OUTPUT_DIR / "basal-diagnostics"

XY_COUNT = max(7, int(os.environ.get("POLATORY_BASAL_XY_COUNT", "25")))
Z_STEP = float(
    os.environ.get(
        "POLATORY_BASAL_Z_STEP",
        str(max(float(suite.SURFACE_RESOLUTION) * 0.5, 1.0)),
    )
)
EVALUATION_BATCH_SIZE = max(
    1, int(os.environ.get("POLATORY_BASAL_EVALUATION_BATCH_SIZE", "100000"))
)
if not np.isfinite(Z_STEP) or Z_STEP <= 0.0:
    raise ValueError("POLATORY_BASAL_Z_STEP must be a positive finite distance.")


def _requested_cases():
    requested = (
        os.environ.get("POLATORY_BASAL_CASES", "").strip()
        or os.environ.get("POLATORY_BENCHMARK_CASES", "").strip()
        or os.environ.get("POLATORY_BENCHMARK_CASE", "").strip()
        or "S3_R100,S3_R500"
    )
    names = [item.strip().upper() for item in requested.split(",") if item.strip()]
    if not names:
        raise ValueError("No benchmark cases were requested.")

    previous_max_cases = suite.MAX_CASES
    suite.MAX_CASES = 0
    try:
        available = _ORIGINAL_AVAILABLE_CASES()
    finally:
        suite.MAX_CASES = previous_max_cases

    by_name = {case.name.upper(): case for case in available}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            f"Unknown benchmark case(s) {missing}. Available cases: "
            + ", ".join(sorted(by_name))
        )
    return [by_name[name] for name in names]


suite.available_cases = _requested_cases


class _CapturingBuilder:
    """Delegate to the production builder while retaining its returned domains."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._wrapped = _ORIGINAL_BUILDER(*args, **kwargs)
        _CAPTURE["builder"] = self._wrapped

    def build_from_inputs(self, *args: Any, **kwargs: Any):
        domains = list(self._wrapped.build_from_inputs(*args, **kwargs))
        _CAPTURE["domains"] = domains
        return domains

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _capturing_structural_factory(*args: Any, **kwargs: Any):
    structural = _ORIGINAL_STRUCTURAL_FACTORY(*args, **kwargs)
    _CAPTURE["structural"] = structural
    return structural


suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder = _CapturingBuilder
polatory.StructuralInterpolant3 = _capturing_structural_factory


def _evaluate_batched(
    evaluator: Callable[[np.ndarray], Any],
    query: np.ndarray,
) -> np.ndarray:
    values = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), EVALUATION_BATCH_SIZE):
        stop = min(start + EVALUATION_BATCH_SIZE, len(query))
        values[start:stop] = np.asarray(
            evaluator(query[start:stop]), dtype=np.float64
        ).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Structural field returned non-finite diagnostic values.")
    return values


def _zero_crossings(volume: np.ndarray, z_coordinates: np.ndarray) -> dict[str, np.ndarray]:
    lower = np.asarray(volume[:, :, :-1], dtype=np.float64)
    upper = np.asarray(volume[:, :, 1:], dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(volume))))
    tolerance = np.finfo(np.float64).eps * scale
    lower_sign = np.where(np.abs(lower) <= tolerance, -tolerance, lower)
    upper_sign = np.where(np.abs(upper) <= tolerance, -tolerance, upper)
    crossing = ((lower_sign < 0.0) & (upper_sign > 0.0)) | (
        (lower_sign > 0.0) & (upper_sign < 0.0)
    )
    crossing_count = np.sum(crossing, axis=2).astype(np.int64)
    has_crossing = crossing_count > 0

    first_index = np.argmax(crossing, axis=2)
    last_index = crossing.shape[2] - 1 - np.argmax(crossing[:, :, ::-1], axis=2)

    def interpolate(indices: np.ndarray) -> np.ndarray:
        gathered_lower = np.take_along_axis(lower, indices[:, :, None], axis=2)[:, :, 0]
        gathered_upper = np.take_along_axis(upper, indices[:, :, None], axis=2)[:, :, 0]
        denominator = gathered_upper - gathered_lower
        fraction = np.divide(
            -gathered_lower,
            denominator,
            out=np.full_like(gathered_lower, 0.5),
            where=np.abs(denominator) > tolerance,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        base = z_coordinates[indices]
        return base + fraction * (z_coordinates[indices + 1] - base)

    lowest = interpolate(first_index)
    highest = interpolate(last_index)
    lowest[~has_crossing] = np.nan
    highest[~has_crossing] = np.nan
    return {
        "count": crossing_count,
        "lowest": lowest,
        "highest": highest,
        "has": has_crossing,
    }


def _mode_summary(values: np.ndarray, bin_width: float) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {
            "count": 0,
            "mode_z": None,
            "mode_count": 0,
            "mode_fraction": 0.0,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
            "standard_deviation": None,
        }

    rounded = np.round(finite / bin_width) * bin_width
    coordinates, counts = np.unique(rounded, return_counts=True)
    mode_index = int(np.argmax(counts))
    return {
        "count": int(len(finite)),
        "mode_z": float(coordinates[mode_index]),
        "mode_count": int(counts[mode_index]),
        "mode_fraction": float(counts[mode_index] / len(finite)),
        "minimum": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10.0)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "maximum": float(np.max(finite)),
        "standard_deviation": float(np.std(finite)),
    }


def _domain_coverage(points: np.ndarray, domains: Sequence[Any]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.int64)
    for domain in domains:
        minimum = np.asarray(domain.bbox_min, dtype=np.float64)
        maximum = np.asarray(domain.bbox_max, dtype=np.float64)
        counts += np.all(points >= minimum[None, :], axis=1) & np.all(
            points <= maximum[None, :], axis=1
        )
    return counts


def _save_crossing_map(
    path: Path,
    values: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    title: str,
) -> None:
    figure = plt.figure(figsize=(8, 6))
    image = plt.imshow(
        values.T,
        origin="lower",
        extent=(
            float(x_coordinates[0]),
            float(x_coordinates[-1]),
            float(y_coordinates[0]),
            float(y_coordinates[-1]),
        ),
        aspect="equal",
    )
    plt.colorbar(image, label="Lowest zero-crossing Z")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_crossing_histogram(
    path: Path,
    lowest_crossings: np.ndarray,
    domain_minimum_z: np.ndarray,
    title: str,
) -> None:
    finite = np.asarray(lowest_crossings, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    figure = plt.figure(figsize=(8, 5))
    if len(finite):
        bin_count = max(10, min(80, int(np.ceil((finite.max() - finite.min()) / Z_STEP))))
        plt.hist(finite, bins=bin_count)
    for value in np.asarray(domain_minimum_z, dtype=np.float64):
        plt.axvline(float(value), linewidth=0.4, alpha=0.15)
    plt.xlabel("Lowest zero-crossing Z")
    plt.ylabel("Vertical probe count")
    plt.title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _diagnose_case(
    case: Any,
    result: dict[str, Any],
    points: np.ndarray,
) -> dict[str, Any]:
    structural = _CAPTURE.get("structural")
    domains = _CAPTURE.get("domains")
    builder = _CAPTURE.get("builder")
    if structural is None or not domains:
        raise RuntimeError("The diagnostic did not capture the fitted structural model.")

    generated_path = suite.ROOT / result["generated_obj"]
    generated = suite.polydata(generated_path)
    bounds = generated.bounds
    x_coordinates = np.linspace(float(bounds[0]), float(bounds[1]), XY_COUNT)
    y_coordinates = np.linspace(float(bounds[2]), float(bounds[3]), XY_COUNT)
    z_coordinates = np.arange(
        float(suite.MODEL_MIN[2]),
        float(suite.MODEL_MAX[2]) + 0.5 * Z_STEP,
        Z_STEP,
        dtype=np.float64,
    )

    x_grid, y_grid, z_grid = np.meshgrid(
        x_coordinates,
        y_coordinates,
        z_coordinates,
        indexing="ij",
    )
    query = np.column_stack(
        [x_grid.ravel(order="C"), y_grid.ravel(order="C"), z_grid.ravel(order="C")]
    )
    print(
        f"[{case.name}] Basal diagnostic sampling {len(query):,} scalar nodes "
        f"({XY_COUNT} x {XY_COUNT} vertical probes, dz={Z_STEP:g} m)…",
        flush=True,
    )

    raw_evaluator = getattr(structural, "_raw_evaluate", structural.evaluate)
    raw_values = _evaluate_batched(raw_evaluator, query)
    final_values = _evaluate_batched(structural.evaluate, query)
    shape = (len(x_coordinates), len(y_coordinates), len(z_coordinates))
    raw_volume = raw_values.reshape(shape, order="C")
    final_volume = final_values.reshape(shape, order="C")
    raw_crossings = _zero_crossings(raw_volume, z_coordinates)
    final_crossings = _zero_crossings(final_volume, z_coordinates)

    xy_grid_x, xy_grid_y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
    final_has = final_crossings["has"]
    final_points = np.column_stack(
        [
            xy_grid_x[final_has],
            xy_grid_y[final_has],
            final_crossings["lowest"][final_has],
        ]
    )
    final_domain_counts = np.full(final_has.shape, -1, dtype=np.int64)
    final_nearest_data = np.full(final_has.shape, np.nan, dtype=np.float64)
    if len(final_points):
        final_domain_counts[final_has] = _domain_coverage(final_points, domains)
        final_nearest_data[final_has] = np.asarray(
            cKDTree(np.asarray(points, dtype=np.float64)).query(final_points, k=1)[0],
            dtype=np.float64,
        )

    domain_minimum = np.vstack(
        [np.asarray(domain.bbox_min, dtype=np.float64) for domain in domains]
    )
    domain_maximum = np.vstack(
        [np.asarray(domain.bbox_max, dtype=np.float64) for domain in domains]
    )
    internal_radii: list[float] = []
    diagnostics = getattr(builder, "diagnostics_", None)
    if diagnostics is not None:
        internal_radii = [
            float(item.internal_radius) for item in getattr(diagnostics, "postcluster", [])
        ]

    raw_summary = _mode_summary(raw_crossings["lowest"], Z_STEP)
    final_summary = _mode_summary(final_crossings["lowest"], Z_STEP)
    domain_min_z_summary = _mode_summary(domain_minimum[:, 2], Z_STEP)
    mode_z = final_summary.get("mode_z")
    alignment = (
        float(np.min(np.abs(domain_minimum[:, 2] - float(mode_z))))
        if mode_z is not None
        else None
    )

    calibration = getattr(structural, "calibration_", None)
    summary: dict[str, Any] = {
        "case": case.name,
        "strength": float(case.strength),
        "trend_range": float(case.trend_range),
        "xy_probe_count_per_axis": int(XY_COUNT),
        "vertical_step": float(Z_STEP),
        "scalar_nodes": int(len(query)),
        "domain_count": int(len(domains)),
        "domain_bbox_min_z": domain_minimum[:, 2].tolist(),
        "domain_bbox_max_z": domain_maximum[:, 2].tolist(),
        "domain_minimum_z_summary": domain_min_z_summary,
        "internal_radii": internal_radii,
        "raw_lowest_crossing": raw_summary,
        "final_lowest_crossing": final_summary,
        "final_mode_to_nearest_domain_min_z": alignment,
        "final_crossing_domain_count": _mode_summary(
            final_domain_counts[final_domain_counts >= 0].astype(np.float64), 1.0
        ),
        "final_crossing_nearest_data_distance": _mode_summary(
            final_nearest_data, max(Z_STEP, 1.0)
        ),
        "support_calibration": calibration,
    }

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "x": xy_grid_x.ravel(order="C"),
            "y": xy_grid_y.ravel(order="C"),
            "raw_crossing_count": raw_crossings["count"].ravel(order="C"),
            "raw_lowest_z": raw_crossings["lowest"].ravel(order="C"),
            "raw_highest_z": raw_crossings["highest"].ravel(order="C"),
            "final_crossing_count": final_crossings["count"].ravel(order="C"),
            "final_lowest_z": final_crossings["lowest"].ravel(order="C"),
            "final_highest_z": final_crossings["highest"].ravel(order="C"),
            "active_domain_boxes_at_final_lowest": final_domain_counts.ravel(order="C"),
            "nearest_data_distance_at_final_lowest": final_nearest_data.ravel(order="C"),
        }
    )
    frame.to_csv(DIAGNOSTIC_DIR / f"{case.name}_vertical_crossings.csv", index=False)
    (DIAGNOSTIC_DIR / f"{case.name}_basal_diagnostic.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _save_crossing_map(
        DIAGNOSTIC_DIR / f"{case.name}_raw_lowest_crossing.png",
        raw_crossings["lowest"],
        x_coordinates,
        y_coordinates,
        f"{case.name}: raw lowest zero crossing",
    )
    _save_crossing_map(
        DIAGNOSTIC_DIR / f"{case.name}_final_lowest_crossing.png",
        final_crossings["lowest"],
        x_coordinates,
        y_coordinates,
        f"{case.name}: V2 final lowest zero crossing",
    )
    _save_crossing_histogram(
        DIAGNOSTIC_DIR / f"{case.name}_final_lowest_crossing_histogram.png",
        final_crossings["lowest"],
        domain_minimum[:, 2],
        f"{case.name}: lowest crossing distribution and domain minima",
    )

    print(
        f"[{case.name}] Basal diagnostic: raw mode={raw_summary['mode_z']}, "
        f"raw mode fraction={raw_summary['mode_fraction']:.3f}; "
        f"final mode={final_summary['mode_z']}, "
        f"final mode fraction={final_summary['mode_fraction']:.3f}; "
        f"nearest domain-min alignment={alignment}.",
        flush=True,
    )
    return summary


def _diagnostic_build_case(
    case: Any,
    points: np.ndarray,
    indicators: np.ndarray,
    trend_vertices: np.ndarray,
    trend_faces: np.ndarray,
) -> dict[str, Any]:
    _CAPTURE.update({"structural": None, "domains": None, "builder": None})
    result = _ORIGINAL_BUILD_CASE(
        case,
        points,
        indicators,
        trend_vertices,
        trend_faces,
    )
    result["basal_range_diagnostic"] = _diagnose_case(case, result, points)
    return result


suite.build_case = _diagnostic_build_case


def _write_cross_case_comparison() -> None:
    summaries: list[dict[str, Any]] = []
    for path in sorted(DIAGNOSTIC_DIR.glob("*_basal_diagnostic.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        return
    summaries.sort(key=lambda item: (float(item["strength"]), float(item["trend_range"])))
    comparison = {
        "cases": [
            {
                "case": item["case"],
                "strength": item["strength"],
                "trend_range": item["trend_range"],
                "raw_mode_z": item["raw_lowest_crossing"]["mode_z"],
                "raw_mode_fraction": item["raw_lowest_crossing"]["mode_fraction"],
                "raw_p10_p90_span": (
                    item["raw_lowest_crossing"]["p90"]
                    - item["raw_lowest_crossing"]["p10"]
                    if item["raw_lowest_crossing"]["p90"] is not None
                    else None
                ),
                "final_mode_z": item["final_lowest_crossing"]["mode_z"],
                "final_mode_fraction": item["final_lowest_crossing"]["mode_fraction"],
                "final_p10_p90_span": (
                    item["final_lowest_crossing"]["p90"]
                    - item["final_lowest_crossing"]["p10"]
                    if item["final_lowest_crossing"]["p90"] is not None
                    else None
                ),
                "mode_to_nearest_domain_min_z": item[
                    "final_mode_to_nearest_domain_min_z"
                ],
            }
            for item in summaries
        ]
    }
    (DIAGNOSTIC_DIR / "cross_case_basal_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(
        "PROGRESS\tWrote cross-case basal comparison to "
        f"{DIAGNOSTIC_DIR / 'cross_case_basal_comparison.json'}",
        flush=True,
    )


print(
    "PROGRESS\tDiagnostic mode: compare range-driven lowest zero crossings, current "
    "topology-local V2 field, finite-domain minima, and active domain coverage.",
    flush=True,
)


if __name__ == "__main__":
    exit_code = suite.main()
    _write_cross_case_comparison()
    raise SystemExit(exit_code)
