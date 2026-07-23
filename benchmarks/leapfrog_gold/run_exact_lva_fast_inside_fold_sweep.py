"""Run the fold LVA sweep with local domains and inside-only component filtering.

This is the faster alternative to ``run_exact_lva_full_model_fold_sweep.py``.
It keeps the confirmed lower-Z full-depth correction from the fold runner, but does
not extend every automatic domain across the complete 5x model box. The generated
mesh is then split into connected components and unsupported enclosing shells are
removed. Only the cleaned data-supported OBJ is retained.

The common benchmark defaults remain 50,000,000 base cells and 256 scalar slabs.
For this fold runner only, either limit can be disabled by setting its environment
variable to zero before Python starts:

- POLATORY_FOLD_MAX_TOTAL_BASE_CELLS=0
- POLATORY_FOLD_MAX_CHUNKS=0

A positive value replaces the corresponding default with that explicit limit.
Use the same remaining environment variables as
``run_exact_lva_full_depth_fold_sweep.py``.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault(
    "POLATORY_FOLD_OUTPUT_NAME",
    "fold-fast-inside-only-sweep-5x-extent",
)

import run_exact_lva_full_depth_fold_sweep as fold  # noqa: E402

suite = fold.suite
diagnostic = fold.diagnostic

_BASE_BUILD_CASE = suite.build_case
_BASE_GENERATE_ISOSURFACE = suite.SAFE_MESHER.generate_safe_isosurface
_CURRENT_POINTS: np.ndarray | None = None
_LIMIT_MESSAGE_PRINTED = False


def _configured_limit(name: str, current: int) -> int:
    """Read a positive integer limit; zero means no practical Python-side limit."""
    text = os.environ.get(name, "").strip()
    if not text:
        return int(current)
    value = int(text)
    if value < 0:
        raise ValueError(f"{name} must be zero or a positive integer")
    return int(sys.maxsize if value == 0 else value)


def _apply_fold_meshing_limits() -> None:
    """Override limits after the common benchmark resets its safe defaults."""
    global _LIMIT_MESSAGE_PRINTED
    cells = _configured_limit(
        "POLATORY_FOLD_MAX_TOTAL_BASE_CELLS",
        int(suite.SAFE_MESHER.MAX_TOTAL_BASE_CELLS),
    )
    chunks = _configured_limit(
        "POLATORY_FOLD_MAX_CHUNKS",
        int(suite.SAFE_MESHER.MAX_CHUNKS),
    )
    suite.SAFE_MESHER.MAX_TOTAL_BASE_CELLS = cells
    suite.SAFE_MESHER.MAX_CHUNKS = chunks

    if not _LIMIT_MESSAGE_PRINTED:
        cell_text = "unlimited" if cells == sys.maxsize else f"{cells:,}"
        chunk_text = "unlimited" if chunks == sys.maxsize else f"{chunks:,}"
        print(
            "PROGRESS\tFold meshing safety limits: "
            f"base cells={cell_text}, scalar slabs={chunk_text}. "
            "The scalar field remains slab-streamed; disabling these guards does not "
            "make the computation small.",
            flush=True,
        )
        _LIMIT_MESSAGE_PRINTED = True


def _support_distance(points: np.ndarray) -> tuple[float, float]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        median_spacing = float(suite.SURFACE_RESOLUTION)
    else:
        nearest = np.asarray(
            suite.cKDTree(points).query(points, k=2)[0][:, 1],
            dtype=np.float64,
        )
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        median_spacing = (
            float(np.median(nearest))
            if len(nearest)
            else float(suite.SURFACE_RESOLUTION)
        )

    explicit = os.environ.get(
        "POLATORY_FOLD_COMPONENT_SUPPORT_DISTANCE", ""
    ).strip()
    if explicit:
        threshold = float(explicit)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "POLATORY_FOLD_COMPONENT_SUPPORT_DISTANCE must be positive and finite"
            )
    else:
        multiplier = float(
            os.environ.get("POLATORY_FOLD_COMPONENT_SPACING_MULTIPLIER", "2")
        )
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(
                "POLATORY_FOLD_COMPONENT_SPACING_MULTIPLIER must be positive and finite"
            )
        threshold = max(
            3.0 * float(suite.SURFACE_RESOLUTION),
            multiplier * median_spacing,
        )
    return float(threshold), float(median_spacing)


def _component_from_region(connected: Any, region_id: int) -> Any:
    region_values = np.asarray(connected.cell_data["RegionId"], dtype=np.int64)
    selected = connected.extract_cells(region_values == int(region_id))
    return selected.extract_surface().triangulate().clean()


def _merge_components(components: list[Any]) -> Any:
    merged = components[0].copy(deep=True)
    for component in components[1:]:
        merged = merged.merge(component, merge_points=False)
    return merged.extract_surface().triangulate().clean()


def _filter_supported_components(output_obj: Path, points: np.ndarray) -> dict[str, Any]:
    output_obj = Path(output_obj)
    points = np.asarray(points, dtype=np.float64)
    mesh = suite.pv.read(output_obj).extract_surface().triangulate().clean()
    connected = mesh.connectivity()
    if "RegionId" not in connected.cell_data:
        return {
            "raw_component_count": 1,
            "kept_component_count": 1,
            "support_filter_applied": False,
        }

    region_ids = sorted(
        int(value)
        for value in np.unique(
            np.asarray(connected.cell_data["RegionId"], dtype=np.int64)
        )
    )
    threshold, median_spacing = _support_distance(points)
    min_points = max(
        int(os.environ.get("POLATORY_FOLD_COMPONENT_MIN_SUPPORT_POINTS", "3")),
        int(
            math.ceil(
                float(
                    os.environ.get(
                        "POLATORY_FOLD_COMPONENT_MIN_SUPPORT_FRACTION", "0.01"
                    )
                )
                * len(points)
            )
        ),
    )

    point_cloud = suite.pv.PolyData(points)
    components: list[Any] = []
    records: list[dict[str, Any]] = []
    for region_id in region_ids:
        component = _component_from_region(connected, region_id)
        measured = point_cloud.compute_implicit_distance(component)
        distances = np.abs(
            np.asarray(measured["implicit_distance"], dtype=np.float64)
        )
        support_count = int(np.count_nonzero(distances <= threshold))
        record = {
            "region_id": int(region_id),
            "vertices": int(component.n_points),
            "triangles": int(component.n_cells),
            "bounds": [float(value) for value in component.bounds],
            "minimum_data_distance": float(np.min(distances)),
            "median_data_distance": float(np.median(distances)),
            "p90_data_distance": float(np.percentile(distances, 90.0)),
            "support_distance": float(threshold),
            "support_point_count": support_count,
            "support_fraction": float(support_count / len(points)),
            "kept": bool(support_count >= min_points),
        }
        records.append(record)
        components.append(component)

    kept_indices = [index for index, record in enumerate(records) if record["kept"]]
    if not kept_indices:
        best = int(
            np.argmin([record["median_data_distance"] for record in records])
        )
        records[best]["kept"] = True
        records[best]["fallback_closest_component"] = True
        kept_indices = [best]

    cleaned = _merge_components([components[index] for index in kept_indices])
    cleaned.save(output_obj)

    stale_raw = output_obj.with_name(output_obj.stem + "_all_components.obj")
    if stale_raw.exists():
        stale_raw.unlink()

    report = {
        "cleaned_obj": str(output_obj),
        "raw_component_count": int(len(records)),
        "kept_component_count": int(len(kept_indices)),
        "median_input_spacing": float(median_spacing),
        "support_distance": float(threshold),
        "minimum_support_points": int(min_points),
        "components": records,
    }
    report_path = output_obj.with_name(output_obj.stem + "_components.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"PROGRESS\tInside-only component filter kept {len(kept_indices)}/{len(records)} "
        f"components; threshold={threshold:g}, median input spacing={median_spacing:g}.",
        flush=True,
    )
    return report


def _filtered_generate_isosurface(*args: Any, **kwargs: Any):
    # run_real_gold_suite.py resets its conservative defaults immediately before
    # calling us. Apply fold-specific overrides here so zero/unlimited is honoured.
    _apply_fold_meshing_limits()
    result = _BASE_GENERATE_ISOSURFACE(*args, **kwargs)
    output_obj = kwargs.get("output_obj")
    if output_obj is None:
        raise RuntimeError("The fold mesher did not provide output_obj")
    if _CURRENT_POINTS is None:
        raise RuntimeError("Fold component filtering has no current input points")
    _filter_supported_components(Path(output_obj), _CURRENT_POINTS)
    return result


suite.SAFE_MESHER.generate_safe_isosurface = _filtered_generate_isosurface


def _inside_only_build_case(
    case: Any,
    points: np.ndarray,
    indicators: np.ndarray,
    trend_vertices: np.ndarray,
    trend_faces: np.ndarray,
) -> dict[str, Any]:
    global _CURRENT_POINTS
    _CURRENT_POINTS = np.asarray(points, dtype=np.float64)
    try:
        result = _BASE_BUILD_CASE(
            case,
            points,
            indicators,
            trend_vertices,
            trend_faces,
        )
    finally:
        _CURRENT_POINTS = None

    output_path = suite.ROOT / result["generated_obj"]
    component_report_path = output_path.with_name(
        output_path.stem + "_components.json"
    )
    if component_report_path.is_file():
        result["component_filter"] = json.loads(
            component_report_path.read_text(encoding="utf-8")
        )
    return result


suite.build_case = _inside_only_build_case

print(
    "PROGRESS\tFast inside-only fold mode enabled: automatic domains remain local "
    "except for the confirmed lower-Z extension, and unsupported disconnected outer "
    "shells are removed after meshing. Only cleaned OBJs are retained.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(fold.main())
