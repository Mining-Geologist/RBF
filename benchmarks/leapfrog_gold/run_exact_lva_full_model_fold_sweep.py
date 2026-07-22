"""Run the fold LVA sweep with full-model domains and support-aware components.

The 5x model extent can expose a second zero surface at the finite automatic-domain
envelope. This runner keeps the same data, LVA matrices, support memberships and
local RBF models, but extends every domain evaluation box to the complete model
bounds. After meshing, disconnected components with no nearby input support are
removed from the primary OBJ.

By default only the cleaned, data-supported OBJ is retained. Set
``POLATORY_FOLD_KEEP_ALL_COMPONENTS=1`` only when the unfiltered diagnostic OBJ is
also required.

Use the same environment variables as run_exact_lva_full_depth_fold_sweep.py.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

# Keep this run separate from the earlier full-depth-only fold output.
os.environ.setdefault(
    "POLATORY_FOLD_OUTPUT_NAME",
    "fold-exact-lva-full-model-sweep-5x-extent",
)

import run_exact_lva_full_depth_fold_sweep as fold  # noqa: E402

suite = fold.suite
diagnostic = fold.diagnostic

_BASE_DOMAIN_BUILDER = suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder
_BASE_BUILD_CASE = suite.build_case
_BASE_GENERATE_ISOSURFACE = suite.SAFE_MESHER.generate_safe_isosurface
_CURRENT_POINTS: np.ndarray | None = None

_KEEP_ALL_COMPONENTS = os.environ.get(
    "POLATORY_FOLD_KEEP_ALL_COMPONENTS", "0"
).strip().casefold() in {"1", "true", "yes", "on"}


class _FullModelDomainBuilder:
    """Extend all six faces of each automatic domain to the model bbox."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._wrapped = _BASE_DOMAIN_BUILDER(*args, **kwargs)

    def build_from_inputs(self, *args: Any, **kwargs: Any):
        original_domains = list(self._wrapped.build_from_inputs(*args, **kwargs))
        bbox_min = np.asarray(suite.MODEL_MIN, dtype=np.float64)
        bbox_max = np.asarray(suite.MODEL_MAX, dtype=np.float64)
        extended_domains = [
            fold.sweep.full_depth.polatory.StructuralDomain3(
                np.asarray(domain.anisotropy, dtype=np.float64),
                bbox_min.copy(),
                bbox_max.copy(),
                np.asarray(domain.support_indices, dtype=np.int64).tolist(),
                np.asarray(domain.model_parameters, dtype=np.float64).tolist(),
            )
            for domain in original_domains
        ]

        # Diagnostics and CSV exports must see the domains actually used to fit.
        diagnostic._CAPTURE["domains"] = extended_domains
        print(
            "PROGRESS\tExtended every automatic domain to the complete model bbox: "
            f"min={bbox_min.tolist()}, max={bbox_max.tolist()}. LVA matrices, support "
            "indices and local RBF parameters are unchanged.",
            flush=True,
        )
        return extended_domains

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder = _FullModelDomainBuilder


def _support_distance(points: np.ndarray) -> tuple[float, float]:
    """Return the component support threshold and median input spacing."""
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
    """Keep connected isosurfaces supported by nearby input data."""
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
        # Conservative fallback: retain the component that is globally closest to
        # the input samples instead of producing an empty mesh.
        best = int(
            np.argmin([record["median_data_distance"] for record in records])
        )
        records[best]["kept"] = True
        records[best]["fallback_closest_component"] = True
        kept_indices = [best]

    raw_path = output_obj.with_name(output_obj.stem + "_all_components.obj")
    if _KEEP_ALL_COMPONENTS:
        shutil.copy2(output_obj, raw_path)
    elif raw_path.exists():
        # Remove a stale diagnostic OBJ from an earlier run in the same folder.
        raw_path.unlink()

    cleaned = _merge_components([components[index] for index in kept_indices])
    cleaned.save(output_obj)

    report = {
        "raw_obj": str(raw_path) if _KEEP_ALL_COMPONENTS else None,
        "cleaned_obj": str(output_obj),
        "raw_component_count": int(len(records)),
        "kept_component_count": int(len(kept_indices)),
        "median_input_spacing": float(median_spacing),
        "support_distance": float(threshold),
        "minimum_support_points": int(min_points),
        "kept_all_components_obj": bool(_KEEP_ALL_COMPONENTS),
        "components": records,
    }
    report_path = output_obj.with_name(output_obj.stem + "_components.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    removed = len(records) - len(kept_indices)
    extra = f" Raw OBJ: {raw_path.name}" if _KEEP_ALL_COMPONENTS else ""
    print(
        f"PROGRESS\tComponent support filter kept {len(kept_indices)}/{len(records)} "
        f"components and removed {removed}; threshold={threshold:g}, "
        f"median input spacing={median_spacing:g}.{extra}",
        flush=True,
    )
    return report


def _filtered_generate_isosurface(*args: Any, **kwargs: Any):
    result = _BASE_GENERATE_ISOSURFACE(*args, **kwargs)
    output_obj = kwargs.get("output_obj")
    if output_obj is None:
        raise RuntimeError("The fold mesher did not provide output_obj")
    if _CURRENT_POINTS is None:
        raise RuntimeError("Fold component filtering has no current input points")
    _filter_supported_components(Path(output_obj), _CURRENT_POINTS)
    return result


suite.SAFE_MESHER.generate_safe_isosurface = _filtered_generate_isosurface


def _support_aware_build_case(
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


suite.build_case = _support_aware_build_case

print(
    "PROGRESS\tFull-model fold correction enabled: all automatic domain boxes span "
    "the model bbox, and disconnected isosurfaces without nearby data support are "
    "removed from the primary OBJ. "
    + (
        "Raw all-component OBJs will also be preserved."
        if _KEEP_ALL_COMPONENTS
        else "Only cleaned data-supported OBJs will be retained."
    ),
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(fold.main())
