"""Overlap-and-own meshing for the process-isolated LVA application.

The native isosurface extractor is evaluated on padded chunks.  Complete triangles
are then assigned to exactly one unpadded core by their cell centres.  Unlike a
geometric clip, this does not cut triangles on the core planes and therefore cannot
manufacture a new flat wall or pinched termination at a temporary chunk boundary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvista as pv

import polatory
from polatory import three as p3


def _owned_cell_indices(
    mesh: pv.DataSet,
    core_lower: np.ndarray,
    core_upper: np.ndarray,
    model_upper: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    centres = np.asarray(mesh.cell_centers().points, dtype=float)
    lower = np.asarray(core_lower, dtype=float)
    upper = np.asarray(core_upper, dtype=float)
    model_upper = np.asarray(model_upper, dtype=float)

    owned = np.all(centres >= lower[None, :] - tolerance, axis=1)
    for axis in range(3):
        is_last = abs(float(upper[axis] - model_upper[axis])) <= tolerance
        if is_last:
            owned &= centres[:, axis] <= upper[axis] + tolerance
        else:
            # Half-open ownership makes every overlapping triangle belong to one
            # core only, while retaining the triangle itself without cutting it.
            owned &= centres[:, axis] < upper[axis] - tolerance
    return np.flatnonzero(owned).astype(np.int64)


def install_chunk_overlap(safe_module: Any) -> None:
    """Replace ``generate_safe_isosurface`` on the supplied v3-safe module."""

    if getattr(safe_module, "_overlap_chunk_meshing_installed", False):
        return

    original = safe_module.generate_safe_isosurface

    def generate_overlap_isosurface(
        structural: Any,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        resolution: float,
        refine: int,
        output_obj: Path,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        bbox_min_array = np.asarray(bbox_min, dtype=float)
        bbox_max_array = np.asarray(bbox_max, dtype=float)
        resolution_value = float(resolution)
        plan = safe_module.grid_plan(
            bbox_min_array,
            bbox_max_array,
            resolution_value,
        )

        if plan["chunk_total"] == 1:
            return original(
                structural,
                bbox_min_array,
                bbox_max_array,
                resolution_value,
                refine,
                output_obj,
                progress,
            )

        if plan["total"] > safe_module.MAX_TOTAL_BASE_CELLS:
            suggested = safe_module.recommended_resolution(
                bbox_min_array,
                bbox_max_array,
                resolution_value,
                safe_module.MAX_TOTAL_BASE_CELLS,
            )
            raise MemoryError(
                f"Resolution {resolution_value:g} creates {plan['total']:,} base "
                f"cells. The safety limit is "
                f"{safe_module.MAX_TOTAL_BASE_CELLS:,}. Use approximately "
                f"{suggested:.6g} or coarser, or reduce the bounding box."
            )
        if plan["chunk_total"] > safe_module.MAX_CHUNKS:
            raise MemoryError(
                f"The mesh needs {plan['chunk_total']:,} chunks, above the safety "
                f"limit of {safe_module.MAX_CHUNKS}. Use a coarser resolution or "
                "smaller extent."
            )

        cores = list(
            safe_module.chunk_bounds(
                bbox_min_array,
                bbox_max_array,
                resolution_value,
                plan["cells"],
                plan["chunks"],
            )
        )
        field = polatory.StructuralRbfFieldFunction(structural)

        # Keep two complete base cells on each side of a core. The padded chunks
        # therefore extract the same crossing triangles before ownership is decided.
        padding = 2.0 * resolution_value
        tolerance = max(1.0e-8 * resolution_value, 1.0e-9)
        progress(
            f"Fine grid: {plan['total']:,} base cells. Processing safely in "
            f"{plan['chunk_total']:,} overlapping chunks "
            f"{tuple(int(value) for value in plan['chunks'])}; complete triangles "
            "will be assigned by cell centre without clipping…"
        )

        combined: pv.DataSet | None = None
        with tempfile.TemporaryDirectory(prefix="polatory_lva_overlap_chunks_") as directory:
            folder = Path(directory)
            for index, (core_lower, core_upper) in enumerate(cores, start=1):
                progress(f"Generating padded mesh chunk {index:,}/{len(cores):,}…")
                padded_lower = np.maximum(
                    bbox_min_array,
                    np.asarray(core_lower, dtype=float) - padding,
                )
                padded_upper = np.minimum(
                    bbox_max_array,
                    np.asarray(core_upper, dtype=float) + padding,
                )

                padded_bbox = p3.Bbox(
                    padded_lower.reshape(1, 3),
                    padded_upper.reshape(1, 3),
                )
                result = polatory.Isosurface(
                    padded_bbox,
                    resolution_value,
                    np.eye(3),
                ).generate(field, isovalue=0.0, refine=int(refine))

                part_path = folder / f"part_{index:04d}.obj"
                result.export_obj(str(part_path))
                if not safe_module.obj_has_vertices(part_path):
                    continue

                part = pv.read(part_path).extract_surface().triangulate().clean()
                if part.n_points == 0 or part.n_cells == 0:
                    continue

                owned_indices = _owned_cell_indices(
                    part,
                    np.asarray(core_lower, dtype=float),
                    np.asarray(core_upper, dtype=float),
                    bbox_max_array,
                    tolerance,
                )
                if len(owned_indices) == 0:
                    continue

                retained = part.extract_cells(owned_indices)
                retained = retained.extract_surface().triangulate().clean()
                if retained.n_points == 0 or retained.n_cells == 0:
                    continue

                if combined is None:
                    combined = retained
                else:
                    combined = combined.merge(retained, merge_points=True)
                    combined = combined.extract_surface().triangulate().clean()

        if combined is None or combined.n_points == 0 or combined.n_cells == 0:
            raise RuntimeError(
                "No zero isosurface was found inside the selected model extent."
            )

        progress(
            "Joining overlap-owned triangles and writing the final result without "
            "temporary chunk-plane cuts…"
        )
        safe_module.write_obj(combined, output_obj)
        return plan

    safe_module.generate_safe_isosurface = generate_overlap_isosurface
    safe_module._overlap_chunk_meshing_installed = True
