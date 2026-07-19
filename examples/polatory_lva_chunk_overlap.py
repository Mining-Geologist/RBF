"""Overlap-and-trim meshing for the process-isolated LVA application.

The native isosurface extractor is evaluated on padded chunks, then every result is
trimmed back to its unpadded core before the pieces are joined.  This keeps any
surface generated at a temporary chunk boundary outside the retained region and
prevents internal flat walls or pinched terminations in the final OBJ.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvista as pv

import polatory
from polatory import three as p3


def _core_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[float, ...]:
    return (
        float(lower[0]),
        float(upper[0]),
        float(lower[1]),
        float(upper[1]),
        float(lower[2]),
        float(upper[2]),
    )


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

        # Preserve the original single-grid path exactly.  No artificial internal
        # boundary exists when the complete model is meshed in one native call.
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

        # Two base cells are sufficient to move the temporary native boundary away
        # from the retained core, including refined cells adjacent to that boundary.
        padding = 2.0 * resolution_value
        progress(
            f"Fine grid: {plan['total']:,} base cells. Processing safely in "
            f"{plan['chunk_total']:,} overlapping chunks "
            f"{tuple(int(value) for value in plan['chunks'])}; temporary chunk "
            "faces will be trimmed before joining…"
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

                # Remove everything in the padding collar.  Any cap or flat wall made
                # at the padded native bbox is therefore discarded rather than merged
                # into the final model.
                retained = part.clip_box(
                    bounds=_core_bounds(core_lower, core_upper),
                    invert=False,
                    crinkle=False,
                    merge_points=True,
                )
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
            "Joining trimmed chunk cores and writing the final result without "
            "temporary chunk-face geometry…"
        )
        safe_module.write_obj(combined, output_obj)
        return plan

    safe_module.generate_safe_isosurface = generate_overlap_isosurface
    safe_module._overlap_chunk_meshing_installed = True
