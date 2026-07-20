"""Globally aligned, slab-streamed isosurface extraction for structural LVA models.

The previous memory-safe path called Polatory's adaptive isosurface extractor once per
3-D chunk.  Even with overlap, each call refined its own temporary lattice, so adjacent
chunks could disagree and leave axis-aligned stairs, shelves, open seams, or artificial
flat terminations.

This module samples ``StructuralInterpolant3.evaluate`` on one globally aligned regular
grid.  The scalar field is evaluated in memory-safe slabs, and every neighbouring slab
shares the exact same boundary-node values.  Marching cubes is then run on those slabs
and the resulting vertices are merged on the common grid planes.  The field itself is
unchanged; only the surface extraction path is replaced.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvista as pv

try:
    from skimage.measure import marching_cubes
except ImportError as error:  # pragma: no cover - exercised by deployment validation
    raise ImportError(
        "The globally aligned LVA mesher requires scikit-image. Install it with: "
        "python -m pip install scikit-image"
    ) from error


EVALUATION_BATCH_SIZE = int(
    os.environ.get("POLATORY_GLOBAL_GRID_EVALUATION_BATCH_SIZE", "250000")
)
MAX_POINTS_PER_SLAB = int(
    os.environ.get("POLATORY_GLOBAL_GRID_MAX_POINTS_PER_SLAB", "2000000")
)


def _grid_geometry(
    minimum: np.ndarray,
    maximum: np.ndarray,
    requested_resolution: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    requested_resolution = float(requested_resolution)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("Meshing bounds must be three-dimensional.")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("Meshing bounds must be finite.")
    span = maximum - minimum
    if not np.all(span > 0.0):
        raise ValueError("Every meshing-box span must be positive.")
    if not requested_resolution > 0.0:
        raise ValueError("Isosurface resolution must be positive.")

    # Use an integer number of globally uniform cells on every axis.  The resulting
    # spacing is never coarser than the requested resolution, and the final node lands
    # exactly on the user-supplied maximum rather than creating a short last cell.
    cells = np.maximum(np.ceil(span / requested_resolution).astype(np.int64), 1)
    spacing = span / cells.astype(float)
    coordinates = [
        minimum[axis]
        + np.arange(int(cells[axis]) + 1, dtype=np.float64) * spacing[axis]
        for axis in range(3)
    ]
    for axis in range(3):
        coordinates[axis][-1] = maximum[axis]
    return cells, spacing, coordinates


def _evaluate_layers(
    structural: Any,
    slab_axis: int,
    plane_axes: tuple[int, int],
    plane_first: np.ndarray,
    plane_second: np.ndarray,
    slab_coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate complete globally aligned node layers in bounded point batches."""

    first_flat = np.asarray(plane_first, dtype=np.float64).ravel(order="C")
    second_flat = np.asarray(plane_second, dtype=np.float64).ravel(order="C")
    plane_count = len(first_flat)
    layer_count = len(slab_coordinates)
    total = plane_count * layer_count
    values = np.empty(total, dtype=np.float32)

    for start in range(0, total, EVALUATION_BATCH_SIZE):
        stop = min(start + EVALUATION_BATCH_SIZE, total)
        linear = np.arange(start, stop, dtype=np.int64)
        layer_indices = linear // plane_count
        plane_indices = linear - layer_indices * plane_count

        query = np.empty((stop - start, 3), dtype=np.float64)
        query[:, slab_axis] = slab_coordinates[layer_indices]
        query[:, plane_axes[0]] = first_flat[plane_indices]
        query[:, plane_axes[1]] = second_flat[plane_indices]

        evaluated = np.asarray(structural.evaluate(query), dtype=np.float64).reshape(-1)
        if len(evaluated) != len(query):
            raise RuntimeError(
                "Structural field evaluation did not return one value per grid node."
            )
        if not np.all(np.isfinite(evaluated)):
            raise RuntimeError("Structural field evaluation returned non-finite values.")
        values[start:stop] = evaluated.astype(np.float32, copy=False)

    return values.reshape(
        (layer_count, plane_first.shape[0], plane_first.shape[1]),
        order="C",
    )


def _polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    face_stream = np.empty((len(faces), 4), dtype=np.int64)
    face_stream[:, 0] = 3
    face_stream[:, 1:] = np.asarray(faces, dtype=np.int64)
    return pv.PolyData(
        np.asarray(vertices, dtype=np.float64),
        face_stream.ravel(order="C"),
    )


def install_global_grid_meshing(safe_module: Any) -> None:
    """Replace ``generate_safe_isosurface`` on the supplied v3-safe module."""

    if getattr(safe_module, "_global_grid_meshing_installed", False):
        return

    def generate_global_grid_isosurface(
        structural: Any,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        resolution: float,
        refine: int,
        output_obj: Path,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        minimum = np.asarray(bbox_min, dtype=float)
        maximum = np.asarray(bbox_max, dtype=float)
        cells, spacing, coordinates = _grid_geometry(
            minimum,
            maximum,
            float(resolution),
        )
        total_cells = int(math.prod(int(value) for value in cells))
        if total_cells > int(safe_module.MAX_TOTAL_BASE_CELLS):
            suggested = safe_module.recommended_resolution(
                minimum,
                maximum,
                float(resolution),
                int(safe_module.MAX_TOTAL_BASE_CELLS),
            )
            raise MemoryError(
                f"Resolution {float(resolution):g} creates {total_cells:,} base cells. "
                f"The safety limit is {int(safe_module.MAX_TOTAL_BASE_CELLS):,}. Use "
                f"approximately {suggested:.6g} or coarser, or reduce the bounding box."
            )

        node_counts = cells + 1
        # Stream along the longest axis so each scalar plane has the fewest nodes.
        slab_axis = int(np.argmax(node_counts))
        plane_axes_list = [axis for axis in range(3) if axis != slab_axis]
        plane_axes = (plane_axes_list[0], plane_axes_list[1])
        plane_first, plane_second = np.meshgrid(
            coordinates[plane_axes[0]],
            coordinates[plane_axes[1]],
            indexing="ij",
        )
        plane_points = int(plane_first.size)
        if plane_points <= 0:
            raise RuntimeError("The global scalar grid contains no plane nodes.")

        # A slab with N cells needs N+1 scalar layers.  Adjacent slabs reuse the exact
        # same boundary layer, so no independently refined chunk surface can appear.
        cells_per_slab = max(
            1,
            int(MAX_POINTS_PER_SLAB // max(plane_points, 1)) - 1,
        )
        cells_per_slab = min(cells_per_slab, int(cells[slab_axis]))
        slab_count = int(math.ceil(int(cells[slab_axis]) / cells_per_slab))
        if slab_count > int(safe_module.MAX_CHUNKS):
            raise MemoryError(
                f"The globally aligned grid needs {slab_count:,} scalar slabs, above "
                f"the safety limit of {int(safe_module.MAX_CHUNKS):,}. Increase "
                "POLATORY_GLOBAL_GRID_MAX_POINTS_PER_SLAB, use a coarser resolution, "
                "or reduce the bounding box."
            )

        axis_names = ("X", "Y", "Z")
        progress(
            f"Global scalar grid: {tuple(int(value) for value in node_counts)} nodes; "
            f"streaming {slab_count:,} aligned slab(s) along "
            f"{axis_names[slab_axis]} with shared boundary values…"
        )

        all_vertices: list[np.ndarray] = []
        all_faces: list[np.ndarray] = []
        vertex_offset = 0
        previous_top: np.ndarray | None = None
        slab_cell_start = 0

        for slab_index in range(slab_count):
            slab_cell_stop = min(
                slab_cell_start + cells_per_slab,
                int(cells[slab_axis]),
            )
            node_start = slab_cell_start
            node_stop = slab_cell_stop + 1
            slab_coordinates = coordinates[slab_axis][node_start:node_stop]
            progress(
                f"Evaluating aligned scalar slab {slab_index + 1:,}/{slab_count:,} "
                f"({node_stop - node_start:,} node layers)…"
            )

            if previous_top is None:
                volume = _evaluate_layers(
                    structural,
                    slab_axis,
                    plane_axes,
                    plane_first,
                    plane_second,
                    slab_coordinates,
                )
            else:
                volume = np.empty(
                    (
                        len(slab_coordinates),
                        plane_first.shape[0],
                        plane_first.shape[1],
                    ),
                    dtype=np.float32,
                )
                volume[0] = previous_top
                if len(slab_coordinates) > 1:
                    volume[1:] = _evaluate_layers(
                        structural,
                        slab_axis,
                        plane_axes,
                        plane_first,
                        plane_second,
                        slab_coordinates[1:],
                    )
            previous_top = np.asarray(volume[-1], dtype=np.float32).copy()

            minimum_value = float(np.min(volume))
            maximum_value = float(np.max(volume))
            if minimum_value <= 0.0 <= maximum_value:
                local_vertices, local_faces, _, _ = marching_cubes(
                    volume,
                    level=0.0,
                    spacing=(
                        float(spacing[slab_axis]),
                        float(spacing[plane_axes[0]]),
                        float(spacing[plane_axes[1]]),
                    ),
                    allow_degenerate=False,
                    method="lewiner",
                )

                vertices = np.empty_like(local_vertices, dtype=np.float64)
                origins = (
                    float(coordinates[slab_axis][node_start]),
                    float(coordinates[plane_axes[0]][0]),
                    float(coordinates[plane_axes[1]][0]),
                )
                vertices[:, slab_axis] = local_vertices[:, 0] + origins[0]
                vertices[:, plane_axes[0]] = local_vertices[:, 1] + origins[1]
                vertices[:, plane_axes[1]] = local_vertices[:, 2] + origins[2]

                all_vertices.append(vertices)
                all_faces.append(np.asarray(local_faces, dtype=np.int64) + vertex_offset)
                vertex_offset += len(vertices)

            slab_cell_start = slab_cell_stop

        if not all_vertices or not all_faces:
            raise RuntimeError(
                "No zero isosurface was found inside the selected model extent."
            )

        progress(
            "Merging shared global-grid vertices and removing duplicate slab-boundary "
            "nodes…"
        )
        vertices = np.concatenate(all_vertices, axis=0)
        faces = np.concatenate(all_faces, axis=0)
        mesh = _polydata(vertices, faces)
        tolerance = max(
            1.0e-8 * float(np.linalg.norm(maximum - minimum)),
            1.0e-9,
        )
        mesh = mesh.clean(tolerance=tolerance, absolute=True)
        mesh = mesh.triangulate()

        # ``refine`` belonged to Polatory's independently adaptive native lattice.
        # Global marching cubes already interpolates every crossing on shared grid
        # edges.  Keeping topology globally consistent is more important than applying
        # a second per-slab refinement that could reintroduce cracks.
        if int(refine) > 0:
            progress(
                "Global-grid topology is active; native per-chunk refine passes are "
                "intentionally skipped to preserve shared slab boundaries."
            )

        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=False,
        )
        progress(
            f"Global-grid surface assembled: {mesh.n_points:,} vertices, "
            f"{mesh.n_cells:,} triangles, {boundary.n_cells:,} boundary-edge cells."
        )
        safe_module.write_obj(mesh, Path(output_obj))

        peak_cells = int(
            cells_per_slab
            * int(cells[plane_axes[0]])
            * int(cells[plane_axes[1]])
        )
        return {
            "cells": cells,
            "total": total_cells,
            "chunks": np.asarray(
                [slab_count if axis == slab_axis else 1 for axis in range(3)],
                dtype=np.int64,
            ),
            "chunk_total": slab_count,
            "peak": peak_cells,
            "spacing": spacing,
            "slab_axis": slab_axis,
        }

    safe_module.generate_safe_isosurface = generate_global_grid_isosurface
    safe_module._global_grid_meshing_installed = True
