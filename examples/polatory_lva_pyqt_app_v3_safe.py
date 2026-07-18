"""Memory-safe launcher for ``polatory_lva_pyqt_app_v2.py``.

Place this file in the same directory as ``polatory_lva_pyqt_app_v2.py`` and run::

    python polatory_lva_pyqt_app_v3_safe.py

The original application performs the complete isosurface extraction in one native
Polatory call. A smaller numeric resolution means a finer 3-D grid, so reducing
25 to 10 can increase the number of base cells by roughly 15.6 times. Large native
allocations may terminate the whole process before Python can raise an exception.

This launcher replaces only the model worker. It splits fine isosurface grids into
aligned boxes, generates them sequentially, joins the surfaces, and preserves the
original user interface and output layers.
"""

from __future__ import annotations

import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pyvista as pv

import polatory
from polatory import three as p3

try:
    import polatory_lva_pyqt_app_v2 as app
except ImportError as error:
    raise SystemExit(
        "Place polatory_lva_pyqt_app_v3_safe.py beside "
        "polatory_lva_pyqt_app_v2.py, then run the v3 file."
    ) from error


# Keep each native meshing call close to the grid size that already worked at
# resolution 25. Fine jobs are divided automatically instead of being rejected.
MAX_CELLS_PER_CHUNK = 250_000
MAX_TOTAL_BASE_CELLS = 30_000_000
MAX_CHUNKS = 128


def grid_plan(
    minimum: np.ndarray,
    maximum: np.ndarray,
    resolution: float,
) -> dict[str, Any]:
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    resolution = float(resolution)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("Meshing bounds must be three-dimensional.")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("Meshing bounds must be finite.")
    spans = maximum - minimum
    if not np.all(spans > 0.0):
        raise ValueError("Every meshing-box span must be positive.")
    if not resolution > 0.0:
        raise ValueError("Isosurface resolution must be positive.")

    cells = np.maximum(np.ceil(spans / resolution).astype(np.int64), 1)
    total = int(math.prod(int(value) for value in cells))

    chunks = np.ones(3, dtype=np.int64)
    while True:
        peak_axis = np.ceil(cells / chunks).astype(np.int64)
        peak = int(math.prod(int(value) for value in peak_axis))
        if peak <= MAX_CELLS_PER_CHUNK:
            break
        candidates = np.flatnonzero(chunks < cells)
        if len(candidates) == 0:
            break
        loads = cells[candidates] / chunks[candidates]
        axis = int(candidates[int(np.argmax(loads))])
        chunks[axis] += 1

    peak_axis = np.ceil(cells / chunks).astype(np.int64)
    peak = int(math.prod(int(value) for value in peak_axis))
    chunk_total = int(math.prod(int(value) for value in chunks))
    return {
        "cells": cells,
        "total": total,
        "chunks": chunks,
        "chunk_total": chunk_total,
        "peak": peak,
    }


def recommended_resolution(
    minimum: np.ndarray,
    maximum: np.ndarray,
    current: float,
    limit: int,
) -> float:
    spans = np.asarray(maximum, dtype=float) - np.asarray(minimum, dtype=float)

    def count(cell_size: float) -> int:
        cells = np.maximum(np.ceil(spans / cell_size).astype(np.int64), 1)
        return int(math.prod(int(value) for value in cells))

    low = max(float(current), np.finfo(float).eps)
    if count(low) <= limit:
        return low
    high = max(float(np.max(spans)), low)
    while count(high) > limit:
        high *= 2.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if count(middle) <= limit:
            high = middle
        else:
            low = middle
    return high


def chunk_bounds(
    minimum: np.ndarray,
    maximum: np.ndarray,
    resolution: float,
    cells: np.ndarray,
    chunks: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    chunks = np.asarray(chunks, dtype=np.int64)

    edges: list[np.ndarray] = []
    for axis in range(3):
        cell_count = int(cells[axis])
        chunk_count = int(chunks[axis])
        axis_edges = np.asarray(
            [(index * cell_count) // chunk_count for index in range(chunk_count + 1)],
            dtype=np.int64,
        )
        axis_edges[-1] = cell_count
        edges.append(axis_edges)

    for ix in range(int(chunks[0])):
        for iy in range(int(chunks[1])):
            for iz in range(int(chunks[2])):
                lower_index = np.array(
                    [edges[0][ix], edges[1][iy], edges[2][iz]], dtype=np.int64
                )
                upper_index = np.array(
                    [edges[0][ix + 1], edges[1][iy + 1], edges[2][iz + 1]],
                    dtype=np.int64,
                )
                lower = minimum + lower_index * float(resolution)
                upper = minimum + upper_index * float(resolution)
                upper = np.minimum(upper, maximum)
                for axis in range(3):
                    if upper_index[axis] == cells[axis]:
                        upper[axis] = maximum[axis]
                if np.all(upper > lower):
                    yield lower, upper


def obj_has_vertices(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return any(line.startswith("v ") for line in handle)


def write_obj(mesh: pv.DataSet, path: Path) -> None:
    surface = mesh.extract_surface().triangulate().clean()
    points = np.asarray(surface.points, dtype=float)
    raw_faces = np.asarray(surface.faces, dtype=np.int64)
    if len(points) == 0 or len(raw_faces) == 0:
        raise RuntimeError("The generated zero isosurface is empty.")
    faces = raw_faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise RuntimeError("The merged isosurface could not be triangulated.")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Polatory chunked isosurface\n")
        for x, y, z in points:
            handle.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for first, second, third in faces[:, 1:4]:
            handle.write(
                f"f {int(first) + 1} {int(second) + 1} {int(third) + 1}\n"
            )


def generate_safe_isosurface(
    structural: Any,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: float,
    refine: int,
    output_obj: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    plan = grid_plan(bbox_min, bbox_max, resolution)
    if plan["total"] > MAX_TOTAL_BASE_CELLS:
        suggested = recommended_resolution(
            bbox_min,
            bbox_max,
            resolution,
            MAX_TOTAL_BASE_CELLS,
        )
        raise MemoryError(
            f"Resolution {resolution:g} creates {plan['total']:,} base cells. "
            f"The safety limit is {MAX_TOTAL_BASE_CELLS:,}. Use approximately "
            f"{suggested:.6g} or coarser, or reduce the bounding box."
        )
    if plan["chunk_total"] > MAX_CHUNKS:
        raise MemoryError(
            f"The mesh needs {plan['chunk_total']:,} chunks, above the safety "
            f"limit of {MAX_CHUNKS}. Use a coarser resolution or smaller extent."
        )

    field = polatory.StructuralRbfFieldFunction(structural)
    if plan["chunk_total"] == 1:
        progress(
            f"Generating one mesh grid with {plan['total']:,} base cells…"
        )
        bbox = p3.Bbox(bbox_min.reshape(1, 3), bbox_max.reshape(1, 3))
        result = polatory.Isosurface(
            bbox,
            float(resolution),
            np.eye(3),
        ).generate(field, isovalue=0.0, refine=int(refine))
        result.export_obj(str(output_obj))
        return plan

    progress(
        f"Fine grid: {plan['total']:,} base cells. Processing safely in "
        f"{plan['chunk_total']:,} aligned chunks "
        f"{tuple(int(v) for v in plan['chunks'])}…"
    )

    combined: pv.DataSet | None = None
    all_chunks = list(
        chunk_bounds(
            bbox_min,
            bbox_max,
            resolution,
            plan["cells"],
            plan["chunks"],
        )
    )
    with tempfile.TemporaryDirectory(prefix="polatory_lva_chunks_") as directory:
        folder = Path(directory)
        for index, (lower, upper) in enumerate(all_chunks, start=1):
            progress(f"Generating mesh chunk {index:,}/{len(all_chunks):,}…")
            bbox = p3.Bbox(lower.reshape(1, 3), upper.reshape(1, 3))
            result = polatory.Isosurface(
                bbox,
                float(resolution),
                np.eye(3),
            ).generate(field, isovalue=0.0, refine=int(refine))
            part_path = folder / f"part_{index:04d}.obj"
            result.export_obj(str(part_path))
            if not obj_has_vertices(part_path):
                continue
            part = pv.read(part_path).extract_surface().triangulate().clean()
            if part.n_points == 0 or part.n_cells == 0:
                continue
            if combined is None:
                combined = part
            else:
                combined = combined.merge(part, merge_points=True)
                combined = combined.extract_surface().triangulate().clean()

    if combined is None or combined.n_points == 0 or combined.n_cells == 0:
        raise RuntimeError(
            "No zero isosurface was found inside the selected model extent."
        )
    progress("Joining chunk boundaries and writing the final result…")
    write_obj(combined, output_obj)
    return plan


def safe_worker_run(self: Any) -> None:
    temp_obj: str | None = None
    try:
        if not hasattr(polatory, "AutomaticStructuralDomainBuilder3"):
            raise RuntimeError(
                "AutomaticStructuralDomainBuilder3 is unavailable. Reinstall "
                "Polatory from feature/automatic-subdomainer and restart the app."
            )

        points = self.payload["points"]
        indicators = self.payload["indicators"]
        trend_vertices = self.payload["trend_vertices"]
        trend_faces = self.payload["trend_faces"]
        parameters = self.payload["parameters"]
        bbox_min = self.payload["bbox_min"]
        bbox_max = self.payload["bbox_max"]

        self.progress.emit("Calculating Leapfrog-compatible indicator distances…")
        value_info = polatory.leapfrog_indicator_values3(
            points,
            indicators,
            fit_accuracy=float(parameters["fit_tolerance"]),
        )
        values = np.asarray(value_info.values, dtype=float)

        self.progress.emit("Creating structural LVA input…")
        trend_input = polatory.StructuralTrendInput3(
            trend_vertices,
            trend_faces,
            float(parameters["strength"]),
            float(parameters["trend_range"]),
        )

        rbf = p3.CovSpheroidal3(
            [float(parameters["sill"]), float(parameters["base_range"])]
        )
        model = p3.Model(rbf, int(parameters["poly_degree"]))
        model.nugget = float(parameters["nugget"])
        model_parameters = np.asarray(
            model.parameters, dtype=float
        ).reshape(-1).tolist()

        trend_type = {
            "Strongest along inputs": (
                polatory.StructuralTrendType.STRONGEST_ALONG_INPUTS
            ),
            "Blending": polatory.StructuralTrendType.BLENDING,
            "Non-decaying": polatory.StructuralTrendType.NON_DECAYING,
        }[parameters["trend_type"]]

        self.progress.emit("Building automatic structural domains…")
        builder = polatory.AutomaticStructuralDomainBuilder3(
            centroid_count=int(parameters["centroid_count"]),
            minimum_cluster_fraction=float(
                parameters["minimum_cluster_fraction"]
            ),
            maximum_cluster_fraction=float(
                parameters["maximum_cluster_fraction"]
            ),
            consistency_threshold=float(parameters["consistency_threshold"]),
            base_range=float(parameters["base_range"]),
            support_multiplier=int(parameters["support_multiplier"]),
            minimum_support_points=int(parameters["minimum_support_points"]),
        )
        domains = builder.build_from_inputs(
            points,
            [trend_input],
            model_parameters=model_parameters,
            trend_type=trend_type,
        )
        diagnostics = builder.diagnostics_
        labels = np.asarray(builder.labels_, dtype=np.int64)
        if diagnostics is None:
            raise RuntimeError("The automatic builder returned no diagnostics.")

        self.progress.emit(
            f"Fitting {diagnostics.final_domain_count} structural domains…"
        )
        structural = polatory.StructuralInterpolant3(
            model,
            outside_value=float(parameters["outside_value"]),
            blend_power=float(parameters["blend_power"]),
            alignment_strength=float(parameters["alignment_strength"]),
        )
        structural.fit(
            points,
            values,
            domains,
            tolerance=float(value_info.fit_accuracy),
            max_iter=int(parameters["max_iterations"]),
        )

        predictions = np.asarray(structural.evaluate(points), dtype=float)
        errors = predictions - values
        training_rmse = float(np.sqrt(np.mean(errors**2)))
        training_max_abs = float(np.max(np.abs(errors)))

        file_descriptor, temp_obj = tempfile.mkstemp(
            prefix="polatory_lva_", suffix=".obj"
        )
        os.close(file_descriptor)
        plan = generate_safe_isosurface(
            structural=structural,
            bbox_min=np.asarray(bbox_min, dtype=float),
            bbox_max=np.asarray(bbox_max, dtype=float),
            resolution=float(parameters["isosurface_resolution"]),
            refine=int(parameters["isosurface_refine"]),
            output_obj=Path(temp_obj),
            progress=self.progress.emit,
        )

        self.progress.emit("Sampling the LVA field for 3-D display…")
        lva_dimension = int(parameters["lva_grid_dimension"])
        lva_dimensions = (lva_dimension, lva_dimension, lva_dimension)
        lva_points = app.structured_points(bbox_min, bbox_max, lva_dimensions)
        non_decaying = trend_type == polatory.StructuralTrendType.NON_DECAYING
        lva_matrices = polatory.sample_single_input_anisotropies3(
            lva_points,
            trend_input,
            non_decaying=non_decaying,
        )
        lva_eigenvalues, _ = np.linalg.eigh(lva_matrices)
        lva_ratio = lva_eigenvalues[:, -1] / lva_eigenvalues[:, 0]

        glyph_dimension = int(parameters["lva_glyph_dimension"])
        glyph_dimensions = (glyph_dimension, glyph_dimension, glyph_dimension)
        glyph_points = app.structured_points(
            bbox_min, bbox_max, glyph_dimensions
        )
        glyph_matrices = polatory.sample_single_input_anisotropies3(
            glyph_points,
            trend_input,
            non_decaying=non_decaying,
        )
        glyph_eigenvalues, glyph_eigenvectors = np.linalg.eigh(glyph_matrices)
        glyph_ratio = glyph_eigenvalues[:, -1] / glyph_eigenvalues[:, 0]
        principal_axes = glyph_eigenvectors[:, :, -1]

        result = {
            "temp_obj": temp_obj,
            "values": values,
            "predictions": predictions,
            "labels": labels,
            "centroid_points": np.asarray(
                diagnostics.centroid_points, dtype=float
            ),
            "centroid_labels": np.asarray(
                diagnostics.centroid_labels, dtype=np.int64
            ),
            "centroid_grid_shape": tuple(diagnostics.centroid_grid_shape),
            "merge_count": int(diagnostics.merge_count),
            "domain_count": int(diagnostics.final_domain_count),
            "minimum_points": int(diagnostics.minimum_points),
            "maximum_points": int(diagnostics.maximum_points),
            "training_rmse": training_rmse,
            "training_max_abs": training_max_abs,
            "fit_tolerance": float(value_info.fit_accuracy),
            "clipping_distance": float(value_info.clipping_distance),
            "data_diagonal": float(value_info.data_diagonal),
            "lva_points": lva_points,
            "lva_dimensions": lva_dimensions,
            "lva_ratio": lva_ratio,
            "glyph_points": glyph_points,
            "glyph_axes": principal_axes,
            "glyph_ratio": glyph_ratio,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }
        self.progress.emit(
            f"Meshing complete: {plan['total']:,} base cells in "
            f"{plan['chunk_total']:,} chunk(s); largest chunk "
            f"{plan['peak']:,} cells."
        )
        self.finished.emit(result)
    except Exception:
        if temp_obj and Path(temp_obj).exists():
            try:
                Path(temp_obj).unlink()
            except OSError:
                pass
        self.failed.emit(traceback.format_exc())


# Replace the original QThread worker while keeping the complete v2 interface.
app.ModelWorker.run = safe_worker_run


if __name__ == "__main__":
    raise SystemExit(app.main())
