"""Scalable LVA-field launcher for the Polatory PyQt application.

Place this file beside ``polatory_lva_pyqt_app_v2.py``,
``polatory_lva_pyqt_app_v3_safe.py`` and
``polatory_lva_pyqt_app_v4_lva_safe.py``, then run::

    python polatory_lva_pyqt_app_v5_streamed_lva.py

The old display path sampled an N x N x N volume even though the UI displayed
only three centre planes. This launcher samples only those planes, reducing the
LVA display workload from O(N^3) to O(N^2). It also computes the recovered
single-input ratio directly instead of allocating one 3 x 3 matrix and running
an eigensolver for every display point.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pyvista as pv

import polatory

try:
    from scipy.spatial import cKDTree
except ImportError as error:
    raise SystemExit(
        "This launcher requires SciPy. Install it with: python -m pip install scipy"
    ) from error

import polatory_lva_pyqt_app_v4_lva_safe as v4

v3 = v4.v3
app = v4.app

# This protects the VTK/GPU display path. The spin box accepts larger values so
# the app can report a normal validation error rather than terminating natively.
MAX_LVA_DIMENSION = 1500
QUERY_CHUNK_SIZE = 250_000

_original_structured_points = app.structured_points
_original_sample_anisotropies = polatory.sample_single_input_anisotropies3
_original_eigh = np.linalg.eigh
_original_window_init = app.MainWindow.__init__
_original_model_parameters = app.MainWindow.model_parameters
_original_model_finished = app.MainWindow.model_finished
_worker_state = threading.local()


class PlanePointArray(np.ndarray):
    """Three concatenated centre planes carrying their common dimension."""

    dimension: int

    def __new__(cls, points: np.ndarray, dimension: int):
        instance = np.asarray(points, dtype=np.float32).view(cls)
        instance.dimension = int(dimension)
        return instance

    def __array_finalize__(self, source: Any) -> None:
        if source is not None:
            self.dimension = int(getattr(source, "dimension", 0))


class LvaRatioProxy:
    """Avoid allocating anisotropy matrices solely to recover eigenvalue ratios."""

    def __init__(self, ratios: np.ndarray) -> None:
        self.ratios = np.asarray(ratios, dtype=np.float32)


def _plane_points(
    minimum: np.ndarray,
    maximum: np.ndarray,
    dimension: int,
) -> PlanePointArray:
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    centre = 0.5 * (minimum + maximum)
    axes = [
        np.linspace(minimum[i], maximum[i], dimension, dtype=np.float32)
        for i in range(3)
    ]

    planes: list[np.ndarray] = []
    specifications = (
        (0, axes[1], axes[2]),
        (1, axes[0], axes[2]),
        (2, axes[0], axes[1]),
    )
    for fixed_axis, first_axis, second_axis in specifications:
        first, second = np.meshgrid(first_axis, second_axis, indexing="ij")
        points = np.empty((dimension * dimension, 3), dtype=np.float32)
        varying_axes = [axis for axis in range(3) if axis != fixed_axis]
        points[:, fixed_axis] = np.float32(centre[fixed_axis])
        points[:, varying_axes[0]] = first.ravel(order="F")
        points[:, varying_axes[1]] = second.ravel(order="F")
        planes.append(points)

    return PlanePointArray(np.concatenate(planes, axis=0), dimension)


def scalable_structured_points(
    minimum: np.ndarray,
    maximum: np.ndarray,
    dimensions: tuple[int, int, int],
) -> np.ndarray:
    """Replace only the first worker grid request with three centre planes."""
    if getattr(_worker_state, "active", False):
        call_index = int(getattr(_worker_state, "structured_call", 0))
        _worker_state.structured_call = call_index + 1
        if call_index == 0:
            if not (dimensions[0] == dimensions[1] == dimensions[2]):
                raise ValueError("The LVA display grid must use equal dimensions.")
            dimension = int(dimensions[0])
            return _plane_points(minimum, maximum, dimension)
    return _original_structured_points(minimum, maximum, dimensions)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 0.0
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0

    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0])
    return normals


def scalable_sample_anisotropies(
    points: np.ndarray,
    input_: object,
    *,
    non_decaying: bool = False,
) -> Any:
    """Directly sample LVA ratios for the plane-only display request."""
    if not isinstance(points, PlanePointArray):
        return _original_sample_anisotropies(
            points,
            input_,
            non_decaying=non_decaying,
        )

    vertices = np.asarray(input_.vertices, dtype=np.float64)
    faces = np.asarray(input_.faces, dtype=np.int64)
    strength = float(input_.strength)
    range_ = float(input_.range)
    normals = _vertex_normals(vertices, faces)
    tree = cKDTree(vertices)

    ratios = np.empty(len(points), dtype=np.float32)
    query_points = np.asarray(points, dtype=np.float32)
    for start in range(0, len(query_points), QUERY_CHUNK_SIZE):
        stop = min(start + QUERY_CHUNK_SIZE, len(query_points))
        try:
            distances, _ = tree.query(
                query_points[start:stop],
                k=1,
                workers=-1,
            )
        except TypeError:
            distances, _ = tree.query(query_points[start:stop], k=1)
        distances = np.asarray(distances, dtype=np.float64)
        if non_decaying:
            influence = np.ones(len(distances), dtype=np.float64)
        else:
            influence = np.exp(-distances / range_)
            influence[distances >= 4.0 * range_] = 0.0
        ratios[start:stop] = 1.0 + (strength - 1.0) * influence

    # Keep a reference to normals so this function validates and prepares the
    # same reference orientation field as the original sampler. The principal
    # axes used by glyphs still follow the original full-matrix path below.
    del normals
    return LvaRatioProxy(ratios)


def scalable_eigh(value: Any, *args: Any, **kwargs: Any):
    if isinstance(value, LvaRatioProxy):
        ratios = np.asarray(value.ratios, dtype=np.float64)
        tangent = ratios ** (-1.0 / 3.0)
        normal = ratios ** (2.0 / 3.0)
        eigenvalues = np.column_stack([tangent, tangent, normal])
        return eigenvalues, None
    return _original_eigh(value, *args, **kwargs)


def scalable_worker_run(self: Any) -> None:
    _worker_state.active = True
    _worker_state.structured_call = 0
    try:
        v3.safe_worker_run(self)
    finally:
        _worker_state.active = False
        _worker_state.structured_call = 0


def _lva_multiblock(
    points: PlanePointArray,
    ratios: np.ndarray,
) -> pv.MultiBlock:
    dimension = int(points.dimension)
    count = dimension * dimension
    dimensions = (
        (1, dimension, dimension),
        (dimension, 1, dimension),
        (dimension, dimension, 1),
    )
    names = ("X centre plane", "Y centre plane", "Z centre plane")

    blocks = pv.MultiBlock()
    for index, (name, grid_dimensions) in enumerate(zip(names, dimensions)):
        start = index * count
        stop = start + count
        grid = pv.StructuredGrid()
        grid.points = np.asarray(points[start:stop], dtype=np.float32)
        grid.dimensions = grid_dimensions
        grid["LVA ratio"] = np.asarray(ratios[start:stop], dtype=np.float32)
        blocks[name] = grid
    return blocks


def scalable_window_init(self: Any) -> None:
    _original_window_init(self)
    self.lva_grid_spin.setRange(2, 100_000)
    self.lva_grid_spin.setToolTip(
        "Samples per axis on each of the three LVA centre planes. v5 samples "
        "O(N^2) plane points instead of an O(N^3) volume. Requests above the "
        f"safe interactive limit ({MAX_LVA_DIMENSION:,}) are rejected normally "
        "instead of crashing VTK or the GPU driver."
    )


def scalable_model_parameters(self: Any) -> dict[str, Any]:
    parameters = _original_model_parameters(self)
    dimension = int(parameters["lva_grid_dimension"])
    if dimension > MAX_LVA_DIMENSION:
        raise ValueError(
            f"LVA grid dimension {dimension:,} exceeds the safe interactive "
            f"limit of {MAX_LVA_DIMENSION:,}. This would create "
            f"{3 * dimension * dimension:,} displayed plane points. The request "
            "was stopped before modelling so the application remains open."
        )
    return parameters


def scalable_model_finished(self: Any, result: dict[str, Any]) -> None:
    plane_points = result.get("lva_points")
    plane_ratios = result.get("lva_ratio")
    if not isinstance(plane_points, PlanePointArray):
        _original_model_finished(self, result)
        return

    # Let the original completion path create every existing model layer using a
    # tiny placeholder LVA grid, then replace only that placeholder layer.
    display_result = dict(result)
    dummy_dimensions = (2, 2, 2)
    dummy_points = _original_structured_points(
        np.asarray(result["bbox_min"], dtype=float),
        np.asarray(result["bbox_max"], dtype=float),
        dummy_dimensions,
    )
    display_result["lva_points"] = dummy_points
    display_result["lva_dimensions"] = dummy_dimensions
    display_result["lva_ratio"] = np.ones(len(dummy_points), dtype=np.float32)
    _original_model_finished(self, display_result)

    try:
        self._remove_layer("LVA field slices")
        blocks = _lva_multiblock(plane_points, np.asarray(plane_ratios))
        self._add_layer(
            "LVA field slices",
            blocks,
            kind="mesh",
            scalars="LVA ratio",
            cmap="viridis",
            opacity=0.82,
            show_edges=False,
        )
        dimension = int(plane_points.dimension)
        self._log(
            f"LVA field display complete: {dimension:,} x {dimension:,} samples "
            "on each of three centre planes."
        )
    except Exception as error:
        self._show_error("The LVA field could not be displayed", error)


app.structured_points = scalable_structured_points
polatory.sample_single_input_anisotropies3 = scalable_sample_anisotropies
np.linalg.eigh = scalable_eigh
app.ModelWorker.run = scalable_worker_run
app.MainWindow.__init__ = scalable_window_init
app.MainWindow.model_parameters = scalable_model_parameters
app.MainWindow.model_finished = scalable_model_finished


if __name__ == "__main__":
    raise SystemExit(app.main())
