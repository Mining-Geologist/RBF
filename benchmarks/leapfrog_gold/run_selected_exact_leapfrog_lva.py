"""Run the selected structural benchmark with the recovered Leapfrog LVA sampler forced.

This diagnostic deliberately replaces the automatic builder's single-input sampler with an
independent implementation of the LVA rules recovered from native Leapfrog projects:

* equal-weight accumulation of unit triangle normals at each vertex,
* nearest mesh vertex in Euclidean distance,
* q = exp(-distance / range),
* q = 0 at distance >= 4 * range,
* ratio = 1 + (strength - 1) * q,
* determinant-one anisotropy with tangent scale ratio**(-1/3) and normal scale
  ratio**(2/3).

The exported Leapfrog glyph base of 4.0 is intentionally absent: it multiplies every glyph
axis equally and therefore cancels from the anisotropy matrix used by Polatory.

The script then runs the normal selected-case suite.  If its output is unchanged, the
remaining mismatch is downstream of LVA sampling (clustering, local support, blending or
field evaluation), not in the recovered single-mesh LVA field.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_selected_no_background_blending as suite  # noqa: E402
import polatory.automatic_domain_builder as automatic_module  # noqa: E402


def exact_leapfrog_single_input_anisotropies3(
    points: np.ndarray,
    input_: Any,
    *,
    non_decaying: bool = False,
) -> np.ndarray:
    """Independent exact single-mesh Leapfrog LVA reconstruction."""
    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(input_.vertices, dtype=np.float64)
    faces = np.asarray(input_.faces, dtype=np.int64)
    strength = float(input_.strength)
    range_ = float(input_.range)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("input vertices must have shape (m, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("input faces must have shape (k, 3)")
    if strength < 1.0:
        raise ValueError("input strength must be at least 1")
    if range_ <= 0.0:
        raise ValueError("input range must be positive")

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 0.0
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid] = 0.0

    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    valid = lengths > 0.0
    vertex_normals[valid] /= lengths[valid, None]
    vertex_normals[~valid] = np.array([0.0, 0.0, 1.0])

    tree = cKDTree(vertices)
    try:
        distances, nearest = tree.query(points, k=1, workers=-1)
    except TypeError:
        distances, nearest = tree.query(points, k=1)
    distances = np.asarray(distances, dtype=np.float64)
    nearest = np.asarray(nearest, dtype=np.int64)

    if non_decaying:
        q = np.ones(len(points), dtype=np.float64)
    else:
        q = np.exp(-distances / range_)
        q[distances >= 4.0 * range_] = 0.0

    ratio = 1.0 + (strength - 1.0) * q
    normals = vertex_normals[nearest]
    projectors = normals[:, :, None] * normals[:, None, :]
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    tangent_scale = ratio ** (-1.0 / 3.0)
    normal_scale = ratio ** (2.0 / 3.0)
    return (
        tangent_scale[:, None, None] * (identity - projectors)
        + normal_scale[:, None, None] * projectors
    )


# Force the automatic builder to use the independent recovered implementation.
automatic_module.sample_single_input_anisotropies3 = (
    exact_leapfrog_single_input_anisotropies3
)

suite.OUTPUT_DIR = suite.ROOT / "benchmark-results" / "exact-leapfrog-lva-forced"
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tExact Leapfrog LVA forced: nearest vertex, equal-weight unit face "
    "normals, exponential decay, hard 4R cutoff and determinant-one anisotropy. "
    "The Leapfrog glyph base 4.0 is display-only and is not applied to the RBF metric.",
    flush=True,
)

if __name__ == "__main__":
    raise SystemExit(suite.main())
