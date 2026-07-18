"""Leapfrog-compatible post-clustering structural-domain construction.

This module implements the part of Leapfrog's structural SubDomainer that can
be reconstructed exactly once a final cluster label is known for every input
point. It intentionally separates *how clusters are found* from *how each
cluster is converted into a local RBF domain*.

The recovered rules are:

1. Average the sampled anisotropy matrices arithmetically over the core points.
2. Scale the local CovSpheroidal3 range by the largest eigenvalue of that mean.
3. Select support points by anisotropic distance to the nearest core point.
4. Limit support to five times the core population by shrinking the radius.
5. Expand the core AABB by the anisotropic internal support radius.

The returned objects are ordinary ``StructuralDomain3`` instances and can be
passed directly to ``StructuralInterpolant3.fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np

from ._structural import (
    StructuralDomain3,
    StructuralDomainBuilder3,
    StructuralTrendType,
)

# CovSpheroidal3 uses rho = distance / range internally with the IMQ term
# 1 / (1 + C rho**2)**(3/2). FastRBF serialises the equivalent internal radius
# range / sqrt(C).
SPHEROIDAL3_C = 7.181510581693163


@dataclass(frozen=True)
class LabeledStructuralDomainDiagnostics3:
    """Recovered construction details for one local structural domain."""

    label: int
    core_indices: np.ndarray
    support_indices: np.ndarray
    anisotropy: np.ndarray
    desired_internal_radius: float
    internal_radius: float
    local_kernel_range: float
    bbox_min: np.ndarray
    bbox_max: np.ndarray


def _nearest_core_distances_numpy(
    transformed_points: np.ndarray,
    transformed_core: np.ndarray,
    *,
    query_chunk_size: int = 4096,
    core_chunk_size: int = 1024,
) -> np.ndarray:
    """Return nearest-core Euclidean distances without a SciPy dependency."""

    result = np.full(len(transformed_points), np.inf, dtype=float)
    for query_start in range(0, len(transformed_points), query_chunk_size):
        query_stop = min(query_start + query_chunk_size, len(transformed_points))
        query = transformed_points[query_start:query_stop]
        minimum_squared = np.full(len(query), np.inf, dtype=float)

        for core_start in range(0, len(transformed_core), core_chunk_size):
            core_stop = min(core_start + core_chunk_size, len(transformed_core))
            core = transformed_core[core_start:core_stop]
            diff = query[:, None, :] - core[None, :, :]
            squared = np.einsum("qci,qci->qc", diff, diff, optimize=True)
            minimum_squared = np.minimum(minimum_squared, squared.min(axis=1))

        result[query_start:query_stop] = np.sqrt(minimum_squared)
    return result


def _nearest_core_distances(
    transformed_points: np.ndarray,
    transformed_core: np.ndarray,
) -> np.ndarray:
    """Use SciPy's KD-tree when available, otherwise a chunked NumPy fallback."""

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        return _nearest_core_distances_numpy(transformed_points, transformed_core)

    tree = cKDTree(transformed_core)
    try:
        distances, _ = tree.query(transformed_points, k=1, workers=-1)
    except TypeError:  # Older SciPy versions do not expose ``workers``.
        distances, _ = tree.query(transformed_points, k=1)
    return np.asarray(distances, dtype=float)


def sample_single_input_anisotropies3(
    points: np.ndarray,
    input_: object,
    *,
    non_decaying: bool = False,
) -> np.ndarray:
    """Sample the recovered single-mesh Leapfrog structural field exactly.

    Triangle normals are normalised before equal-weight accumulation at each
    vertex. The nearest mesh vertex controls the local axis. Decaying trends use
    ``exp(-distance / range)`` inside a hard ``4 * range`` cutoff and zero
    influence outside it.
    """

    points = np.asarray(points, dtype=float)
    vertices = np.asarray(input_.vertices, dtype=float)
    faces = np.asarray(input_.faces, dtype=np.int64)
    strength = float(input_.strength)
    range_ = float(input_.range)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("input vertices must have shape (m, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("input faces must have shape (k, 3)")
    if not range_ > 0.0:
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

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        nearest_indices = np.empty(len(points), dtype=np.int64)
        nearest_squared = np.full(len(points), np.inf, dtype=float)
        for start in range(0, len(vertices), 1024):
            stop = min(start + 1024, len(vertices))
            diff = points[:, None, :] - vertices[None, start:stop, :]
            squared = np.einsum("qvi,qvi->qv", diff, diff, optimize=True)
            local = np.argmin(squared, axis=1)
            local_squared = squared[np.arange(len(points)), local]
            replace = local_squared < nearest_squared
            nearest_squared[replace] = local_squared[replace]
            nearest_indices[replace] = start + local[replace]
        distances = np.sqrt(nearest_squared)
    else:
        tree = cKDTree(vertices)
        try:
            distances, nearest_indices = tree.query(points, k=1, workers=-1)
        except TypeError:
            distances, nearest_indices = tree.query(points, k=1)
        distances = np.asarray(distances, dtype=float)
        nearest_indices = np.asarray(nearest_indices, dtype=np.int64)

    if non_decaying:
        q = np.ones(len(points), dtype=float)
    else:
        q = np.exp(-distances / range_)
        q[distances >= 4.0 * range_] = 0.0

    ratios = 1.0 + (strength - 1.0) * q
    normals = vertex_normals[nearest_indices]
    projectors = normals[:, :, None] * normals[:, None, :]
    identity = np.eye(3)[None, :, :]
    tangent = ratios ** (-1.0 / 3.0)
    normal = ratios ** (2.0 / 3.0)
    return (
        tangent[:, None, None] * (identity - projectors)
        + normal[:, None, None] * projectors
    )


class LabeledStructuralDomainBuilder3:
    """Build Leapfrog-compatible local domains from final point labels."""

    def __init__(
        self,
        base_range: float = 0.0,
        support_multiplier: int = 5,
        spheroidal_c: float = SPHEROIDAL3_C,
        minimum_support_points: int = 1,
    ) -> None:
        if base_range < 0.0:
            raise ValueError("base_range must be non-negative")
        if support_multiplier <= 0:
            raise ValueError("support_multiplier must be positive")
        if not spheroidal_c > 0.0:
            raise ValueError("spheroidal_c must be positive")
        if minimum_support_points <= 0:
            raise ValueError("minimum_support_points must be positive")

        self.base_range = float(base_range)
        self.support_multiplier = int(support_multiplier)
        self.spheroidal_c = float(spheroidal_c)
        self.minimum_support_points = int(minimum_support_points)
        self.diagnostics_: list[LabeledStructuralDomainDiagnostics3] | None = None

    def _resolved_base_range(self, model_parameters: Sequence[float]) -> float:
        if self.base_range > 0.0:
            return self.base_range
        if model_parameters and float(model_parameters[-1]) > 0.0:
            return float(model_parameters[-1])
        raise ValueError(
            "a positive base_range or model_parameters ending in the base range "
            "is required"
        )

    @staticmethod
    def _validate(
        points: np.ndarray,
        labels: np.ndarray,
        anisotropies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        labels = np.asarray(labels)
        anisotropies = np.asarray(anisotropies, dtype=float)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if len(points) == 0:
            raise ValueError("points must not be empty")
        if labels.ndim != 1 or len(labels) != len(points):
            raise ValueError("labels must have shape (n,)")
        if anisotropies.shape != (len(points), 3, 3):
            raise ValueError("anisotropies must have shape (n, 3, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        if not np.all(np.isfinite(anisotropies)):
            raise ValueError("anisotropies must be finite")

        if not np.issubdtype(labels.dtype, np.integer):
            rounded = np.rint(labels)
            if not np.allclose(labels, rounded, rtol=0.0, atol=0.0):
                raise ValueError("labels must contain integer values")
            labels = rounded.astype(np.int64)
        else:
            labels = labels.astype(np.int64, copy=False)

        return points, labels, anisotropies

    def compute(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        anisotropies: np.ndarray,
        model_parameters: Sequence[float] = (),
    ) -> list[LabeledStructuralDomainDiagnostics3]:
        """Compute exact post-clustering domain geometry and support metadata."""

        points, labels, anisotropies = self._validate(points, labels, anisotropies)
        base_range = self._resolved_base_range(model_parameters)
        sqrt_c = sqrt(self.spheroidal_c)
        point_count = len(points)

        diagnostics: list[LabeledStructuralDomainDiagnostics3] = []
        for label in np.unique(labels):
            core_indices = np.flatnonzero(labels == label).astype(np.int64)
            if len(core_indices) == 0:
                continue

            anisotropy = anisotropies[core_indices].mean(axis=0)
            anisotropy = 0.5 * (anisotropy + anisotropy.T)
            eigenvalues = np.linalg.eigvalsh(anisotropy)
            if not np.all(eigenvalues > 0.0):
                raise ValueError(
                    f"mean anisotropy for label {int(label)} is not positive definite"
                )

            desired_local_range = base_range * float(eigenvalues[-1])
            desired_internal_radius = desired_local_range / sqrt_c

            transformed_points = points @ anisotropy
            transformed_core = transformed_points[core_indices]
            distances = _nearest_core_distances(transformed_points, transformed_core)

            maximum_support = min(
                point_count,
                self.support_multiplier * len(core_indices),
            )
            desired_support_count = int(
                np.count_nonzero(distances < desired_internal_radius)
            )

            if desired_support_count <= maximum_support:
                internal_radius = desired_internal_radius
            elif maximum_support < point_count:
                internal_radius = float(
                    np.partition(distances, maximum_support)[maximum_support]
                )
            else:
                internal_radius = desired_internal_radius

            support_indices = np.flatnonzero(
                distances < internal_radius
            ).astype(np.int64)

            if len(support_indices) < self.minimum_support_points:
                needed = min(self.minimum_support_points, point_count)
                nearest = np.argpartition(distances, needed - 1)[:needed]
                support_indices = np.sort(nearest.astype(np.int64))
                internal_radius = float(
                    np.nextafter(distances[nearest].max(), np.inf)
                )

            local_kernel_range = internal_radius * sqrt_c
            core_min = points[core_indices].min(axis=0)
            core_max = points[core_indices].max(axis=0)
            inverse = np.linalg.inv(anisotropy)
            expansion = internal_radius * np.linalg.norm(inverse, axis=1)
            bbox_min = core_min - expansion
            bbox_max = core_max + expansion

            diagnostics.append(
                LabeledStructuralDomainDiagnostics3(
                    label=int(label),
                    core_indices=core_indices,
                    support_indices=support_indices,
                    anisotropy=anisotropy,
                    desired_internal_radius=float(desired_internal_radius),
                    internal_radius=float(internal_radius),
                    local_kernel_range=float(local_kernel_range),
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                )
            )

        self.diagnostics_ = diagnostics
        return diagnostics

    def build(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        anisotropies: np.ndarray,
        model_parameters: Sequence[float],
    ) -> list[StructuralDomain3]:
        """Return ``StructuralDomain3`` objects ready for interpolation."""

        parameters = [float(value) for value in model_parameters]
        if not parameters:
            raise ValueError(
                "model_parameters are required so each local kernel range can be set"
            )

        diagnostics = self.compute(points, labels, anisotropies, parameters)
        domains: list[StructuralDomain3] = []
        for item in diagnostics:
            local_parameters = parameters.copy()
            local_parameters[-1] = item.local_kernel_range
            domains.append(
                StructuralDomain3(
                    anisotropy=item.anisotropy,
                    bbox_min=item.bbox_min,
                    bbox_max=item.bbox_max,
                    support_indices=item.support_indices.tolist(),
                    model_parameters=local_parameters,
                )
            )
        return domains

    def build_from_inputs(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
        trend_type: object = StructuralTrendType.STRONGEST_ALONG_INPUTS,
    ) -> list[StructuralDomain3]:
        """Sample structural anisotropy from meshes, then build labeled domains."""

        inputs = list(inputs)
        if len(inputs) == 1:
            non_decaying = trend_type == StructuralTrendType.NON_DECAYING
            anisotropies = sample_single_input_anisotropies3(
                points,
                inputs[0],
                non_decaying=non_decaying,
            )
        else:
            samples = StructuralDomainBuilder3().sample(
                np.asarray(points, dtype=float),
                inputs,
                trend_type,
            )
            anisotropies = np.asarray(samples.anisotropies, dtype=float)

        return self.build(
            points,
            labels,
            anisotropies,
            model_parameters,
        )


LeapfrogLabeledDomainBuilder3 = LabeledStructuralDomainBuilder3
