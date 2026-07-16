"""Deterministic clustered structural-domain builder for local-varying anisotropy.

The builder samples the structural anisotropy at every interpolation point,
clusters points jointly in active spatial coordinates and standardized
anisotropy-matrix space, selects an actual sampled matrix medoid for each
cluster, and expands each core by the anisotropic local RBF range.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Sequence

import numpy as np

from ._structural import (
    StructuralDomain3,
    StructuralDomainBuilder3,
    StructuralTrendType,
)


@dataclass(frozen=True)
class _KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


class ClusteredStructuralDomainBuilder3:
    """Build overlapping structural domains by deterministic matrix-space clustering.

    Parameters
    ----------
    domain_count:
        Number of domains. ``0`` selects ``ceil(sqrt(n_points) / 3)``.
    base_range:
        Base RBF range before applying ``local_range_scale``. ``0`` derives it
        from the final model parameter when available, otherwise from the
        largest structural-trend range.
    local_range_scale:
        Scale applied to the base RBF range for every local solve. The tested
        default is ``0.8``.
    restarts:
        Number of deterministic k-means++ restarts.
    random_seed:
        Seed controlling the deterministic restarts.
    maximum_iterations:
        Maximum Lloyd iterations per restart.
    minimum_support_points:
        Minimum number of interpolation points in an expanded domain box.
    spatial_weight, anisotropy_weight:
        Optional feature-group weights. Defaults reproduce the validated
        implementation.
    """

    def __init__(
        self,
        domain_count: int = 0,
        base_range: float = 0.0,
        local_range_scale: float = 0.8,
        restarts: int = 20,
        random_seed: int = 42,
        maximum_iterations: int = 200,
        minimum_support_points: int = 4,
        spatial_weight: float = 1.0,
        anisotropy_weight: float = 1.0,
    ) -> None:
        if domain_count < 0:
            raise ValueError("domain_count must be non-negative")
        if base_range < 0.0:
            raise ValueError("base_range must be non-negative")
        if not local_range_scale > 0.0:
            raise ValueError("local_range_scale must be positive")
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        if maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if minimum_support_points <= 0:
            raise ValueError("minimum_support_points must be positive")
        if not spatial_weight > 0.0:
            raise ValueError("spatial_weight must be positive")
        if not anisotropy_weight > 0.0:
            raise ValueError("anisotropy_weight must be positive")

        self.domain_count = int(domain_count)
        self.base_range = float(base_range)
        self.local_range_scale = float(local_range_scale)
        self.restarts = int(restarts)
        self.random_seed = int(random_seed)
        self.maximum_iterations = int(maximum_iterations)
        self.minimum_support_points = int(minimum_support_points)
        self.spatial_weight = float(spatial_weight)
        self.anisotropy_weight = float(anisotropy_weight)

        self.labels_: np.ndarray | None = None
        self.active_axes_: np.ndarray | None = None
        self.medoid_indices_: np.ndarray | None = None
        self.features_: np.ndarray | None = None
        self.domain_count_: int | None = None
        self.local_range_: float | None = None
        self.inertia_: float | None = None

    @staticmethod
    def _matrix_features(anisotropies: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                anisotropies[:, 0, 0],
                anisotropies[:, 0, 1],
                anisotropies[:, 0, 2],
                anisotropies[:, 1, 1],
                anisotropies[:, 1, 2],
                anisotropies[:, 2, 2],
            ]
        )

    @classmethod
    def _detect_active_axes(
        cls,
        points: np.ndarray,
        anisotropies: np.ndarray,
        number_of_bins: int = 8,
        variation_threshold: float = 0.02,
    ) -> np.ndarray:
        matrix_features = cls._matrix_features(anisotropies)
        global_mean = matrix_features.mean(axis=0)
        active = np.zeros(3, dtype=bool)

        for axis in range(3):
            coordinate_min = float(points[:, axis].min())
            coordinate_max = float(points[:, axis].max())
            span = coordinate_max - coordinate_min
            if not span > 0.0:
                continue

            normalized = (points[:, axis] - coordinate_min) / span
            bins = np.clip(
                np.floor(normalized * number_of_bins).astype(np.int64),
                0,
                number_of_bins - 1,
            )

            maximum_variation = 0.0
            for bin_id in range(number_of_bins):
                indices = bins == bin_id
                if not np.any(indices):
                    continue
                local_mean = matrix_features[indices].mean(axis=0)
                maximum_variation = max(
                    maximum_variation,
                    float(np.linalg.norm(local_mean - global_mean)),
                )

            active[axis] = maximum_variation > variation_threshold

        if not np.any(active):
            active[:] = True
        return active

    @staticmethod
    def _kmeans(
        features: np.ndarray,
        cluster_count: int,
        restarts: int,
        random_seed: int,
        maximum_iterations: int,
    ) -> _KMeansResult:
        random = np.random.default_rng(random_seed)
        point_count = len(features)
        best: _KMeansResult | None = None

        for _ in range(restarts):
            centers = np.empty((cluster_count, features.shape[1]), dtype=float)
            first_index = int(random.integers(point_count))
            centers[0] = features[first_index]

            minimum_distance_squared = np.sum(
                (features - centers[0]) ** 2,
                axis=1,
            )

            for center_id in range(1, cluster_count):
                total = float(minimum_distance_squared.sum())
                if total <= 0.0:
                    point_index = int(random.integers(point_count))
                else:
                    probabilities = minimum_distance_squared / total
                    point_index = int(random.choice(point_count, p=probabilities))

                centers[center_id] = features[point_index]
                new_distance_squared = np.sum(
                    (features - centers[center_id]) ** 2,
                    axis=1,
                )
                minimum_distance_squared = np.minimum(
                    minimum_distance_squared,
                    new_distance_squared,
                )

            labels: np.ndarray | None = None

            for _ in range(maximum_iterations):
                distance_squared = np.sum(
                    (
                        features[:, None, :]
                        - centers[None, :, :]
                    )
                    ** 2,
                    axis=2,
                )
                new_labels = np.argmin(distance_squared, axis=1)

                if labels is not None and np.array_equal(new_labels, labels):
                    break
                labels = new_labels

                for cluster_id in range(cluster_count):
                    indices = np.flatnonzero(labels == cluster_id)
                    if len(indices) > 0:
                        centers[cluster_id] = features[indices].mean(axis=0)
                    else:
                        centers[cluster_id] = features[
                            int(random.integers(point_count))
                        ]

            assert labels is not None
            inertia = float(np.sum((features - centers[labels]) ** 2))
            candidate = _KMeansResult(labels.copy(), centers.copy(), inertia)
            if best is None or candidate.inertia < best.inertia:
                best = candidate

        assert best is not None
        return best

    @staticmethod
    def _matrix_medoid(
        anisotropies: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        local = anisotropies[indices]
        average = local.mean(axis=0)
        distances = np.linalg.norm(local - average, axis=(1, 2))
        local_medoid = int(np.argmin(distances))
        global_medoid = int(indices[local_medoid])
        return global_medoid, anisotropies[global_medoid]

    @staticmethod
    def _support_indices(
        points: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> np.ndarray:
        return np.flatnonzero(
            np.all(
                (points >= minimum) & (points <= maximum),
                axis=1,
            )
        ).astype(np.int64)

    def _resolved_domain_count(self, point_count: int) -> int:
        if self.domain_count > 0:
            return min(self.domain_count, point_count)
        return min(point_count, max(2, int(ceil(sqrt(point_count) / 3.0))))

    def _resolved_base_range(
        self,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
    ) -> float:
        if self.base_range > 0.0:
            return self.base_range
        if len(model_parameters) >= 3 and float(model_parameters[-1]) > 0.0:
            return float(model_parameters[-1])

        maximum_input_range = max(float(item.range) for item in inputs)
        if not maximum_input_range > 0.0:
            raise ValueError("could not resolve a positive base range")
        return maximum_input_range

    def build(
        self,
        points: np.ndarray,
        inputs: Sequence[object],
        trend_type: object = StructuralTrendType.STRONGEST_ALONG_INPUTS,
        model_parameters: Sequence[float] = (),
    ) -> list[StructuralDomain3]:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if len(points) == 0:
            raise ValueError("points must not be empty")
        if not inputs:
            raise ValueError("inputs must not be empty")

        samples = StructuralDomainBuilder3().sample(
            points,
            list(inputs),
            trend_type,
        )
        anisotropies = np.asarray(samples.anisotropies, dtype=float)
        if anisotropies.shape != (len(points), 3, 3):
            raise RuntimeError("unexpected sampled anisotropy shape")

        active_axes = self._detect_active_axes(points, anisotropies)
        maximum_input_range = max(float(item.range) for item in inputs)

        spatial = (
            points[:, active_axes]
            - points[:, active_axes].mean(axis=0)
        ) / maximum_input_range

        matrix_features = self._matrix_features(anisotropies)
        matrix_mean = matrix_features.mean(axis=0)
        matrix_scale = matrix_features.std(axis=0)
        matrix_scale[matrix_scale < 1e-12] = 1.0
        matrix_features = (matrix_features - matrix_mean) / matrix_scale

        features = np.column_stack(
            [
                self.spatial_weight * spatial,
                self.anisotropy_weight * matrix_features,
            ]
        )

        cluster_count = self._resolved_domain_count(len(points))
        result = self._kmeans(
            features,
            cluster_count,
            self.restarts,
            self.random_seed,
            self.maximum_iterations,
        )

        base_range = self._resolved_base_range(inputs, model_parameters)
        local_range = self.local_range_scale * base_range
        if not local_range > 0.0:
            raise ValueError("resolved local range must be positive")

        domains: list[StructuralDomain3] = []
        medoid_indices: list[int] = []

        for cluster_id in range(cluster_count):
            core_indices = np.flatnonzero(result.labels == cluster_id)
            if len(core_indices) == 0:
                continue

            medoid_index, anisotropy = self._matrix_medoid(
                anisotropies,
                core_indices,
            )
            medoid_indices.append(medoid_index)

            core_minimum = points[core_indices].min(axis=0)
            core_maximum = points[core_indices].max(axis=0)

            inverse = np.linalg.inv(anisotropy)
            expansion = local_range * np.linalg.norm(inverse, axis=1)
            bbox_minimum = core_minimum - expansion
            bbox_maximum = core_maximum + expansion

            support_indices = self._support_indices(
                points,
                bbox_minimum,
                bbox_maximum,
            )

            attempts = 0
            while (
                len(support_indices) < self.minimum_support_points
                and attempts < 8
            ):
                expansion *= 1.25
                bbox_minimum = core_minimum - expansion
                bbox_maximum = core_maximum + expansion
                support_indices = self._support_indices(
                    points,
                    bbox_minimum,
                    bbox_maximum,
                )
                attempts += 1

            local_parameters = list(float(value) for value in model_parameters)
            if local_parameters:
                local_parameters[-1] = local_range

            domains.append(
                StructuralDomain3(
                    anisotropy=anisotropy,
                    bbox_min=bbox_minimum,
                    bbox_max=bbox_maximum,
                    support_indices=support_indices.tolist(),
                    model_parameters=local_parameters,
                )
            )

        self.labels_ = result.labels.copy()
        self.active_axes_ = active_axes.copy()
        self.medoid_indices_ = np.asarray(medoid_indices, dtype=np.int64)
        self.features_ = features.copy()
        self.domain_count_ = len(domains)
        self.local_range_ = float(local_range)
        self.inertia_ = float(result.inertia)

        return domains


def fit_from_meshes_clustered(
    interpolant: object,
    points: np.ndarray,
    values: np.ndarray,
    inputs: Sequence[object],
    tolerance: float,
    *,
    trend_type: object = StructuralTrendType.STRONGEST_ALONG_INPUTS,
    model_parameters: Sequence[float] = (),
    max_iter: int = 100,
    accuracy: float = float("inf"),
    builder: ClusteredStructuralDomainBuilder3 | None = None,
) -> list[StructuralDomain3]:
    """Build clustered domains, fit a structural interpolant, and return domains."""

    if builder is None:
        builder = ClusteredStructuralDomainBuilder3()

    domains = builder.build(
        points,
        inputs,
        trend_type,
        model_parameters,
    )
    interpolant.fit(
        points,
        values,
        domains,
        tolerance,
        max_iter,
        accuracy,
    )
    return domains
