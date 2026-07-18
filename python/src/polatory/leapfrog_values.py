"""Leapfrog-compatible binary indicator value preprocessing.

Leapfrog does not fit the input +/-1 indicator values directly. For each point
it first calculates the Euclidean distance to the nearest point belonging to
the opposite class, assigns the opposite sign of the input indicator, and clips
the magnitude using the data bounding-box diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeapfrogIndicatorValues3:
    """Recovered binary values and automatic fitting scales."""

    values: np.ndarray
    raw_signed_distances: np.ndarray
    data_diagonal: float
    fit_accuracy: float
    clipping_distance: float


def _nearest_distances_numpy(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    query_chunk_size: int = 4096,
    reference_chunk_size: int = 1024,
) -> np.ndarray:
    result = np.full(len(query), np.inf, dtype=float)
    for query_start in range(0, len(query), query_chunk_size):
        query_stop = min(query_start + query_chunk_size, len(query))
        local_query = query[query_start:query_stop]
        minimum_squared = np.full(len(local_query), np.inf, dtype=float)

        for reference_start in range(0, len(reference), reference_chunk_size):
            reference_stop = min(
                reference_start + reference_chunk_size,
                len(reference),
            )
            local_reference = reference[reference_start:reference_stop]
            diff = local_query[:, None, :] - local_reference[None, :, :]
            squared = np.einsum("qri,qri->qr", diff, diff, optimize=True)
            minimum_squared = np.minimum(
                minimum_squared,
                squared.min(axis=1),
            )

        result[query_start:query_stop] = np.sqrt(minimum_squared)
    return result


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        return _nearest_distances_numpy(query, reference)

    tree = cKDTree(reference)
    try:
        distances, _ = tree.query(query, k=1, workers=-1)
    except TypeError:
        distances, _ = tree.query(query, k=1)
    return np.asarray(distances, dtype=float)


def leapfrog_indicator_values3(
    points: np.ndarray,
    indicators: np.ndarray,
    *,
    fit_accuracy: float = 0.0,
    clipping_distance: float = 0.0,
) -> LeapfrogIndicatorValues3:
    """Convert binary signed indicators to Leapfrog's fitting values.

    Parameters
    ----------
    points:
        Point coordinates with shape ``(n, 3)``.
    indicators:
        A one-dimensional signed binary array. Positive indicators receive
        negative output values and negative indicators receive positive output
        values, matching the supplied WolfPass SDF convention.
    fit_accuracy:
        Optional explicit fitting accuracy. ``0`` uses
        ``1e-5 * data_bbox_diagonal``.
    clipping_distance:
        Optional explicit magnitude cap. ``0`` uses
        ``1e-2 * data_bbox_diagonal``, exactly 1000 times the automatic fitting
        accuracy.
    """

    points = np.asarray(points, dtype=float)
    indicators = np.asarray(indicators, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if len(points) == 0:
        raise ValueError("points must not be empty")
    if indicators.ndim != 1 or len(indicators) != len(points):
        raise ValueError("indicators must have shape (n,)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    if not np.all(np.isfinite(indicators)):
        raise ValueError("indicators must be finite")
    if np.any(indicators == 0.0):
        raise ValueError("indicators must be non-zero")

    signs = np.sign(indicators)
    unique_signs = np.unique(signs)
    if len(unique_signs) != 2 or not np.array_equal(
        unique_signs,
        np.array([-1.0, 1.0]),
    ):
        raise ValueError("indicators must contain both positive and negative values")

    data_diagonal = float(
        np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    )
    if not data_diagonal > 0.0:
        raise ValueError("point bounding box must have a positive diagonal")

    if fit_accuracy == 0.0:
        fit_accuracy = 1.0e-5 * data_diagonal
    elif not fit_accuracy > 0.0:
        raise ValueError("fit_accuracy must be positive or zero for automatic")

    if clipping_distance == 0.0:
        clipping_distance = 1.0e-2 * data_diagonal
    elif not clipping_distance > 0.0:
        raise ValueError(
            "clipping_distance must be positive or zero for automatic"
        )

    nearest_opposite = np.empty(len(points), dtype=float)
    positive = signs > 0.0
    negative = ~positive
    nearest_opposite[positive] = _nearest_distances(
        points[positive],
        points[negative],
    )
    nearest_opposite[negative] = _nearest_distances(
        points[negative],
        points[positive],
    )

    raw_signed_distances = -signs * nearest_opposite
    values = np.clip(
        raw_signed_distances,
        -clipping_distance,
        clipping_distance,
    )

    return LeapfrogIndicatorValues3(
        values=values,
        raw_signed_distances=raw_signed_distances,
        data_diagonal=data_diagonal,
        fit_accuracy=float(fit_accuracy),
        clipping_distance=float(clipping_distance),
    )
