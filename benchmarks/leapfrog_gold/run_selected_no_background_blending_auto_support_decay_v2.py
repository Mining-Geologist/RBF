"""Run one selected case with topology-local, fully data-driven support completion.

Version 1 learned two distance populations correctly, but then applied the learned decay
radially to every point at the same normalized data distance.  That can erode valid remote
parts of the geological surface even when only one attached branch is unsupported.

This version keeps the same data-driven population fit, then spatially localizes the
correction using the raw zero-surface itself:

1. Probe the raw ``background_blending=False`` field on the automatic calibration grid.
2. Classify raw zero-crossing cells by their fitted supported/unsupported posterior.
3. Build spatial indices for both crossing populations.
4. Modify the field only where a query point is both statistically more likely to belong
   to the unsupported population and spatially closer to an unsupported crossing branch.

The 0.5 boundaries below are posterior decision boundaries, not geological distances or
case tuning.  There are no case names, range thresholds, strength rules, metre values, or
dataset-specific support multipliers.  If either population cannot be established, the
raw fitted field remains unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polatory
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_selected_no_background_blending_auto_support_decay as v1  # noqa: E402

suite = v1.suite
_BASE_FACTORY = v1._BASE_FACTORY
_OUTSIDE_VALUE = float(suite.OUTSIDE_VALUE)


def _posterior_unsupported(
    values: np.ndarray,
    supported_center: float,
    unsupported_center: float,
    standard_deviation: float,
    unsupported_fraction: float,
) -> np.ndarray:
    """Return the fitted two-population posterior without physical thresholds."""
    x = np.asarray(values, dtype=np.float64)
    variance = max(float(standard_deviation) ** 2, np.finfo(np.float64).tiny)
    unsupported_weight = float(
        np.clip(unsupported_fraction, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    )
    supported_weight = 1.0 - unsupported_weight
    log_supported = (
        math.log(supported_weight)
        + v1._log_normal_density(x, float(supported_center), variance)
    )
    log_unsupported = (
        math.log(unsupported_weight)
        + v1._log_normal_density(x, float(unsupported_center), variance)
    )
    maximum = np.maximum(log_supported, log_unsupported)
    supported_score = np.exp(log_supported - maximum)
    unsupported_score = np.exp(log_unsupported - maximum)
    return unsupported_score / (supported_score + unsupported_score)


def _decision_ramp(probability: np.ndarray) -> np.ndarray:
    """Map the natural MAP boundary to zero and certainty to one, smoothly."""
    t = np.clip(2.0 * np.asarray(probability, dtype=np.float64) - 1.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _TopologyLocalAutoSupportInterpolant(v1._AutoSupportDecayInterpolant):
    """Learn unsupported crossings, but apply completion only near those branches."""

    def __init__(self, wrapped: Any) -> None:
        super().__init__(wrapped)
        self._supported_crossing_tree: cKDTree | None = None
        self._unsupported_crossing_tree: cKDTree | None = None

    def calibrate_support(
        self,
        bbox_min: Any,
        bbox_max: Any,
        resolution: float,
        progress: Callable[[str], None],
    ) -> None:
        # Version 1 performs the unitless local-spacing calculation, raw-field probing,
        # BIC model selection and two-population fit.
        super().calibrate_support(bbox_min, bbox_max, resolution, progress)
        calibration = self.calibration_ or {}
        if not bool(calibration.get("support_decay_enabled", False)):
            return

        supported_center = float(calibration["supported_center"])
        unsupported_center = float(calibration["unsupported_center"])
        standard_deviation = float(calibration["shared_standard_deviation"])
        unsupported_fraction = float(calibration["unsupported_fraction"])
        if not (
            np.isfinite(supported_center)
            and np.isfinite(unsupported_center)
            and unsupported_center > supported_center
            and np.isfinite(standard_deviation)
            and standard_deviation > 0.0
        ):
            progress("Topology-local support calibration rejected an invalid mixture fit.")
            return

        minimum = np.asarray(bbox_min, dtype=np.float64)
        maximum = np.asarray(bbox_max, dtype=np.float64)
        span = maximum - minimum
        target_nodes = max(v1._PROBE_TARGET_NODES, 8)
        node_density = (target_nodes / float(np.prod(span))) ** (1.0 / 3.0)
        node_counts = np.maximum(np.rint(span * node_density).astype(np.int64) + 1, 3)
        while int(np.prod(node_counts)) > 2 * target_nodes:
            axis = int(np.argmax(node_counts))
            node_counts[axis] = max(3, int(node_counts[axis]) - 1)

        coordinates = [
            np.linspace(minimum[axis], maximum[axis], int(node_counts[axis]))
            for axis in range(3)
        ]
        grid = np.meshgrid(*coordinates, indexing="ij")
        query = np.column_stack([axis.ravel(order="C") for axis in grid])
        raw_flat = np.empty(len(query), dtype=np.float64)
        batch_size = max(v1._PROBE_BATCH_SIZE, 1)
        for start in range(0, len(query), batch_size):
            stop = min(start + batch_size, len(query))
            raw_flat[start:stop] = self._raw_evaluate(query[start:stop]).reshape(-1)

        field_scale = max(1.0, float(np.max(np.abs(raw_flat))))
        zero_tolerance = np.finfo(np.float64).eps * field_scale
        raw_flat[np.abs(raw_flat) <= zero_tolerance] = -zero_tolerance
        volume = raw_flat.reshape(tuple(int(value) for value in node_counts), order="C")
        corners = (
            volume[:-1, :-1, :-1],
            volume[1:, :-1, :-1],
            volume[:-1, 1:, :-1],
            volume[:-1, :-1, 1:],
            volume[1:, 1:, :-1],
            volume[1:, :-1, 1:],
            volume[:-1, 1:, 1:],
            volume[1:, 1:, 1:],
        )
        cell_minimum = np.minimum.reduce(corners)
        cell_maximum = np.maximum.reduce(corners)
        crossing_indices = np.argwhere(
            (cell_minimum <= 0.0) & (cell_maximum >= 0.0)
        )
        if len(crossing_indices) == 0:
            progress("Topology-local calibration found no raw zero-crossing cells.")
            return

        crossing_centers = np.empty((len(crossing_indices), 3), dtype=np.float64)
        for axis in range(3):
            lower = coordinates[axis][crossing_indices[:, axis]]
            upper = coordinates[axis][crossing_indices[:, axis] + 1]
            crossing_centers[:, axis] = 0.5 * (lower + upper)

        normalized_distance = self._normalized_distance(crossing_centers)
        log_distance = np.log1p(normalized_distance)
        posterior = _posterior_unsupported(
            log_distance,
            supported_center,
            unsupported_center,
            standard_deviation,
            unsupported_fraction,
        )
        unsupported_mask = posterior > 0.5
        supported_mask = ~unsupported_mask
        if not np.any(supported_mask) or not np.any(unsupported_mask):
            progress(
                "Topology-local calibration could not retain both crossing populations; "
                "the fitted field remains unchanged."
            )
            return

        self._supported_crossing_tree = cKDTree(crossing_centers[supported_mask])
        self._unsupported_crossing_tree = cKDTree(crossing_centers[unsupported_mask])
        calibration.update(
            {
                "topology_localization_enabled": True,
                "supported_crossing_cells": int(np.count_nonzero(supported_mask)),
                "unsupported_crossing_cells": int(np.count_nonzero(unsupported_mask)),
            }
        )
        progress(
            "Topology-local calibration retained "
            f"{np.count_nonzero(supported_mask):,} supported and "
            f"{np.count_nonzero(unsupported_mask):,} unsupported raw crossing cells; "
            "decay will be confined to the unsupported branch neighbourhood."
        )

    def evaluate(self, points: Any, *args: Any, **kwargs: Any):
        query = np.asarray(points, dtype=np.float64)
        raw_values = self._raw_evaluate(query, *args, **kwargs)
        original_shape = raw_values.shape
        flat_values = raw_values.reshape(-1)
        if query.ndim != 2 or query.shape[1] != 3 or len(query) != len(flat_values):
            raise ValueError(
                "Structural evaluation points must have shape (m, 3) and match the "
                "number of returned values."
            )

        calibration = self.calibration_ or {}
        if (
            self._supported_crossing_tree is not None
            and self._unsupported_crossing_tree is not None
            and bool(calibration.get("topology_localization_enabled", False))
        ):
            normalized_distance = self._normalized_distance(query)
            log_distance = np.log1p(normalized_distance)
            posterior = _posterior_unsupported(
                log_distance,
                float(calibration["supported_center"]),
                float(calibration["unsupported_center"]),
                float(calibration["shared_standard_deviation"]),
                float(calibration["unsupported_fraction"]),
            )
            statistical_weight = _decision_ramp(posterior)

            distance_supported = np.asarray(
                self._supported_crossing_tree.query(query, k=1)[0], dtype=np.float64
            )
            distance_unsupported = np.asarray(
                self._unsupported_crossing_tree.query(query, k=1)[0], dtype=np.float64
            )
            denominator = distance_supported + distance_unsupported
            spatial_probability = np.divide(
                distance_supported,
                denominator,
                out=np.zeros_like(distance_supported),
                where=denominator > 0.0,
            )
            spatial_weight = _decision_ramp(spatial_probability)
            weight = statistical_weight * spatial_weight
            flat_values = (1.0 - weight) * flat_values + weight * _OUTSIDE_VALUE

        exact_zero = flat_values == 0.0
        if np.any(exact_zero):
            nonzero = np.abs(flat_values[~exact_zero])
            scale = max(1.0, float(np.max(nonzero)) if nonzero.size else 1.0)
            flat_values = flat_values.copy()
            flat_values[exact_zero] = -np.finfo(np.float64).eps ** 0.5 * scale
        return flat_values.reshape(original_shape)


def _topology_local_factory(*args: Any, **kwargs: Any):
    return _TopologyLocalAutoSupportInterpolant(_BASE_FACTORY(*args, **kwargs))


polatory.StructuralInterpolant3 = _topology_local_factory
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-topology-local-auto-support-decay"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: fully data-driven topology-local support calibration; "
    "supported surface branches remain untouched and decay is confined to spatially "
    "unsupported raw zero-surface branches.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
