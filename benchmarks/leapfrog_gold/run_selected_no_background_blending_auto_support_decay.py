"""Run one selected case with fully data-driven structural-field support completion.

This diagnostic contains no case names, range thresholds, absolute support distances, or
strength-specific tuning.  It learns where the fitted field stops being supported from
that case's own input geometry and raw zero-surface behaviour:

1. Local data scale is estimated at every input point from its automatically selected
   neighbourhood size (cube root of the sample count).
2. Before final meshing, the raw background_blending=False field is sampled on a coarse
   globally aligned probe grid covering the requested model extent.
3. Distances from raw zero-crossing cells to the data are normalized by the local data
   scale, making the calculation independent of coordinate units and dataset density.
4. A deterministic two-population model separates the near-data supported crossings from
   a far-field crossing population such as the attached basal pancake.
5. The two learned population centres define a smooth transition toward OUTSIDE_VALUE.

The fitted field remains untouched throughout the learned supported population.  Range,
strength, data spacing, model extent, and field behaviour affect the result only through
the model and measurements themselves; none are mapped to hand-picked decay distances.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polatory
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Installs the selected-case filter and StructuralInterpolant3 background_blending=False.
import run_selected_no_background_blending as selected  # noqa: E402

suite = selected.suite
_BASE_FACTORY = polatory.StructuralInterpolant3
_OUTSIDE_VALUE = float(suite.OUTSIDE_VALUE)
_ORIGINAL_MESHER = suite.SAFE_MESHER.generate_safe_isosurface

# These settings limit diagnostic work only; they do not encode geological distances.
_PROBE_TARGET_NODES = int(os.environ.get("POLATORY_AUTO_SUPPORT_PROBE_NODES", "180000"))
_PROBE_BATCH_SIZE = int(os.environ.get("POLATORY_AUTO_SUPPORT_PROBE_BATCH_SIZE", "250000"))
_EM_MAX_ITERATIONS = int(os.environ.get("POLATORY_AUTO_SUPPORT_EM_MAX_ITERATIONS", "200"))


def _log_normal_density(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (
        np.log(2.0 * math.pi * variance)
        + ((values - mean) * (values - mean)) / variance
    )


def _fit_shared_variance_mixture(values: np.ndarray) -> dict[str, float | bool]:
    """Fit one- and two-population 1-D Gaussian models and select with BIC.

    Equal variance makes the supported-population probability monotonic with distance.
    The two starting centres are selected by splitting the sorted samples into equal-count
    halves, so initialization also contains no dataset-specific distance or quantile.
    """

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) < 16:
        return {
            "use_two_populations": False,
            "supported_center": float(np.min(x)) if len(x) else 0.0,
            "unsupported_center": float(np.max(x)) if len(x) else 0.0,
            "bic_one": float("inf"),
            "bic_two": float("inf"),
            "shared_standard_deviation": 0.0,
            "unsupported_fraction": 0.0,
        }

    ordered = np.sort(x)
    split = len(ordered) // 2
    mean_supported = float(np.mean(ordered[:split]))
    mean_unsupported = float(np.mean(ordered[split:]))
    total_variance = float(np.var(ordered))
    variance_floor = max(total_variance * np.finfo(np.float64).eps ** 0.5, 1.0e-12)
    variance = max(total_variance, variance_floor)
    supported_weight = 0.5

    previous_log_likelihood = -np.inf
    for _ in range(max(_EM_MAX_ITERATIONS, 1)):
        log_supported = (
            math.log(max(supported_weight, np.finfo(float).tiny))
            + _log_normal_density(ordered, mean_supported, variance)
        )
        log_unsupported = (
            math.log(max(1.0 - supported_weight, np.finfo(float).tiny))
            + _log_normal_density(ordered, mean_unsupported, variance)
        )
        maximum = np.maximum(log_supported, log_unsupported)
        denominator = maximum + np.log(
            np.exp(log_supported - maximum) + np.exp(log_unsupported - maximum)
        )
        responsibility_supported = np.exp(log_supported - denominator)
        responsibility_unsupported = 1.0 - responsibility_supported

        count_supported = float(np.sum(responsibility_supported))
        count_unsupported = float(np.sum(responsibility_unsupported))
        if count_supported <= 1.0 or count_unsupported <= 1.0:
            break

        new_supported_weight = count_supported / len(ordered)
        new_mean_supported = float(
            np.sum(responsibility_supported * ordered) / count_supported
        )
        new_mean_unsupported = float(
            np.sum(responsibility_unsupported * ordered) / count_unsupported
        )
        new_variance = float(
            (
                np.sum(
                    responsibility_supported
                    * (ordered - new_mean_supported)
                    * (ordered - new_mean_supported)
                )
                + np.sum(
                    responsibility_unsupported
                    * (ordered - new_mean_unsupported)
                    * (ordered - new_mean_unsupported)
                )
            )
            / len(ordered)
        )
        new_variance = max(new_variance, variance_floor)
        log_likelihood = float(np.sum(denominator))

        supported_weight = new_supported_weight
        mean_supported = new_mean_supported
        mean_unsupported = new_mean_unsupported
        variance = new_variance
        if abs(log_likelihood - previous_log_likelihood) <= (
            np.finfo(float).eps ** 0.5 * max(1.0, abs(log_likelihood))
        ):
            break
        previous_log_likelihood = log_likelihood

    if mean_supported > mean_unsupported:
        mean_supported, mean_unsupported = mean_unsupported, mean_supported
        supported_weight = 1.0 - supported_weight

    one_mean = float(np.mean(ordered))
    one_variance = max(float(np.var(ordered)), variance_floor)
    one_log_likelihood = float(
        np.sum(_log_normal_density(ordered, one_mean, one_variance))
    )

    log_supported = (
        math.log(max(supported_weight, np.finfo(float).tiny))
        + _log_normal_density(ordered, mean_supported, variance)
    )
    log_unsupported = (
        math.log(max(1.0 - supported_weight, np.finfo(float).tiny))
        + _log_normal_density(ordered, mean_unsupported, variance)
    )
    maximum = np.maximum(log_supported, log_unsupported)
    two_log_likelihood = float(
        np.sum(
            maximum
            + np.log(
                np.exp(log_supported - maximum)
                + np.exp(log_unsupported - maximum)
            )
        )
    )

    sample_count = float(len(ordered))
    bic_one = -2.0 * one_log_likelihood + 2.0 * math.log(sample_count)
    bic_two = -2.0 * two_log_likelihood + 4.0 * math.log(sample_count)
    use_two = bool(
        np.isfinite(bic_two)
        and bic_two < bic_one
        and mean_unsupported > mean_supported
    )
    return {
        "use_two_populations": use_two,
        "supported_center": mean_supported,
        "unsupported_center": mean_unsupported,
        "bic_one": bic_one,
        "bic_two": bic_two,
        "shared_standard_deviation": float(math.sqrt(variance)),
        "unsupported_fraction": float(1.0 - supported_weight),
    }


class _AutoSupportDecayInterpolant:
    """Fit normally, then learn a support-confidence field before meshing."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._tree: cKDTree | None = None
        self._local_scales: np.ndarray | None = None
        self._supported_center: float | None = None
        self._unsupported_center: float | None = None
        self.calibration_: dict[str, Any] | None = None

    def fit(self, points: Any, *args: Any, **kwargs: Any):
        fit_points = np.asarray(points, dtype=np.float64)
        if fit_points.ndim != 2 or fit_points.shape[1] != 3:
            raise ValueError("Structural fit points must have shape (n, 3).")
        if len(fit_points) < 2:
            raise ValueError("Automatic support calibration needs at least two points.")

        self._tree = cKDTree(fit_points)
        # The neighbourhood size follows sample count rather than a geological distance.
        neighbour_count = min(
            len(fit_points),
            max(2, int(math.ceil(len(fit_points) ** (1.0 / 3.0))) + 1),
        )
        neighbour_distances = np.asarray(
            self._tree.query(fit_points, k=neighbour_count)[0],
            dtype=np.float64,
        )
        if neighbour_distances.ndim == 1:
            neighbour_distances = neighbour_distances[:, None]
        local_scales = neighbour_distances[:, -1]
        valid = np.isfinite(local_scales) & (local_scales > 0.0)
        if not np.any(valid):
            raise ValueError("Input points do not define a positive local spacing scale.")
        replacement = float(np.median(local_scales[valid]))
        local_scales = np.where(valid, local_scales, replacement)
        self._local_scales = local_scales
        self.calibration_ = {
            "input_points": int(len(fit_points)),
            "neighbour_count": int(neighbour_count - 1),
            "local_scale_min": float(np.min(local_scales)),
            "local_scale_median": float(np.median(local_scales)),
            "local_scale_max": float(np.max(local_scales)),
        }
        return self._wrapped.fit(points, *args, **kwargs)

    def _raw_evaluate(self, points: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.asarray(
            self._wrapped.evaluate(points, *args, **kwargs),
            dtype=np.float64,
        )

    def _normalized_distance(self, query: np.ndarray) -> np.ndarray:
        if self._tree is None or self._local_scales is None:
            raise RuntimeError("Structural interpolant must be fitted before calibration.")
        distances, nearest = self._tree.query(query, k=1)
        distances = np.asarray(distances, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)
        return distances / self._local_scales[nearest]

    def calibrate_support(
        self,
        bbox_min: Any,
        bbox_max: Any,
        resolution: float,
        progress: Callable[[str], None],
    ) -> None:
        if self._supported_center is not None:
            return
        if self._tree is None or self._local_scales is None:
            raise RuntimeError("Structural interpolant must be fitted before calibration.")

        minimum = np.asarray(bbox_min, dtype=np.float64)
        maximum = np.asarray(bbox_max, dtype=np.float64)
        span = maximum - minimum
        if minimum.shape != (3,) or maximum.shape != (3,) or not np.all(span > 0.0):
            raise ValueError("Automatic support calibration needs valid 3-D bounds.")

        target_nodes = max(_PROBE_TARGET_NODES, 8)
        node_density = (target_nodes / float(np.prod(span))) ** (1.0 / 3.0)
        node_counts = np.maximum(np.rint(span * node_density).astype(np.int64) + 1, 3)
        # Correct rounding so the probe remains close to its requested computational budget.
        while int(np.prod(node_counts)) > 2 * target_nodes:
            axis = int(np.argmax(node_counts))
            node_counts[axis] = max(3, int(node_counts[axis]) - 1)

        coordinates = [
            np.linspace(minimum[axis], maximum[axis], int(node_counts[axis]))
            for axis in range(3)
        ]
        grid = np.meshgrid(*coordinates, indexing="ij")
        query = np.column_stack([axis.ravel(order="C") for axis in grid])
        progress(
            "Auto-calibrating field support on a data-scaled probe grid "
            f"{tuple(int(value) for value in node_counts)} ({len(query):,} nodes)…"
        )

        raw_flat = np.empty(len(query), dtype=np.float64)
        for start in range(0, len(query), max(_PROBE_BATCH_SIZE, 1)):
            stop = min(start + max(_PROBE_BATCH_SIZE, 1), len(query))
            raw_flat[start:stop] = self._raw_evaluate(query[start:stop]).reshape(-1)
        if not np.all(np.isfinite(raw_flat)):
            raise RuntimeError("Raw structural field returned non-finite probe values.")

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
        if len(crossing_indices) < 16:
            self._supported_center = float("inf")
            self._unsupported_center = float("inf")
            assert self.calibration_ is not None
            self.calibration_.update(
                {
                    "probe_node_counts": node_counts.tolist(),
                    "raw_zero_crossing_cells": int(len(crossing_indices)),
                    "support_decay_enabled": False,
                    "reason": "too few raw zero-crossing cells for population separation",
                }
            )
            progress("Auto support calibration found no separable far-field population.")
            return

        centers = np.empty((len(crossing_indices), 3), dtype=np.float64)
        for axis in range(3):
            lower = coordinates[axis][crossing_indices[:, axis]]
            upper = coordinates[axis][crossing_indices[:, axis] + 1]
            centers[:, axis] = 0.5 * (lower + upper)

        normalized_distance = self._normalized_distance(centers)
        log_distance = np.log1p(normalized_distance)
        mixture = _fit_shared_variance_mixture(log_distance)
        enabled = bool(mixture["use_two_populations"])
        if enabled:
            self._supported_center = float(mixture["supported_center"])
            self._unsupported_center = float(mixture["unsupported_center"])
        else:
            self._supported_center = float("inf")
            self._unsupported_center = float("inf")

        assert self.calibration_ is not None
        self.calibration_.update(
            {
                "probe_node_counts": node_counts.tolist(),
                "raw_zero_crossing_cells": int(len(crossing_indices)),
                "normalized_crossing_distance_min": float(np.min(normalized_distance)),
                "normalized_crossing_distance_median": float(
                    np.median(normalized_distance)
                ),
                "normalized_crossing_distance_max": float(np.max(normalized_distance)),
                "support_decay_enabled": enabled,
                **mixture,
            }
        )
        if enabled:
            progress(
                "Auto support calibration separated supported and unsupported raw "
                "zero-crossing populations: normalized log-distance centres "
                f"{self._supported_center:.6g} and {self._unsupported_center:.6g}."
            )
        else:
            progress(
                "Auto support calibration found one statistically preferred crossing "
                "population; the fitted field will remain unchanged."
            )

    def evaluate(self, points: Any, *args: Any, **kwargs: Any):
        query = np.asarray(points, dtype=np.float64)
        values = self._raw_evaluate(query, *args, **kwargs)
        original_shape = values.shape
        flat_values = values.reshape(-1)
        if query.ndim != 2 or query.shape[1] != 3 or len(query) != len(flat_values):
            raise ValueError(
                "Structural evaluation points must have shape (m, 3) and match the "
                "number of returned values."
            )

        if (
            self._supported_center is not None
            and self._unsupported_center is not None
            and np.isfinite(self._supported_center)
            and self._unsupported_center > self._supported_center
        ):
            normalized_distance = self._normalized_distance(query)
            log_distance = np.log1p(normalized_distance)
            t = np.clip(
                (log_distance - self._supported_center)
                / (self._unsupported_center - self._supported_center),
                0.0,
                1.0,
            )
            weight = t * t * (3.0 - 2.0 * t)
            flat_values = (1.0 - weight) * flat_values + weight * _OUTSIDE_VALUE

        exact_zero = flat_values == 0.0
        if np.any(exact_zero):
            nonzero = np.abs(flat_values[~exact_zero])
            scale = max(1.0, float(np.max(nonzero)) if nonzero.size else 1.0)
            flat_values = flat_values.copy()
            flat_values[exact_zero] = -np.finfo(np.float64).eps ** 0.5 * scale
        return flat_values.reshape(original_shape)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _auto_support_factory(*args: Any, **kwargs: Any):
    return _AutoSupportDecayInterpolant(_BASE_FACTORY(*args, **kwargs))


def _calibrating_mesher(
    structural: Any,
    bbox_min: Any,
    bbox_max: Any,
    resolution: float,
    refine: int,
    output_obj: Path,
    progress: Callable[[str], None],
):
    calibrate = getattr(structural, "calibrate_support", None)
    if callable(calibrate):
        calibrate(bbox_min, bbox_max, resolution, progress)
    return _ORIGINAL_MESHER(
        structural=structural,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        resolution=resolution,
        refine=refine,
        output_obj=output_obj,
        progress=progress,
    )


polatory.StructuralInterpolant3 = _auto_support_factory
suite.SAFE_MESHER.generate_safe_isosurface = _calibrating_mesher
suite.OUTPUT_DIR = (
    suite.ROOT
    / "benchmark-results"
    / "diagnostic-auto-calibrated-support-decay"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"

print(
    "PROGRESS\tDiagnostic mode: fully data-driven support calibration from local input "
    "spacing and the raw fitted field; no case/range/strength-specific decay values.",
    flush=True,
)


if __name__ == "__main__":
    raise SystemExit(suite.main())
