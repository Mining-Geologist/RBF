"""Run one or more real-gold cases with automatic domain boxes expanded globally.

This is a diagnostic only.  It preserves the recovered automatic-domain anisotropies,
support indices, model parameters, and fitted structural interpolant, but removes finite
axis-aligned domain cutoffs by making every recovered domain active across the complete
model extent plus one base-range halo.  Comparing this output with the normal finite-
domain output isolates whether shelves, vertical walls, and forced bottoms originate at
the automatic domain bounding boxes rather than in the isosurface extractor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polatory

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_real_gold_suite as suite  # noqa: E402


_BASE_BUILDER = suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder


class FullExtentDiagnosticBuilder:
    """Delegate clustering, then remove only the finite axis-aligned box cutoffs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._builder = _BASE_BUILDER(*args, **kwargs)

    def build_from_inputs(
        self,
        points: np.ndarray,
        inputs: Sequence[object],
        model_parameters: Sequence[float],
        trend_type: object = polatory.StructuralTrendType.STRONGEST_ALONG_INPUTS,
    ) -> list[Any]:
        domains = list(
            self._builder.build_from_inputs(
                points,
                inputs,
                model_parameters=model_parameters,
                trend_type=trend_type,
            )
        )

        halo = float(suite.BASE_RANGE)
        bbox_min = np.asarray(suite.MODEL_MIN, dtype=float) - halo
        bbox_max = np.asarray(suite.MODEL_MAX, dtype=float) + halo
        expanded: list[Any] = []
        for domain in domains:
            expanded.append(
                polatory.StructuralDomain3(
                    np.asarray(domain.anisotropy, dtype=float),
                    bbox_min,
                    bbox_max,
                    np.asarray(domain.support_indices, dtype=np.int64).tolist(),
                    np.asarray(domain.model_parameters, dtype=float)
                    .reshape(-1)
                    .tolist(),
                )
            )

        print(
            "PROGRESS\tDiagnostic mode: expanded every recovered automatic domain "
            "across the model extent plus one base-range halo.",
            flush=True,
        )
        return expanded

    @property
    def diagnostics_(self) -> Any:
        return self._builder.diagnostics_

    @property
    def labels_(self) -> Any:
        return self._builder.labels_

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)


suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder = FullExtentDiagnosticBuilder
suite.OUTPUT_DIR = suite.ROOT / "benchmark-results" / "domain-diagnostic-full-extent"
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"


if __name__ == "__main__":
    raise SystemExit(suite.main())
