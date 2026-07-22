"""Test the basal domain-boundary hypothesis with full-depth domain boxes.

This diagnostic preserves the recovered Leapfrog LVA sampler, automatic domain
partition, support memberships, local RBF models, and all meshing settings.  The
only change is that every automatic domain evaluation box is extended downward
to the benchmark model minimum Z before fitting.  This prevents the local field
from disappearing at a finite domain floor and falling immediately to the
constant outside value.

The baseline exact-LVA runner remains unchanged.  Results are written to a
separate output directory so the two runs can be compared directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import polatory
import run_selected_exact_leapfrog_lva as baseline

suite = baseline.suite
diagnostic = baseline.diagnostic

suite.OUTPUT_DIR = (
    suite.ROOT / "benchmark-results" / "exact-leapfrog-lva-full-depth-domains"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"
diagnostic.DIAGNOSTIC_DIR = suite.OUTPUT_DIR / "basal-diagnostics"
baseline.CSV_DIR = suite.OUTPUT_DIR / "inspection-csv"

_BASE_CAPTURE_BUILDER = suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder
_MODEL_MIN_Z = float(suite.MODEL_MIN[2])


class _FullDepthDomainBuilder:
    """Extend only the lower Z face of each returned automatic domain."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._wrapped = _BASE_CAPTURE_BUILDER(*args, **kwargs)

    def build_from_inputs(self, *args: Any, **kwargs: Any):
        original_domains = list(self._wrapped.build_from_inputs(*args, **kwargs))
        extended_domains = []
        changed = 0

        for domain in original_domains:
            bbox_min = np.asarray(domain.bbox_min, dtype=np.float64).copy()
            bbox_max = np.asarray(domain.bbox_max, dtype=np.float64).copy()
            original_min_z = float(bbox_min[2])
            bbox_min[2] = min(original_min_z, _MODEL_MIN_Z)
            if bbox_min[2] < original_min_z:
                changed += 1

            extended_domains.append(
                polatory.StructuralDomain3(
                    np.asarray(domain.anisotropy, dtype=np.float64),
                    bbox_min,
                    bbox_max,
                    np.asarray(domain.support_indices, dtype=np.int64).tolist(),
                    np.asarray(domain.model_parameters, dtype=np.float64).tolist(),
                )
            )

        # The basal diagnostic and CSV exporter must inspect the domains that are
        # actually fitted, rather than the pre-extension copies captured by the
        # delegated diagnostic builder.
        diagnostic._CAPTURE["domains"] = extended_domains

        print(
            f"PROGRESS\tExtended the lower Z face of {changed}/{len(extended_domains)} "
            f"automatic domains to model Z={_MODEL_MIN_Z:g}; X/Y and upper Z faces, "
            "support indices, anisotropy matrices and model parameters are unchanged.",
            flush=True,
        )
        return extended_domains

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


suite.LVA_WORKER.FiniteLvaGeodesicAutomaticBuilder = _FullDepthDomainBuilder

print(
    "PROGRESS\tFull-depth-domain test enabled. This is a single-variable diagnostic: "
    "only automatic-domain bbox_min_z is extended to MODEL_MIN[2].",
    flush=True,
)


if __name__ == "__main__":
    exit_code = suite.main()
    diagnostic._write_cross_case_comparison()
    raise SystemExit(exit_code)
