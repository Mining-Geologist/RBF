"""Process worker for exact-support LVA with selectable domain extent.

This worker deliberately reuses the same recovered production corrections as the
confirmed benchmark:

* exact nearest-vertex, equal-face-normal LVA sampling with the hard 4R cutoff;
* finite LVA-geodesic automatic domains;
* ``background_blending=False``;
* topology-local, data-driven unsupported-branch completion;
* optional lower-Z-only full-depth domain extension to the user model minimum;
* globally aligned slab-streamed marching cubes.

The model extent, dataset, reference mesh, strength, range and domain-extent mode come
from the GUI, so no WolfPass coordinates or case names are embedded here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polatory
import polatory.automatic_domain_builder as automatic_module
from polatory.labeled_domain_builder import sample_single_input_anisotropies3

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MODULES = ROOT / "benchmarks" / "leapfrog_gold"
if str(BENCHMARK_MODULES) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_MODULES))

# Importing this benchmark module installs exactly the no-background-blending,
# topology-local support-completion wrapper and its calibration-before-meshing hook.
# It does not execute a benchmark because its main block is not entered.
import run_selected_no_background_blending_auto_support_decay_v2 as parity_support  # noqa: E402,F401
import polatory_lva_worker_process_v3 as worker  # noqa: E402

# Make the exact recovered sampler explicit on both Python paths used by the
# automatic SubDomainer and the finite-geodesic propagation grid.
automatic_module.sample_single_input_anisotropies3 = sample_single_input_anisotropies3
polatory.sample_single_input_anisotropies3 = sample_single_input_anisotropies3

_BASE_BUILDER = worker.FiniteLvaGeodesicAutomaticBuilder


class ExactFullDepthAutomaticBuilder:
    """Apply only the confirmed lower-Z domain extension after normal clustering."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._wrapped = _BASE_BUILDER(*args, **kwargs)

    def build_from_inputs(self, *args: Any, **kwargs: Any):
        original_domains = list(self._wrapped.build_from_inputs(*args, **kwargs))
        model_min_z = float(worker.v2._USER_BBOX_MIN[2])
        domains: list[Any] = []
        changed = 0

        for domain in original_domains:
            bbox_min = np.asarray(domain.bbox_min, dtype=np.float64).copy()
            bbox_max = np.asarray(domain.bbox_max, dtype=np.float64).copy()
            original_min_z = float(bbox_min[2])
            bbox_min[2] = min(original_min_z, model_min_z)
            changed += int(bbox_min[2] < original_min_z)
            domains.append(
                polatory.StructuralDomain3(
                    np.asarray(domain.anisotropy, dtype=np.float64),
                    bbox_min,
                    bbox_max,
                    np.asarray(domain.support_indices, dtype=np.int64).tolist(),
                    np.asarray(domain.model_parameters, dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                )
            )

        print(
            "PROGRESS\tExact full-depth mode: extended only the lower Z face of "
            f"{changed}/{len(domains)} automatic domains to model Z={model_min_z:g}; "
            "X/Y, upper Z, support memberships, anisotropy and local RBF parameters "
            "are unchanged.",
            flush=True,
        )
        return domains

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _domain_extent_mode() -> str:
    value = os.environ.get("POLATORY_DOMAIN_EXTENT_MODE", "full_depth")
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "full": "full_depth",
        "full_depth": "full_depth",
        "exact": "full_depth",
        "exact_full_depth": "full_depth",
        "finite": "finite",
        "finite_domains": "finite",
        "finite_automatic_domains": "finite",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            "POLATORY_DOMAIN_EXTENT_MODE must be 'full_depth' or 'finite', "
            f"not {value!r}."
        ) from error


def main() -> int:
    mode = _domain_extent_mode()
    if mode == "finite":
        worker.FiniteLvaGeodesicAutomaticBuilder = _BASE_BUILDER
        print(
            "PROGRESS\tDomain extent mode: Finite automatic domains. Recovered local "
            "LVA-geodesic bounds are retained; no face is extended to the model boundary.",
            flush=True,
        )
    else:
        worker.FiniteLvaGeodesicAutomaticBuilder = ExactFullDepthAutomaticBuilder
        print(
            "PROGRESS\tDomain extent mode: Exact full-depth. Lower Z faces may be "
            "extended to the model minimum after finite-domain construction.",
            flush=True,
        )
    return worker.main()


if __name__ == "__main__":
    raise SystemExit(main())
