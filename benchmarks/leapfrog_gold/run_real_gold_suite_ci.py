"""CI entry point for the real Leapfrog gold benchmark.

The compiled ``StructuralInterpolant3`` binding exposes ``background_blending`` as a
keyword-only constructor option on the Windows wheel.  The benchmark originally
passed it as a fifth positional argument, while the interactive worker already uses
the supported API shape.  Keep the benchmark source unchanged and adapt that one
constructor call here so the production path can be replayed headlessly.
"""

from __future__ import annotations

import polatory


_original_structural_interpolant = polatory.StructuralInterpolant3


def _structural_interpolant_compat(*args, **kwargs):
    if len(args) == 5 and "background_blending" not in kwargs:
        args, background_blending = args[:4], args[4]
        kwargs["background_blending"] = background_blending
    return _original_structural_interpolant(*args, **kwargs)


polatory.StructuralInterpolant3 = _structural_interpolant_compat

from run_real_gold_suite import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
