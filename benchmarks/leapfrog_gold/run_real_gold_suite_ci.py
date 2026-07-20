"""CI entry point for the real Leapfrog gold benchmark.

The production worker chain replaces ``polatory.StructuralInterpolant3`` while it is
imported.  That Python wrapper exposes ``background_blending`` as a keyword-only
option.  Import the full benchmark first, then adapt the final constructor object so
the benchmark's historical five-positional-argument call reaches the production
wrapper using its supported API.
"""

from __future__ import annotations

import run_real_gold_suite as suite


_original_structural_interpolant = suite.polatory.StructuralInterpolant3


def _structural_interpolant_compat(*args, **kwargs):
    if len(args) == 5 and "background_blending" not in kwargs:
        args, background_blending = args[:4], args[4]
        kwargs["background_blending"] = background_blending
    return _original_structural_interpolant(*args, **kwargs)


suite.polatory.StructuralInterpolant3 = _structural_interpolant_compat


if __name__ == "__main__":
    raise SystemExit(suite.main())
