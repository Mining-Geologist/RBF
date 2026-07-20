"""Compatibility entry point for the process-isolated LVA worker.

The structural-domain implementation lives in ``polatory_lva_worker_process_v3``.
This entry point enables two corrections used by the Leapfrog-style launcher:

* overlapping chunk extraction without retaining temporary chunk faces;
* smooth completion of finite local-domain weights into the outside field.
"""

from polatory_lva_chunk_overlap import install_chunk_overlap
import polatory_lva_worker_process_v3 as worker


install_chunk_overlap(worker.v2.v8.v5.v3)

_native_structural_interpolant = worker.v2.v8.v5.v3.polatory.StructuralInterpolant3


def _leapfrog_structural_interpolant(
    base_model,
    outside_value=-1.0,
    blend_power=1.0,
    alignment_strength=0.0,
    background_blending=True,
):
    """Create the native interpolant while preserving the full constructor API.

    Leapfrog-style isolated runs default to background blending, but callers may
    still pass the fifth argument either positionally or by keyword.  Keeping the
    native-compatible signature is important for headless benchmarks and any future
    non-GUI use of the worker module.
    """
    return _native_structural_interpolant(
        base_model,
        float(outside_value),
        float(blend_power),
        float(alignment_strength),
        bool(background_blending),
    )


# The worker is a separate process, so this factory affects only isolated LVA runs
# and does not alter the normal Polatory API in the GUI process.
worker.v2.v8.v5.v3.polatory.StructuralInterpolant3 = (
    _leapfrog_structural_interpolant
)
main = worker.main


if __name__ == "__main__":
    raise SystemExit(main())
