# Symmetric minimum/maximum vein-thickness correction (19 October 2026)

## Verified numerical finding

A read-only Ghidra static analysis of an owner-supplied Leapfrog executable identified two independently computed reference-relative wall offsets, a signed difference between them, clipping to saved minimum/maximum thickness bounds, and a symmetric correction factor. A separate private read-only PE scalar audit **confirmed this factor is -0.5**, rather than assuming it from appearances.

For provisional wall offsets `H` and `F`, signed thickness `t=H-F`, and bounded thickness `tc=clamp(t,min,max)`, the equivalent **original mathematical reconstruction** is:

- corrected upper wall `H' = H - (t-tc)/2`
- corrected lower wall `F' = F + (t-tc)/2`.

These satisfy `H'-F' = tc` and `(H'+F')/2 = (H+F)/2`. For example, an original 12m wall separation with max thickness 10m moves each boundary inward by 1m without shifting the centre.

**Evidence basis:** private successful Ghidra run 39337513831; private successful independent constant verification 39338595140. No decompiled vendor code, private project data or binaries are published here.

The native compiled refined-wall evaluator also calls a common conformal reference-mesh distance routine and an RBF evaluation interface to combine a reference-relative distance with the corrected offset. The existence of these calls is verified; the exact reference projection geometry and kernel have not yet been fully recovered.

## Scope and limits

The formula applies to the normal wall-thickness enforcement stage, when applicable. It does **not** establish the native policies for pinchout where minimum thickness is disabled, the ordering of max-thickness versus synthetic reversal in all cases, contact snapping, reference-mesh construction, multiple-intersection branch selection or final clipping/boolean rules.

It also **does not identify** the interpolation kernel for the automatic folded reference surface. Earlier native A-G tests matched synthetic pinchout intervals by raw-mesh intersection; those use the previously constructed reference and should not be conflated with reconstructing it from midpoint data.

## Independent RSGeo-compatible implementation

- `benchmarks/vein_parity/native_wall_thickness_clamp.py`
- `benchmarks/vein_parity/test_native_wall_thickness_clamp.py`

Five synthetic tests validate symmetric expansion/contraction, limits, unchanged midpoint, lack of constraints, and invalid configuration. All five passed locally; GitHub Actions includes a matching independent CI workflow. The public main branch remains unchanged.
