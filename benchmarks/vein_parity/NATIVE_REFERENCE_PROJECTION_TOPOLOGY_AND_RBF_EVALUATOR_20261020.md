# Native reference-mesh projection and RBF-evaluator investigation (20 October 2026)

**Method:** read-only Ghidra 12.1.4 static analysis of owner-provided commercial executable, combined with independent PE method-table inspection. The decompiled commercial pseudocode and owner project files remain PRIVATE in the owner's connected repository and its access-controlled CI artifacts; this public branch only records aggregate engineering conclusions. No Leapfrog executable was launched or patched.

## New confirmed structure

Private Ghidra run **39339077379** completed successfully and decompiled bounded native helper functions connected to `PySurfaceOffsetInterpolant` and `PyRefinedWallsFunction`. A subsequent **private analysis reader run 39345282397** retrieved the saved results and verified the following:

1. The common conformal-distance wrapper at the compiled level delegates to a **reference-mesh geometric query** rather than simply taking the closest vertex. A projection query distinguishes 2D and 3D reference meshes through separate native paths (three-dimensional path `FUN_14032b180`, two-dimensional path `FUN_140329ff0`).
2. The reference-mesh query returns more than a distance: projected geometry, **face/edge/vertex classification**, and orientation/adjacency information. In ambiguous edge or vertex cases the query examines **connected incident triangle indices**, vectors/directions and signs to select a consistent result. The final distance sign can be inverted by an orientation option. This is genuine compiled topology-aware geometric processing and is compatible with the owner's overturned multiple-intersection benchmark.
3. The downstream refined-wall evaluator first queries this reference/conformal projection, then evaluates two stored wall fields at the projected 3D point through a native RBF evaluation dispatcher. The evaluator takes min/max thickness limits, computes a signed wall separation, and performs a **midpoint-preserving symmetric thickness correction** with independently verified coefficient **-0.5**. It adds the corrected wall offset to the signed conformal reference distance.
4. The refined-wall evaluator optionally checks and populates a **proximity cache** around recently evaluated sample positions; this is a speed optimization, not an additional geological constraint proven here.
5. The native RBF evaluation dispatcher is a generic numerical engine with saved evaluation accuracy settings, optional low-pass filter width, centre-count handling, and alternative evaluation paths. Those details could account for discrepancies with an untuned SciPy RBF surrogate, but **do not establish which kernel/degree/smoothing settings are selected for automatic folded references**, nor prove optional filtering is enabled in a particular vein model.

## Relationship to previous independent user benchmarks

We previously reproduced the signed HW/FW label for **95/95 real logged wall pairs** in five owner-provided synthetic native vein benchmarks with a drillhole-aware geometric surrogate: use a vertical reference-ray crossing closest in drilled depth, with nearest-face fallback if there is no reference-ray crossing. It produced 93/95 direct ray branches and 2/95 fallbacks in the five benchmark configurations, including all 15 overturned fold wall pairs. This **predictive agreement applies only after the correct native reference mesh is already supplied**.

The newly recovered compiled reference-query implementation is *more complicated* than that surrogate (explicit edge/vertex adjacency), hence **95/95 observational parity does not imply exact internal code correspondence**. Complex folded surfaces require consistent sheet/branch association at ambiguous intersections.

The independent thickness-correction module and tests already reside on the research branch:
- `native_wall_thickness_clamp.py` and `test_native_wall_thickness_clamp.py`;
- `reference_raycast_wall_polarity.py` and `test_reference_raycast_wall_polarity.py`.

The source-code-independent tests passed on GitHub Actions.

## Follow-up static work already initiated

The owner's private Ghidra workflow **39345407077** targets the 3D/2D reference-query subroutines and the two native RBF evaluation paths `FUN_140372110`/`FUN_140372920`. Until its result is inspected, we cannot claim the exact triangle intersection or RBF summation formula for these helpers.

**Remaining largest unknown:** the **automatic construction, fold topology, and orientation/handedness** of the `VeinReferenceSurfaceBlock` input reference itself. Neither decompiling the downstream offset evaluator nor recovering another model class's intrusion kernel resolves it. Reversing CSV drillhole order was previously proven not to change the synthetic folded reference. Further investigation should focus on reference input topology and generator entry points, not just downstream wall offsets.

All changes isolated to research branches. Do not merge to main based on these partial findings.
