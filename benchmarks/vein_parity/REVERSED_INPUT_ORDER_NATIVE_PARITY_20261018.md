# Reversed drillhole CSV order: native reference is numerically identical — 2026-10-18

**Experiment:** User independently imported a reversed-record-order copy of the original `04_MULTIPLE_INTERSECTIONS` synthetic drillhole set into Leapfrog Geo and regenerated the same vein using visually matching settings. The user reported all references, hangingwall/footwall surfaces and final solids looked identical and supplied the newly exported reference OBJ for quantitative comparison.

**Files compared:** original `Vein_Benchmark_Extracted/04_MULTIPLE_INTERSECTIONS/Reference_Surface.obj` from the owner's original benchmark archive, and new `06_Multiple_intersections_reversed - 06_Multiple_intersections_reversed_vein Reference Surface.obj` uploaded in the follow-up conversation. Only this **reference OBJ** was independently checked numerically; other surfaces/solids remain user-reported visual comparisons.

| Property | Original | Reversed |
|---|---:|---:|
| Vertices | 349 | 349 |
| Triangles | 606 | 606 |
| Mesh bounds | [-121,-121,-10]–[121,121,210] m | Identical |
| Vertex index ordering | baseline | **Identical** |
| Triangle index arrays | baseline | **Exactly identical** |

- Maximum absolute coordinate-axis difference: **4.9725912277e-9 m**.
- Maximum corresponding vertex Euclidean distance: **6.92038826145e-9 m**.
- Mean corresponding vertex distance: **3.87536048963e-9 m**.
- Root-mean-square corresponding vertex distance: **4.11012583933e-9 m**.
- All 349 coordinate triplets agree after rounding to eight decimal places.

This is stronger than simply checking mesh-to-mesh nearest distances: with identical triangle connectivity and vertex order, matching barycentric positions on corresponding triangles differ by **at most 6.93e-9 m** by triangle inequality, giving a certified upper bound on symmetric Hausdorff surface distance. The tiny differences arise from the original archive's eight-decimal OBJ output versus the new export's greater printed precision; **not** evidence of changed interpolation.

## Scientific conclusion

The simple hypothesis that **reversing source CSV drillhole record order alone switches the fold's reference topology/handedness is falsified for this one benchmark under these reimport conditions**. Deprioritize this form of ordering sensitivity. This observation does **not** prove the internal algorithm is order invariant: it may sort records deterministically by ID/XYZ, reuse cached state, or initialize the topology from a deterministic geometric prior.

The previously identified **mirror ambiguity of midpoint positions still exists mathematically**, and the native folded output remains highly asymmetric despite symmetric midpoint coordinates. This new experiment places more weight on fixed/deterministic tie-breaking, boundary conditions, and potentially additional native geometry input.

## Most discriminating next native experiment

In a **new clone** of the flat or curved benchmark, change wall **thickness only** by moving the top/bottom lithological contacts equal distances in opposite directions along each downhole trace while preserving each target interval's measured-depth midpoint. Also preserve hole IDs, collars, survey, model boundary, other settings, and reference resolution. Export new reference mesh. A changed reference despite strictly identical midpoint input coordinates demonstrates additional wall/segment dependence; an unchanged reference supports (but does not prove) a midpoint+boundary construction. Separately perform a **boundary-only** perturbation to distinguish clipping from actual fitting.

Do **not** claim the default native interpolation algorithm is recovered by this experiment. The `main` branch is unchanged.

## Reproducibility

An independent `compare_leapfrog_reference_order.py` tool parses OBJ vertex/face arrays, directly compares index order, and calculates the certified corresponding-surface distance bound. No owner-provided OBJ files are committed to this public repository.
