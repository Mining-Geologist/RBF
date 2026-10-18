# Next offline reference-manifold investigation — 2026-10-18

**Status: experimental geometric reconstruction. Not recovered native Leapfrog code or end-to-end parity.** All input XYZ and native reference meshes came from the owner's previously supplied saved `Vein_Benchmark`. The actual reference OBJ meshes were used only for scoring candidate geometry after construction. No Leapfrog binary or live runtime executed; `main` remains unchanged.

## A new numerical clue: broad attitude is identifiable, local curvature is strongly attenuated

The original curved reference midpoint cloud has a principal-component normal roughly `(-.060,-.035,-.998)`. The saved Leapfrog curved reference face normals have area-weighted mean absolute dot product **~0.9996** with that direction. Native global reference attitude is consistent with the midpoint cloud's best-fit plane, but its geometry departs from a direct through-midpoint interpolation.

For example, fitting quadratic surfaces over the midpoint array yields second-order scaled coefficients X²≈-7, Y²≈-4, while fitting the saved surface itself gives about X²≈-1.39, Y²≈-1.28. This is **qualitatively consistent with strong curvature smoothing**, but not proof of the native process.

## Original prototype tested

Two different surface classes are needed:

1. **Non-overturned single-chart vein**: robust Huber-reweighted quadratic heightfield with affine drift and ridge penalty on quadratic terms. The penalty **λ=3** was selected after inspecting the *saved curved reference*, therefore all curved gains are **in-sample calibration**, NOT blind validation. A single duplicate XY point is collapsed; multiple repeated XY sites with separate Z are rejected as folded, rather than spuriously interpreted as a heightfield.
2. **Synthetic folded multi-intersection**: a genuinely connected **3D parametrized ribbon** made from three Y cross sections, each with a five-point [nose, upper inner, upper outer, lower outer, lower inner] ordering. Shape-preserving cubic interpolation in XZ, continued in Y, supports multiple elevations at the same XY. This is specific to the given symmetric synthetic U-fold and is not a general-purpose native fold reconstructor.

## Measured errors

| Case | Previous baseline | Experimental reference | Comparison region/metric |
|---|---:|---:|---|
| Curved | 1.137m best simple affine plane or 1.570m robust midpoint TPS | **0.416m** | 153 native reference vertices, interior vertical RMSE |
| Curved | 2.476m robust TPS | **0.712m** | 293 reference vertices, full vertical RMSE |
| End-of-hole | **1.496m** robust midpoint TPS | 1.600m (worse) | 85 reference vertices, interior vertical RMSE |
| Flat/pinchout | 1.780m datum difference | 1.780m datum difference | Flat input midpoint at Z100 vs saved flat reference at Z101.78 |
| Folded multi-hit | 18.025m horizontal Z100 plane | **9.337m** connected 3D ribbon | 127 saved reference vertices in a *restricted central* X/Y/Z region; nearest 3D distance to Y-extruded section profile, NOT full mesh parity |

For the curved shape, ridge λ≈2.5–3 minimizes native *reference* RMSE. However, within the older independent MQ1 wall approximation λ≈10 can give smaller predicted **wall** elevation error, despite worse reference surface RMSE. This demonstrates why fitting walls alone cannot uniquely recover native reference construction, and why point-cloud-only geometry and constrained wall interpolation must be checked separately.

For end-hole, four extreme native reference-input values (Z=1,4,196,198m) demand robust handling; our new model still has ~1.6m error. For the folded reference, a 3D connected ribbon improves over a wrong horizontal sheet, but remaining >9m disagreement means branch geometry is still far from parity. Reference construction is still **unsolved**.

## Reproduction

The actual tested standalone scripts, native reference meshes, decoded native reference-input midpoint CSVs, five predicted experimental OBJ surfaces, numerical scores and diagnostic cross-section plots are bundled in the conversation artifact **`Leapfrog_Vein_Reference_Manifold_Research_20261018.zip`**. Code dependencies: NumPy, SciPy, trimesh, matplotlib. No vendor runtime or licence is required. Since the archive includes owner-provided native mesh data, **do not commit it to a public repository**.

Research path: `benchmarks/vein_parity/`. Existing source modules `robust_midpoint_reference.py`, `reference_relative_wall_model.py`, `vein_segment_orientation.py` remain independent and unchanged.

## Most valuable next native target

The project's stored reference manager identifies **`engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock`** and its input `MergedLocationsBlock`, without normals or scalar values. This is a **distinct process** from the native `PySurfaceOffsetInterpolant.init_from_mesh_and_points` API already traced in Ghidra; the latter concerns reference-*relative wall offsets*, not necessarily reference surface creation.

Next look for the reference-manifold generator invoked by `VeinReferenceSurfaceBlock` in the permitted owner-saved static runtime. Source filenames or native wrapper registration strings alone do not establish the algorithm. Independently implement topology-aware implicit surfacing and validate new geometries before claiming native parity.
