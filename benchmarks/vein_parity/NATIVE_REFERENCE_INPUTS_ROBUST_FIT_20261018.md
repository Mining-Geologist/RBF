# Vein reference surface: exact saved inputs, robust-fit counterexample, affine gauge (2026-10-18)

**Result: partial scientific advance, NOT native-reference reconstruction.** All experiments below were run offline on the owner-supplied `Vein_Benchmark.zip` with independently authored NumPy/SciPy code. No Leapfrog software, native solver or Windows runner was used. Main branch unchanged.

## Direct finding from the actual Leapfrog project object graph

The saved `VeinReferenceSurfaceBlock.points` references a `MergedLocationsBlock`. Its retained metadata has `_hasNormals=False`, `_hasValues=False`, `_hasNormalSigns=False`. The `ResultStore.locations` field is an inline `lfdata-1.0` numerical payload and was decoded through a symbolic pickle reader (never imports vendor code). **These are the native midpoint inputs**, not an inferred HW/FW arithmetic mean.

| Case | Native midpoint inputs | Reference output Z range | Native reference-input distance to mesh |
|---|---|---|---|
| Flat | 25 at Z=100 | constant 101.78 | all 1.78 m |
| Pinchout A | 5 at Z=100 | constant 101.78 | all 1.78 m |
| Curved | 25, Z 85.36–102.04 | 82.542–105.525 | median 1.454 m, max 2.029 m |
| Multiple intersections | 15, including 6 XY sites with two Z values | −10–210 | median 8.497 m, max 45.679 m |
| End-of-hole | 26, including four extreme points at Z=1, 4, 196, 198 | 95.504–104.150 | median 1.253 m, four extremes 93.5–99.4 m away |

**Interpretation:** The native reference is **not an exact interpolating surface through the input midpoint positions**. The end-hole case is a particularly strong counterexample: four input points lie roughly 93–99 m from the saved reference mesh. The native process effectively suppresses or handles these outliers. Flat/curved point-to-native-ref residuals also rule out exact interpolation. The status of these points as accepted reference inputs is verified; the internal fitting mechanism remains unknown.

## Independent reference constructor tested

An explicitly non-native candidate:
1. Collapse a **single** multiply valued XY site to its mean Z; reject >=2 multisheet XY sites and route to a yet-to-be-developed full 3D manifold builder.
2. Fit a robust affine plane with soft-L1 residual, f_scale=2m.
3. Reject XY sites whose deviation from that plane exceeds 7m. These thresholds were chosen **without inspecting native reference geometry**.
4. Fit a thin-plate-spline reference heightfield on remaining sites. Select the smoothing parameter from .01/1/20/100/500/2000 by **midpoint leave-one-out error**, without reading the saved reference mesh.
5. Only after prediction, evaluate vertical error at native reference vertices.

The end-hole case has 25 distinct XY sites (one duplicate-site pair collapsed), and two rejected extreme sites; 23 stable sites remain. Native input has four extreme *observations*, two paired at one XY site. This restores the broad z≈100 reference trend without access to the native surface.

### Reference vertical RMSE, interior (metres)

| Model | Ordinary least-squares plane on ALL native midpoints | Robust plane | Robust LOOCV thin-plate spline |
|---|---:|---:|---:|
| Flat | 1.780 | 1.780 | 1.780 |
| Curved | 1.137 | 1.162 | 1.570 |
| End-hole | **7.460** | **1.603** | **1.496** |

For end-hole, the robust midpoint fit cuts error by ~80% versus the unfiltered least-squares plane, but absolute ~1.5m reference error is still unacceptable for high-precision wall modelling. The curved reference becomes **worse** under native-independent TPS fitting. The folded model is intentionally not flattened into a global XY heightfield.

### End-to-end conditional wall check using SAME first-order MQ1 radial kernel and affine drift

Native HW/FW contacts and native reversed pinchout controls are kept fixed; the only changed input is the reference estimate. Native wall vertices are used **only** to score errors.

| Model | Wall | Direct wall-Z fit | With saved native reference | With independent point-cloud reference |
|---|---|---:|---:|---:|
| Flat | HW/FW | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Pinchout A | HW/FW | 0.104 / 0.144 | 0.104 / 0.144 | 0.104 / 0.144 |
| Curved | HW/FW | 0.064 / 0.064 | **0.01859 / 0.01847** | **0.19960 / 0.19965** |
| End-hole | HW/FW | 4.011 / 4.442 | 3.752 / 4.237 | 4.010 / 4.443 |

All entries are vertical elevation RMSE in metres, restricted to interior native wall vertices. This is **negative evidence for replacing the saved reference with our current midpoint builder**. The curved surface shape has not been reconstructed from raw midpoints.

## Important exact mathematical property: affine gauge invariance

For a wall reconstruction `W(x,y)=R(x,y)+RBF_affine(z_i-R(x_i,y_i))`, with the interpolator reproducing all affine polynomials exactly, replacing `R` with `R + c + a*x + b*y` does not change any predicted wall elevations, because that added affine function cancels exactly in the fitted residual.

Independently tested with 25 random control sites and 120 held-out positions: the largest numerical difference after adding a **27.5m vertical shift and arbitrary X/Y slopes** was **7.11e−14 m**.

**Consequently, native flat reference at Z=101.78 vs midpoint cloud at Z=100 is NOT the key obstacle** for this particular affine-polynomial wall reconstruction. The decisive problem is reconstructing non-affine curvature and 3D surface topology. This is a property of *our* tested MQ1+affine model, **not proof that Leapfrog uses the same affine drift**.

## Limits and next target

- Native reference construction from point-only inputs is not recovered; it can generate a manifold with overhangs/branches and large deviations from certain input midpoints.
- The native saved input block offers no explicit structural normals/values. Any exact full 3D reference construction likely involves a point-cloud orientation/meshing algorithm or further model context not yet exposed.
- Next: examine the *reference-surface generation* code path (distinct from `PySurfaceOffsetInterpolant.init_from_mesh_and_points`) and develop a full implicit 3D point-cloud manifold algorithm. For reference validation, compare signed point-to-mesh distances, local normals, fold branch assignment and mesh intersection topology, rather than only height RMSE.
- The separately recovered native `asphere` contact-snap field remains a subsequent stage and is not a substitute for base/reference construction.

Code: `robust_midpoint_reference.py`, `test_robust_midpoint_reference.py`, `benchmark_robust_midpoint_reference.py`. Private saved project arrays are NOT published in this branch. The user-facing conversation ZIP contains owner-only reference input CSVs and native OBJ meshes to reproduce the numerical scores.
