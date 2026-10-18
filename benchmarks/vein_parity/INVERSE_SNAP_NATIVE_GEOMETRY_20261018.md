# Inverting the native vein HW/FW snap fields: quantitative recovery (2026-10-18)

Scope: read-only, offline reconstruction from the user's saved Leapfrog Geo 2026.1 project. No running Leapfrog software, no licensing/network interaction, no proprietary source or binary checked into this public repository.

## What is established

The saved **three non-planar vein cases** contain two native FastRBF `Solution3` records (HW and FW): basis `asphere`, range parameter 10m, degree -1, RBF coefficients and 3D centres. Their evaluation is independently reproduced by the same piecewise `asphere` formula used in `replay_saved_asphere.py`.

**New inverse-snap numerical test**: for each saved final HW/FW mesh vertex `v`, evaluate saved native RBF `f(v)`, take normalized final vertex normal `n(v)`, and estimate pre-snap position `v_original = v + f(v)n(v)` for HW, or `v_original = v - f(v)n(v)` for FW (the native wall triangle normals have opposite side orientation conventions). This is a first-order inversion; original pre-snap normals are not preserved.

For each case, compare the 3D Euclidean point-to-triangle distance from the native **saved RBF centres** to (i) final mesh, (ii) inverse-corrected mesh, and (iii) deliberately wrong-polarity control mesh. Those centres were not fitted by our inverse algorithm and are part of the user-owned saved native model.

| Geometry | Wall | Final mesh distance RMSE m | Correct inverse RMSE m | Opposite direction RMSE m | Recovered distance median m |
|---|---|---:|---:|---:|---:|
| Curved | HW | 0.097174 | **0.001088** | 0.193251 | 0.000357 |
| Curved | FW | 0.096724 | **0.001094** | 0.192344 | 0.000481 |
| End-of-hole | HW | 1.555021 | **0.382071** | 2.613496 | 0.01913 |
| End-of-hole | FW | 1.415054 | **0.642781** | 2.609972 | 0.02771 |
| Multiple intersections | HW | 2.211531 | **0.796940** | 3.682642 | 0.161455 |
| Multiple intersections | FW | 2.218623 | **0.950628** | 3.557609 | 0.191453 |

The curved model's RBF centres lie within about **1 mm RMSE of the inverse-corrected mesh**; reversed signs worsen the fit by ~2x. This is compelling direct geometric evidence that the saved `asphere` field encodes **contact-to-surface normal displacements / snap correction**. End-hole and folded cases improve but are not reproduced exactly using final-face vertex normals; local reference geometry and nonlinear/large displacement may matter.

`Hangingwall_Display` and `Hangingwall_Interpolated` have different triangulation sizes but **identical geometric support** within numeric precision for all six test scenarios, as do the two FW meshes. They are not separate unsnapped/snap reference surfaces.

## Experimental reconstruction of initial wall-offset field

Using native saved RBF centres as *proxy base-wall points* (not ordinary drilling inputs), project their coordinates and recovered pre-snap wall vertices onto the saved reference mesh; assign signed normal distances to each. Fit an independent scalar reference-offset RBF using projected 3D reference coordinates. This is a research diagnostic and does not recover exact native `init_from_mesh_and_points` implementation.

A sweep of linear distance, first-order multiquadric, compact spherical, saved asphere and exponential kernels with range 1/10/50/150/300/650/1000m, degree 0 and 1, found:
- Curved model: best mean HW/FW signed-offset RMSE around **0.050 m** using MQ1 with ~10m scale and degree 0; conventional spherical 300m degree 0 about 0.055m.
- Multiple intersections: about **2.55m** best mean signed-offset RMSE among candidates (exponential R1000, degree 0).
- End-of-hole: about **15.26m** best mean signed-offset RMSE among candidates (MQ1 R50, degree 1). The large extreme deviations indicate reference projection / branch matching / conformity needs reconstruction, and should not be presented as satisfactory models.

The experiment fits *saved intermediate RBF centres* and validates against the first-order inverse mesh; it is **not a blinded out-of-sample test of the complete vein algorithm**. The native original reference-surface construction, conformal projection and initial HW/FW surface interpolation remain unverified.

## Implementation / CI

- Independently authored inverse operation: `benchmarks/vein_parity/invert_native_asphere_snap.py`.
- Synthetic polarity/shape tests: `benchmarks/vein_parity/test_invert_native_asphere_snap.py`.
- CI: `.github/workflows/offline-inverse-vein-snap.yml`.
- User-facing reproduction kit has original Python evaluator against owner-provided NPZ inputs, six recovered pre-snap OBJ meshes and numerical tabulation. User benchmark raw arrays are not committed to public GitHub.
- New private static Ghidra audit focuses on `PySurfaceOffsetInterpolant.init_from_mesh_and_points`: https://github.com/Mining-Geologist/LF_SC/actions/workflows/ghidra-vein-offset-survey.yml.

## Limits of confidence

1. Snap field identity `asphere`, parameters and geometric displacement **strongly confirmed for these three saved native `Solution3` cases**.
2. Mesh-normal inverse is an approximation, exact only for sufficiently small displacement over smooth geometry.
3. `init_from_mesh_and_points` is a separate native operation; Ghidra method labels alone are not mathematical equations.
4. Flat and pinchout models do not persist these snap `Solution3` records, so none of these tests identifies the first-stage pinchout wall interpolation kernel.
5. Before commercial implementation, validate mesh interactions, constraints and snapping options with distinct datasets, and respect software licence/interoperability conditions.

## Independent contact-to-wall cross-check

On the curved saved model, the **original observed HW/FW contact points** are on the final native wall meshes (both median and 95th percentile distance = 0m within numerical tolerance). After reversing the exact native RBF field, their median distance from the recovered pre-snap meshes becomes **0.06824m HW** / **0.08079m FW**, with 95th percentiles **0.16689m** / **0.17329m**. This is the expected direction: the native saved RBF moves the original base wall towards contact observations.

On the end-of-hole and multiple-intersection examples, some contacts lie far outside the represented final wall branch (end-of-hole p95 9–10m), so a 100% snap control does not imply every interpreted 3D point is on the exported final mesh. The high-percentile outliers further limit using their whole input sets as direct kernel-calibration targets.

No RBF fitting was performed for this inversion: all coefficients and scales were frozen from the actual saved native solutions.

## Second-stage improvement: solving the inverse deformation self-consistently

The first-order inverse evaluates final-face normals and the RBF field at final mesh vertices. But the snap mechanism likely evaluates these quantities on the **original, pre-snap wall**, particularly problematic for large deformation and folded geometry.

A new **damped fixed-point iteration** solves the proposed equation `x_base = x_final + polarity*n(x_base)*f(x_base)` using native saved `asphere` coefficients, updated surface normals and updated evaluation coordinates. Each iteration is selected solely by its **internal fixed-point residual**, not by proximity to the saved native centres. The inferred polarity is + for HW and − for FW, matching the project mesh winding. Damping 0.65, up to 24 iterations.

| Case | HW first-order RMSE m | HW iterative RMSE m | FW first-order RMSE m | FW iterative RMSE m |
|---|---:|---:|---:|---:|
| Curved | 0.001088 | **0.000021** | 0.001094 | **0.000022** |
| End of hole | 0.382071 | **0.080131** | 0.642781 | **0.099127** |
| Multiple intersections | 0.796940 | **0.242117** | 0.950628 | **0.565933** |

RMSE is the nearest point-on-triangulation distance from **saved native RBF centres to the inverse-recovered wall**. This is a *model-internal geometric consistency check*, not an independently exported original wall mesh. It is strong support for the hypothesized normal-displacement correction, but it cannot establish exact native topology/constraints or reconstruct the initial reference/wall surface.

- New independent code: `benchmarks/vein_parity/invert_asphere_snap_iterative.py`.
- Synthetic known-forward-deformation test: `test_invert_asphere_snap_iterative.py`.
- GitHub numerical workflow: `.github/workflows/offline-iterative-vein-snap.yml`.

Complex folds can converge only partially or oscillate because triangle normals and correspondence change with deformation; the residual-based stopping rule mitigates but cannot remove these effects.
