# Real multi-vein system — custom reference, wall RBF validation and blocked contact tests (2026-10-18)

**Research-only result, not vendor kernel recovery.** Offline, read-only analysis of user-supplied private Leapfrog Geo real multi-vein project. Proprietary project archive, coordinates, reference/wall meshes, proprietary serialized objects and individual control records are not published. No native vendor code executed; serialized objects inspected as data without unpickling. This report intentionally uses anonymous child identifiers.

## Exact native custom-reference reuse
Four different real child veins refer to the same first vein's final hangingwall as a custom reference. Independently decoded each child reference mesh: all four have **43,784 vertices and 85,839 triangles** with bit-identical vertices and face-index order to that first final HW mesh. The reference is strongly inclined and curved; it is not a global XY elevation heightfield. This confirms an explicit production geometry dependency, not only a saved graph pointer.

## Real native raw-wall comparison
Construct a global best-fit reference-aligned PCA chart `(u,v,w)`. For each original **true logged** HW/FW contact, intersect a ray along the PCA normal with the saved reference mesh. Fit two separate, exact-interpolating **linear 2D degree-0 RBF** models to (a) absolute contact `w`, and (b) the signed offset `w - w_reference(u,v)`. Independent prediction targets are cross-sections of each saved **raw/pre-pinchout** native vein mesh at 1,100 fixed random chart sites, retaining only sites with one reference and exactly two raw-wall intersections.

| Anonymous real child | Wall | Valid native raw section sites | Absolute model RMSE (m) | Reference-relative model RMSE (m) |
|---|---|---:|---:|---:|
| A (41 contacts/wall) | HW | 1,004 | 10.589 | **3.613** |
| A | FW | 1,004 | 9.233 | **1.853** |
| B (25 contacts/wall) | HW | 889 | 8.362 | **5.215** |
| B | FW | 889 | 7.387 | **3.095** |
| C (3 contacts/wall) | HW | 1,100 | 4.381 | **0.863** |
| C | FW | 1,100 | 5.887 | **0.357** |

A fourth real child has only two original controls/wall and is not suitable for identifying a 2D RBF kernel using this comparison. Two of the main child cases show nearly tied relative linear vs multiquadric candidates: this **does not distinguish their native kernels**. Third child has too few controls for reliable kernel-family discrimination.

For A/B/C, logged HW and FW controls project to the **same side of the inherited reference** in this chart (except one contact lying at the reference itself). The custom reference is clearly *not required* to lie halfway between the child HW and FW.

## Independent geological-control validation

We also withheld ~20% of true original HW/FW contacts in **contiguous along-strike blocks** and refit candidate 2D RBFs to the remaining 80%. Prediction target is held-out real geological contact, not a fitted native surface:

| Original control subset | Absolute linear d0 RMSE | Reference-relative linear d0 RMSE |
|---|---:|---:|
| A HW (41) | 23.32m | 23.36m |
| A FW (41) | 30.03m | **26.86m** |
| B HW (25) | 23.95m | **16.22m** |
| B FW (25) | 22.40m | **15.53m** |

Reference-relative improves 3/4 held-out subsets; one is nearly tied. Metre-scale raw-mesh residuals and tens-of-metres blocked extrapolation error remain, so this **is not an accurate Leapfrog-equivalent solver**.

An exploratory 2x scaling of the chart's short axis improves relative blocked-CV scores to A HW/FW ~22.02/25.44m and B HW/FW ~15.30/14.38m. These are *post hoc* anisotropy experiments, not proof of native anisotropy parameters.

## Implications for RSGeo and next steps
- Independent real-project evidence reinforces **reference-relative wall interpolation**, including inherited curved references and contacts entirely on one side of the reference.
- A fixed PCA-aligned global chart likely misses native **spatially varying reference normals/tangents**, structural directions, anisotropic distances, and local surface parametrization. Don't generalize linear-vs-MQ comparisons from the synthetic flat cases.
- Next research priority: build **reference-following local coordinates/normal offsets** and compare against the same frozen real native meshes and held-out contacts. Separately study termination/boolean and automatic folded-reference generation.
- The owner's private conversation bundle includes read-only reproducible Python scripts, per-model numerical tables and full error diagnostics; only aggregate data are disclosed here.
- The public `main` branch is unchanged.
