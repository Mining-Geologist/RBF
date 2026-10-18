# Native test G — held-out wall-kernel discrimination and mesh-geometry isolation (2026-10-18)

**Status:** User completed preregistered G run in Leapfrog Geo and provided saved project `Vein_Benchmark(5).zip`. We inspected it offline read-only (SQLite, pickletools, numeric LFData / Parquet decode). Proprietary project, native raw mesh, and per-drillhole coordinates are **not committed to this public repository**; full reproducible package returned privately to the owner.

## Saved native control state

G `VeinReferenceSurfaceBlock` 13960: resolution **20 m**. G `VeinBlock` 13966: resolution **10 m**, contact snap fraction **1.0**. Pinchout on empty traces: **true**. Five actual target observations, 28 synthetic pinchout holes including 8 new near the vein centres.

Native reference mesh payload is **identical** to previous D–F reference mesh (SHA-256 `b0a37bb46aec9529ece7c3a26ed576f8ec65288b245e00f383e6366b2b319a73`), even when the five real thicknesses change nonuniformly around identical input midpoints. Native raw/pre-pinchout G vein is *different* from D: **1802 vertices, 3600 triangles** (D: 1792/3580).

G native `segments` field uses Parquet `PAR1` for its 56x3 floating point contact points (in contrast with D's inline numpy bytes), so decoder must support both. Native saved `collar_ids` and `depths` decode 28 artificial pinchout intervals. Independent XY-ray/triangle intersections into the separately saved G raw mesh reproduce **all 56 endpoints** to max absolute **2.842170943e-14 m**. Pinchout widths range **5.833531–17.485909 m**, despite original wall midpoint MD100 constant.

## Blind/preregistered model ranking (28 saved pseudo-interval widths, m RMSE)

| Frozen 2D thickness RBF candidate | All 28 | Existing 20 | Newly added 8 |
|---|---:|---:|---:|
| **linear r kernel, constant polynomial (degree=0), zero smoothing** | **0.049899** | **0.008539** | **0.092370** |
| MQ ε=1, degree 0 | 0.114304 | 0.061711 | 0.190285 |
| MQ ε=0.1, degree 0 | 0.706976 | 0.639640 | 0.852353 |
| linear r, degree 1 | 1.221255 | 1.443861 | 0.091025 |
| thin-plate spline, degree 1 | 2.851692 | 3.290475 | 1.180872 |

This is a **genuinely held-out** trial of the five candidates fixed in the G input ZIP before the user ran Leapfrog. 2D linear-degree-0 remains the clear overall winner. But the 49.9-mm error close to the five controls is not exact native interpolation parity. Worst added near-control location (+0,+12) had ~0.193319 m direct analytic model thickness difference.

## Critical new result: separating kernel mismatch from triangle meshing

**Post-hoc numerical diagnostic, NOT independent native mesher parity.** Take the exact same frozen linear RBF and evaluate its 2D width `t(x,y)` at the XY locations of **native saved G raw-mesh vertices**, preserve the native face/triangle list, set top/bottom predicted elevations `z=100±t/2` from each vertex's native wall polarity, and intersect 28 rays with this **re-triangulated RBF model**.

| Same RBF, different evaluation geometry | All 28 RMSE | Existing 20 | Added 8 |
|---|---:|---:|---:|
| Evaluate analytical interpolator at drillhole XY | 0.049899m | 0.008539m | 0.092370m |
| **Evaluate on native vertices and native triangle topology, then intersect** | **0.009226m** | **0.006065m** | **0.014351m** |

At (+0,+12), discrepancy falls **+193.32 mm -> -23.23 mm**. This is major supporting evidence that native polygonal discretization, and not just the kernel field, contributes to observed pinchout-width differences. Since this test conditions on native mesh geometry, it is *not* a blind prediction of mesh vertex placement or overall geometric parity.

Direct native vertex audit further gives ~0.0192m width RMSE for a frozen linear-degree-0 model against raw-mesh vertex Z-derived thicknesses (excluding the near-midline sidewall), with 0.00242m RMSE in outer x/y>=80m regions and larger deviations near controls (up to 0.216m). Remaining difference may reflect local discretization or native control handling/snap; do not claim exact code/kernel.

## Conclusions for the general reconstruction

- **Confirmed numerically on G**: RAW vein geometry upstream, pinchout pseudo-intervals from independent desurveyed ray/mesh intersections, identical reference despite nonuniform thickness, separate downstream pinchout refinement.
- **Strong support** for **linear 2D degree-zero radial** as *one component* of preliminary wall interpolation for these flat models, not a recovered vendor kernel.
- **Unsolved**: generation/orientation of folded references, exact native interpolation kernel and constraints, raw wall mesher, snapping, parameter dependence, multi-branch pairing and final solid operations.
- **Next decisive parameter control**: user should clone G and change **only vein surface resolution 10m -> 5m** while preserving reference resolution 20m, snap 100%, pinchout on/empty on and all inputs. No new drillhole set required. Compare the new native 28 computed intervals and raw mesh with both pre-registered predictions and a separately constructed finer mesh. This will reveal whether residual errors shrink with resolution, rather than guessing by kernel tuning.
- Later: G with snap=50% only, holding resolution fixed, to trace which stage the snapping parameter affects.

Research branch only; public `main` untouched.
