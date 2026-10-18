# Case F curved-desurvey validation and case G kernel preregistration — 2026-10-18

**Source:** Owner's updated `Vein_Benchmark(4).zip` project (all data remains private). We used read-only SQLite object inspection, pickletools opcode scanning, NumPy numeric decoding and independently implemented minimum-curvature geometry, followed by ray/triangle intersections and continuous curve/triangle plane root refinement. **No vendor code was executed and no owner mesh data has been committed.**

## Held-out F results

Three previously inclined WASTE-only holes were replaced with three-station surveys (MD0, MD80, MD200) changing both azimuth and dip, holding the five true VEIN drillholes and all input midpoint geometry fixed. The previously **pre-registered** predictions using 0.1m MD curved-path linearization already matched the new native F output with maximum discrepancy **2.07732213653e−7m**. Native artificial depth values are:

| Hole | Native from–to measured depths, m |
|---|---|
| VH_X-080_Y+000 | 104.95508183468030–116.68397453594883 |
| VH_X+080_Y+000 | 103.12568433034562–114.67549876631472 |
| VH_X+000_Y-080 | 100.19307299175888–110.33188355032966 |

We subsequently eliminated linearization error by evaluating the exact continuous minimum-curvature spherical path between survey stations and root-solving signed distances to the original indexed raw-vein triangles with Brent's method. **All 40 native artificial MD endpoints from all 20 eligible F WASTE holes matched**:
- Maximum MD discrepancy **1.5489831639570184e−12m**;
- MD RMSE **2.868868540945067e−13m**;
- Maximum native saved 3D intersection-coordinate error **1.4073242413538485e−12m**.

This is a strong behavioral reconstruction of **minimum-curvature desurvey -> preliminary vein intersection -> artificial reversed pinchout constraints** for the tested simple, single-crossing case, including varying azimuth/dip. It does not establish the specific internal vendor implementation or how intersections are paired if >2.

## Upstream/downstream dependency isolation

D, E and F save exactly the same indexed **raw/pre-pinchout vein mesh**, 1792 vertices and 3580 triangles; in F the raw mesh coordinates and connectivity were byte-for-byte equal in NumPy arrays to D. The reference surface's inline mesh SHA256 is **b0a37bb46aec9529ece7c3a26ed576f8ec65288b245e00f383e6366b2b319a73**, unchanged from A to F, with surface resolution 20m. The vein surface resolution is 10m, snap fraction 1.0.

Yet downstream outputs vary when only the WASTE-hole surveys change:

| Geometry | D | E | F |
|---|---|---|---|
| Raw vein | 1792 / 3580 | identical | identical |
| Refined vein vertices/faces | 400 / 796 | 368 / 724 | 365 / 722 |
| HW vertices/faces | 1070 / 1960 | 1070 / 1960 | 1070 / 1960 |
| FW vertices/faces | 1090 / 1996 | 1099 / 2014 | 1097 / 2010 |

The HW mesh **also differs in vertex values and connectivity** despite equal counts. This isolates *upstream preliminary wall interpolation* from *downstream pinchout constraints and final wall/solid refinement*.

## Scope relative to original vein reverse-engineering

**Directly useful:** desurvey parity, exact pseudo-interval placement on drillholes, reversed wall contact generation, stage boundaries, test fixtures to estimate wall interpolation without confusing intersection errors.

**Not yet solved:** default folded reference topology/handedness (separately observed Y-mirror midpoint ambiguity), complete wall interpolation kernel/stiffness/trend/anisotropy, native wall snapping and final boolean/solid generation, multi-hit raw-volume intersection pairing, and parameter transfer to overturned references. Native D uniform-to-nonuniform experiment gave ~0.0077m RMSE to an **untuned linear 2D degree-0 RBF** surrogate against 20 generated widths. This suggests a kernel family but is not proof of the native kernel; F raw mesh is invariant to D and thus provides *no new independent wall-kernel training case*.

## Preregistered G: test the kernel itself rather than pinchout mechanics

We independently constructed an owner-facing CSV bundle `Leapfrog_Pinchout_G_Kernel_Discrimination_Test_20261018.zip` (not committed) with 33 collars, 66 survey rows and 41 logged geology rows. The five true target intervals retain midpoints MD100 but have irregular thicknesses:
- (-40,-40) 8m, (40,-40) 13m, (0,0) 18m, (-40,40) 7m, (40,40) 10m.

Eight additional near-field fully logged WASTE holes at XY (4,0),(12,0),(20,0),(0,4),(0,12),(-12,0),(8,8),(16,16) are included. The original three unlogged collars remain unlogged; 'pinchout on empty traces' must stay ON. No overlapping logs. Candidate frozen predictions attached to user-facing ZIP: 2D linear degree-0/degree-1, multiquadric epsilon=1/0.1 degree-0, thin plate spline degree-1, all exact-fit and **not tuned to G output**.

At new WASTE point (12,0), example candidate predicted raw thickness values: linear degree-0 **16.442m**, MQ epsilon1 **16.573m**, MQ epsilon0.1 **17.385m**, TPS **17.663m**. Actual native G output is not yet available; these are predictions, NOT experimental findings.

**Native G run:** one new model with original D settings (20m ref resolution, 10m vein resolution, 100% snapping, pinchout on with empty traces). **Optional parameter sensitivity:** in a separate cloned project change ONLY the vein surface resolution from 10m to 5m. Export complete project states and compare raw geometry to eliminate meshing artifacts before selecting a candidate interpolation family.

**Main branch untouched.**