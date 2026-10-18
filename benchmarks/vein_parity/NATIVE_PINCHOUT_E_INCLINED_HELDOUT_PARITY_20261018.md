# Native case E inclined drillholes: genuinely held-out raw-vein intersection parity (2026-10-18)

**Status:** user ran independently prepared case E in Leapfrog Geo; new native saved project `Vein_Benchmark(3).zip` was decoded offline using static pickle opcodes / numeric LFData processing. No vendor binaries executed; no owner mesh data committed.

## Pre-registered predictions versus new native output

The case-E CSVs and `preregistered_raw_mesh_predictions.json` were prepared **before** the user ran Leapfrog. Only three WASTE drillholes had nonvertical (constant) survey orientations; the original five target VEIN drillholes and input geology were preserved.

| Hole | Azimuth/dip | Pre-registered MD from D raw-vein mesh | Native E synthetic interval MD |
|---|---|---|---|
| VH_X-080_Y+000 | 90° / 60° | 108.55986460315783–122.93599994106442 | 108.55986460315785–122.93599994106445 |
| VH_X+080_Y+000 | 270° / 70° | 101.17509043666486–111.82501387697808 | 101.17509043666486–111.82501387697808 |
| VH_X+000_Y-080 | 0° / 75° | 98.79533396047422–108.33277166832629 | 98.79533396047420–108.33277166832633 |

Maximum pre-registered endpoint error **4.263256414560601e−14m (6 endpoints)**. No fitted calibration to native case E.

## End-to-end independent verification

Static-decoded new E raw vein mesh: **1,792 vertices, 3,580 faces**. It is **exactly vertex-, face- and triangle-ID-identical** to D's saved raw mesh. Using all 20 collar coordinates and independently computed straight-hole direction vectors, ray/triangle intersections reproduce **all 40 E native artificial endpoint MDs**: max absolute error **4.263256414560601e−14m**, endpoint RMSE **1.852868764786606e−14m**.

The saved reference mesh payload SHA-256 remains `b0a37bb46aec9529ece7c3a26ed576f8ec65288b245e00f383e6366b2b319a73`, identical in A through E. The original five native reference input midpoint XYZ coordinates are also unchanged.

By contrast, the refined solid changes from D 400 vertices / 796 faces to E 368 vertices / 724 faces. Both final HW/FW meshes also change. This is a clean confirmation that shifting artificial constraints along an inclined hole affects *downstream refinement*, not raw/reference construction.

**Conclusion:** In these simple, single-crossing flat models, the synthetic pinchout MD endpoints are determined by the intersections of the desurveyed drillhole trace with the preliminary/raw vein boundaries. It is not snapping WASTE sample intervals to the reference or placing arbitrary fixed-width intervals. The precise vendor implementation could use mathematically equivalent wall intersections instead of triangles; these observations don't prove internal code.

## Limits and next information gain

Case E tests **three straight inclined traces only**; dogleg/curved desurvey integration, intersection pairing with more than two hits, pinchout eligibility with partially logged data, and native raw-wall interpolation family remain open. Strong next test: changed survey orientations at intermediate MD with non-collinear desurvey; more decisive next topology test: one drillhole intersecting disconnected raw-vein lobes / repeated folds with more than two surface crossings.

A portable, read-only verification script and 20-hole result table were bundled in the user-facing conversation ZIP `Leapfrog_Pinchout_E_Validation_20261018.zip`. User-supplied Leapfrog data remains private and out of this repository; `main` unchanged.
