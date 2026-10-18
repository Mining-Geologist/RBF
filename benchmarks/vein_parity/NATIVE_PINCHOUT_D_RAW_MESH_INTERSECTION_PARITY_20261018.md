# D non-uniform vein test: native raw-vein intersection reproduces every artificial pinchout — 2026-10-18

**Status: strongest confirmed mechanism yet for the flat, vertical, single-crossing benchmark; not a recovered native implementation or a general multiple-crossing solution.** Owner-provided `Vein_Benchmark(2).zip` saved project was inspected offline. No Leapfrog executable or licensed module was executed; native meshes and data remain private and are not committed.

## Proven numerical placement for A/B/C/D

We decoded the saved `FastRBF.objects.Mesh3` raw/pre-pinchout vein volume from the project's LFData, reading its Python pickle bytecode **without unpickling** and the embedded Parquet arrays using Apache Thrift metadata and system Zstd decompression (Parquet BYTE_STREAM_SPLIT for float64 coordinates, PLAIN for int32 face indices).

For each of the 20 logged/non-logged pinchout-eligible vertical drillholes, we performed an independent barycentric XY point-in-triangle computation for its ray, collected the two raw vein outer boundary Z-intersections, and converted Z to MD with a 200m collar datum.

| Case | Actual native generated pseudo-intervals | Hole count | Maximum endpoint absolute error, independently intersecting raw mesh |
|---|---|---:|---:|
| A | MD96–104 everywhere | 20 | 2.84e-14m |
| B | MD94–106 everywhere | 20 | 2.84e-14m |
| C | MD96–104 everywhere | 20 | 2.84e-14m |
| D | Nonuniform MD, width 7.1020–10.3377m | 20 | 2.84e-14m |

**Total verified: 80/80 two-ended native artificial pinchout intervals match the pre-pinchout raw vein boundaries to floating-point precision.** All artificial HW point XYZ locations are the **lower** raw vein trace intersections and all artificial FW point XYZ locations are the **upper** ones: reversed relative to normal HW/FW contacts, exactly as expected for the native synthetic constraints. The *final* interpolated HW/FW surfaces are different from these initial raw intersections.

This resolves the placement question **for the four flat, vertically drilled, single-crossing benchmark cases**: raw vein triangle intersection -> synthetic MD interval -> reversed HW/FW constraints -> refined walls and final vein. It does NOT prove which implementation Leapfrog uses internally (mesh-intersection versus algebraically equivalent surface intersection), how it handles curved or overturned surfaces, or branch selection for multiple crossings.

## D: directly observed spatial variability

D changes only the center observed VEIN interval from MD96–104 to MD92–108 (the other four stay MD96–104), keeping all five reference-input midpoint XYZ values identical. The native reference **mesh payload** is also bit-identical across A/B/C/D. Among 20 generated pinchouts:
- minimum virtual interval width: **7.102003465313416m** (corners X±80,Y±80);
- maximum virtual interval width: **10.337676822839455m** (near X±40,Y0);
- maximum absolute midpoint depth shift from MD100: **0.003701768715985m**.
This observed width field includes extrapolation *below 8m*, even though all original thickness observations are >=8m, consistent with oscillatory linear RBF-type interpolation.

## A versus corrected C: exactly equal final geometry

The updated C project now stores `snap_fraction=1.0` (A/B/D also 1.0). After independently decoding its native raw/refined vein meshes and final HW/FW mesh LFData, **every vertex coordinate and every indexed triangle is exactly equal** between A and C:

| Geometry | Vertices | Triangles | A vs C |
|---|---:|---:|---|
| Raw vein | 1784 | 3564 | exact geometry equality |
| Refined vein | 384 | 764 | exact geometry equality |
| Final HW | 1066 | 1952 | exact geometry equality |
| Final FW | 1094 | 2004 | exact geometry equality |

A/C serialized raw/refined result files have different checksums due to persistence metadata, so these are **decoded geometric comparisons**, not naive file-hash assertions. WASTE subdivision therefore has zero observed geometric effect on all four products in this controlled case, not just on virtual interval depths.

## Independent RBF interpolation clue: linear kernel unexpectedly close

Fit a 2D **linear RBF** with a constant background (degree=0, zero smoothing; no parameter tuning to D outputs) to the five TRUE target thicknesses at X/Y `(-40,-40),(40,-40),(0,0),(-40,40),(40,40)`, values 8/8/16/8/8m. At the 20 pinchout collar XY locations, native calculated interval widths match this kernel with RMSE **0.007695515m**, max absolute error **0.0179774m**.

A separate test against the saved raw-vein surface triangles confirms top elevation fits of **0.01079m RMSE** (1,280 nearly horizontal triangle centroids) and bottom fits **0.00964m RMSE** (1,312). On these faces median absolute discrepancies are each <0.001m; differences are consistent with mesh discretization and/or differing native implementation details. For comparison, fixed MQ1 epsilon=1 predicts virtual widths with ~0.05436m RMSE, 2D thin-plate spline ~2.775m. This is **strong surrogate evidence for a first-order linear radial basis in raw wall construction**, **NOT proof of the proprietary kernel** or independently validated parity in new geometries.

## Next unresolved experiments

1. Nonvertical, inclined holes through the same preliminarily generated raw vein: validate ray-triangle trace intersections along desurveyed drillhole polylines and MD interpolation.
2. Multi-branch raw vein crossings: test which crossings are paired and excluded; current four cases each cross exactly twice.
3. A curved reference case: decide whether generated artificial intervals follow raw volume intersections with multiple crossing outcomes.
4. Native wall kernel controlled **held-out** test using a fresh nonuniform geometry to separate linear RBF from MQ1 and triangulation artifacts. Avoid overfitting to five centers.

Local owner-facing reproducible bundle: `Leapfrog_Pinchout_D_Native_Intersection_Results_20261018.zip`, including complete table of 80 segments, independently decoded cross-section predictions, a 20-hole spatial plot, and pure Python scripts for LFData mesh decoding and triangle intersection. This archive is not pushed to the public research branch. The public `main` branch is unchanged.
