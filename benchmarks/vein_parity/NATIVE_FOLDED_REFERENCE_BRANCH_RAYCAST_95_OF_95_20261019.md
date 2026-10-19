# Folded reference orientation: ray-first geometric contact matching, 95/95 (2026-10-19)

**Status: independent numerical behavioral parity against the owner's five synthetic native Leapfrog examples; NO native internal projection algorithm recovered.** This document reports only aggregate findings. The owner's project, mesh vertices and source labels remain private. The implementation `reference_raycast_wall_polarity.py` and synthetic tests do not contain owner data.

## Direct 95/95 result

The first Euclidean-nearest reference triangle normal sign rule predicted 94/95 HW/FW polarity assignments. The one mismatch concerned a folded midpoint ~45.68m from the nearest surface branch, with a differently oriented candidate just 0.27m farther.

A simple, independently implemented alternative:
1. For each original **vertical** drillhole VEIN segment midpoint `p=(x,y,z)`, compute all intersections between the line `(x,y,*)` and the wound native reference triangles.
2. Choose the intersection closest in *measured vertical depth* to the segment midpoint (not nearest Euclidean surface point).
3. If the reference sheet **does not intersect the vertical line** at that x/y, use the 3D Euclidean-nearest triangle as a fallback.
4. Predict signed HW/FW vertical polarity from the **negative sign of the chosen wound face normal's z component**.

| Case | Matched / saved contact pairs | Vertical-ray hits |
|---|---:|---:|
| Flat baseline | 25/25 | 25 |
| Pinchout A | 5/5 | 5 |
| Curved variable thickness | 25/25 | 25 |
| Folded / multiple intersections | **15/15** | **13** |
| End of hole | 25/25 | 25 |
| **Total** | **95/95** | **93** |

On the folded model, the reference mesh has several overlapping triangle sheets under XY projection, so a vertical ray can have **up to five candidate intersections** at the same x/y location. Selecting a branch using vertical depth gives the correct reference polarity even where Euclidean 3D distance chooses the wrong sheet. Two folded control XY positions have no vertical reference intersection, and the Euclidean fallback classifies both saved contacts correctly.

**This is NOT a proof Leapfrog's actual native implementation uses the ray-first/nearest-depth/fallback rule.** The controls are all vertical drilling; a more general algorithm must use the actual desurveyed path, and native projection may be based on reference charts, conformal interpolation or structural correspondence instead. A model-generated HW/FW contact table can also be a consequence of reference construction, not a source of information used by the reference builder itself.

## What changed for RSGeo

We now have a tested, simple recipe for assigning HW/FW labels to geological contact pairs *given a previously constructed folded reference*: determine the corresponding local orientation branch in a drillhole-aware way, and use its oriented normal sign. For overturned geometry, never treat nearest triangle as the universal structural correspondence; multiple crossings and missing local reference coverage need explicit handling.

## Relationship to native binary audit

A separate private Ghidra 12.1.4 analysis of the owner's Leapfrog executable completed successfully (run **39324069518**) and located Python-binding method registrations for `PySurfaceOffsetInterpolant.init_from_mesh_and_points`, `init_from_mesh_and_rbf`, `conformal_distances_to_mesh`, and the refined-walls conformal distance evaluator. Another successful run (**39325510855**) recovered the complete offset-interpolant method registration table and its callable pointer slots. These private static results support investigating *reference-conformal evaluation and offset fitting*, **but do not establish the actual surface construction kernel or how native projected contact branches are chosen**.

Generic public code:
- `benchmarks/vein_parity/reference_raycast_wall_polarity.py`
- `benchmarks/vein_parity/test_reference_raycast_wall_polarity.py`

Private owner reproducibility bundle: `Leapfrog_Folded_Reference_Ghidra_and_Wall_Orientation_20261019.zip` with per-case 95-row CSVs and the owner-provided folded reference OBJ.

The public/private main branches remain untouched.
