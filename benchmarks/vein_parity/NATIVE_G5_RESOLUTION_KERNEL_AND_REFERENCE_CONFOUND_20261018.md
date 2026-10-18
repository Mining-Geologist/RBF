# Native G5 wall-kernel and resolution audit (2026-10-18)

**Status:** User supplied updated Leapfrog Geo project with the original G10 and a newly created G5 model. Read-only static inspection of SQLite, serialized native numeric outputs and LFData geometry. The user's source project and proprietary saved meshes have not been committed. Vendor code was not executed.

## Critical saved-setting difference

| Parameter | Original G10 | New G5 |
|---|---:|---:|
| Vein mesh resolution | **10 m** | **5 m** |
| Reference mesh resolution | **20 m** | **10 m** |
| Snap to contacts | 100% | 100% |
| Actual input midpoints | 5 controls at z=100 m | Identical |
| Native saved reference elevation | z=101.78 m | z=100.89 m |
| Preliminary vein vertices / triangles | 1,802 / 3,600 | 5,038 / 10,072 |
| Final refined vein vertices / triangles | 272 / 524 | 719 / 1,418 |
| Final HW vertices / triangles | 1,118 / 2,053 | 3,031 / 5,702 |
| Final FW vertices / triangles | 1,102 / 2,020 | 3,011 / 5,662 |

**The new model changes TWO parameters.** It is not a pure vein-resolution test; the reference plane shifts by 0.89 m. Do not attribute changes to vein resolution alone or extrapolate a linear reference-offset formula from only these two settings.

## Exact native pinchout mechanism remains valid

The 28 original WASTE drillholes in each model still generate exactly 28 saved virtual pinchout intervals. Independent vertical-ray intersections with each model's *saved* pre-pinchout raw mesh reproduce all 56 generated interval endpoints in each model to max absolute **2.842170943e−14 m**.

At matching 28 XY holes, G5 minus G10 virtual interval widths vary by **−0.0261829 to +0.1717517 m**, with **0.0405841 m RMS** change. The largest center-depth shift is **0.0457594 m**.

## Frozen kernel candidates against independent G5 native data

The five candidate fields were specified before the user's previous G10 native result. The new G5 case is a new resolution-level test with **no parameter fitting** to G5 or G10.

| Raw-wall thickness field | G10 direct RBF RMSE | G5 direct RBF RMSE |
|---|---:|---:|
| **linear radial r + constant degree-0** | **0.049899m** | **0.011750m** |
| multiquadric ε=1, degree-0 | 0.114304m | 0.090020m |
| multiquadric ε=0.1, degree-0 | 0.706976m | 0.693116m |
| linear radial r + affine degree-1 | 1.221255m | 1.218268m |
| thin-plate spline, degree-1 | 2.851692m | 2.846496m |

Linear degree-0 G5 model: **3.7249 mm RMSE on the original 20 holes** and **21.1782 mm** on 8 additional near-control holes; all 28 11.7498 mm. This is a **76.5% reduction in direct RMSE**, supportive of mesh-discretization effects, but changes in both reference and vein resolution prohibit a strict causal claim.

### Post-hoc native-meshing control

Evaluate each frozen candidate on the native mesh's own XY vertices, retain the native face connectivity, re-intersect 28 hole XY rays against the resulting piecewise-planar surrogate. Using native XY vertices and triangles is a **post-hoc geometry diagnostic, NOT independently generated mesher parity**.

For the linear degree-0 candidate:
- G10 raw thickness error: direct 49.90 mm RMSE; **native-layout 9.23 mm**.
- G5 raw thickness error: direct 11.75 mm RMSE; **native-layout 9.86 mm**.
- G5 old 20 holes, native-layout: **1.14 mm**; new 8: **18.36 mm**.

The remaining ~10mm RMSE on the native mesh topology is **not** eliminated by the refinement. Do not claim that the exact native kernel, mesher or wall snapping is solved.

Additional independent numerical support: wall Z values at native raw-mesh vertices restricted to |x|<100, |y|<100 (avoiding vertical side clipping) have RMSE ~**14.55 mm G10** and **3.15 mm G5** relative to z=100±linearRBF(x,y)/2. Five genuine control holes still honour original observed thicknesses exactly under native-mesh ray intersections.

## Next one-factor experiment: G5 with 20m reference

**Duplicate the new G5 vein model, then change only its Reference Surface resolution from 10m to 20m.** Leave its vein resolution at 5m, snap 100%, same logs, same bounding volume, pinchout ON including empty traces. This creates the missing third factor-combination:

- G10: ref20, vein10 (existing)
- G5: ref10, vein5 (existing)
- **G5_ref20: ref20, vein5 (requested next)**

G5_ref20 versus G10 tests vein-resolution effects with reference resolution controlled. G5_ref20 versus G5 tests reference-resolution effects at the same vein resolution. No new drillhole import required. Save complete owner project (`.aproj` and `.aproj_data`).

Remaining larger goals: folded-reference handedness, exact wall-field formulation, meshing, multiple-intersection pairing, parameter handling and final solid operations. Owner-data tables/decoder scripts are available in the separately delivered private conversation ZIP. Public `main` unchanged.
