# Native snap-fraction isolation and curved-reference wall RBF tests (2026-10-18)

## Scope / provenance
Read-only offline experiments on owner-supplied Leapfrog Geo projects (1) and (2), and native original curved-reference model 03. We statically parsed the native SQLite / LFData data without executing vendor code or unpickling classes. Public research branch only; proprietary project and native mesh assets remain private. Full private reproducibility package: `Leapfrog_Snap_Curved_Native_Analysis_20261018.zip` delivered separately in the conversation.

## Test 1: 50% vs 100% snapping, controlled case C

Saved `VeinBlock` OID 10223: `snap_fraction=0.5` in project (1), `1.0` in project (2); reference resolution 20m, vein resolution 10m. The native reference mesh result and all saved generated pinchout `collar_ids`, `depths`, and `segments` arrays are **byte-identical**. All five real VEIN intervals are MD96–104m, with observed HW Z104m and FW Z96m.

| Persisted geometry | 50% snapping | 100% snapping |
|---|---:|---:|
| Raw/pre-pinchout vein vertices/triangles | 1755/3506 | 1784/3564 |
| Refined solid vertices/triangles | 367/730 | 384/764 |
| Final HW vertices/triangles | 1046/1912 | 1066/1952 |
| Final FW vertices/triangles | 1084/1984 | 1094/2004 |

**Key distinction:** DIFFERENT raw vertex counts do **not** imply a changed raw physical wall surface. We independently evaluated native 50% and 100% raw meshes via barycentric face intersections on the **same 35×35 interior XY grid** (1225 sites, 2450 upper/lower Z intersections). Their raw upper and lower wall elevations match with **1.05e−14m RMSE, 2.84e−14m maximum**. Both equal Z104 and Z96. The raw mesh is re-tessellated, not deformed geometrically at these sampled locations.

**Downstream surface changes:** On the same 1225 XY locations, 50%→100% final HW elevations differ with **0.180516m RMSE** (maximum 1.377731m, p95 .391606m); final FW **0.240506m RMSE** (maximum 1.745593m, p95 .546214m). Refined final volume has 522 matched intersection depths with .442883m Z RMSE and 20 XY locations having mismatching intersection counts; this is NOT a full volumetric-overlap or surface-Hausdorff score.

At four observed off-centre VEIN collars (XY±40,±40), 50% final HW **Z102.622269**, FW **Z97.745593** vs 100% HW **Z104**, FW **Z96**, matching true logged contact elevations exactly. At centre XY0,0: 50% HW Z103.416727 / FW Z97.410115; 100% HW Z104 / FW Z96. Snap fraction is genuinely affecting final contact conformity and solid geometry, not virtual pinchout placement for this particular flat constant-thickness setup. Corrected C100 final HW/FW and refined/raw meshes also match A100's geometry exactly, confirming the prior WASTE sample-boundary control.

**Limits:** No 0% snap native run, so no validated continuous snap-fraction formula. Raw geometry invariance proven on this flat case, not on curved/nonuniform cases.

## Test 2: raw preliminary walls in native curved-reference model 03

Take 25 native real HW and FW contacts with irregular surface curvature. Use native **saved curved reference mesh** as known geometric conditioning input; it is NOT reconstructed from midpoints here. Fit separate exact 2D SciPy RBF models for each wall, all untuned to the native raw mesh. Compare:

- absolute wall Z: `RBF_i(Z_wall_i)`
- reference-relative wall offsets: `R_native(x,y)+RBF_i(Z_wall_i-R_native(x_i,y_i))`.

At 1521 XY sites (3042 individual wall elevations), independent barycentric vertical-triangle intersection against saved native raw `VeinBlock` 4059:

| Frozen kernel / polynomial | Absolute-wall analytic RMSE | Reference-relative analytic RMSE |
|---|---:|---:|
| Linear radial `r`, degree 0 | **0.067153m** | **0.013873m** |
| Multiquadric epsilon1 degree0 | 0.063066m | **0.016557m** |
| Thin-plate spline degree1 | 0.107051m | 0.100045m |

Reference-relative linear remains .013195m RMSE on 856 XY locations >15m from any original control, versus .075821m for absolute linear, showing this is not exclusively contact-site fitting.

**Crucial mesh-conditioned qualification:** Evaluate each RBF model at the *native raw mesh XY vertices*, assign upper/lower wall and preserve native triangles before intersecting the same 1521 XY rays: relative linear RMSE **.015472m**, relative multiquadric **.010059m**. At 588 interior native mesh vertices, direct reference-relative wall errors are .020153m linear vs .014662m MQ. Candidate kernel ranking therefore **depends on meshing/control choices**; the flat G/G5 benchmark favours linear degree0, but these new curved outputs do NOT prove one exact universal native kernel. Fitting to native vertex layout is POST-HOC and cannot be represented as independently predicting the mesher.

## Conclusion / next native experiment

Strongly supported *architecture*: native reference geometry is a trend/coordinate frame; HW and FW can be interpolated as offsets relative to that frame; preliminary raw walls are distinct from generated synthetic pinchout constraints, which in our earlier A–F tests exactly matched raw wall/drillhole intersections; final wall interpolation/contact snap and solid refinement are downstream.

**Unsolved:** native automatic reference construction and 3D folded orientation/handedness, real interpolator kernel and polynomial/anisotropy, native mesher, parameter behavior in general curved and nonuniform cases, multiple-crossing branch pairing, and solid boolean details.

**Suggested next owner action (not needed to validate results in this report):** duplicate existing non-uniform-thickness G5 vein; change ONLY snap-to-contacts 100%→50% while keeping all drillholes and coupled resolution settings fixed. This tests whether flat-C raw-shape invariance transfers to non-uniform geometry and separates final snap sensitivity from width interpolation. No new drillhole set needed. Later independently perturb folded reference boundary/orientation; flat or gently curved reference parity cannot identify overturned fold topology.

Main branch unchanged.