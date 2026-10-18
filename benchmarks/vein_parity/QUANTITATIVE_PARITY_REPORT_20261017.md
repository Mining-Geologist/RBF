# Leapfrog Geo vein parity: saved-mesh numerical comparison (2026-10-17)

This report documents an independent offline approximation to six owner-exported Leapfrog Geo 2026.1.1 vein cases. The raw user-owned geometry and binary project are not published here. Original 3D triangulations, when available, came directly from the owner's saved project, not from our code.

**Confirmed run settings**: 100% snap to data (stored fraction 1.0 and inherited snap mode); vein resolution 10 m, reference resolution 20 m; single-pass meshing; adaptive off; minimum thickness 0; maximum thickness disabled.

**Direct saved-project observations**:
- Flat observed HW contacts Z=104 and FW Z=96; saved curved reference plane Z=101.78, while midpoint Z=100. Thus the reference isn't universally the arithmetic middle.
- Pinchout A created 17 flipped outside-hole HW/FW pairs; enabling completely unlogged holes adds 3 more to B (20 pairs). Saved final solid volume A 29,348.3m3; B 17,344.4m3. The raw unrefined body is 387,200m3 in both.
- Original observed contact elevations are honored to numerical precision by saved interpolation meshes in flat, curved and pinchout tests; native interpolated wall meshes are not exact at all synthetic outside points.
- These are evidence for reversed signed HW/FW separation driving pinchout, not an identification of the hidden RBF implementation.

**Independent original numerical model**: sample saved observed plus flipped outside points, solve augmented radial basis interpolation equations for separate scalar HW(x,y)/FW(x,y) with constant drift and an isotropic spherical covariance kernel phi(r)=1-1.5*(r/L)+0.5*(r/L)^3 if r<L, otherwise zero. Solid exists only where HW>=FW and within both planes. This is not claimed to be the native Leapfrog kernel, and the XY representation cannot model overturned multiply-valued surfaces.

**One fixed kernel range L=650 m**, selected on pinchout A HW surface fit and applied without changing the range to other heightfield-compatible cases:

| Test | HW RMSE m | FW RMSE m | Vertical-prism IoU | Volume difference |
|---|---:|---:|---:|---:|
| Flat | 0 | 0 | 1.000 | 0 |
| Pinchout A, empty holes off | 0.125 | 0.154 | 0.930 | -7.2% |
| Pinchout B, empty holes on | 0.131 | 0.175 | 0.872 | -12.5% |
| Curved and variable thickness | 0.203 | 0.202 | 0.916 | +0.03% |
| End-of-hole | 4.120 | 4.568 | 0.826 | -0.04% |

These are *wall-surface height errors* and an *integrated vertical interval overlap*, not exact 3D mesh Boolean Dice or Hausdorff distance. Volume agreement in the curved/end-hole tests is insufficient to imply surface fidelity.

**Alternative L=180 m** chosen on pinchout A to improve signed-thickness overlap yields A 0.935 IoU with -3.1% volume difference; when applied unchanged to B gives 0.893 IoU with -9.7% volume difference. It does not prove a native range of 180m. Exponential and other covariance families can give similarly plausible surfaces, so kernel identification remains open.

Original offline Python solver, seven watertight positive-volume OBJ meshes, numerical CSV, overlays and exact target meshes are bundled in the user-facing `Leapfrog_Vein_Parity_Quantitative_Comparison_20261017.zip`, **not committed to this public research branch**. The independent multiple-intersection case is *not modelled* by the simple heightfield algorithm and requires a reference-mesh-normal 3D approach.

The documented order of operations also matters: pinchout before maximum-thickness limiting, and snap to drilling after thickness computation (Seequent 2026.1 vein docs: https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm).

**Next benchmark**: switch only snap ON/OFF using identical data; repeat with 10m vs 5m surface resolution, then test a custom planar reference surface, one nonzero max-thickness value, and logged-hole distance control. Hold out the new native meshes during interpolation calibration. No vendor software or binary execution was used in this research run.

**Additional exact eligibility check:** On the reconstructed valid-only geology table, independent rule "all logged non-vein holes; optionally also holes with no valid logs" predicts precisely the saved 17-hole empty-OFF and 20-hole empty-ON selections (zero differences in either hole-ID set). The three new holes contain six original geology rows excluded by import validation, therefore "empty" refers to absence of **accepted** lithology, not necessarily missing raw CSV rows. This benchmark does not validate the logged-hole distance threshold or manual exclusion logic.
