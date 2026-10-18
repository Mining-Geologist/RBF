# Kernel family comparison against saved vein HW/FW meshes (2026-10-17)

**Status:** 17 independently evaluated RBF/covariance families, using 5 non-overturned tests recovered from the user-owned Vein_Benchmark project. No Leapfrog binary executed; no private mesh or drillhole data committed. Main branch unchanged.

## Provenance of names (do not conflate)

**Saved Leapfrog numeric models:** LinearVariogram, SpheroidalVariogram alpha=3; FastRBF saved local interpolants have basis `asphere`, and compiled binary analysis exposed `FastRBF V 1.4.2` (engine version). These are **not** direct identification of the RBF for vein HW/FW. Previous independent numerical field parity on separate WolfPass scalar models was ~1.49e-5 heldout RMSE for 3D linear-distance + constant drift, and ~7.48e-6 for calibrated Spheroidal3 + constant drift (150 held-out query points).

**Our independently authored Polatory/Jizai code:** Biharmonic2D/3D, Triharmonic2D/3D; CovSpherical, CovExponential, CovGaussian, CovCubic, CovGeneralizedCauchy3/5/7/9, CovSpheroidal3/5/7/9. These class names in our source must **not** be presented as native Leapfrog binary function strings.

## New five-case offline test

Compared 17 radial families. Each HW/FW fit used saved **observed plus reversed pinchout interpolation point constraints only**; saved surface mesh vertices were used **only for evaluation**. A common linear/affine polynomial background term, no nugget, no smoothing for all except Triharmonic2D (degree2 needed for admissibility). Candidate ranges 90/180/300/450/650/1000 m. Every family's best range selected using pinchout variant A *only*, then frozen for B, curved and end-hole tests. The comparison is an XY heightfield approximation, **not a verified 3D reference-surface field**. The multiple-intersections benchmark is unsuitable for the heightfield and excluded.

### Mean of HW/FW interior vertex RMSE in metres

| Kernel (range chosen from A) | Pinchout A | Pinchout B | Curved | End-hole |
|---|---:|---:|---:|---:|
| Conventional spherical, L650 | **0.1415** | **0.1532** | **0.0697** | **4.3409** |
| Spheroidal3, R1000 | 0.1424 | 0.1546 | 0.0741 | 4.3470 |
| Saved-basis `asphere`, a373.157 (same Spheroidal3) | 0.1424 | 0.1546 | 0.0741 | 4.3470 |
| Linear distance = Biharmonic3D radial form | 0.1423 | 0.1546 | 0.0772 | 4.3466 |
| Spheroidal5/7/9, R1000 | 0.1423 | 0.1546 | 0.0772 | 4.3466 |
| Exponential, R1000 | 0.1467 | 0.1574 | 0.0885 | 4.3578 |
| Gaussian, R90 | 3.3219 | 1.6027 | 0.1104 | 14.0712 |
| 2D thin-plate biharmonic | 1.0644 | 0.5504 | 0.2006 | 7.2866 |

The other tested classes (CovCubic, Cauchy3/5/7/9, Triharmonic3D/2D) did not improve the overall observed ranking.

### Signed HW/FW vertical prism overlap with saved wall surfaces

| Case | Linear-distance | Spheroidal3 (R1000) | Spherical (L650) | Spherical (L180; separately chosen) |
|---|---:|---:|---:|---:|
| Pinchout A | 0.9281 | 0.9281 | 0.9304 | **0.9305** |
| Pinchout B | 0.8698 | 0.8698 | 0.8720 | **0.8930** |
| Curved | 0.9726 | 0.9742 | 0.9764 | **0.9795** |
| End-hole | 0.8284 | 0.8276 | 0.8271 | 0.7980 |

The L180 spherical option was selected in a **separate prior pinchout-volume experiment** and should not be treated as an uncontaminated winner of the range-frozen RMSE comparison. The overlap is a vertical-prism proxy, not exact Boolean 3D IoU.

## Mechanistic insight

The actual Polatory CovSpheroidal3 function is linear within its inner radius:
`phi(r)=1-2.009875543958482*r/R` for `r/R<0.18657871684006438`; otherwise `0.8734640537108553/(1+7.181510581693163*(r/R)^2)^1.5`.
The older saved `asphere` basis is `1-0.75*u` for `u<0.5`, else `0.8734640537108553/(1+u^2)^1.5`.
With `u=r/(0.3731574337*R)`, these two formulas agree within **2.6e-11** on 100,001 tested distances, and give the same fitted curved/pinchout field at selected points within ~5e-11. Other Spheroidal5/7/9 families also have linear near-fields. Hence the strong match from linear distance is mathematically compatible with a spheroidal kernel of sufficiently long range. These tests cannot identify the proprietary vein kernel uniquely.

**Interpretation:** No newly tested family clearly outperforms our best existing spherical surrogate. Native vein algorithm remains unproven. Large end-hole ~4m errors across families indicate missing constraints/reference/snap treatment more than a kernel defect. Next high-information experiment: exact native snap-OFF controls and/or true 3D reference-normal wall interpolation.

Source-only reproducible script and all scores are included in the separate user-facing offline experiment zip, which does NOT publish private native mesh inputs.