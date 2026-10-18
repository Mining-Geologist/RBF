# Binary-audit follow-up: turbo-rbf multiquadric candidates and real vein meshes (2026-10-17)

The user correctly identified that our **actual private Leapfrog PE static audit**, rather than Jizai's source tree, recorded C++ identifier strings of `turbo_rbf::basic_functions` in `Leapfrog.exe`.

Original source: https://github.com/Mining-Geologist/LF_SC/blob/analysis/leapfrog-binary-audit-20261014/analysis-output/leapfrog_static_audit.md

## Verified strings and interpretation

The audit contains names for `BiharmonicParameters2/3`, `GaussianParameters`, `MultiquadricParameters<1>/<3>/<5>`, `InverseMultiquadricParameters<1>`, `ExponentialParameters`, `CubicParameters`, `TriharmonicParameters2/3`, `SphericalParameters`, `TestLinearParameters` and the `PyInit_turbo_rbf_spaces` module entry point.

It also exposes `PySurfaceOffsetInterpolant`, `init_from_mesh_and_points`, `init_from_mesh_and_rbf`, and `PyRefinedWallsFunction.conformal_distances_to_mesh`. These offset-surface references are arguably more relevant to vein surfacing than the global numeric model's choice of radial kernel.

**Static names alone do not show which basis Leapfrog selects for veins, nor the native multiquadric shape convention or polynomial degree.**

## Independently evaluated, standard-form radial candidates

Tested MQ1 `sqrt(1+(r/a)^2)`, MQ3 `(1+(r/a)^2)^(3/2)`, MQ5 `(1+(r/a)^2)^(5/2)`, IMQ1 `(1+(r/a)^2)^(-1/2)`, and linear distance `-r/a`, with identical HW/FW observed+pinchout points and degree-one affine drift for first-order functions. MQ3/5 also tested with degree-two drift. No smoothing. No vendor library executed.

Calibrated the shape parameter **only on pinchout A** across a grid 0.1–2800 m; then froze the selected range for other cases. Score is average HW and FW elevation RMSE measured at interior vertices of independently saved native meshes. Original references never used for fitting.

| Case | Prior spherical covariance L650 RMSE (m) | MQ1 a1 RMSE (m) |
|---|---:|---:|
| Flat | 0.0000 | 0.0000 |
| Pinchout A (calibration) | 0.1415 | **0.1241** |
| Pinchout B (settings holdout, shared geometry) | 0.1532 | **0.1319** |
| Curved geometry | 0.0697 | **0.0638** |
| End-of-hole | 4.3409 | **4.2265** |

MQ1 a1 improved each non-flat wall fit relative to the earlier spherical L650 surrogate. IMQ1 optimum on A within searched parameters: 0.5113 m. MQ3 >=2.3301 m, MQ5 >=6.9871 m on A, substantially worse. The 'TestLinear' hypothesis gives 0.1423/0.1546 m on A/B.

## Vertical-prism signed wall overlap on 161x161 regular grid

This is **not** exact 3D mesh Boolean volume intersection.

| Scenario | MQ1 a1 | Linear distance | Spherical L650 | Spherical L180 |
|---|---:|---:|---:|---:|
| Pinchout A | **0.9460** | 0.9281 | 0.9304 | 0.9305 |
| Pinchout B | **0.9134** | 0.8698 | 0.8720 | 0.8930 |
| Curved | 0.9769 | 0.9726 | 0.9764 | **0.9795** |
| End-hole | 0.8240 | 0.8284 | 0.8271 | 0.7980 |

Pinchout A/B wall-prism volume bias for MQ1 a1: -1.43% / -5.14%, compared with -6.28% / -11.77% for spherical L650. Although this is a promising change, absolute wall mismatch on the end-hole model remains greater than 4m.

## Interpretation and limitations

- MQ1's finite shape rounds the distance near zero. In the limit `a→0`, it approaches a scaled linear-distance RBF, explaining why MQ1 and TestLinear can produce similar surfaces.
- The exact turbo-rbf MQ parameterization was **not recovered**; the formulas here are independent standard hypotheses derived from the binary identifier names.
- Shape `a=1m` was selected using the native pinchout A mesh (calibration), so improvement on A is **in-sample**; pinchout B shares underlying dataset with A. Curved geometry is more independent but still requires different reference and end-of-hole handling to establish native parity.
- A larger MQ1 shape (6m) yields curved-case error around 0.0326m, but grossly worsens pinchout A (0.3047m). We did not change the A-calibrated parameter for held-out scoring.
- The native `PySurfaceOffsetInterpolant` and conformal wall distance components should be examined in a **true 3D reference-mesh-relative interpolation experiment**. More XY radial kernel fitting will not solve overturned/multiple-hit cases.
- The independent portable reproduction and full score/shape sweeps are in the user's downloadable ZIP; no private reference mesh, project or vendor binary is committed to this public research repo.

**Decision:** MQ1 a1 becomes a leading **research surrogate for planar/pinching heightfield veins**, not a verified Leapfrog-native formula. Do not replace production RSGeo/Jizai on this evidence alone.
