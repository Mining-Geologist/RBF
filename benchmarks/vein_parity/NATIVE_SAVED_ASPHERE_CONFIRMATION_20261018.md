# Native Leapfrog 2026.1 vein FastRBF: directly saved solutions, Ghidra offsets (2026-10-18)

**Conclusion, with unusually strong evidence:** The user's saved native HW/FW **snap/refinement solutions** (three nontrivial vein test cases) use the explicitly persisted basis **`asphere`**, parameters **`[10.0, -3]`**, **`degree=-1`**, **`rho=0.0`**, fitting method **`gmres`**, engine **`FastRBF V 1.4.2`**. The complete 3D centres, fitted scalar coefficients, polynomial origin, bounds and tolerance fields are also persisted in each `Solution3`. This does **not** establish the basis that originally generated the whole reference mesh or the unsnapped wall surfaces.

## Evidence from Ghidra static binary inspection

Private, read-only headless Ghidra workflow: https://github.com/Mining-Geologist/LF_SC/actions/runs/39095895314
Private metadata artifact: `private-ghidra-vein-offset-methods`.
- `PySurfaceOffsetInterpolant.init_from_mesh_and_rbf` at `0x1401fb580`;
- `PySurfaceOffsetInterpolant.conformal_distances_to_mesh` at `0x1401f9680`;
- `PyRefinedWallsFunction.conformal_distances_to_mesh` at `0x1401f6c30`;
- `PySurfaceOffsetInterpolant.set_params_for_approx_eval` at `0x1401f9070`.

Ghidra recovered 8,334 functions and 130 candidate functions associated with 176 method/string identifiers. These are actual relevant Cython method registrations and function addresses, but they do **not** alone establish numerical semantics or arbitrary call sequencing. No native software was launched, patched, or instrumented; proprietary binary content remains in the private repository.

## Independent numeric reconstruction from the actual saved Solution3 coefficients

For an owner-owned `Vein_Benchmark.zip`, private symbolic decoding (without importing any proprietary Python classes) revealed the saved `Solution3` at HW/FW for **curved**, **multiple-intersections**, and **end-of-hole** cases. Flat and two pinchout controls have `hw_interp/fw_interp=None` in the saved Vein records; their original wall interpolator must not be inferred from this evidence.

Independent evaluator (without native binary):

- `u = ||x-centre_i|| / 10`
- `phi(u) = 1 - 0.75*u` for `u<0.5`
- `phi(u) = 0.8734640537108553 / (1+u*u)^1.5` otherwise
- `f(x)=sum_i coeff_i*phi(u_i)` (the serialized degree is -1).

The original **observed wall-contact** nearest each saved RBF centre was compared to that centre. Projecting the contact-centre vector onto local *final wall* face normals yields a signed separation; HW and FW carry opposite normal-polarity conventions. Compare directly to `f(centre)`.

### Independent physical residuals

| Model | HW controls | HW signed offset RMSE | FW controls | FW signed offset RMSE |
|---|---:|---:|---:|---:|
| Curved | 25 | **0.0000265 m** | 25 | **0.0000310 m** |
| End-of-hole | 23 | 0.4239 m, median abs 0.00175 m | 23 | 0.5435 m, median abs 0.00293 m |
| Multiple intersections | 14 | 0.8897 m, median abs 0.0450 m | 15 | 0.8482 m, median abs 0.0892 m |

The curved model reproduces all **50 contacts within 1 cm**, with actual RMSE ~0.03 mm. At other cases, RMS is elevated by several cross-branch/nearest-contact or reference-normal ambiguities, though nearest-centre to observed-contact **magnitude vs abs(saved coeff)** correlations remain 0.9989–0.9998 in all six side/model combinations. Final wall contacts selected by the nearest-centre mapping are within 5 cm of their saved wall meshes.

For the same saved curved coefficients and geometry, changing the asphere physical scale from 10.0m to 9.5m increases HW RMSE from **0.0000265 to 0.0003283 m**, while 10.5m gives **0.0003586 m**. The exact saved 10m parameter performs far better. If we wrongly substitute spherical covariance at short range with the **same native coefficients** HW residual rises to **0.0024065 m**; naive affine-linear radial or MQ1 forms with the same native coefficients give much larger RMS. This is direct evidence for the saved asphere evaluator; those alternate-kernel tests keep native coefficients fixed and therefore are not fair *re-fit* comparisons.

This confirms an *RBF snap/correction field* with near-millimetric or micrometric geometric agreement, not that the complete wall interpolator is itself this one RBF. The saved vein `snap_distance=10.0` equals the saved RBF range parameter on the models studied, suggesting they are coupled, but resolution is also 10.0, so controlled additional tests would be needed to identify which setting governs the parameter.

## Separate 3D reference-normal test

An independent 3D nearest-face-normal signed-offset surrogate was also built, preserving the saved HW/FW reference meshes for geometric input. It is not a re-evaluation of the native saved `Solution3`: it fits new generic 3D RBFs using observed contacts. Flat and pinching planes remain compatible; 3D folded reference is not a single-valued XY heightfield. Example signed normal-distance RMSE on end-of-hole with constant drift and ordinary linear3D is **~1.70m FW / ~1.58m HW**, compared with previous ~4m vertical-elevation errors from a different comparison metric. Different metrics must not be conflated or reported as a precise percentage improvement.

## Remaining before complete vein parity

- Reconstruct native **reference surface construction** from midpoints, orientation and planning settings.
- Recover **unsnapped wall base** interpolation and how `PySurfaceOffsetInterpolant` combines its reference mesh with fitted correction solutions.
- Establish precise input point orientation, folded branch selection and how segment exclusion works.
- Reproduce `Pinchout` outside segment generation and folded boolean mesh classification across alternate settings.
- Recreate boundary clipping, meshing resolution and the final "Snap to Data 100%" transform.
- Verify against changed settings and truly independent models.

Keep separate from `main` and commercial production until these are validated. Proprietary native arrays and binaries are NOT in the public repository. See `replay_saved_asphere.py` and synthetic `test_replay_saved_asphere.py` for the clean independent implementation. This report's geometric residuals are computed offline from owner-owned saved reference data.
