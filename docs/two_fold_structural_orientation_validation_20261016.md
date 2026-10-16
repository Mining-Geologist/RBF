# Structural LVA orientation — independent two-fold audit (2026-10-16)

## Result

Using two **user-owned original Leapfrog project archives**, we extracted saved `SubDomainer.DomainInfo.average_mtx` from symbolic Pickle data and reproduced their domain-mean tensors using the same fixed mathematical rule, **without tuning**:

```
nearest = argmin_mesh_vertex ||x - v||
d = ||x - v[nearest]||
n = normalized_stored_gradient[nearest]
q = exp(-d/R) if d < 4 R else 0
r = 1 + (S - 1) q
M(x) = r^(-1/3) (I - n n^T) + r^(2/3) n n^T
A_domain = mean(M(x) for x in original domain input rows)
```

## Asymmetric fold (LVA_Test2)

- 1,995 original interpolation points, 8,601 mesh vertices, 16,800 triangles, 15 saved domains, S=5, R=100.
- Original `StructuralInterpolant.seeds` has **two** modified tail points (indices 1993 and 1994) relocated to extreme bounding-box coordinates. Canonical XYZ comes from the independent original input-point table in the user-owned reverse-engineering bundle. The first 1,993 saved RBF seed coordinates and original input coordinates agree to numeric precision.
- Nearest **mesh vertex** for both normal and distance, then arithmetic core averaging, reproduces all 15 original saved `average_mtx` matrices: **full RMSE 2.75e-16, maximum absolute error 1.72e-15**.
- There are **zero** nearest-vertex ties within 1e-8.
- Exact same vertices, normals, domains and parameters with **closest-triangle distance** instead yields ~3.62e-4 RMSE; interpolated triangle normals yield ~1.81e-3 RMSE even with nearest-vertex distance. These controls directly discriminate the rule.

## Symmetric fold (original 1,666-point case)

- 1,666 input seeds, 7,381 mesh vertices, 14 saved domains, S=5, R=100.
- Two terminal RBF seed coordinates were also replaced by bounding-box extremes. A correction inferred from the original regularly spaced drillhole intervals makes the Y-only matrix residual ~1.48e-16, versus ~1.54e-4 without corrections. **These two corrections are inferred, not independently verified against a saved source input table**, unlike the asymmetric-fold corrections.
- **147 points** have tied nearest mesh vertices (within 1e-8); 8/14 saved domains contain **no** such points and reproduce their full 3×3 `average_mtx` to numerical precision. All six remaining mismatched domains contain ties.
- The native reference off-diagonal XZ entry falls inside the interval of possible nearest-vertex selections in **every** domain; in domains 210 and 211 it is at an extremal endpoint to numeric tolerance.
- Changing SciPy KD-tree construction/leaf size changes the residual considerably, without resolving exact proprietary tie behavior: best among 80 tested settings RMSE ~0.00903; default nearest-vertex ~0.02277. This is **not** a justified inference that Leapfrog uses that best SciPy configuration.

## What is and is not proved

**Strongly established for this single-input trend type:** distance-dependent attenuation, volume-preserving SPD tensor, nearest vertex/its saved gradient, domain core arithmetic mean, and the reason the first folded mesh exhibits orientation mismatches. All of this was checked against the matrices actually serialized in the original projects, not merely against a derived `point_matrices` NPZ.

**Still not established:** the exact implementation-level rule for nearest-vertex ties (particularly at perfectly symmetric folds), multiple competing structural trend meshes, other trend modes, automatic SubDomainer clustering/order, and numerical field parity at arbitrary query XYZ for unseen models.

## Reproduction and source handling

- Committed lightweight test: `benchmarks/leapfrog_gold/validate_orientation_saved_domains.py` (expects supplied original project archive and restricted symbolic parser modules on `--helpers-dir`).
- Complete user-facing test kit: `Leapfrog_Orientation_Two_Fold_Validation_20261016.zip`, with cross-platform decoder helpers, fuller six-candidate comparison, output CSVs, and two sparse coordinate correction CSVs.
- Neither original Leapfrog project ZIP nor proprietary Leapfrog binary is committed.

## Next decisive experiment

1. Construct two non-symmetric fold perturbations that eliminate nearest-vertex ties, preserving S,R and core membership as far as possible.
2. Compare per-domain saved native tensors against the unchanged vertex equation; test whether any remaining mismatch can occur without a tied neighbour.
3. On a controlled exact-tie case, independently capture nearest-vertex index at the real orientation evaluator if possible. Do not fit per-domain tie choices to saved reference averages and claim that recovered selections generalize.
4. Keep automatic clustering as a separate unresolved problem: exact orientation evaluation does **not** resolve the real SubDomainer merge sequence.
