# Structural LVA port to Jizai 1.0.0 — research baseline

This research branch is derived from `fix/jizai-1.0.0-build`; do not merge to main until the tests pass.

## Current native execution

- Workflow: `.github/workflows/jizai-lva-baseline.yml`.
- Test: `ci/jizai_lva_baseline.py`.
- Build: upstream `unageek/jizai` tag `v1.0.0`, GCC 13 and the existing `ci/patch_jizai.py` fixes.
- Artifacts: synthetic held-out predictions and `jizai_lva_baseline/result.json`.
- No original Leapfrog or Rocscience data is copied to this branch.

## API verified in upstream v1.0.0 sources

- `jizai.CovSpheroidal3(psill, range, dim=3)`
- `jizai.Model(rbf, poly_degree=0)`
- `model.anisotropies = [matrix]`
- `jizai.Interpolant(model).fit(...)` and `.evaluate(...)`
- C++ `RbfBase::evaluate` applies the anisotropy transform to the coordinate difference before calling the isotropic covariance function.
- Upstream Python bindings contain no `StructuralInterpolant3`, `AutomaticStructuralDomainBuilder3`, or `StructuralTrendInput3` custom extensions. Do not label the regular anisotropy matrix or two-component orientation prototype “LVA parity.”

## Previous work is still available in other branches

- `feature/structural-lva`: native C++ structural interpolator and trend/domain primitives.
- `feature/leapfrog-parity`: value preprocessing, recovered local support/range/bbox rules, oracle-label path.
- `feature/automatic-subdomainer`: deterministic centroid-grid domain clustering and WolfPass diagnostic suites.
- Reference: `docs/structural_lva.md` in `feature/automatic-subdomainer`.

The previous branch's evidence claims (not yet retested in Jizai):

1. Signed binary class indicators should be converted to nearest-opposite-class distances and clipped using the input bbox diagonal.
2. Single-input trend matrix uses the nearest mesh normal and an exponentially decaying, compact-cutoff strength.
3. Given final domain labels, support radius, local range, representative matrix and expanded bbox are well constrained by previous Leapfrog exported objects.
4. Automatic clustering, blending weights, and multiple-input orientation rules remain research questions.

## Reproducible next experiments

1. Confirm Jizai isotropic spheroidal kernel and global anisotropy convention numerically against an independent dense covariance solver on synthetic data (current workflow).
2. Evaluate a synthetic smoothly varying orientation field from separate `cos(2 theta)` and `sin(2 theta)` interpolation, reporting held-out angular errors against a global orientation baseline (current workflow).
3. Port pure NumPy single-input structural trend tensor and the signed-indicator preprocessing, test against synthetic analytic fields, and preserve its clear provenance.
4. Port the oracle-label local-domain Jizai interpolation first, measuring scalar-field residuals per domain before attempting global blending or surface extraction.
5. Only after the oracle-label path passes, port and compare automatic partitions from the legacy centroid-grid SubDomainer on multiple reference examples.
6. Compare each stage with actual Leapfrog exported evaluation CSVs. Report RMSE, max error, sign agreement, domain/support membership, and mesh distances separately.

**Important:** Numerical equivalence of isotropic/global anisotropic Jizai does not by itself validate structural LVA or the full Leapfrog implicit modelling pipeline.
