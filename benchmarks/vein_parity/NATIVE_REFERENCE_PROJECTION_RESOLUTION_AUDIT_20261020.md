# Reference projection and resolution control — 20 October 2026

Independent read-only analysis of the user's private compiled program found a spatially accelerated 3D reference-mesh projection with triangle-interior, edge, vertex and adjacency handling. The native offset evaluator adds a field-based correction to a signed conformal distance. This is consistent with reference-relative HW/FW construction, **not a recovery of automatic folded-reference generation**.

The generic RBF backend has distinct value-only and value-plus-gradient paths, plus approximate evaluation settings for resolution and a maximum input-point distance from the reference. A backend evaluation-mode switch is not proof of a particular RBF kernel.

Separately, a frozen analytical 2D linear RBF field was sampled on independent piecewise-linear triangular grids:
- Synthetic grid 10 m: RMSE **0.306046455 m**
- Synthetic grid 5 m: RMSE **0.080251799 m**
- Improvement ratio: **3.81358×**, with **the same RBF** at both resolutions.

Earlier user-run Leapfrog G10/G5 comparisons improved by ~4.25×. The similarity of ratios is consistent with discretization playing a major role, but the synthetic grids, absolute errors and native settings differ. It is not proof of meshing or kernel parity.

Reproducibility:
- `reference_branch_ambiguity.py` and `test_reference_branch_ambiguity.py` (5 passing tests)
- `linear_rbf_mesh_resolution_experiment.py` and `test_linear_rbf_mesh_resolution_experiment.py` (2 passing tests, CI run 39346364897)
- Private Ghidra run 39345407077; private docstring scan 39345955844. Vendor binaries and decompiled contents are confined to the owner-private research repository.

**Still unknown:** the automatic folded reference builder's initial topology and handedness, and the precise kernel/configuration chosen by that builder. No change to main.
