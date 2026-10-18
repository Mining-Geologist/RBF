# Linear radial-basis hypothesis: independent vein kernel comparison (2026-10-17)

**Outcome:** A true linear RBF is competitive, but the saved surfaces do not uniquely identify it as Leapfrog's native interpolator. Importantly, much of the improvement on the curved vein comes from giving the model a linear **polynomial trend**, not changing just the radial kernel.

Dataset: user-owned original `Vein_Benchmark.zip`, with previously exported observed hangingwall/footwall and reversed pinchout control points; independently extracted Leapfrog Geo 2026.1.1 interpolated HW/FW OBJ references. Snapping 100%, vein resolution 10 m, minimum thickness 0, maximum disabled. Model fits only constraints; native wall vertices are used only for scoring.

Candidate linear radial basis is SciPy `kernel='linear'`, proportional to **−distance** in the contact/reference XY plane; its RBF is **not** equivalent to a plane fit, triangular piecewise-linear surface or a triangular compact covariance. Spherical covariance was previously evaluated as `max(0, 1−1.5r/L+0.5(r/L)^3)`. Both models were run with either constant drift or exactly the same affine trend (degree one), zero smoothing and the same input points.

## Matched degree-1 comparison: RMSE of interior Leapfrog HW and FW surface vertices (metres)

| Scenario | Linear RBF, affine HW/FW | Spherical L=650, affine HW/FW | Spherical L=180, affine HW/FW |
|---|---:|---:|---:|
| Flat | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| Pinchout A empty OFF | 0.128 / 0.157 | 0.127 / 0.156 | See raw CSV |
| Pinchout B empty ON | 0.133 / 0.177 | 0.131 / 0.175 | See raw CSV |
| Curved variable thickness | **0.0775 / 0.0769** | **0.0699 / 0.0696** | **0.0596 / 0.0613** |
| End of hole | 4.120 / 4.573 | 4.113 / 4.569 | See raw CSV |

Original previously reported spherical L=650 used only constant drift, yielding curved 0.203 / 0.202 m. With the same affine drift that the linear RBF used, spherical performs BETTER: 0.070 / 0.070 m. This isolates a major confound in the earlier comparison. The spherical L=180 comparison is separately chosen on pinchout A, not evidence of a native kernel/range.

## Signed-thickness grid comparison (using original spherical L=650 constant drift, versus linear with affine drift)

| Scenario | Linear vertical-prism IoU | Prior spherical vertical-prism IoU |
|---|---:|---:|
| Flat | 1.000 | 1.000 |
| Pinchout A | 0.928 | 0.930 |
| Pinchout B | 0.870 | 0.872 |
| Curved | **0.973** | 0.916 |
| End-of-hole | 0.828 | 0.826 |

These are **2D-parameterization-based integrated vertical intersections**, not exact 3D Boolean mesh overlaps. Piecewise-linear Delaunay was also tested and had poor extrapolation plus <54% target-vertex coverage, so it is not equivalent to a global linear RBF.

## Interpretation and next discriminating experiment

- A simple linear RBF can match much of the observed shape and has no length parameter in its exact interpolation case.
- The curved-plane trend captures much of the smooth global structure; kernel comparisons must keep the drift degree identical.
- Pinchout A/B do not distinguish the linear radial kernel from a spherical radial covariance under this 2D approximation.
- The end-of-hole ~4–5 m wall errors in both methods indicate the *contact constraints/reference/snap/geometry* are not fully modelled. Changing only the kernel will not solve these mismatches.
- To identify native behaviour, independently vary snap to data 100% versus off, and compare saved interpolated walls before/after snap; then use synthetic isolated non-coplanar/non-heightfield contacts and fit candidate kernels *without choosing range from that same target surface*. The actual 3D local-coordinate field may be different from global XY.

Numerical scripts and all CSV results are in the accompanying conversation artifact. No native Leapfrog binary executed and none of the owner's private reference meshes are published to this research repository.
