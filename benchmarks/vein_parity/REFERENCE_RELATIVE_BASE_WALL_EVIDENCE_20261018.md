# Reference-relative vein walls: matched-kernel and manifold falsification, 2026-10-18

**Scope:** independently authored offline reconstructions using the user's previously saved Leapfrog Geo 2026.1.1 `Vein_Benchmark` project; all inputs read from owner-exported HW/FW contact CSVs, saved reference OBJ, and target HW/FW mesh OBJ. No native binary is executed. Default run settings: 100% snapping, surface resolution 10m, minimum thickness 0, maximum disabled. Public research branch only; `main` untouched.

## Hypothesis tested

Do the initial HW/FW walls behave like independent scalar offsets from a 3D reference surface, followed by the separately recovered `asphere` contact correction?

Three original model classes were compared at fixed first-order multiquadric radial kernel `sqrt(1+(r/1m)^2)` and identical affine polynomial drift; `a=1m` was selected in an **earlier** Pinchout A comparison, not tuned against the new curved result:

- **Absolute:** `HW(x,y) = RBF(x,y,observed_hangingwall_height)`, independently for FW.
- **Relative:** `HW(x,y) = ReferenceZ(x,y) + RBF(x,y, observed_hangingwall_height - ReferenceZ(contact_xy))`; independently for FW.
- Both independently use the same observed + persisted generated pinchout constraints. Native HW/FW vertices are used ONLY for scoring, after fitting. The native reference mesh is an **input** to the relative prediction, not inferred from drillholes.
- All vertex scores are vertical-elevation RMSE. **inside98** denotes native HW/FW vertices having |x|<=98m and |y|<=98m; **all** includes outskirts to 114.4m.
- Our `ReferenceRelativeWalls` code guards against conflicting Z for identical XY and mixed face orientation of projected reference triangles; it is not a full 3D-fold algorithm.

## Matched experiment: native wall-vertex RMSE (m)

| Case / selection | Model | HW | FW |
|---|---|---:|---:|
| Flat / all | absolute MQ1 | 0.0000 | 0.0000 |
| Flat / all | reference-relative MQ1 | 0.0000 | 0.0000 |
| Pinchout A / all | absolute and reference-relative | 0.1160 | 0.1388 |
| Pinchout B / all | absolute and reference-relative | 0.0824 | 0.1266 |
| Curved / inside98 | absolute MQ1 | 0.063993 | 0.063635 |
| Curved / inside98 | **reference-relative MQ1** | **0.018133** | **0.019377** |
| Curved / all | absolute MQ1 | 0.1908 | 0.1831 |
| Curved / all | **reference-relative MQ1** | **0.0511** | **0.0522** |
| End of hole / inside98 | absolute MQ1 | 4.0107 | 4.4423 |
| End of hole / inside98 | relative MQ1 | 3.7486 | 4.2335 |

The flat/pinchout cases have planar saved references, so a constant reference offset is already representable by the affine polynomial and cannot distinguish the two methods. The curved geometry shows ~70% improvement from the reference trend alone. **This is strong functional evidence for reference-relative wall interpolation, not proof of the exact native kernel or initial reference generation.**

## Direct 3D normal-offset test (different metric)

Predict signed surface-normal offset at reference-vertex positions using identical MQ1 controls, move the vertex either vertically or along its saved reference normal, then compare its 3D closest-point distance to saved HW/FW triangles:

| Curved case | Vertical movement | Reference-normal movement |
|---|---:|---:|
| HW 3D point-to-native RMSE | 0.037192m | 0.036706m |
| FW 3D point-to-native RMSE | 0.037644m | 0.033618m |

This small improvement is compatible with conformal reference-relative geometry but does **not** prove exact normal displacement, especially since the reference is nearly horizontal (|n_z| >0.9945 on every face).

## Critical failure cases and geometry diagnostics

- End-of-hole: ~35% of final HW/FW vertices are more than 10m away from the native reference, some up to ~90m; native HW/FW vertices span Z ~8 to 192m while the reference spans Z ~95.5 to 104.15m. These extreme connected branches do not fit a single-valued small-offset model.
- Multiple intersections: the native reference surface has significant vertical/overturned faces and regions of opposite projected orientation. It is not a valid global XY heightfield. The new code explicitly rejects projected fold topologies where passed face winding is mixed.
- Generic **intrinsic graph-Laplacian** offset interpolation, using barycentric face projections of real control points, with and without affine trend, was implemented and falsified. With affine trend its curved full-surface HW/FW signed-offset RMSE remains ~0.284/0.286m, versus ~0.051/0.052m vertical errors for the simpler MQ1 reference-relative heightfield (different geometric error measure, so treat only as negative trend, not a direct score comparison). The fold and end-hole models regress substantially; nearest reference-face assignment or point-to-branch ambiguity remains unresolved. This approach is not retained as a replacement.
- Original **reference construction** is separately unknown. Flat observed HW/FW midpoints are exactly Z=100, but stored reference plane is Z=101.78. On the curved case, the best tested least-squares plane fitted only to arithmetic midpoints has ~1.137m RMSE against the saved reference, while midpoint linear/TPS/MQ1 interpolants range ~1.20m and above. These figures evaluate **generated reference geometry**, not a wall fit. Native construction may depend on normal vectors, boundary conditions, model extent, smoothing or additional hidden controls.
- The saved reference OBJ is a conditioning input, and synthetic outside interval coordinates for pinchout are *also* already extracted from Leapfrog's project. Therefore neither reference construction nor pinchout generation from raw logs is solved by these experiments.

## Native binary evidence

Prior private Ghidra read-only analysis on `Leapfrog.exe` found `PySurfaceOffsetInterpolant.init_from_mesh_and_points` at 0x1401fc6a0 and a delegate at 0x1402d4e80, with a downstream numerically active 0x1402fc8a0. The two-hop report (73 functions) shows 30 P-code floating point arithmetic instructions in 0x1402fc8a0 but does **not** identify an RBF kernel or the initial wall algorithm. This stage does not claim native source extraction.

Separately, saved curved/end-hole/multiple-hit `Solution3` Snap correction objects *do* explicitly contain an `asphere` FastRBF basis, scale 10, degree -1. Its field and inverse-normal displacement have been independently validated. This is a **subsequent correction stage**, not proof that the underlying HW/FW base mesh uses the same RBF.

## Deliverables and reproduction

- `benchmarks/vein_parity/reference_relative_wall_model.py`: independently authored conditional MQ1 wall interpolation and fold guard.
- `benchmarks/vein_parity/test_reference_relative_wall_model.py`: synthetic tests; no private data.
- `benchmarks/vein_parity/benchmark_reference_relative_walls.py`: optional owner-local saved-mesh comparison CLI, holding HW/FW reference meshes out of training.
- This experiment's portable research ZIP includes aggregate CSVs and local offline tests for midpoint reference hypothesis, normal vs vertical offsets, and intrinsic Laplacian negative result; none uses native Leapfrog runtime.

**Next most decisive test**: independently recover the precise reference surface generator (orientation + smoothing + translation) from midpoint, structure and model bounds; then fit signed wall offsets and the independently recovered native `asphere` snap field, reconstruct the full meshes without using the saved native reference as an input. For folded and end-of-hole, also recover reference-manifold parameterization and branch association. A no-snap native project would be especially useful to validate pre-snap surface reconstruction, but is not needed to continue static research.
