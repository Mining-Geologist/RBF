# Local-normal reference-following wall reconstruction — verified against real multi-vein models (2026-10-18)

**Status:** A new, independent 3D reference-following RBF surrogate was implemented and compared to the original, privately supplied multi-vein saved project. No native vendor code, copied source or proprietary geometry has been published. All project inspection was read-only and used static serialization/numeric parsing. This is behavioral evidence, not recovered Leapfrog source code.

## Method

The four child veins reuse a single identical native parent hangingwall reference (43,784 vertices / 85,839 triangles). The new model takes this known parent reference **as input**. For every observed true HW/FW contact, it finds the nearest point on the reference, estimates a smoothly varying local reference normal, projects the contact displacement onto that normal, then fits an exact-interpolating 2D degree-0 linear RBF to these signed offsets in a reference PCA chart. At **every inherited reference mesh vertex** it predicts the normal offset and **moves that vertex along its own locally rotating normal**, preserving reference mesh connectivity. It interpolates HW and FW separately.

The matched global-reference model also retains the **same parent mesh topology**, same contacts and identical evaluation rays, but moves each reference vertex along a fixed global PCA normal. Neither surrogate was fitted to the saved native raw walls. This is a substantially stronger forward 3D geometry comparison than previous analytic global-chart predictions.

## Native preliminary-wall parity (held out from fitting)

Measured at ~450–550 reproducibly selected fully valid reference/native projected chart-ray locations per child:

| Child | Wall | Global reference-offset linear RBF RMSE | New local-normal linear RBF RMSE | Reduction |
|---|---|---:|---:|---:|
| Child A / Schist2 | HW | 3.7764m | **3.4385m** | 8.9% |
| Child A / Schist2 | FW | 1.9546m | **0.5737m** | 70.6% |
| Child B / Schist3 | HW | 5.0943m | **3.3595m** | 34.1% |
| Child B / Schist3 | FW | 2.8130m | **0.8911m** | 68.3% |
| Child C / Schist4 | HW | 0.8812m | **0.0730m** | 91.7% |
| Child C / Schist4 | FW | 0.3695m | **0.0774m** | 79.0% |

Local geometry intersections have >99.7% valid single-hit coverage. Results are not exactly matched pointwise for the ≤1 locally missing hit; track this limitation. Child C has only 3 genuine contacts per wall, so cannot reliably identify a kernel. A different reference sampling seed (98765 rather than 1234) and more candidate search triangles (650 versus 400) reproduces the improvements: Schist2 FW 0.58m; Schist3 FW 0.48m; Schist4 HW/FW ~0.08/0.07m.

Alternative local multiquadric epsilon1 is comparable to linear: slightly better on hangingwalls, generally worse on footwalls. **No unique kernel identified.** Local normal directions differ from the global PCA normal by median ~6.0°, 95th ~17.95°, max ~61.43°.

## Independent blocked geological-contact prediction: mixed outcome

On the same five spatially contiguous withheld-fold partitions for each original wall dataset:

| Child and observed contacts | Global relative linear RMSE | Local normal linear offset RMSE |
|---|---:|---:|
| A HW (41) | 23.361m | 22.978m |
| A FW (41) | 26.863m | 25.770m |
| B HW (25) | **16.216m** | 17.426m |
| B FW (25) | **15.532m** | 15.798m |

**This contrast is critical:** excellent agreement with native *modeled* wall surfaces is NOT evidence of accurate prediction of unseen real geology. When evaluating the local model in true 3D, additional tangential projection error increases errors in child B even further (roughly 19.02m HW / 16.34m FW). Long-range, strongly curved reference-to-contact correspondences remain ambiguous.

## Architectural inference / constraints

These results substantially strengthen the reference-local wall-generation hypothesis: offset surfaces are well approximated by moving the inherited triangulated reference along rotating normals with interpolated signed offsets, rather than by globally vertical/global PCA-normal elevation fields.

**Remaining unsolved:** autonomous primary curved/folded reference construction and its handedness, exact kernel/anisotropy/conditioning, structural local chart/geodesic coordinates, long-distance projection ambiguity, control handling, mesher/clipping, final snapping and booleans, multiple folded intersections, and inter-vein truncation and chronology. Neither result reconstructs an automatic reference from drillhole midpoints. Progress toward all of Leapfrog's vein functionality remains approximate and is not implied by a high score on a subset.

Generic standalone code and synthetic tests are released on this **research branch only**. All owner-specific validation scripts, IDs, native project data and geometry stay within the private conversation archive `Leapfrog_Local_Reference_Normal_RBF_Investigation_20261018.zip`. Public main branch untouched.
