# Native folded reference: exact midpoint-orientation non-identifiability — 2026-10-18

**Scope:** independent static/geometry analysis of the owner's saved `Vein_Benchmark`, not native solver recovery. User-owned benchmark meshes, contacts and decoded point arrays were analyzed locally and are NOT committed to this public repository. The pure Python test instrument `midpoint_orientation_identifiability.py` and synthetic regression tests are committed.

## Main result: the native reference and its mirror fit the *same* raw geological input positions equally well

In saved case `04_MULTIPLE_INTERSECTIONS`, 15 native reference-input midpoint points form three ordered rows (five at Y=-40, five at Y=0, five at Y=+40). Consider reflection `T(x,y,z)=(x,-y,z)`. The **unordered** reference midpoint cloud is exactly invariant under T (tolerance 1e-7m). So is the **unordered union** of the 15 native HW and 15 native FW contact XYZ coordinates. However, the HW-labelled and FW-labelled *subsets individually* are not reflection-invariant: wall polarity has been switched in several rows *after* reference generation, consistent with earlier persisted reference->wall-generator DAG findings.

Create an independent, mirrored copy `T(R)` of the **native** 349-vertex/606-face reference mesh `R`. Measure the exact closest-point distance from **each of the 15 original midpoint samples** to triangles in R and T(R), via `trimesh.proximity.closest_point_naive`:

| Quantity | R (native) | T(R) (mirrored native) |
|---|---:|---:|
| Midpoint-to-surface Euclidean RMSE | **17.396852661513 m** | **17.396852661513 m** |
| Midpoint-to-surface Euclidean median | 8.497181332137 m | 8.497181332137 m |
| Midpoint-to-surface Euclidean max | 45.679461709885 m | 45.679461709885 m |
| Mean native vertex-to-mirrored surface distance | (not applicable) | **43.664500677672 m** |
| RMSE native vertex-to-mirrored surface distance | (not applicable) | **62.362523214269 m** |

This equality is a **mathematical consequence** of the input cloud's mirror invariance, not an accidental numerical coincidence: for any candidate surface S and symmetric input set P, the *multiset* of distances `{dist(p,S):p∈P}` equals that to `T(S)`. Intrinsic area and curvature norms are also preserved by rigid reflections. Consequently, even a symmetric geometric objective containing midpoint distance and intrinsic smoothness **cannot select a unique handed fold** without additional orientation / initialization / ordering / structural constraints.

## Midpoint-only 2D interpolation also cannot choose the correct orientation

As an **explicitly post-hoc witness**, choose one normal `n = (-0.35457005,-0.75983370,-0.54491543)` recovered via a feasible hemisphere LP from native face normals, and its reflected normal `T(n) = (-0.35457005,+0.75983370,-0.54491543)`. Their unoriented acute angular separation is **81.1009°**. Fit identical 2D TPS interpolants (with degree-1 drift, 40m coordinate scale, isotropic smoothing) in these two frames, evaluating held-out **midpoints only** by leave-one-point-out CV.

At smoothing=1.0, both midpoint CV RMSEs are **33.865898990546 m**, identical within floating-point error. The equality persists for smoothing values 0.001, .01, .1, 1, 5, 20 and 100. When these two candidate charts are evaluated **post hoc against native vertex normal coordinates**, however, RMSEs are **44.25347 m vs 74.58960 m**. Neither is parity. The reflected normal is not even a valid global graph normal *for the native mesh* (minimum face dot = -0.9435 versus +0.0851), although it is an equally good input-only score candidate.

Point-cloud PCA selects a near-vertical `(0.025,0,0.9997)` normal. Its acute angle to the native **output-mesh** least-variance PCA normal is **48.06746°**. This is another warning that fitting a midpoint covariance plane does not recover the folded reference's spatial orientation.

**Restrictions:** orientation normal and native-vertex targets were obtained from the saved native output *only for testing non-identifiability*; no independent oracle-free orientation predictor is being claimed. Errors are local to this synthetic saved dataset. The possibility that the runtime uses extra inputs from the `FaultBlockBoundaryBlock` or indirectly through `raw_vein` remains open.

## Cross-case controls: closest midpoint distances to reference and mirrored reference

| Case | Unordered midpoint cloud symmetric? | Native ref RMSE (m) | Mirrored ref RMSE (m) |
|---|---|---:|---:|
| 01 Flat | yes | 1.7800 | 1.7800 |
| 02 Pinchout A | yes | 1.7800 | 1.7800 |
| 03 Curved | **no** | 1.3785 | 4.1841 |
| 04 Multiple intersections | **yes** | **17.3969** | **17.3969** |
| 05 End of hole | no | 37.8509 | 37.8090 |

The flat/pinchout controls behave as expected. The end-hole values average over included extreme contacts, so their large RMSE is not surprising; they are NOT an interpolation quality measure relative to the native reference.

## New key clue: input ordering *itself* breaks the mirror symmetry

The **ordered** 15-point array is grouped into the Y=-40 row first, then Y=0, then Y=+40. The reflection maps the array onto the **permutation** `[10,11,12,13,14,5,6,7,8,9,0,1,2,3,4]`: 10 of the 15 records change position. Thus a solver consuming the order of midpoint insertion as a seed or traversal order is **not constrained** to produce a mirror-invariant result, even when the *unordered* point set is symmetric. This is a concrete possible symmetry-breaking mechanism, not proof that Leapfrog actually does so.

## Next decisive controlled native comparison

The single most informative control is **reorder-only**, not a newly tuned interpolant:

- Clone the saved `04_MULTIPLE_INTERSECTIONS` model.
- Preserve collar/survey/interval XYZ positions and all modelling settings.
- Reverse the order of borehole records (or logically equivalent input row/insertion order) without changing the geological point **set**.
- Regenerate the **default curved reference**. Export and compare reference OBJ before any downstream wall fit; compare surface-to-surface distance, chart normal, and mirror discrepancy.

If reference *shape/orientation changes substantially*, index/insertion-order/initialization sensitivity is implicated. If unchanged, order sensitivity for this perturbation becomes less likely and attention shifts to hidden boundary/geometry context and native deterministic orientation priors. A mirror-coordinate-only paired test should follow, preserving IDs and sequence explicitly. If the native output mirrors with the inputs, the algorithm may still be reflection-equivariant while the first run happened to use an asymmetric seed/state; determine the actual seed by repeated rebuilds and reinsertion order.

**Do not assume the reference depends solely on the midpoint cloud.** The saved runtime DAG has the fault-block boundary as a declared input, and the official documentation mentions wall-related context. Our symmetry theorem only applies when all geometrically relevant input *sets and configuration* are symmetric and the solver has equivariant initialization.

## Reproduction

In `benchmarks/vein_parity/` run:

```
python -m unittest -v test_midpoint_orientation_identifiability.py
python midpoint_orientation_identifiability.py \
  --midpoints path/to/04_MULTIPLE_INTERSECTIONS_reference_input_midpoints.csv \
  --reference path/to/04_MULTIPLE_INTERSECTIONS/Reference_Surface.obj \
  --hangingwall path/to/Observed_hangingwall_points.csv \
  --footwall path/to/Observed_footwall_points.csv
```

Inputs are user-owned, intentionally not shipped in this public repository. `main` unchanged.
