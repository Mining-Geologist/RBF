# Folded native reference is a globally oblique 2D graph — 2026-10-18

**Research status: new geometric fact, not recovery of Leapfrog's algorithm.** The owner-provided `Vein_Benchmark` OBJ meshes and midpoint/location CSVs were used **only locally** for analysis. No vendor binaries or private samples are committed here. Code: `fold_chart_audit.py`; synthetic unit tests: `test_fold_chart_audit.py`.

## 1. Exact y-mirror asymmetry

In case `04_MULTIPLE_INTERSECTIONS`, the **15 native saved midpoint XYZ samples** are exactly invariant under `(x,y,z) -> (x,-y,z)` to 8 decimals. The reference surface, however, is highly asymmetric.

With a deterministic seed=42, uniformly sample 900 points by native triangle area, reflect each through Y=0, then compute its **exact closest point on the native reference triangles** using `trimesh.proximity.closest_point_naive`. This avoids nearest-vertex sampling artifacts.

| Case | Mean 3D mirror distance | 95th percentile | Interior mean | Midpoints y-mirror invariant? |
|---|---:|---:|---:|---|
| Flat | <1e-12 m | <1e-12 m | <1e-12 m | yes |
| Pinchout A | <1e-12 m | <1e-12 m | <1e-12 m | yes |
| Curved | 4.229 m | 7.997 m | 2.120 m | no |
| Multiple intersections | **48.111 m** | **118.147 m** | **20.962 m** | **yes** |
| End of hole | 0.216 m | 0.500 m | 0.236 m | no |

The interior restriction is `|X|<80, |Y|<60, 20<Z<180`; it is not a claim of geologically meaningful clip. Mean values depend on deterministic area sampling but are stable in order of magnitude. Flat/pinchout control symmetry is precise.

**Falsification:** a reflection-equivariant, uniquely specified interpolator that consumes only these symmetric midpoint locations (and a symmetric domain) could not produce this highly asymmetric surface. At least one of the assumptions—input completeness, reflection equivariance, unique solution, or symmetric initialization—must fail. This does not tell which one fails.

## 2. Actual native fold IS a heightfield in an oblique plane

All of the `04_MULTIPLE_INTERSECTIONS` native reference faces are connected, consistently wound, and form an orientable open disk with Euler characteristic 1, one boundary loop and no non-manifold edges.

Solve the linear feasibility problem `face_normal_i · n >= 1` for an unbounded n, normalize it, and use n as the normal of a candidate oblique 2D chart. For this saved surface one feasible direction is:

```
n = (-0.354570052610, -0.759833702480, -0.544915426803)
minimum(face_normal_i dot n) = 0.085113323866
```

Then project the mesh triangles to the plane orthogonal to n, and check for **positive-area overlaps of non-vertex-sharing triangles** (Shapely STRtree, tolerance 1e-7). On this mesh:

| Projection | Positive-area overlapping nonadjacent triangle pairs | Sum overlap area |
|---|---:|---:|
| Global XY plane | **550** | ~3,373.68 m² |
| Feasible oblique plane | **0** | 0 m² |

Combined with the strictly positive face-normal alignment, the no-overlap test shows this *particular native reference mesh* is representable as a **single-valued heightfield above an oblique plane**. This is a concrete correction to earlier test scripts that rejected *any* 2D representation when the same XY location had multiple Z samples. They were justified in rejecting a **global XY heightfield**, not **every possible geological 2D chart**.

**Important qualification:** The geometric chart was recovered **from the native output mesh**, not inferred from the 15 midpoint positions. This is a post-hoc existence certificate, not an independent reconstruction or evidence that Leapfrog's native generator actually fits a 2D TPS.

## 3. Orientation is still the hard part

A preliminary exploratory 2D thin-plate spline fit to the saved native midpoints *in the post-hoc native-derived plane* still has ~44.25 m normal-displacement RMSE against the native reference at smoothing=1. The native surface extends and bends considerably outside the modest 15-point midpoint cloud. Recovering a chart alone is **far from parity**.

## 4. Wall-contact polarity is not safely treated as upstream input

At X=0 and the lower intersection midpoint Z=78.5m, recorded contact assignment changes from HW Z82/FW Z75 at Y=-40 and Y=0 to HW Z75/FW Z82 at Y=+40. At X=40 the inversion occurs already at Y=0. This pattern is consistent with a twisted reference/orientation field but should **not** be used to claim wall labels caused the surface asymmetry. The saved dependency graph shows the ordinary wall-point generator consumes the already-created reference surface. This may be orientation *feedback in the saved output*, not the input to the initial reference builder.

## 5. Next discriminating native experiments

1. Duplicate the identical multiple-intersection project; reflect all collars and drillhole coordinates across Y=0 while keeping the symmetric model domain and settings unchanged. Export reference OBJ for old/new projects. If the algorithm is reflection-equivariant, the transformed outputs must match within meshing error. Otherwise ordering, initialization, an unexposed directional rule, or other hidden inputs are involved.
2. Reorder the hole IDs/ingestion order while keeping exact XYZ points/intervals. Measure reference displacement and face-normal differences. Any large change establishes order/seed dependence.
3. Vary only the reference resolution, keeping data/geometry unchanged, to distinguish a geometric choice from discretization/winding artifacts.
4. Test independent recovery of the oblique orientation from *only* midpoint/segment data; score blind on a new topology before adjusting a wall fit.

**Do not merge an automatically inferred oblique chart into main** based on this one post-hoc geometry. In particular the Fold case remains unsolved as a forward prediction from raw drillholes.

Source-based context: the Seequent Veins reference documentation describes a curved best-fit reference from vein data (https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm), while custom interpolant meshes are a distinct option (https://www.seequent.com/new-feature-alert-improve-your-vein-modelling-using-a-custom-reference-mesh/).
