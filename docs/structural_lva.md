# Structural LVA prototype

The `feature/structural-lva` branch introduced a native Polatory structural-LVA
path. The `feature/leapfrog-parity` branch added exact value preprocessing and
post-clustering construction recovered from the supplied WolfPass Leapfrog
benchmark. The `feature/automatic-subdomainer` branch adds a standalone,
deterministic geometric-centroid SubDomainer that does not require decoded
Leapfrog point labels.

## Standalone automatic Python API

```python
import numpy as np
import polatory
from polatory import three as p3

rbf = p3.CovSpheroidal3([100.0, 400.0])
model = p3.Model(rbf, 0)
model.nugget = 0.0

trend_input = polatory.StructuralTrendInput3(
    vertices=mesh_vertices,
    faces=mesh_faces.astype(np.int64),
    strength=5.0,
    range=100.0,
)

value_info = polatory.leapfrog_indicator_values3(points, sdf_indicators)

builder = polatory.AutomaticStructuralDomainBuilder3(
    centroid_count=6000,
    minimum_cluster_fraction=0.001,
    maximum_cluster_fraction=0.10,
    consistency_threshold=0.60,
    base_range=400.0,
)
domains = builder.build_from_inputs(
    points,
    [trend_input],
    model_parameters=np.asarray(model.parameters, dtype=float).tolist(),
)

structural = polatory.StructuralInterpolant3(
    model,
    outside_value=-1.0,
    blend_power=1.0,
)
structural.fit(
    points,
    value_info.values,
    domains,
    tolerance=value_info.fit_accuracy,
)
predictions = structural.evaluate(query_points)
```

The automatic builder contains no benchmark coordinates, case names, domain
counts or precomputed point labels. Its pipeline is:

1. factor the requested centroid count into a data-aspect-ratio-aware grid;
2. sample the structural LVA matrix at every centroid;
3. connect face-neighbouring grid cells;
4. greedily merge adjacent cells in decreasing SPD-matrix consistency order;
5. enforce the requested minimum and maximum core populations;
6. assign a canonical automatic label to every interpolation point;
7. pass those labels to `LabeledStructuralDomainBuilder3` for the recovered
   support, local-range, representative-matrix and bounding-box rules.

The complete runnable example is
`examples/standalone_automatic_lva.ipynb`.

## Oracle-label Python API

The oracle path remains useful for isolating errors after automatic clustering:

```python
builder = polatory.LabeledStructuralDomainBuilder3(base_range=400.0)
domains = builder.build_from_inputs(
    points,
    labels,
    [trend_input],
    model_parameters=np.asarray(model.parameters, dtype=float).tolist(),
)
```

## Recovered binary indicator values

Leapfrog does not pass the imported `+1/-1` SDF column directly to FastRBF.
For every point it calculates the Euclidean distance to the nearest point in
the opposite class and reverses the indicator sign:

```text
raw_value_i = -sign(indicator_i) * nearest_opposite_distance_i
```

Let `D` be the diagonal length of the interpolation-data bounding box. The
automatic scales in this benchmark are exactly:

```text
fit_accuracy = 1e-5 * D
value_clip   = 1e-2 * D = 1000 * fit_accuracy
value_i      = clip(raw_value_i, -value_clip, value_clip)
```

For WolfPass, `D = 1903.0467631734136`, therefore Leapfrog fits with
`0.019030467631734136` accuracy and clips the signed values at
`+/-19.030467631734136`. The values reconstructed independently from all local
serialized FastRBF solutions agree to about `1e-11`.

`leapfrog_indicator_values3` implements this transform. This correction is
important: fitting the original `+1/-1` column produces a materially different
zero surface even when the structural domains are otherwise exact.

## Recovered single-input field

For nearest equal-weight mesh-vertex normal `n`, distance `d`, input strength
`s`, and input range `R`:

```text
q = exp(-d / R),  d < 4 R
q = 0,            d >= 4 R
r = 1 + (s - 1) q
M = r^(-1/3) (I - n n^T) + r^(2/3) (n n^T)
```

The hard cutoff at exactly four input ranges is required to reproduce the two
50 m range benchmarks. Omitting it leaves a weak non-zero trend in Leapfrog
regions that are exactly isotropic.

## Exactly recovered local spheroidal kernel

The serialized local solutions use FastRBF `asphere` with order `-3`. Written
using its stored internal radius `a`, the unit-sill covariance is exactly the
same kernel as Polatory `CovSpheroidal3`:

```text
u = distance / a
phi(u) = 1 - 0.75 u                         for u < 0.5
phi(u) = 0.8734640537108553/(1+u^2)^(3/2)  otherwise
```

The corresponding Polatory range is
`a * sqrt(7.181510581693163)`. Evaluating the serialized coefficients with this
formula recovers one common training-value vector from all overlapping local
solutions to numerical precision. This confirms the kernel equation and range
conversion independently of the final meshes.

## Exactly recovered post-cluster construction

Once the final point labels are known, the local domain construction is
recovered for all 82 domains across the nine WolfPass strength/range cases.

For one cluster with core point index set `C`:

1. Sample `M_i` at every interpolation point using the field above.
2. Use the arithmetic matrix mean
   `M_C = mean(M_i for i in C)`.
3. Let `lambda_max` be the largest eigenvalue of `M_C`.
4. For a user-facing CovSpheroidal3 base range `B`, define
   `desired_local_range = B * lambda_max`.
5. Convert to Leapfrog/FastRBF's internal radius using
   `a_desired = desired_local_range / sqrt(7.181510581693163)`.
6. Transform all points by `x -> x M_C` and calculate each point's distance to
   its nearest transformed core point.
7. Support is the strict set `distance < a`. If this would contain more than
   `5 * |C|` points, reduce `a` to the first excluded order statistic so the
   five-times-core support cap is respected.
8. The actual local Polatory kernel range is
   `a * sqrt(7.181510581693163)`.
9. Expand the core point AABB by
   `a * row_norm(inv(M_C))` on each world coordinate axis.

The reconstructed support membership is exact for every benchmark domain. The
maximum comparison errors against the serialized Leapfrog interpolants are
approximately:

- average matrix: `8.9e-16`;
- internal support radius: `2.2e-8`;
- blending/culling box: `3.6e-8`;
- transformed local centres mapped back to input points: `7.0e-10`.

`LabeledStructuralDomainBuilder3` implements these rules. The automatic builder
uses it internally after generating labels, so there is only one implementation
of the recovered post-cluster rules.

## Parity status and validation boundary

The supplied serialized objects preserve final point assignments but not the
temporary geometric leaf/neighbour queues. The automatic implementation is
therefore a deterministic reconstruction of the observed 6000-centroid,
adjacency-constrained architecture, not a copy of hidden Leapfrog source code.

The following items still require empirical regression across independent
Leapfrog exports before universal exact-parity can be claimed:

1. exact tie-breaking in geometric centroid partition and adjacent merge order;
2. the exact local-function blend weight in overlap regions;
3. the relationship between requested fit accuracy and FastRBF stopping;
4. multiple-input `BLENDING` orientation and strength equations;
5. global mean trend and compatibility-version interactions.

Use two separate validation layers:

- **Oracle-label benchmark:** feed decoded final labels to
  `LabeledStructuralDomainBuilder3`. Remaining mesh difference is caused by the
  local solver, blending or isosurface extraction.
- **Automatic benchmark:** generate labels using
  `AutomaticStructuralDomainBuilder3`, compare partitions first, then compare
  the final scalar field and surface.

A strict parity claim requires one unchanged automatic implementation to pass
every supplied strength/range case, including the 50 m cutoff cases, without
case-specific parameter tuning, followed by held-out datasets not used during
reconstruction.
