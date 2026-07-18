# Structural LVA prototype

The `feature/structural-lva` branch introduced a native Polatory structural-LVA
path. The `feature/leapfrog-parity` branch adds exact value preprocessing and
post-clustering construction recovered from the supplied WolfPass Leapfrog
benchmark.

## Python API

```python
import numpy as np
import polatory
from polatory import three as p3

rbf = p3.CovSpheroidal3([100.0, 400.0])
model = p3.Model(rbf, 0)

trend_input = polatory.StructuralTrendInput3(
    vertices=mesh_vertices,
    faces=mesh_faces.astype(np.int64),
    strength=5.0,
    range=100.0,
)

# Leapfrog does not fit +/-1 indicators directly.
value_info = polatory.leapfrog_indicator_values3(points, sdf_indicators)

# labels contains one recovered or externally generated final cluster ID per
# interpolation point.
builder = polatory.LabeledStructuralDomainBuilder3(base_range=400.0)
domains = builder.build_from_inputs(
    points,
    labels,
    [trend_input],
    model_parameters=[100.0, 400.0],
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

Once the final point labels are known, the local domain construction is now
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

`LabeledStructuralDomainBuilder3` implements these rules. It is deliberately
separate from automatic clustering so interpolation and blending can be tested
against Leapfrog without conflating them with the SubDomainer partition.

## What remains before full automatic parity

The remaining unknowns are now limited to:

1. Leapfrog's initial geometric mini-cluster grid and deterministic adjacent
   merge ordering;
2. the exact local-function blending weight in overlap regions;
3. the exact relationship between Leapfrog's requested fit accuracy and the
   stopping behaviour of its approximate FastRBF solver;
4. multiple-input `BLENDING` orientation and strength equations;
5. global mean trend and compatibility-version interactions.

The former fixed `0.8 * base_range` local range and axis-aligned-box support
heuristics are not Leapfrog rules and should not be used for parity claims.
The current `StructuralInterpolant3` box smoothstep weighting also remains an
experimental approximation until the overlap function is identified.

## Validation strategy

Use two separate benchmarks:

- **Oracle-label benchmark:** feed the decoded final labels to
  `LabeledStructuralDomainBuilder3`. Any remaining mesh difference is caused by
  the local solver, blending or isosurface extraction—not clustering or value
  preprocessing.
- **Automatic benchmark:** generate labels from the replacement SubDomainer and
  compare label partitions first, then run the same exact post-cluster builder.

A parity claim requires one unchanged implementation to pass every supplied
strength/range case, including the 50 m cutoff cases, without case-specific
parameter tuning.
