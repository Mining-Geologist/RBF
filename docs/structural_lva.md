# Structural LVA prototype

The `feature/structural-lva` branch introduced a native Polatory structural-LVA
path. The `feature/leapfrog-parity` branch adds an exact post-clustering path
recovered from the supplied WolfPass Leapfrog benchmark.

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
structural.fit(points, values, domains, tolerance=1e-6)
predictions = structural.evaluate(query_points)
```

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
3. multiple-input `BLENDING` orientation and strength equations;
4. global mean trend and compatibility-version interactions.

The former fixed `0.8 * base_range` local range and axis-aligned-box support
heuristics are not Leapfrog rules and should not be used for parity claims.
The current `StructuralInterpolant3` box smoothstep weighting also remains an
experimental approximation until the overlap function is identified.

## Validation strategy

Use two separate benchmarks:

- **Oracle-label benchmark:** feed the decoded final labels to
  `LabeledStructuralDomainBuilder3`. Any remaining mesh difference is caused by
  the local solver, blending or isosurface extraction—not clustering.
- **Automatic benchmark:** generate labels from the replacement SubDomainer and
  compare label partitions first, then run the same exact post-cluster builder.

A parity claim requires one unchanged implementation to pass every supplied
strength/range case, including the 50 m cutoff cases, without case-specific
parameter tuning.
