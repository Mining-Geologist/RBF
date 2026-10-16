# Structural nearest-vertex ties and independent clustering falsification

Date: 2026-10-16. Research branch: research/structural-tensor-inversion-20261016.

## A. Exact tie-decision inverse from real saved matrices

User-provided original symmetric-fold Leapfrog archive; 1,666 stored RBF seeds; 7,381 saved structural-mesh vertices; 14 original SubDomainer domain-average 3x3 tensors; one input mesh, strength 5, decay range 100. No proprietary source code or project archive is committed.

The previously independently validated pointwise tensor is:

    M_i = r_i^(-1/3) I + (r_i^(2/3) - r_i^(-1/3)) n_i n_i^T
    r_i = 1 + (5 - 1) exp(-d_i / 100),   d_i < 400
    r_i = 1 otherwise

For each of 147 input locations with two nearest mesh vertices tied within 1e-8, construct tensors M_i(0), M_i(1). All other locations keep the unique nearest mesh vertex. Saved domain means supply an exact discrete constraint, **not an independently predicted label**:

    sum_{i in domain} M_i(native_choice_i) = |domain| * saved_average_mtx

Exhaustive subset enumeration for domains with at most 18 ties used all nine matrix entries with a Frobenius sum-residual cutoff of 1e-8.

| Saved domain ID | Tied rows | Compatible binary selections | Identifiable individual choices |
|---|---:|---:|---:|
| 4 | 5 | 1 | 5 |
| 7 | 18 | 48 | 7 |
| 210 | 7 | 1 | 7 |
| 211 | 7 | 1 | 7 |
| 2 | 60 | not enumerated | undetermined |
| 3 | 50 | not enumerated | undetermined |

All eight tie-free domains reconstruct exactly with canonical points. **26 individual nearest-vertex choices are uniquely constrained** in four small domains. A further 11 choices within domain 7 are not individually identifiable from its saved mean, despite 48 completely valid 18-bit solutions.

This discovery does **not depend on the two previously inferred end-of-list coordinate corrections**: those two rows belong to saved domains 5 and 9, whereas all 26 uniquely identified choices are in domains 4, 7, 210 and 211.

The domain 210 and 211 selections consistently favour opposite sides of the fold in their own small point groups, including cases sharing x/y but differing in z. These original saved tensors therefore impose real tie-policy constraints instead of just an overall surface error.

### No simple known-neighbour policy reproduces these selections

Tested 80 configurations of SciPy cKDTree (leafsize 1..1000, balanced/unbalanced, compact/uncompacted), 38 scikit-learn KDTree/BallTree configurations, plus a baseline selection. **No configuration reproduced all 26**. The best returned **16/26**.

Limitations: this is a score against 26 **reference-derived** selections, not held-out independent validation. The saved means do not identify a unique global native nearest-neighbour algorithm. Arbitrarily fitting a custom selection rule to these 26 choices would be overfitting.

Reproducible diagnostic: `benchmarks/leapfrog_gold/infer_native_tie_choices.py` accepts a locally supplied private NPZ and deliberately leaves non-identifiable choices unresolved.

## B. Clustering generalization on an independent asymmetric-fold project

Tested the unchanged previously corrected geometric-centroid prototype in `leapfrog_automatic_domain_builder_fixed.py` on user-owned `LVA_Test2.aproj_data.zip`. There are 1,995 saved original input positions, 8,601 mesh vertices, 15 saved Leapfrog domains and no tied nearest-vertex input points (within 1e-8). Recovered single-mesh structural tensors match all 15 saved matrices to about 2.75e-16.

Kept prototype parameters fixed: centroid_count=6000, min fraction=0.001, max fraction=0.1, consistency=0.6, six-adjacent greedy centroid merge. Point coordinates corrected using independently supplied exact original input table rather than two saved RBF bounding-box replacements.

| Requested centroids | Actual grid | Prototype domains | Leapfrog domains | Adjusted Rand Index |
|---:|---:|---:|---:|---:|
| 1000 | 1152, 18x8x8 | 14 | 15 | 0.455059327 |
| 6000 | 6272, 32x14x14 | 13 | 15 | 0.484990200 |

For 6000 requested centroids, **12 of 15** saved native domains occupy more than one prototype coarse group, accounting for **91.58%** of the 1,995 native input points. Therefore no algorithm that *only splits* these coarse prototype groups can reproduce the native partition exactly.

Taking the oracle labels and splitting every coarse group at true-label boundaries yields ARI **0.632018279**. This is a reference-assisted diagnostic comparison, **not a mathematically established upper bound** on ARI under all possible coarse refinements. It is not deployable and does not resolve cross-boundary merging.

Results are independent of the 9 WolfPass model variants previously used to study clustering (average original-stage ARI approximately 0.441).

## C. Implications / remaining experiments

1. Nearest-mesh-vertex normal evaluation is numerically demonstrated for unambiguous single-mesh points; the exact native tie resolver is still missing.
2. Saved domain averages identify 26 native choices but do not explain the native spatial-query data structure. Capturing the real orientation evaluator's **per-point chosen vertex indices** in a controlled project would discriminate the remaining tie mechanism.
3. Automatic domain construction must recover the initial grid/leaf process and **cross-coarse merging or reassignment**; splitting a 6000-centroid graph alone is insufficient on both WolfPass and this independent asymmetric fold.
4. Direct capture of actual second-stage candidate edges, heap invalidations, and merge decisions is higher-information than fitting partition heuristics on saved final labels.
5. Keep the quantitative overall-reconstruction estimate near 73%; these results narrow uncertainty without delivering an exact automatic clustering implementation.

No original Leapfrog executable, copied proprietary source code, or user-owned project archives are stored here.
