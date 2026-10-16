# Offline real-point SubDomainer experiment (2026-10-16)

**Result: direct clustering of actual input points underperformed the existing geometric-grid baseline.** All ten evaluations were run offline against previously exported user-owned references, without launching Leapfrog, changing an installed program, or using the Windows runner.

## Hypothesis and method

Replace the 6,000-centroid geometric grid with the original XYZ drillhole samples themselves. Build a symmetric 32-nearest-neighbour graph; assign every sample its independently reconstructed local SPD anisotropy matrix; dynamically build connected microclusters of at most four points. Then greedily merge adjacent regions using reciprocal determinant of the population-weighted mean anisotropy matrix, recomputing that matrix and the region-adjacency graph after each union. Cap final domain size at 15% of input samples and use the independently configured 0.6 consistency threshold.

The parameters were selected using six WolfPass development configurations. Saved Leapfrog reference memberships were consulted **only to score the results after prediction**. The chosen configuration was then frozen for three withheld WolfPass configurations and an independent geometrically different asymmetric-fold model.

## Results: adjusted Rand index (higher is better)

| Case | Grid baseline | Real-point hypothesis | Real-point K |
|---|---:|---:|---:|
| S2_R50 (development) | 0.388150 | 0.382744 | 9 |
| S2_R100 (development) | 0.464231 | 0.436364 | 8 |
| S3_R100 (development) | 0.479154 | 0.455985 | 9 |
| S3_R300 (development) | 0.425568 | 0.457471 | 9 |
| S5_R100 (development) | 0.435910 | 0.415731 | 9 |
| S5_R300 (development) | 0.534997 | 0.538989 | 8 |
| **Development mean** | **0.454668** | 0.447881 | — |
| S3_R500 (held-out) | 0.460857 | 0.472858 | 9 |
| S5_R50 (held-out) | **0.363328** | 0.291382 | 9 |
| S5_R500 (held-out) | 0.419151 | **0.484192** | 9 |
| Independent fold (held-out) | **0.484990** | 0.463614 | 8 |
| **Held-out mean** | **0.432082** | 0.428012 | — |
| **Overall mean** | **0.445634** | 0.439933 | — |

The three held-out WolfPass settings slightly improved on average, but those cases share input geometry with development. The independent fold **worsened** by 0.021376 ARI; four out of ten cases improved while six regressed.

The algorithm is mathematically valid but does not show convincing equivalence to native Leapfrog domain formation. The poorer result may reflect how broad real-point k-nearest-neighbour edges cross data gaps and how synthetic centroid population constraints differ from real-point population limits.

## Decision

**Keep the existing grid-based baseline. Do not merge the experimental real-point clusterer into main/Jizai.** A more promising follow-up is to preserve the grid topology for stage one and use real data occupancy to guide stage-two region growth and restricted cross-grid merging. This follow-up has not been evaluated in this experiment.

Research source:
- `benchmarks/leapfrog_gold/real_point_domain_graph.py`
- `benchmarks/leapfrog_gold/test_real_point_domain_graph.py`
- `.github/workflows/offline-real-point-domaining.yml`

The original project archives, points, and reference labels remain offline and are not committed.
