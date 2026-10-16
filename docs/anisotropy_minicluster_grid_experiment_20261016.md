# Offline anisotropy-aware mini-cluster test — 2026-10-16

No Leapfrog process, native runtime, license connection, or Windows runner used.

## Method
Starting from the established 6000-centroid (6272 on independent fold), six-face-neighbour grid and pointwise structural anisotropy matrices, form connected mini-clusters by initially ranking face-adjacent edges on reciprocal-determinant matrix consistency. Accept an edge only when the dynamically updated merged aggregate reaches a specified consistency, and when the merged mini-cluster has no more than 2, 4, or 8 centroids. Some tests restrict these premerges to aligned 2x2x2 tiles. Then conduct exactly the existing global greedy adjacency-aware merge with threshold 0.6 and maximum fraction 0.10; map real training points to their grid cells.

The model never accesses true/reference labels during inference; ARI is computed afterward. Configuration variants were fixed before evaluation on the four held-out cases. Unlike a dynamic real-data subclustering stage, this is a premerge experiment on grid centroids, not a verified recreation of Leapfrog.

## Scores

| Mode | Development mean ARI (six cases) | Held-out mean ARI (three WolfPass + independent fold) | Ten-case mean |
|---|---:|---:|---:|
| Existing six-neighbour baseline | 0.454668 | 0.432082 | 0.445634 |
| Mini-cluster at most 2, consistency >= .995 | 0.420671 | 0.433086 | 0.425637 |
| Mini-cluster at most 4, consistency >= .995 | 0.454111 | 0.466290 | 0.458982 |
| Mini-cluster at most 4, consistency >= .995, local 2-cell tiles | 0.441494 | 0.455895 | 0.447255 |
| Mini-cluster at most 8, consistency >= .99, local 2-cell tiles | 0.400583 | 0.452549 | 0.421369 |

Independent folded model: existing baseline 0.484990; best ten-case candidate (four-cell non-tiled) 0.477759, a small decrease. Therefore the ten-case mean improvement should **not** be interpreted as convincing independent generalization. The held-out WolfPass parameter settings share geometrical input data with the development settings.

Individual large gains: S3_R300 improves 0.425568->0.478252 with four-cell; S5_R300 improves 0.534997->0.572832. Important regressions: S2_R100 0.464231->0.388793; S5_R100 0.435910->0.418490.

## Conclusion

Mini-cluster grouping measurably changes subsequent greedy merge ordering, in both beneficial and harmful directions. The six-neighbour baseline continues to be the recommended stable implementation; new result belongs on research branch only. The next step is occupancy-aware *real input point* microclusters, and an explicitly updated domain adjacency graph. Test against geometrically independent models. Do not tune using reference K or fold oracle labels.
