# Public clues for offline structural-domain research (2026-10-16)

## High-confidence documentation
Seequent 2026.1 explicitly describes (i) proximity/orientation-based small clusters formed from input points, (ii) optional fixed three-dimensional centroid partitioning for non-triaxial trend types, and (iii) adjacent cluster agglomeration subject to point-count bounds and anisotropy consistency. Cluster membership depends on the model's input data and is not just a property of the trend mesh.
https://help.seequent.com/Geo/2026.1/en-GB/Content/concepts/structural-trends-intro.htm
https://help.seequent.com/Geo/2026.1/en-GB/Content/structural-data/structural-trends.htm

## Numerical clue from official release-note UI
A consistency threshold input 0.8 is displayed as 97.79% merge consistency (2025.1).
A threshold input 0.6 is displayed as 95.02% merge consistency (2025.3).
These match 100*(input)**(1/10) to two decimals. This appears to be an exact UI transformation, but it is NOT evidence for a particular tensor statistic or merge-ordering rule.
https://files.seequent.com/PDFs/Leapfrog2025.1MajorRelease_NotesGeo_ENG_v4.pdf
https://files.seequent.com/PDFs/Leapfrog-Geo-2025.3.1-Point-Release-Notes-ENG.pdf

## First offline falsification experiments
1. Audit the current 6000-centroid grid implementation: rectilinear cell counts, offsets, face/edge/corner neighbor rules, treatment of empty cells, and how real XYZ are assigned to grid leaves. Report sensitivity of partition scores without selecting heldout hyperparameters.
2. Implement a second adaptive minimum-size micro-cluster layer within geometric cells before any global merge; compare KD/octree, nearest-centroid and simple size-limited flood fill. Avoid fixed oracle K.
3. On the resulting region adjacency graph, update a cluster's anisotropy aggregate and point count after every candidate union. Reject unions greater than the configured maximum. Compare greedy priorities (highest compatibility, smallest region first, closest graph-distance).
4. Compare displayed consistency conversion with the historic 1/det(avg SPD) hypothesis: applying x**0.1 to both threshold and score should not alter decisions if the same monotonic transform is used; if decisions differ, there is an inconsistency in score handling.
5. Validate boundary topology across all nine archived WolfPass settings and geometrically independent folds. Track ARI, max/min domain population violations, connectivity, and domain count; do not use true labels to select clustering decisions.

## Related algorithms: inspiration, NOT evidence of Leapfrog internals
- Martin and Boisvert, 2017, iterative refinement of implicit boundary models with local anisotropy and partition of unity: https://doi.org/10.1016/j.cageo.2017.07.003
- Cavoretto et al., 2018, anisotropic RBF partition-of-unity patches: https://arxiv.org/abs/1811.05193
- Felzenszwalb and Huttenlocher, 2004, dynamic graph-region merge criterion: https://www.cs.cornell.edu/~dph/papers/seg-ijcv.pdf

Old Seequent version-2 documentation additionally mentions retained clusters on small dataset updates; stable geometric cell-based IDs may matter for exact reproducibility:
https://help.seequent.com/Geo/4.1/en-GB/Content/concepts/trends.htm

All above should be tested offline using saved user-exported benchmark artifacts. No further program instrumentation is requested.
