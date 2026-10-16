# Offline voxel-neighbourhood falsification (2026-10-16)

**Result:** changing a corrected 6000-centroid geometric merge grid from 6-connected to 18-/26-connected adjacency does **not** improve overall agreement with saved reference domain memberships. The tested 6-connected model remains the best of these three fixed rules.

The controlled test kept anisotropy fields, exact grid dimensions, matrix aggregation, greedy dynamic merge priorities, consistency cutoff 0.6, centroid population limits, and final point assignment fixed. It varied only the topological connectivity between neighbouring centroid cells.

| Model | 6-face ARI | 18-face+edge ARI | 26-all ARI |
|---|---:|---:|---:|
| S2_R50 | 0.388150 | 0.395324 | 0.353814 |
| S2_R100 | 0.464231 | 0.435233 | 0.435233 |
| S3_R100 | 0.479154 | 0.377387 | 0.377387 |
| S3_R300 | 0.425568 | 0.464129 | 0.464129 |
| S3_R500 | 0.460857 | 0.476316 | 0.476316 |
| S5_R50 | 0.363328 | 0.362093 | 0.360869 |
| S5_R100 | 0.435910 | 0.447797 | 0.447797 |
| S5_R300 | 0.534997 | 0.456368 | 0.456368 |
| S5_R500 | 0.419151 | 0.476530 | 0.476530 |
| Asymmetric fold (independent) | 0.484990 | 0.443513 | 0.443513 |
| **Mean, 9 WolfPass** | **0.441261** | 0.432353 | 0.427605 |
| **Mean, all 10** | **0.445634** | 0.433469 | 0.429195 |

Six development cases: mean 6=0.454668, 18=0.429373, 26=0.422455.
Four held-out cases: mean 6=0.432082, 18=0.439613, 26=0.439307.
The fold case is a genuinely different geometry, and it became worse with both diagonal-connectivity variants. Improvements on the withheld WolfPass parameter settings do not compensate for degradation elsewhere.

The 6000-centroid grid shape for WolfPass is (16,25,15): edge counts 16985 (6), 49037 (18), 69197 (26). Fold shape: (32,14,14), 6272 seeds: edges 17724, 51108, 72064.

The **same numeric candidate** across models can change merge order when extra edges are added, making the method quite sensitive to early greedy decisions. Static 18/26 topology is not an appropriate automatic default. More important next hypotheses: anisotropy-filtered adjacency, better mini-cluster construction, and recomputed real-point merging across grid partition boundaries.

Method: owner-exported archived labels were read offline; no active Leapfrog executable was launched. Exact reference labels were used only to compute ARI after each predicted clustering. Original geometry is not committed. Source-only adjacency implementation is in `benchmarks/leapfrog_gold/grid_topology_edges.py`; isolated adjacency validation is in `test_grid_topology_edges.py`. A reproducible benchmark loader and full 30-row CSV are also included in the accompanying local research package. Production source and `main` are unchanged.