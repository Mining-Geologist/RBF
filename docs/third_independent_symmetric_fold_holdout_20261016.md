# Third independent fold: frozen hybrid SubDomainer test (2026-10-16)

**Outcome: neutral, not an improvement.** The frozen hybrid and the corrected six-neighbour grid both achieve ARI **0.487750629** on a newly evaluated, geometrically independent symmetric-fold project. The hybrid rejected a 23.53% proposed point reassignment and accepted zero domain merges; no domain was created. There was **no change to Jizai/main** and no execution of Leapfrog or the Windows runner.

## Source and trust limits

Input: user-owned, previously saved **Same project after adding LVA.zip** (WithLVA project). Independent of the WolfPass input geometry and the already tested asymmetric-fold LVA_Test2 geometry.

This project includes **1,666 modelling points**, an orientation mesh with **7,381 vertices**, **14 saved local RBF core-domain memberships**, strength 5, range 100, **147 points** with ambiguous equal-distance nearest structural vertices, and **two original endpoint XYZ coordinates inferred** from known drillhole spacing. Reference membership comes from the serialized local RBF domain core indices, **not** an independently verified user-facing "Extract Clusters to Points" export. This distinction matters: post-processing may alter apparent cluster sizes and labels.

Reference domain labels are assembled only to score predictions; neither the corrected grid builder nor `fit_hybrid` receives them. Frozen parameters are exactly those reported in `docs/offline_hybrid_real_point_merge_20261016.md`.

## Primary outcome

| Measurement | Grid | Frozen hybrid |
|---|---:|---:|
| Number of predicted domains | 12 | 12 |
| Saved domain count | 14 | 14 |
| ARI vs saved memberships | 0.487750629 | 0.487750629 |
| NMI vs saved memberships | 0.673499237 | 0.673499237 |
| Accepted real-point domain merges | — | 0 |
| Accepted domain births | — | 0 |
| Point-refinement proposal | — | 392/1666 points, rejected by 10% change guard |

The grid shape was 30 × 15 × 14 centroids. A changed candidate from the refinement would have altered 23.53% of memberships; the gate deliberately kept the grid result. This is conservative behaviour, **not evidence of better source parity**.

## Prespecified uncertainty/sensitivity checks (do not tune on results)

Three structural-vertex tie decisions were evaluated: SciPy KD-tree default, smallest eligible vertex index, and largest eligible vertex index. All three produced **the same final ARI (0.487750629), 12 domains, zero merges and a rejected refinement**. The proposed fraction moved varied between 0.2173 and 0.2395, confirming the unaccepted candidate is sensitive to tie handling even while final labels are stable. None of the geometric centroids were tied; 147 real points were tied.

Using the **raw stored RBF seed XYZ** instead of two independently unverified inferred endpoint XYZ gave an ARI of **0.487385644** for both grid and frozen hybrid. Change: -0.000365. Caution: raw RBF substitute seeds are not validated original geological input coordinates; this is a sensitivity check, not an alternative authoritative benchmark.

## Updated evidence across distinct geometries

| Input geometry | Corrected grid ARI | Frozen hybrid ARI | Improvement |
|---|---:|---:|---:|
| WolfPass (nine parameter settings on shared geometry) | 0.441261 mean | about 0.4636 mean | Yes on aggregate; not all settings |
| Asymmetric fold (1995 points) | 0.484990 | 0.484990 | **None**, guard rejected |
| Symmetric fold (1666 points) | 0.487751 | 0.487751 | **None**, guard rejected |

Thus the hybrid's improvements are **restricted to parameter variants of one geometry**. There is not yet evidence that the hybrid generalizes better than the base grid across independent geological inputs. Keep it experimental and do not merge.

## Reproduction

The standalone offline script `benchmarks/leapfrog_gold/validate_saved_symmetric_holdout.py` reads only previously saved user-owned project files, together with separate research-authored, safe symbolic archive parsers and the corrected automatic builder, available in the downloadable offline reproduction kit. It computes all 3 tie-policy variations × 2 endpoint variants and writes a JSON results file. Private model arrays and archives are not committed. Dependencies: numpy, scipy, scikit-learn, thrift, local zstd runtime. No native Leapfrog libraries or licenses are required.

Follow-up should target why *both independent folds* trigger >10% proposed reassignments, using reference-free geometric instability diagnostics. Do not simply loosen the guard because that degraded the asymmetric-fold case from 0.4850 to 0.4113 in earlier experiments.
