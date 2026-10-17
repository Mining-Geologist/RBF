# Independent symmetric fold & exploratory local-graph coherence diagnostic (2026-10-16)

## Confirmed prospective test: frozen hybrid did not improve the third geometry

A previously exported owner-owned symmetric-fold project has 1,666 real modelling points and 14 saved local RBF domain cores. The corrected grid predicts 12 domains, ARI 0.487750629; the frozen hybrid predicts the same 12, ARI 0.487750629, accepting zero merges or births. Its 10% safeguard rejected an independently proposed ~22–24% reassignment.

This was genuinely a held-out geological **geometry**, different from WolfPass and the asymmetric fold. The reference is **persisted local RBF core membership**, not a verified exported categorical "Extract Clusters to Points" truth. There are 147 equal-distance orientation-mesh vertex ties, and two original endpoint coordinates are *inferred*; three prespecified tie rules did not alter the final frozen prediction. Raw saved RBF substitute endpoints instead changed ARI from 0.487750629 to 0.487385644.

The source-independent symbolic archive parser and grid/point clusterers were run offline; no Leapfrog binary, native testing or connection was used.

## Post-hoc diagnostic: the hard 10% safeguard is suppressing a useful correction on one fold

Because the frozen hybrid made no progress on either independent fold, we evaluated what **unrestricted real-point refinement** would have produced, changing **only the maximum-change guard** while keeping all other parameters fixed:

| Independent geometry | Grid ARI | Frozen 10%-guard hybrid | Unrestricted hybrid | Proposed changed fraction |
|---|---:|---:|---:|---:|
| Asymmetric fold, 1995 pts | 0.484990 | 0.484990 | **0.411325** (worse) | 21.40% |
| Symmetric fold, 1666 pts | 0.487751 | 0.487751 | **0.555580** (better) | 21.85% |

The cutoff alone cannot distinguish harmful and helpful refinements: both proposed similarly large changes.

A **reference-free local graph coherence** score is the distance-weighted proportion of k-nearest-neighbour pairs assigned to the same predicted domain, with weights exp(-distance / median graph-edge distance). It changed as follows:

| Geometry | Before refinement | Unrestricted refinement | Change |
|---|---:|---:|---:|
| Asymmetric fold | 0.8291 | 0.8187 | **-0.0104** |
| Symmetric fold | 0.8201 | 0.8590 | **+0.0389** |

This prompted a *new experimental hypothesis*: for large proposed reassignments (>10%), keep the candidate **only when the candidate raises spatial graph coherence**. Small proposals still follow the frozen hybrid.

Retrospective scores across the same **already inspected** eleven configurations:

| Group | Grid mean ARI | Frozen guarded hybrid | Exploratory coherence rule |
|---|---:|---:|---:|
| 9 WolfPass variants sharing geometry | 0.441261 | 0.463613 | 0.463613 |
| 2 independent folded geometries | 0.486370 | 0.486370 | 0.520285 |
| All 11 settings (not independent samples) | 0.449462 | 0.467751 | 0.473917 |

**The new coherence rule was invented after examining the two held-out fold outcomes. Therefore the 0.520285/0.473917 values are exploratory IN-SAMPLE results, not valid hold-out validation.** The S5_R50 WolfPass case still regresses (0.363328 -> 0.354446) even though coherence improves. Coherence is therefore not an oracle for source parity.

## Reproducibility and limits

- Third-project standalone reproducer: `benchmarks/leapfrog_gold/validate_saved_symmetric_holdout.py` (with authored symbolic saved-project parser and corrected grid prototype in the companion local research kit).
- Exploratory proposed gate: `benchmarks/leapfrog_gold/experimental_coherence_guard.py`.
- Synthetic source-independent tests: `test_experimental_coherence_guard.py` and Github CI; no private benchmark points checked into the repo.
- The user-downloadable research kit contains the case-by-case eleven-setting diagnostic CSV and all input-derived **aggregated** scores, but not the original saved project archive.

**Decision:** Leave `main` and released Jizai unchanged. Treat graph coherence as a plausible *unsupervised quality metric*, not a verified correct merger. The next proper test requires a fourth **genuinely unseen** geological geometry or a measurement with an independent physical/geological quality criterion defined *before* looking at reference domain labels. Further tuning on these same two folds risks overfitting.
