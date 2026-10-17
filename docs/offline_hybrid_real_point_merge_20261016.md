# Offline hybrid grid + real-point SubDomainer experiment

**2026-10-16. Research only. No Leapfrog application, Windows runner, licensing services, or native runtime used. No changes to `main`.**

## Method

The new hybrid preserves the corrected geometric grid as an initial, independent clustering stage. After point-to-grid domain assignment, it (1) applies the previously validated point-graph boundary refinement, with an observable 10% maximum reassignment safeguard; (2) constructs a symmetric 8-nearest-neighbour graph over actual drillhole points; (3) merges neighbouring real-point domains in best-first order if both have strong boundary support and compatible mean local anisotropy, and if at least one domain is small. It updates domain membership, tensor averages and the graph between every merge. An optional anisotropy-aware disconnected-component domain-birth routine is implemented, but **disabled** for the evaluated candidate.

The frozen candidate uses normalized determinant-1 local SPD tensors, reciprocal determinant of weighted group-average tensors, minimum score **0.99**, minimum contact fraction **0.10**, smaller-domain population at most **5% of total point count**, maximum merged-domain population **35%**, and **8** point neighbours. Configurations were screened on six development cases before scoring the three held-out WolfPass settings and one distinct folded geometry. Reference labels never enter prediction.

## Benchmark results (adjusted Rand index)

| Model | Set | Grid | Hybrid | Grid domains → Hybrid domains |
|---|---|---:|---:|---|
| S2_R50 | development | 0.388150 | 0.401101 | 12 → 10 |
| S2_R100 | development | 0.464231 | 0.502643 | 11 → 8 |
| S3_R100 | development | 0.479154 | 0.502965 | 10 → 9 |
| S3_R300 | development | 0.425568 | 0.452446 | 10 → 8 |
| S5_R100 | development | 0.435910 | 0.437542 | 10 → 9 |
| S5_R300 | development | 0.534997 | 0.542257 | 10 → 10 |
| S3_R500 | held-out WolfPass | 0.460857 | 0.550336 | 8 → 7 |
| S5_R50 | held-out WolfPass | 0.363328 | 0.354446 | 11 → 11 |
| S5_R500 | held-out WolfPass | 0.419151 | 0.428780 | 10 → 10 |
| Asymmetric fold | independent held-out | 0.484990 | 0.484990 | 13 → 13 |
| **Development mean (6)** | | **0.454668** | **0.473159** | |
| **Held-out mean (4)** | | **0.432082** | **0.454638** | |
| **All ten mean** | | **0.445634** | **0.465751** | |

Overall gain: **+0.020117 ARI**. The four-case held-out gain of +0.022556 is dominated by S3_R500 (+0.089479), which shares its underlying geometry with the development WolfPass cases. The independent fold does **not** improve; its proposed point reassignment was too large, so the guard retained its grid baseline. One held-out case (S5_R50) still regresses by 0.008882. No candidate domains were born in the evaluated runs (the domain-birth module was tested on synthetic examples only).

## Reproduction

Use `benchmarks/leapfrog_gold/hybrid_real_grid_domainer.py` together with `dynamic_domain_refinement.py`. For local owner-exported `*.npz` cases containing `points`, `tensors`, `baseline`, `truth`, run:

`python benchmarks/leapfrog_gold/benchmark_hybrid_real_grid.py --case-dir local_case_cache --out hybrid_results.csv`

The command derives new cluster labels from input XYZ, anisotropy and existing grid labels before reading the saved reference membership for scoring. CI synthetic tests exercise boundary merging, optional domain birth, determinism, population rules and input validation. Original modelling point arrays and reference project files are not committed.

## Decision

This is the strongest **ten-case mean** obtained by the offline two-stage experiments so far, but it does not improve the only geometrically independent held-out model, so it remains experimental and is **not merged into Jizai or the `main` branch**. Next scientific validation requires a truly new geometry where both pointwise tensor reproduction and saved target memberships are available; stop treating additional variants of the same WolfPass geometry as fully independent confirmation.
