# One-time Leapfrog SubDomainer sampling: failure analysis and next decision

Date: 2026-10-16
Private GitHub runtime run: https://github.com/Mining-Geologist/LF_testing/actions/runs/38974374992
Private artifact: one-time-domaining-sample (artifact ID 12078516896)
Research branch: research/structural-tensor-inversion-20261016

## Observed (not inferred)

- GitHub ran the opt-in passive py-spy sampler successfully on the Windows self-hosted runner.
- Profiler discovered 3 Leapfrog background process IDs and ran 582 successful stack-sampling attempts in the recording window.
- Sampler logged 0 matching `domaining.py` frames and stored 0 stack snapshots; the 385-byte JSON contains only the summary. No raw stacks were saved and no `SubDomainer` merge information can be reconstructed retrospectively.
- No py-spy invocation errors were reported.
- Therefore **the test did not recover the native SubDomainer merge sequence**. The absence of matched frames does **not** establish that no merging happened.
- The `LF_testing` inventory and preflight independently verified Windows and bundled Leapfrog Python 3.12 components, but they did not validate a domaining event recorder against the running program.

## Candidate explanations not distinguished by this experiment

- Merge occurred before the profiling loop sampled the relevant frame.
- Domain recomputation wasn't triggered in the exact processing stage our matcher targeted (e.g. cached structure or a different model update).
- Frames were executing under a different process, module path or call signature than the strict matcher anticipated.
- Python-level merge functions ran too briefly relative to the external profiler's sampling window, or work happened at a native layer.

The retained artifact cannot discriminate among these hypotheses. Calling this a successful native trace or attributing a precise root cause would be unjustified.

## Decision: independent offline work is next

Existing benchmark evidence:
- The saved domain tensor rule for one structural input is independently reproduced at machine precision on the asymmetric folded model.
- The candidate geometric-centroid two-stage precursor yields about 0.441 mean adjusted Rand index across 9 WolfPass parameter cases, versus saved Leapfrog partitions.
- In the independent asymmetric folded model its ARI is about 0.485.
- Split-only refinement of the current coarse partitions cannot reproduce native labels because saved native domains frequently cross coarse partition boundaries.
- Static pairwise anisotropy-consistency threshold connected components also fail.

Therefore:
1. Keep running RBF/LVA parity tests against owner-supplied saved benchmark data, without requiring an active Leapfrog instance.
2. Prioritize **cross-coarse domain merger/reassignment and dynamic region-growing hypotheses** over yet another fixed pairwise edge cutoff.
3. Test candidate clusterers on hold-out projects/parameter combinations; always report ARI, exact domain count, and whether the test uses any oracle domain labels.
4. Preserve known-correct structural tensor, per-domain interpolation and blend stages. Keep native-equivalent automatic domain labels explicitly experimental.
5. Do not run another live capture just to hope for favorable sampling. A future authorized runtime experiment must first produce a verifiable test event, log method-entry coverage without leaking arrays, prove process/module discovery, and require separate user approval before production-like use. Avoid silently enabling persistent hooks.

**No percent-progress increase** justified by this failed capture. The research target remains unchanged.
