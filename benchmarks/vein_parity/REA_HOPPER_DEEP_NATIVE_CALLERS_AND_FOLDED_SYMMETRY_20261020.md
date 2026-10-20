# REA/Hopper deep caller analysis and original folded-input invariance — 20 October 2026

**Owner-authorized read-only reverse engineering.** Static analysis of owner-provided commercial Leapfrog PE took place on owner-private connected GitHub runner using **official REA 6.7.0 and signed Hopper 6.6.0 demo**, and an independent standard-library read-only SQLite/pickle-opcode audit of the owner's saved project. No original commercial code, binary or proprietary geological coordinates have been copied into this public repository. Main branches untouched.

## New Hopper native-function evidence

Private successful REA/Hopper run **39407261382** decompiled:
- `0x1402cbe00`, a low-level point/mesh support helper: **10** CFG basic blocks, **7** identified callers, **1** callee. Callers include the previously verified `0x1402d4e80` point-based surface-offset initializer, plus `0x1402d8e40`, `0x1403fa2a0`, `0x1401d4e30`, `0x140200c10`, `0x1402ca730`, and `0x1402cbf20`. Some callers are outside the original offset initialization path, supplying new targets to trace.
- `0x1402d4160`, reference-mesh initialization: **74** basic blocks, **3** callers (`0x1402d3e60`, `0x1402d4600`, `0x1402d4e80`), **8** callees, including two additional geometry helpers `0x14032f510` and `0x14032f550`.

Private successful follow-up REA/Hopper run **39408058647** analyzed two previously uncovered calls:
- `0x1402d8e40`: **11** blocks, **2** callers (`0x14021c120`, `0x14021c310`), **3** callees (`0x14028f5f0`, `0x1402cbe00`, `0x1402d2c30`).
- `0x1403fa2a0`: **65** blocks, **1** caller (`0x1402d4160`), **9** callees including `0x1402cbe00` and multiple mesh helpers.

The new call relationships expand the static target graph, but **do not establish** that any is an `engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock` constructor: the identified routines are still reachable from previously known *offset initialization* functions. Do not interpret such a shared helper as the folded-reference algorithm without a connection to the upstream saved block.

Private follow-up **39408971773** checked Win64 static Cython PyMethodDef pointer registrations for two unexplored callers. It found **no valid direct PyMethodDef** for `0x14021c120`; a pointer to `0x14021c310` exists, but the surrounding candidate is not a valid Python method name/flags pair. By contrast the known `0x1401fc6a0` offset wrapper resolves correctly to the named `init_from_mesh_and_points` records. Thus these two additional callers are not yet identified as public bindings; they require deeper caller/body analysis.

## Actual native source-input invariance: stronger than midpoint reflection

An independent read-only parse of the owner's **Vein_Benchmark(8).zip** SQLite file identified **16** reference builders. Every one has exactly **one enabled** midpoint source, all four stored input point-feature flags false (`_hasNormals`, `_hasNormalSigns`, `_hasValues`, `_hasNoGoValues`), and empty lists of enabled/non-cutting faults.

The original 04 fold:
- 15 midpoint positions, sourced from one VeinSegmentMidpointsBlock with multiple intersections **enabled**, the segment block referring to 30 interval contact points; both endpoint inclusion settings true;
- no saved custom reference mesh, functional transformer or direct HW/FW point controls;
- symmetric model bbox X/Y [−110,+110]m and Z [0,200]m; symmetric reference bbox X/Y [−121,+121]m and Z [−10,210]m;
- **all 25/25 collar positions and all 55/55 geology intervals** are Y-mirror symmetric when compared as physical XYZ/depth/lithology rather than hole-ID strings; survey holes are vertical;
- saved 349-vertex native output sheet is NOT mirror symmetric: mirror-transformed vertices against original triangulated sheet have **43.6645m mean, 62.3625m RMSE** exact nearest triangle distances. This metric is one-way vertex-to-triangle, not exact Hausdorff.
- native reference OBJ exported after reversing original drillhole CSV row order is index-matched within **6.93e−9m** of original.

### Engineering implication

A unique deterministic **reflection-equivariant** surface fitting algorithm fed only the unordered symmetric geometric values and a symmetric model box would have a symmetric output. The observed contradiction indicates some symmetry-breaking orientation initialization, ordered/topological relationships, coordinate convention, solver selection among equivalent solutions, or other context. The saved metadata does not tell us which, so we cannot claim automatic folded-reference parity.

A **hole-ID-only** paired dataset has been generated in the user workspace `Leapfrog_Folded_Vein_HoleID_Ordering_Control_20261020.zip`: both variants have exactly the same numeric coordinates, intervals, survey, table row order and set of 25 drillholes; only the `holeid` assignment is reversed by Y/X-position order. A native reference rebuild under identical model settings would test whether this canonical identifier ordering breaks handedness. The experiment has **not been run in Leapfrog**, so its result is unknown.

The original upstream audit script plus all 16 model metadata summaries is in the owner-only archive `Leapfrog_Folded_Reference_Upstream_Metadata_Audit_20261020.zip`, created outside GitHub. A related generic `reference_input_provenance_audit.py` has three passing synthetic CI tests on the public research branch.

## Source-code location: a separate new packaging clue

Another verified owner-private **read-only** bootstrap audit independently identified a `Zippy` Python application loader in the commercial executable, including `zpyarchive` and `__zpy_decryptor__` metadata for the high-entropy `Leapfrog.zpa` package. This is *evidence about where application-level Python code may reside*, not evidence that the archive's contents have been decoded or the proprietary reference constructor located. The vendor archive and executable must remain private. No claim of archive extraction, encryption circumvention, or recovered vein reference source is made.

## Priority

1. Follow the unexplored native callers only insofar as they reveal a verified entry back into the folded reference's **upstream input processing**; avoid rediscovering downstream offset routines.
2. Reconstruct and test an **input-only** structural sheet/orientation rule against the saved original native reference, with out-of-sample conditions, no native-output oracle at prediction time.
3. If vendor-side control becomes necessary, compare the two identical-geometry, different-hole-ID cases; a changed native sheet identifies ordering/metadata sensitivity.

Overall research progress toward **all** vein modelling functionality should remain about 60%, because the new tests narrow the automatic-reference unknown but do not solve its mathematical constructor.
