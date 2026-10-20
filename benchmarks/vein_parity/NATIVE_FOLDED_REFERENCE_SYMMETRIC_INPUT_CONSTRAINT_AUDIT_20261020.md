# Native 04 folded-vein reference: actual saved upstream constraints (20 Oct 2026)

**Evidence source:** owner-supplied `Vein_Benchmark(8).zip`; read-only SQLite and `pickletools.genops` without Python object deserialization, plus private native reference OBJ exports. This repository contains the aggregate results only, not private geological model/mesh files.

## Strong new negative result: no stored normals, signs, scalar values or active faults

We inspected **all 16** saved `VeinReferenceSurfaceBlock` pipelines in the owner's latest multi-case project. Their `MergedLocationsBlock` metadata establishes:

| Property in saved input | 16-model result |
|---|---|
| Enabled midpoint sources per reference | **exactly one in all 16** |
| `_hasNormals` | **false in all 16** |
| `_hasNormalSigns` | **false in all 16** |
| `_hasValues` | **false in all 16** |
| `_hasNoGoValues` | **false in all 16** |
| Saved model fault systems | **0 enabled / 0 non-cutting faults in all 16** |

For the actual **04_MULTIPLE_INTERSECTIONS** case:
- The sole enabled input is a `VeinSegmentMidpointsBlock`, with **15** midpoints, derived from a `VeinSegmentsBlock` with **multiple intersections enabled**. The underlying segments have **30 boundary points** and both endpoint-inclusion flags are true.
- The reference `_functional_transformer`, `custom_mesh`, `hw_points` and `fw_points` attributes are all **None**; no explicit 3D trend or extra signed points are saved on those direct properties.
- The selected model boundary is an axis-aligned **symmetric box**: X/Y from −110 to 110 m and Z from 0 to 200 m. The saved reference bounding box is X/Y −121 to 121 m, Z −10 to 210 m.
- The saved fault system's enabled and noncutting lists are empty. `MergedLocationsBlock` has no extra input beyond its sole midpoint source and symmetric model boundary in the declared graph.
- The user's reversed-drillhole-record-order copy has the same saved input structure and bounds. The two native reference OBJ exports contain exactly **349 vertices / 606 triangles**, with maximum corresponding 3D vertex separation **6.921e−9 m**, confirming order reversal alone didn't alter the surface.

## What this evidence does and does not establish

The 15 midpoint XYZ cloud is Y-reflection-symmetric, while the original native folded output is strongly Y-asymmetric. Given the symmetric saved bounding extents and no stored point normals or extra scalar values, the output's handedness cannot be explained **simply** by a visible extra orientation value, enabled fault, second source cloud or axis-aligned boundary dimensions.

Potential sources still include (a) a deterministic coordinate-axis/triangulation tie-break or initialization in the automatic reference algorithm, (b) segment relationships or local ordering carried implicitly through input processing, and (c) more complicated model-domain geometry not described by the saved bounding box. These possibilities are not excluded by the metadata audit. A naked symmetry argument alone cannot locate the missing constructor.

No folded-reference kernel or topology-construction formula was recovered in this particular test. The already-recovered Cython `PySurfaceOffsetInterpolant` operates downstream of the reference mesh, so do not confuse its RBF routine with the automatic reference builder.

## Reproduction

The owner has a private reproducibility bundle `Leapfrog_Folded_Reference_Upstream_Metadata_Audit_20261020.zip` containing the read-only metadata auditor, full private numerical metadata JSON and interpretation; no executable/vendor source code. The generic provenance module is also tracked as `reference_input_provenance_audit.py` on this research branch. Its three synthetic tests pass in GitHub Actions.

**Main research direction:** find the upstream `engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock` builder implementation or isolate directional tie-breaking by an XYZ reflection/rotation native test with identical segment topology. The owner's commercial packaged application uses a separate high-entropy `Leapfrog.zpa` with embedded Zippy loader metadata; its plaintext contents are not publicly known.

No main branches modified.


## Extra audit: **entire geology input** is Y-mirror symmetric

A further independent comparison of the original synthetic collar, survey and interval CSVs, after joining numeric geology rows back to XY collar positions, showed:
- all **25 of 25** collar locations have a reflected matching collar at `(x,-y,z)`;
- all **55 of 55** logged geology intervals have a reflected identical record, matching X, reflected Y, from-depth, to-depth and categorical lithology;
- **50** survey records describe vertical drillholes (same survey values after relabeling).
- Yet the 349-vertex native folded reference differs strongly from its own Y-reflection: the exact nearest native triangle distance from mirror-transformed native vertices has **mean 43.6645m; RMS 62.3625m** (and 132.98m at the 95th percentile). Those numbers are a one-way sampled vertex-to-triangle comparison, not a bidirectional Hausdorff distance.

This is stronger than midpoint-cloud symmetry: the **saved geological input itself** is symmetric as a spatial multiset, while the constructed reference is not. Accordingly a purely mirror-equivariant unique geometric fit to unordered geologic observations plus this symmetric model box cannot yield the observed native sheet *without some other symmetry-breaking implementation convention*. Viable mechanisms include hole-ID or stable-order sorting, coordinate-aligned init/tessellation tie-breaking, surface winding rules, or hidden derived topological connection assumptions. The latest native input attributes don't explicitly identify which one.

An owner-only controlled dataset `Leapfrog_Folded_Vein_HoleID_Ordering_Control_20261020.zip` contains two CSV suites with the same numerical collar/survey/geology geometry and row ordering, differing **only** in the hole-ID assignments: IDs enumerated in forward versus reverse Y/X position order. The zipped datasets were validated by joining every numeric record to collars and comparing canonical sorted tuples. The native vendor program has NOT been run on these two new test cases; only user-run native regeneration could determine whether ID ordering changes handedness.
