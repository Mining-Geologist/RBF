# Native vein reference dependency and pipeline-stage audit — 2026-10-18

**Status:** offline static analysis of the owner's saved Leapfrog Geo 2026.1.1 `Vein_Benchmark.aproj` (SQLite and pickle opcodes), plus geometric comparisons of the owner's exported meshes. **Not a recovered proprietary algorithm.** No vendor code was executed, no Python pickle was deserialized, and no proprietary input or output bytes are committed.

## 1. Native project object graph: six references

In the saved SQLite `data` table, each reference is a `engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock`. We parsed its concatenated Python pickle streams with `pickletools.genops` (static opcodes only), followed the stored `_inputs` mapping OID, and resolved the linked block classes.

| Saved case | Reference OID | Midpoint input OID | Other recorded upstream input | Resolution | surface_type |
|---|---:|---:|---|---:|---|
| 01 Flat | 2553 | 2549 | `FaultBlockBoundaryBlock` #2307 | 20.0 | curved |
| 02 Pinchout A | 3060 | 3056 | `FaultBlockBoundaryBlock` #2853 | 20.0 | curved |
| 02 Pinchout B | 3359 | 3515 | `FaultBlockBoundaryBlock` #3353 | 20.0 | curved |
| 03 Curved | 4053 | 4049 | `FaultBlockBoundaryBlock` #3846 | 20.0 | curved |
| 04 Multiple-hit | 4554 | 4550 | `FaultBlockBoundaryBlock` #4336 | 20.0 | curved |
| 05 End-of-hole | 5044 | 5040 | `FaultBlockBoundaryBlock` #4837 | 20.0 | curved |

**Direct persisted attributes in ALL SIX objects:** `points` → `MergedLocationsBlock`, `raw_vein` → corresponding parent `VeinBlock`, `hw_points=None`, `fw_points=None`, `custom_mesh=None`, `resolution=20.0`, `_invert=False`, `surface_type='curved'`.

The recorded upstream `_inputs` mapping contains **two entries** in each case: the merged midpoint locations and the associated fault-block boundary. This elevates the **boundary dependency** from an assumption to direct saved-project evidence. It does **not** prove only those two are used by the algorithm: the `raw_vein` parent pointer may let runtime code retrieve additional geology/wall context without encoding it in `_inputs`. Equally, the `face_dip_values` and `dip_azimuth_values` objects are recorded under `_references`; their presence is not proof that those derived surface attributes serve as structural interpolation inputs.

### Implication for native investigation

The documented reference-builder class is `engine.vein_blocks.reference_surface.VeinReferenceSurfaceBlock`, not `PySurfaceOffsetInterpolant.init_from_mesh_and_points`: the latter concerns a subsequent reference-*relative wall* construction. In the next static/native code search, trace the reference block execution and its `MergedLocationsBlock` and `FaultBlockBoundaryBlock` consumers separately from wall offset/snap functions. Specifically inspect topology generation, smoothing/outlier weights, domain clipping, and whether access through `raw_vein` introduces actual wall geometry.

## 2. Pinchout control: a strong stage-separation test

A and B are saved alternatives of the **same initial geometry**, with the empty-trace pinchout mode changed. Direct SHA-256 and mesh-byte comparison:

- `Reference_Surface.obj`: **identical byte-for-byte**, both 180 vertices / 301 faces, SHA-256 `68907231be1e2ec0962b686e6b745b08811cf885aa66fd63161f8b9b9f8cf7de`.
- `Vein_Solid_Original.obj`: **identical byte-for-byte**, both 1,784 vertices.
- `Hangingwall_Interpolated.obj`: **not identical**, 661 vs 653 vertices.
- `Footwall_Interpolated.obj`: **not identical**, 691 vs 693 vertices.
- `Vein_Solid_Final.obj`: **not identical**, 408 vs 384 vertices.

**Inference limited to this paired case:** changing this pinchout setting affects the interpolated walls and final solid but leaves the reference and original solid unchanged. Thus building pinchout effects directly into the reference mesh would be inconsistent with this observed pair. This does **not** prove all pinchout configurations leave references unchanged, nor establish Leapfrog's general reference algorithm.

## 3. Does the saved HW/FW contact geometry give extra reference constraints?

For case 01 Flat and case 03 Curved, the midpoint input at all **25** XY locations equals the arithmetic mean of the observed HW and FW contact elevations. After rounding horizontal coordinates to 3 decimals, maximum absolute discrepancy was 0.0 m (flat) and 1.42e-14 m (curved). Therefore **adding the same observed HW/FW contact elevations as midpoint *positions* provides no new reference position information** on these synthetic cases. Their separate wall orientations, final interpolated surface geometry, and non-contact control points could still contribute independent information; the result cannot rule that out.

The end-of-hole saved midpoint input contains **26 samples**, including extreme values at 1, 4, 196 and 198 m, while the native reference stays near 95.5–104.2 m. Contact midpoints and input support there have duplicate XY/alternative intervals; do not simply collapse those samples by XYZ indexing or interpret outliers as accepted exact-surface constraints.

## 4. What remains unsolved

Neither this project-graph audit nor the earlier 2D TPS study proves native reference interpolation. The curved 2D surrogate improved in-sample mesh fit to 0.3397 m (interior), but midpoint-only LOOCV preferred a model scoring 1.5333 m; reference construction is **not simply through-midpoint interpolation**. Full fold topology, end-of-hole outlier treatment, the reference builder's boundary interaction, and native interpolation parameter recovery remain unsolved.

**Priority next experiment:** controlled reference-only perturbations in owner-operated Leapfrog (same drillhole midpoint cloud, modify bounding domain and/or wall contact thickness separately); export references after each change. If the midpoint cloud is fixed but the reference changes after changing the wall contact geometry alone, that directly demonstrates a wall-dependent path. If a bounding-only change modifies only cropping rather than local surface shape, constrain the boundary dependency to clipping/meshing. A single saved project snapshot cannot discriminate these behaviors. Any such runtime experiment needs the owner's authorized installation; do not claim one was run.

Related notes: `STRIKE_ALIGNED_2D_REFERENCE_20261018.md`, `NATIVE_REFERENCE_INPUTS_ROBUST_FIT_20261018.md`, `NATIVE_SAVED_ASPHERE_CONFIRMATION_20261018.md`.
