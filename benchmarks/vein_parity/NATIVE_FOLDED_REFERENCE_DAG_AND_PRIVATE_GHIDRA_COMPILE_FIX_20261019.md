# Return to native folded vein reference — saved object-graph and private Ghidra failure audit (2026-10-19)

## Scope

The owner explicitly deprioritized the intrusion analogy and requested renewed reverse engineering of the native *automatic folded vein reference*. The owner's complete updated synthetic Leapfrog project was opened read-only. Static analysis of its persisted SQLite object graph used the original standalone audit script and Python pickle opcodes only: **no proprietary object deserialization or vendor binary execution**. Separate read-only Ghidra analysis was performed through private owner-connected GitHub Actions; this public report does not include binary code or private project data.

## Newly cross-checked dependencies — 16 reference builders

There are **16** saved native VeinReferenceSurfaceBlock objects across the current benchmark. Every reference retains a points block (MergedLocationsBlock), a FaultBlockBoundaryBlock input dependency, and two common global fields:
- `face_dip_values` -> **OID 2302**, FaceDipValuesBlock ("Face dip")
- `dip_azimuth_values` -> **OID 2303**, DipAzimuthValuesBlock ("Dip azimuth")

**Critical qualification:** both objects are the SAME OIDs in all 16 reference builders, including flat, pinchout, curved, multiple-intersection, and end-hole cases. Neither has a declared `_inputs` dependency in the saved object graph. Accordingly, their presence alone is **NOT evidence** that they encode unique folded orientation, that they are populated, or that the algorithm reads them. They may simply be shared default field objects used throughout the model.

For the native folded benchmark, the reference builder OID 4554 specifically has:
- `points` -> MergedLocationsBlock **4550**;
- `_inputs` -> **4550** and FaultBlockBoundaryBlock **4336**;
- `surface_type` = curved;
- `raw_vein` -> **4560** (a backreference/potential downstream association, not a verified reference-fitting input dependency);
- `hw_points` = None and `fw_points` = None;
- `resolution` = 20m;
- common face dip / dip azimuth fields -> 2302 / 2303.

This shows the folded automatic reference has no separate HW/FW point blocks attached as explicit fields, but says nothing conclusive about whether input midpoint generator 4550 incorporates contact order/orientations or whether internal native code derives orientation from the point sequence, extra hidden metadata, and/or model boundary. It is not valid to claim that the native reference uses only XYZ midpoint coordinates.

## Tool failure was an actual Ghidra Java API compilation issue, not failed disassembly

The private repo's headless Ghidra 12.1.4 workflow on 2026-10-19 completed **350 seconds of static autoanalysis** and reported "Analysis succeeded" on the owner-supplied Leapfrog.exe. However, its post-processing Java script `VeinOffsetNativeDossier.java` failed compilation on line 94:
`ReferenceIterator cannot be converted to Reference[]`.

The exact Java source was patched on the **private research branch only** to use `ReferenceIterator` with up to ten incoming-reference samples. Private commit: `278b96446758a769423068151d19d80060baef0a`. This is a confirmed source-code fix for the explicitly observed compiler-type error, **but a new Ghidra decompilation job has not been run successfully after it**. Do not report extracted native pseudocode or new kernel discovery from the failed workflows.

The earlier successful private native-kernel module triage (run 39307411944) inspected geokernel, geolib, tessellation and Python extension modules; broad strings, exports, and symbols are not yet uniquely attributable to the automatic VeinReferenceSurfaceBlock implementation.

## Findings retained from prior controlled experiments

- The multiple-intersection native reference uses one connected, overturned 349-vertex, 606-triangle mesh and is not a global XY heightfield. A well-defined oblique chart can be found *from its native output*.
- Its 15 midpoints are Y-mirror-symmetric while the saved reference is not. Pairwise symmetric midpoint-only fitting cannot uniquely determine output handedness without other constraints/tie-breaking.
- Reversing the source CSV drillhole order yielded indistinguishable native reference indexed meshes (maximum corresponding-vertex difference < 7e-9m).
- Midpoint-only linear-plus-affine proxy, even given an oracle output-derived oblique chart, has tens-of-metres normal-coordinate RMSE against the native reference. Intrusion solution is a **different object family**, and cannot establish the automatic reference interpolant.

## Remaining most useful next reverse-engineering action

Rerun the repaired **PRIVATE Ghidra** workflow `Private Ghidra Leapfrog Vein Dossier` from `research/native-vein-reference-static-audit-20261018`, and read its restricted log/pseudocode artifact before making any claim about functions, kernel, orientation, face/dip constraints, surface meshing, or hidden topology. Do not publish vendor binaries, disassembly or private artifacts. Cross-check the actual discovered functions against the saved `VeinReferenceSurfaceBlock` / `MergedLocationsBlock` / surface-offset pipeline. If no relevant code can be identified, next test a controlled native mirror-polarity change (HW/FW ordering or geological domain orientation) with midpoint XYZ fixed.

**Public main unchanged.**