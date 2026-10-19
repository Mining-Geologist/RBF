# Private Ghidra native surface-offset callback call-graph triage — 2026-10-19

**Result:** Four read-only private Ghidra 12.1.4 workflow passes completed successfully after fixing a Java Ghidra API compilation issue; actual vendor code/binaries/pseudocode are kept as private workflow artifacts and never committed to public GitHub. The targeted results narrow the native Cython wall-offset registration route but **do not recover the automatic folded reference-surface constructor or its interpolation kernel**.

## Confirmed interfaces and call structure

Static analysis of the owner-provided Leapfrog executable identifies entry point PyInit_offset_mesh_functions, a method descriptor registration function, a module initialization function, and a shared Cython function factory. Registration labels include:
- PySurfaceOffsetInterpolant.init_from_mesh_and_points
- PySurfaceOffsetInterpolant.init_from_mesh_and_rbf
- PySurfaceOffsetInterpolant.conformal_distances_to_mesh
- PySurfaceOffsetInterpolant.eval and pointeval
- PySurfaceOffsetInterpolant.get_eval_func and get_transformed_eval_func
- PySurfaceOffsetInterpolant.set_params_for_approx_eval
- PyRefinedWallsFunction.conformal_distances_to_mesh and related evaluation methods.

The complete method registrar is associated with private image code address 0x1401e9300. Its shared initializer is approximately 0x1401e7c50, which constructs each method using a common Cython function factory around address 0x1401edf90. These are **binding-registration/initialization functions**, not demonstrated actual RBF or surface geometry solvers.

The private report successfully traced 12 method descriptor slots: each is zero in the static loaded image and has 3 associated Ghidra references—two data references in the registrar, and one read in the initializer. A direct image-pointer dereference therefore returns null and CANNOT be interpreted as the method's implementation address. The initializer passes those descriptors into the generic Cython function factory along with other runtime data. Further algorithm recovery needs a targeted path through the descriptor creation / Cython wrapper registration itself, or an independently identified native solver from surface-offset objects.

**No new VeinReferenceSurfaceBlock reference-construction kernel or exact local HW/FW conformal distance rule was located** in these runs. The presence of a callable `init_from_mesh_and_rbf` is evidence of a mesh-and-RBF API, not proof of which kernel/order/anisotropy native veins employ. Do not transfer the previously verified intrusion LinearVariogram into the vein reference builder without separate evidence.

## Independent progress on native folded HW/FW assignment

In parallel, a pure geometry surrogate using the saved **native reference mesh as input** predicts all **95/95** original HW/FW assignments over five benchmark cases. For each vertical drillhole original target-segment midpoint, intersect its vertical ray with the reference triangles and choose the closest depth crossing; if none, choose the nearest 3D reference triangle. The sign of the selected reference triangle normal's z-component, with the saved mesh winding, determines the wall assignment. Folded case: 15/15; 13 ray hits and two fallback. Previous Euclidean-only approach: 14/15 on folded, 94/95 overall.

This is **observational parity given a native reference input**, not native automatic reference generation. Since the native wall-point labels are derived downstream of the reference, they cannot causally select the initial folded handedness.

Public generic implementation and unit tests:
- benchmarks/vein_parity/reference_raycast_wall_polarity.py
- benchmarks/vein_parity/test_reference_raycast_wall_polarity.py
- .github/workflows/offline-folded-reference-raycast.yml, passing CI run 39325736766 for related baseline and a later raycast CI success.

Owner-facing private reproducibility archive: Leapfrog_Folded_Reference_Ghidra_and_Wall_Orientation_20261019.zip; includes full 95 per-sample numeric comparisons but no proprietary binary or pseudocode.

## Next investigation

Prioritize `VeinReferenceSurfaceBlock` creation/structural interpolation, rather than repeatedly decompiling the Cython module registrar. To identify the automatic reference's orientation kernel, focus on either the packed Python project/workflow definitions or high-information native controlled perturbations where midpoints remain identical but structural contact labels/geometry or the geological block boundary changes. A dedicated static Ghidra pass can continue through method descriptor creation, but the current pointer slots do not alone supply native solver function addresses.

Both main branches remain unchanged.
