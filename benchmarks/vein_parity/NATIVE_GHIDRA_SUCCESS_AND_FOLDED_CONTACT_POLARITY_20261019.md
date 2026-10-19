# Ghidra callback-registration evidence and native folded-vein wall-polarity audit — 2026-10-19

**Scope:** owner-supplied private Leapfrog executable and synthetic benchmarks, read-only static analysis (Ghidra 12.1.4) and independent triangulated-reference geometry checks. The private vendor binary, pseudocode, serialized native project, contact coordinates and owner benchmark meshes are NOT in this public research branch. No vendor program was executed. No claim that automatic curved/folded reference source or its exact kernel has been recovered.

## Successful private Ghidra pass

The workflow *Private Ghidra Leapfrog Vein Dossier*, completed successfully on private research branch in GitHub Actions run **39324069518**. It first failed on a Ghidra 12.1.4 Java API incompatibility (ReferenceIterator vs Reference[]) but was patched in the private repository. The same Java script then successfully postprocessed the fully analyzed owner-supplied Leapfrog.exe and wrote a restricted artifact with decompiled binding functions.

The static analyzer found 5 native module entry points, including named PyInit__FastRBF, PyInit_offset_mesh_functions, PyInit_structural_interpolant, PyInit_turbo_rbf_spaces and PyInit__isosurfacer; the exact reference-construction algorithm is **not thereby identified**. An identifiable registrar at image address 0x1401e9300 references compiled method-name strings:

- PySurfaceOffsetInterpolant.init_from_mesh_and_points
- PySurfaceOffsetInterpolant.init_from_mesh_and_rbf
- PySurfaceOffsetInterpolant.conformal_distances_to_mesh
- PyRefinedWallsFunction.conformal_distances_to_mesh

This is **verified binary method-registration evidence** that a mesh-based surface-offset interpolant and refined-wall conformal distances exist at a native interface. It is consistent with a reference-relative / conformal wall-processing architecture but **does not identify exact mathematics, the internal method callback addresses, or native VeinReferenceSurfaceBlock reference-building algorithm**. The project class name `VeinReferenceSurfaceBlock` had no plaintext match in the scanned native executable, which is expected for Python-level object names and is not proof of absence of relevant code.

A separate second private workflow run will inspect more of the binding registrar to try to locate callback functions; do not claim that future evidence yet.

## Geometry: HW/FW labels are overwhelmingly explained by reference normal sign

The owner-original 5 synthetic configurations include **95 paired real HW/FW input observations**. We aligned each saved wall-point pair by index (same original interval) and computed its midpoint, closest native reference triangle, Euclidean point-to-triangle distance, and local reference mesh face normal, with the native mesh's wound-normal convention.

In ALL tested cases, predicted sign of (HW_Z - FW_Z) is **negative sign of nearest native reference-face normal's Z component**. The observed samples are vertical drillholes; the sign assignment is a proxy for the local reference orientation along drilling.

| Native test case | Correct / valid paired observations |
|---|---:|
| Flat baseline | **25/25** |
| Pinchout A | **5/5** |
| Curved variable-thickness | **25/25** |
| Multiple intersections, overturned fold | **14/15** |
| End-of-hole | **25/25** |
| **Total** | **94/95** |

For the multiple-intersection folded model, **3 of the 15 HW/FW assignments have reversed vertical polarity**, all on lower fold branch, at synthetic coordinates XY (40,0), (0,40), (40,40). The original midpoint-cloud XYZ collection remains Y-reflection symmetric, but the **saved wall-labelled** clouds are not Y symmetric.

The **only wrong-polarity prediction** occurs for the folded sample located ~**45.6795m** from its nearest reference triangle, precisely the largest midpoint–reference separation in that fold. This geometric closest-point association can select an unrelated fold branch in a strongly overturned model, so the error does not falsify a more robust correspondence algorithm. Using a 40m maximum nearest-reference distance **92/92** assignments match, but this cutoff is post-hoc chosen and must not be treated as validated selection logic.

**Critical causal caveat:** The HW/FW-labelled contact point tables were generated downstream from the native reference, so their spatial polarity asymmetry likely reflects the fitted reference orientation. They are NOT necessarily asymmetric upstream inputs and cannot be used to explain why the reference handedness was selected in the first place. Midpoint-only geometrical mirror ambiguity remains open.

This is a reproducible method for *interpreting/assigning wall labels* once reference orientation is known, not an algorithm for constructing automatic folded surfaces. Overturned folds with multiple nearest faces require a topology-consistent projected-correspondence or ray-tracing method.

## Code and next steps

An independent implementation in benchmarks/vein_parity/reference_contact_orientation_audit.py plus synthetic unit tests contains NO vendor software/data. Both tests passed offline locally. Next run: finish the private Ghidra deeper binding registrar audit and attempt to trace method registration into implementation callback pointers. Future vendor-side experiment with asymmetric contact labels but unchanged midpoint locations could test whether extra structural orientation constraints enter upstream (the saved graph exposes only midpoint locations and model boundary explicitly).

The public and private main branches remain unchanged.
