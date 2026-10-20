# Automatic curved vein reference: focused native-code location audit (2026-10-20)

**Scope:** user-authorized, read-only static reverse engineering of the owner's Leapfrog Geo installation and previous saved synthetic projects. All proprietary program binaries and decompilation remain in the owner-private GitHub repository. Public branch contains only this aggregate report and independently written regression tools. No program execution, patching or licensing work.

## Main result: distinguish the reference generator from the wall-offset solver

We previously identified and decompiled compiled `PySurfaceOffsetInterpolant` and `PyRefinedWallsFunction` interfaces. They project positions onto an **already existing reference mesh**, compute signed conformal distances, interpolate HW/FW offsets, and enforce midpoint-preserving wall-thickness bounds. These are demonstrably *downstream* of construction of the initial folded reference and cannot establish its constructor or RBF kernel.

The `VeinReferenceSurfaceDlg.ui` file bundled with the owner's installation confirms that its **curved reference** setting uses an input-values view, while *planar reference* exposes explicit dip and dip-azimuth controls. Official Seequent 2026.1 documentation independently describes selecting/activating input objects, boundary filtering, and curving the best-fit reference surface (https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm). The absence of an explicit dip widget in the curved editor does not imply no orientation constraints are generated internally.

## Verified compiled-module search

Private GitHub Actions run **39348231365** restored and scanned the actual binaries for the full class strings and related method names:

| Module | Size (bytes) | Found direct class literals? |
|---|---:|---|
| Leapfrog.exe | 6,860,144 | No `VeinReferenceSurfaceBlock` / `MergedLocationsBlock` |
| geolib.dll | 3,706,368 | None |
| mesh_processing.pyd | 246,128 | None |
| gts_delaunay.pyd | 123,760 | None |
| cgal_triangulations.pyd | 834,928 | None |
| geoengine.core.dll | 22,157,312 | None |
| geoengine.map.dll | 9,366,528 | None |

`Leapfrog.exe` *does* contain 23 textual matches for `PySurfaceOffsetInterpolant` and 16 for `PyStructuralInterpolant`, confirming that our scan can locate native binding symbols. Generic “midpoint” strings in a couple of modules are insufficient to tie them to the automatic vein reference.

**Interpretation:** the particular Python block constructor likely lives in the packaged application layer or uses generic compiled geometry routines without exporting its class name. Literal-absence is not proof of absence of native implementation, and the file inventory covers selected relevant modules, not all bundled binaries.

## Investigation of candidate unrelated subsystems

Two independent fast, static inspections of `FastASphere3` logging strings on the owner executable were completed in private runs **39347706587** and **39347871087**. No direct RIP-relative code xrefs, no intermediary eight-byte pointer table entries, and no indirect RIP-pointer xrefs were found for the ASphere diagnostics by these methods. This is a **negative / inconclusive linkage** result, not proof ASphere is unused: compiler optimization and nontrivial indirection can hide references. It should not currently be identified with automatic folded-reference construction.

A separate `PyStructuralInterpolant` native interface includes transformed RBF functions with per-domain blending distances/bounding boxes (confirmed by private scan **39346925489**). Its existence is direct interface evidence, but there is no verified saved-object/caller link to the folded reference builder, so declaring its linear/multiquadric kernel to be native vein-reference behavior is unsupported.

## Saved object-graph constraint remains decisive

In the owner's saved `04_MULTIPLE_INTERSECTIONS` model, `VeinReferenceSurfaceBlock` OID4554 declares upstream `MergedLocationsBlock` OID4550 and `FaultBlockBoundaryBlock` OID4336. Its `hw_points` and `fw_points` fields are None; the separate wall-point generators are downstream. Across 16 saved reference objects, generic `face_dip_values` and `dip_azimuth_values` fields point to the same global objects and are not demonstrated as reference-specific orientation inputs. A `raw_vein` backreference could still provide undeclared context.

The native reference geometry is severely non-Y-symmetric despite exactly Y-mirror-symmetric 15 midpoint XYZ positions. Reversing CSV drillhole row order produced the *same* reference mesh to nanometre export precision. An automatic reference algorithm consuming only unordered symmetric XYZ and using an equivariant uniquely determined objective therefore cannot explain the observed native handed fold without further context, a directional initialization, or symmetry breaking.

## Highest-information next steps

1. **Reference input provenance**: trace `MergedLocationsBlock` into the default curved reference's input-value selections, and inspect the saved `FaultBlockBoundaryBlock` **orientation/filtering**, not just model bounding-box extents. This is the next strongest place to find an asymmetric upstream constraint.
2. **Native package-side search**: inspect any readable application-level code in the owner's installed package through authorized read-only means; direct compiled class-string searches found no concrete constructor in the selected binary modules. Do not assert that the `Leapfrog.zpa` archive is decryptable or simple ZIP/Python plaintext: previous inspection found uniformly high-entropy bytes and no standard archive signatures.
3. **Discriminating native control if needed**: edit only which folded midpoint input values are active (or change boundary filtering), preserve collar/sample XYZ and rebuilding settings, export the new native reference mesh. This would reveal whether the reference generator consumes these selections and whether its handedness changes. Maintain model originals untouched.

This is progress in **locating the missing code**, not progress toward numerical parity for automatic folded reference construction. All work is on research branches; both main branches remain unchanged.
