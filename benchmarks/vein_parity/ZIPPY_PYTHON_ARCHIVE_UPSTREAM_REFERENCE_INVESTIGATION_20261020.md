# Automatic folded reference: upstream application packaging and multi-engine reverse engineering (2026-10-20)

**Scope:** authorized read-only analysis of an owner-provided Leapfrog application copy in the owner's PRIVATE `Mining-Geologist/LF_SC` research branch. Public repository contains aggregate architectural findings only; no vendor executables, source-code extraction, package payloads, license-related artifacts, decompiler outputs or private project geometry. The commercial program was not launched or patched.

## Recovered app-layer architecture: Zippy Python packager

Independent private GitHub Actions, using read-only PE/CAPSTONE static inspection of the owner executable and separate inspection of its `Leapfrog.zpa` archive:

- **39353253916:** finds embedded string `Leapfrog.zpa`, native references to Python `marshal`, `zlib`, and Python code serialization; a distinct native `PyInit__zippylib` module.
- **39353521251:** locates `zippy`, `zpyarchive`, `zpycode`, `__bootstrap__`, `__zpy_decryptor__` / `_zpy_decrypt`, and compiler path indicating `zippy\\__bootstrap__.pyr`, plus internal packager build-marker `zippy\\source\\runw\\hydrostubw.pdb`.
- **39337082017 / 393390... earlier package inspection:** `Leapfrog.zpa` is a **51,875,416-byte high-entropy file**, no ZIP/PYZ or normal zlib stream signatures that can be decoded by tested probes.

**Defensible conclusion:** the application uses a distinct **Zippy-packaged Python application layer**. Its loader has an explicitly named decryption interface and import/bootstrap machinery. It is technically plausible—and consistent with absent plaintext class names in sampled compiled DLL/PYD/EXE files—that the application-side `VeinReferenceSurfaceBlock` implementation is present in this packaged layer. **Neither the complete package format nor the location/source of the actual vein reference constructor has been independently recovered; encryption is suggested but not proven from entropy alone.** It would be irresponsible to claim `VeinReferenceSurfaceBlock` is definitely inside the archive without a safe, authorized inspection of loaded modules.

The bundled GUI `VeinReferenceSurfaceDlg.ui` distinguishes **planar** reference controls (explicit dip/dip-azimuth) from **curved** reference input selection; the latter delegates its orientation/topology to model-input and builder logic. The saved project's declared dependency chain is `VeinReferenceSurfaceBlock → MergedLocationsBlock + FaultBlockBoundaryBlock`, with shared global dip field objects; saved HW/FW point tables are downstream. This is the key unknown **upstream of** all successfully decompiled native wall-offset functions.

## Expanded independent native binary sweep — negative location evidence

Private run **39353072222** scanned a dozen additional geometry modules, including `geokernel.dll`, `geokernel_storage.dll`, `geodist.dll`, `tessellator.dll`, `gts_delaunay.pyd`, `cgal_triangulations.pyd`, `mesh_processing.pyd`, `trimesh._trimesh.pyd`, `modeling_libs._importers.pyd`, `geolib.dll`, `geoengine.core.dll` and `Leapfrog.exe`. These expose generic Delaunay, triangulation, mesh orientation/processing, surface geometry and isosurface APIs; **none exposed the identifiable automatic VeinReferenceSurfaceBlock class name**. This is a string/symbol absence, not mathematical proof the code is missing or exclusively Python.

## Multiple independent tools used against the actual private binary

- Private Ghidra 12.1.4 decompiled signed reference-conformal projection, RBF evaluations, symmetric thickness correction, and auxiliary 2D/3D mesh-query geometry.
- Independent `angr 10.0.4.post1` (run **39349784576**) performed bounded x86-64 control-flow analysis of ten native functions.
- Official REA `rea-agents@6.7.0` with *genuine Hopper Disassembler 6.6.0 demo* (private run **39350167373**), installed from a vendor-signed package under an ephemeral private Linux runner, successfully analyzed native `init_from_mesh_and_points`. The retained private REA evidence contained full assembly/decompiler/CFG data, 169 basic blocks, 208 outgoing references and twelve callees.
- Private run **39350787993** automatically asserted **12/12 exact direct-callee address agreement** between independent REA/Hopper and angr analysis.
- IDA is an optional, separate third-party proprietary tool and upstream MCP integration. The private GitHub runner has neither IDA nor its configured MCP provider. **IDA has NOT been run** and is not a blocker to reporting verified work. No license/activation bypass.

## What this does and does not solve

**Solved / strongly supported:** generic native wall-offset/conformal evaluation call structure and thickness-bound correction, robust independent known-reference HW/FW orientation on all 95 synthetic native labelled samples, and the distinction between continuous field evaluation and meshing error.

**Still unresolved:** the AUTOMATIC curved/folded reference builder's generation from input midpoints/contact topology/boundary and the initial handedness. Reversing raw CSV row order was previously invariant; midpoint cloud has Y-reflection symmetry whereas the native final mesh is asymmetric. No numerical claim of kernel parity or algorithm identity is made.

## Critical correction: official input semantics are broader than the saved direct DAG

The vendor's **Leapfrog Geo 2026.1 Veins** help explicitly states that the drilling-based *initial reference surface* is a best-fit surface using **midpoints and hangingwall/footwall surfaces** (https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm). Our saved-project static graph exposes `MergedLocationsBlock` and model boundary as direct inputs and shows no individual `hw_points`/`fw_points` fields on `VeinReferenceSurfaceBlock`, but these *direct saved fields do not exhaust every possible input and feedback path*. In particular, `raw_vein` or the midpoint generator may carry structural information not shown in the field list.

Therefore the prior implicit assumption that the native reference is necessarily a function **only of an unordered XYZ midpoint cloud** is not established. The symmetric 15-midpoint geometric identifiability result is mathematically valid *for midpoint-only estimators*, but it cannot rule out a proprietary joint fit to wall/structural inputs. A folded reference with Y-asymmetry is quite possible if an unobserved orientation/ordering input is present.

**High-value empirical discriminator:** hold the exact same midpoint XYZ, boundary and mesh resolution; modify HW/FW geological polarity or surface input (without moving contacts), then regenerate and export the reference. If it changes, there is a causal upstream path that the saved direct reference input graph does not currently expose. If unchanged, investigate deterministic orientation and topology tie breaking in the Python constructor.

## Next high-information step

Inspect the owner's original synthetic `04_MULTIPLE_INTERSECTIONS` project with `reference_input_provenance_audit.py` to map the ACTUAL persisted upstream merged-location and boundary configuration, not just generic field names. This remains unrun in the current container because the owner's original .aproj backing file is not accessible in the active model runtime, and the private binary repository does not contain that project.

If application-layer observation later becomes necessary, use a pristine, legitimately authorized Leapfrog installation and **non-invasive** trace/diagnostic observation; do not execute or modify license hooks, tamper with licensing, or assume a protected packaged archive can safely be unpacked. A controlled native experiment changing only reference *input selection* (rather than the drilling geometry) would discriminate whether orientation handedness depends on upstream settings.

Main branches untouched.
