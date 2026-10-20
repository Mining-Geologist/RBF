# Beyond Ghidra: official REA, Hopper demo, and angr cross-provider analysis (2026-10-20)

**Purpose:** In response to the owner asking why the study relied only on Ghidra, we performed independent read-only analyses on the owner-provided Leapfrog executable using the official **REA: Reverse Engineer Anything** CLI, a separately installed **Hopper Disassembler demo**, and the **angr** control-flow analysis framework. The source executable, Ghidra/Hopper pseudocode, and private project data remain restricted to the owner's PRIVATE repository `Mining-Geologist/LF_SC`; this public research branch contains only aggregate findings and generic independent tools.

## REA installed and exercised on the owner's real PE binary

- Verified upstream source: https://github.com/morluto/rea (not similarly named forks); pinned official `rea-agents@6.7.0`.
- Provider discovery on isolated GitHub Actions Linux runner, private run **39349683770**, found Hopper, Ghidra and IDA initially unavailable because the runner did not have those external tools.
- The official REA signed Hopper Linux demo setup initially encountered missing `pkexec`. In a fresh private Ubuntu 24.04 Actions runner, a restricted compatibility helper invoked the runner's ordinary permitted `sudo apt-get` **only for REA's requested package installation**. REA itself verified the vendor-signed package, stable vendor primary fingerprint `7C2D73BAFAF5B9A710A69DA1A26DF815099B1BCD`. After installation `rea doctor --provider hopper --json` reported Hopper **6.6.0**, validated launcher signature `b2271008fb82bf14cfae5c3644cdde2b11fc6804bd7c99d67032ec1b1942d53c`, private Xvfb demo runtime ready, and provider healthy.
- The **actual** REA command `rea function /absolute/path/to/Leapfrog.exe 0x1402d4e80 --provider hopper --json` **completed with exit 0** in private run **39350167373**. Output was a roughly 604 KB structured evidence JSON, saved as a private short-retention GitHub Actions artifact `private-rea-hopper-leapfrog-audit`, NOT published.
- The REA/Hopper function dossier reports 169 basic blocks, 208 outgoing references, 12 resolved direct callees, 1 direct caller, and available assembly/pseudocode/CFG.

## Angr independently decodes the same binary with no Ghidra

Private run **39349784576** installed `angr 10.0.4.post1` and statically inspected ten native functions (Win x64 PE, image base `0x140000000`). No vendor code was executed. The angr bounded 140-block function traversal recovered relevant call paths:
- point-initializer `0x1402d4e80` -> `0x1402d46b0` conformal helper and `0x1402d4160` reference initialization;
- RBF initializer `0x1402d4600` -> `0x1402cb060` mesh projection;
- refined-wall evaluator `0x1402d5ed0` -> `0x140300e80` RBF evaluator twice and `0x1402d46b0` conformal helper once;
- reference projection `0x1402cb060` -> 3D `0x14032b180` and 2D `0x140329ff0` helpers.

These independently corroborate previous Ghidra observations. A bounded disassembly/call inventory does NOT establish complete function semantics or every indirect call.

## Exact Hopper–angr call-set parity verified

A separate owner-private structured evidence reader, run **39350787993**, automatically compared REA/Hopper's twelve observed `init_from_mesh_and_points` direct-callee addresses to the twelve addresses recovered in the angr analysis. **12/12 exact call-set agreement**; no address found only by one tool.

REA/Hopper separately identifies the parent Cython binding wrapper (caller) `0x1401fc6a0`, and its target `0x1402d4e80`. These observations corroborate a concrete native method chain, but NOT a new interpolator formula.

## What the alternative tools have NOT discovered

This pass adds high-confidence independent validation of the *downstream reference-relative surface-offset* solver. It **does not identify** the private application's `VeinReferenceSurfaceBlock` *automatic folded reference generator*, its initial oblique chart direction, kernel, topology or left/right handedness. These may be implemented in the non-plaintext packaged application layer (`Leapfrog.zpa`) or call unnamed native primitives without exporting the reference-builder class. Do not reassign the recovered 3D *intrusion* kernel to the vein-reference generator.

IDA is supported by REA through an **existing configured IDA MCP provider**, but IDA was not installed or connected in the isolated runner. The optional IDA Free integration is GUI-based; any optional cloud decompilation/data transfer and software licensing would need to be evaluated and authorized separately. **No IDA analysis was performed**, and no paid license/authorization was bypassed.

## Research deliverables and next priorities

Public, generic read-only tool `benchmarks/vein_parity/independent_angr_callgraph.py`: accepts an authorized local binary, prints bounded direct callee observations, omits native code and never executes/writes the input.

Owner-private workflows:
- `private-rea-provider-discovery.yml`: `REA` CLI and provider capabilities.
- `private-angr-vein-audit.yml`: independent angr targeted binary CFG.
- `private-rea-hopper-demo.yml`: signed Hopper demo via REA, analysis on owner executable.
- `private-rea-hopper-evidence-reader.yml`: private structured Hopper evidence comparison and automated 12/12 independent call-set assertion.

**Next high-value search:** locate the *upstream curved reference builder/input processing* in the packaged application layer, or establish a native caller linkage from the `MergedLocationsBlock` input and `FaultBlockBoundaryBlock` boundary to a mesh constructor. Spend less time duplicating downstream offset decompilation unless a new engine supplies a concrete additional input, orientation or interpolation clue. Earlier reference-mirror identifiability proof and reverse CSV record-order invariance remain in effect.

No main branches were modified. All binary/data experiments stayed inside the owner-private repository / runner.
