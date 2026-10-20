# Structural interpolant module triage — 20 October 2026

A focused, private, read-only native symbol scan (GitHub Actions run **39346925489**) found a separate `PyStructuralInterpolant` interface. Its embedded signature describes `blended_functions` as a sequence of function-control tuples that include **points, an RBF, a transform, a blending distance, and bounding-box limits**; it also accepts an outside value.

This confirms that the commercial binary includes a domain-bounded blending facility for transformed RBF fields. It is **not direct evidence** that the saved automatic `VeinReferenceSurfaceBlock` uses this facility, a particular RBF kernel, or any particular structural domain setup.

A standalone `FastASphere3` evaluation/gradient diagnostic string also exists, but the current scan establishes no caller linkage between it and the automatic reference builder.

**Action:** deprioritize both names until a saved-project dependency or a compiled caller connects them to native vein reference construction. Continue working on demonstrated reference projection, signed wall offsets, and input-topology selection.

All executable data, disassembly and project geometry remain private; main branches untouched.
