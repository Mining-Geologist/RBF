# Static native method signatures: reference-distance approximation parameters (2026-10-20)

**Private static source:** successful owner-private GitHub Actions PE parser run **39345955844**, which parsed `PySurfaceOffsetInterpolant` method definitions and native embedded docstrings from the owner-provided executable read-only. This public report contains no proprietary code or binary file.

Verified callable methods:
- `init_from_mesh_and_points` and `init_from_mesh_and_rbf` (separate initializers; no informative embedded docstring).
- `eval` and `pointeval` (one-point and array distance evaluation).
- `conformal_distances_to_mesh` (array-oriented conformal distance evaluation).
- `get_eval_func` and `get_transformed_eval_func` (callback for isosurface extraction, including a transformed-evaluator interface).
- **`set_params_for_approx_eval`** explicitly accepts an **evaluation resolution** and a **maximum distance of input points from the reference mesh**. The maximum can be inferred internally for point-built surfaces, and can use the constant distance for a constant-offset surface.

This is **new direct interface evidence for an approximate evaluation stage**. It does NOT establish that any particular Leapfrog vein model uses approximation with particular settings, that resolution changes the analytical RBF interpolant, or that the automatic folded reference generator is the same class.

### Why this matters for the prior user-run native controls

In flat synthetic G/G5, direct 2D linear RBF predictions improved when the user reduced coupled vein/reference resolution, and the remaining residual depended on retaining the native triangle layout. It is therefore necessary to separate:
1. parameters of the **continuous** original RBF interpolant;
2. approximate **evaluation** resolution/distance;
3. the actual **meshing and extraction** of HW/FW and solid surfaces.

The runtime `set_params_for_approx_eval` API supports that decomposition, but whether the earlier G/G5 difference specifically arose in stage 2 or 3 is not yet proven. Native project-dependent stages must be compared before claiming an exact kernel or resolution formula.

Related private 3D/2D reference-projection and RBF evaluation Ghidra pass **39345407077** is running independently. Main repository branches untouched; research branch only.
