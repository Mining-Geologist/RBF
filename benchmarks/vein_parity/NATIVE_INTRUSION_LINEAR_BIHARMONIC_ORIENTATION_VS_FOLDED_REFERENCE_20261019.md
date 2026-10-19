# Native intrusion 3D biharmonic RBF and folded vein comparison — 2026-10-19

Owner-provided synthetic Leapfrog Geo project was inspected read-only via SQLite, pickletools opcode parsing and numeric LFData decoding; vendor code was neither loaded nor executed. Do not publish owner-supplied project, coordinates, model meshes, or native coefficients in this public repository.

## Recovered native intrusion interpolant

The saved SurfaceInterpolantBlock has a LinearVariogram: vrange 200, sill 1.71, nugget 0, degree 1. Its persisted FastRBF.objects.Solution3 has:
- basicfunc = biharm; degree = 1; rho = 0; FastRBF V 1.4.2;
- **17 centres**, composed of original **15 midpoint** positions plus **2 signed orientation/polarity centres**;
- **21 coefficients** (17 radial weights plus an affine polynomial of 4 coefficients), and a saved polynomial offset base;
- native achieved accuracy ~0.001751 in scalar field units.

An independent evaluation of the saved solution, f(x) = sum_i w_i * ||x - c_i|| + a0 + a dot (x - polybase), gives **0.001133 field-value RMSE** on the **3,719 saved native intrusion surface vertices**; median absolute field residual **0.0000673**, max **0.024118**. On the later, subdivided **4,330-vertex** contact-surface output, scalar RMSE **0.001050**. These are scalar-value residuals, NOT metres. This is a directly recovered and independently verified **3D linear radial / biharmonic plus affine drift** interpolant for the user's intrusion, **not** evidence that the automatic vein reference uses this same algorithm.

An offline SciPy 3D linear RBF with those 17 centres, 15 zero observations and two opposite-sign observations regenerates a coarse approximate one-component implicit surface with ~**1.81m symmetric sampled-surface RMSE** to the saved intrusion, using 4m marching-cubes grid; grid discretization and the difference between exact saved and approximate fitted control values are distinct from direct native coefficient recovery.

## Orientation controls explain the different directions

The original 15 midpoint XYZs are exactly symmetric under Y-reflection, but two *additional* saved centres carry near +1.74955 and -1.74955 scalar values. Their position midpoint is identical to an original reference midpoint. Their direction is [-0.57735,+0.57735,+0.57735] (up to sign), separation 113.137m.

- Angle with the native **intrusion mesh PCA normal**: acute **4.771 degrees**.
- Angle with the original native **folded vein reference PCA normal**: **63.043 degrees**.
- Angle with the original reference's favorable oblique chart normal recovered **from the output**: **56.73 degrees**.

Thus the extra sign/orientation supports break midpoint-cloud reflection symmetry and correlate very strongly with the constructed intrusion's direction. The saved record does not independently establish how Leapfrog/UI derived or inserted these controls.

Native 15 midpoint-to-surface Euclidean nearest-triangle distance RMSE: **1.886m to intrusion**, versus **17.397m to the automatic folded vein reference**. The automatic folded reference is not a zero-iso interpolation through every midpoint in the same way.

## The surfaces are not simply rigid copies

- Automatic reference 349 vertices / 606 triangles, one connected open sheet, area ~74,279 m².
- Intrusion isosurface 3,719 vertices / 7,041 triangles; subdivided final contact-surface 4,330 / 8,263; each is one connected sheet, total area ~77,631 m².
- A *different* 1,024-vertex, 1,804-triangle mesh labelled multiple-intersections **Midpoints** has two components, including a small isolated central component. This mesh does NOT belong to the intrusion's own final surface. A connected bulge can still visually appear in the intrusion.
- Symmetric 3D sampled-surface RMS difference without transform ~40.96m; best found among multistart rigid ICP/reflection initializations ~20.52m. This is an approximate point-sampling metric and local ICP result, not a guaranteed global optimum or Hausdorff distance.

## Is rotating the intrusion orientation enough?

No. A matched series of independent 3D linear+affine RBF reconstructions kept all 15 zero midpoint constraints and both signed support magnitudes/separation fixed, varying only the signed-pair direction. Each was meshed at 4m grid spacing:
- Original signed direction, surface-to-native-folded-reference sampled RMS ~42.08m.
- Direction forced to the native **output-derived** oblique chart normal, sampled RMS ~36.59m.
- Direction optimized using the native output vertices as an oracle (a post-hoc diagnostic with scalar field residual objective), sampled RMS ~36.78m.

The native reference remains substantially different even using information from its own output. Additional structural constraints, ordering or topology handling must exist beyond this simple orientation-pair 3D interpolation.

A further 45-point 3D linear-affine surrogate using 15 midpoint zero contacts plus 15 HW-positive / 15 FW-negative contacts yielded TWO components and a ~38.65m sampled RMS difference from the native one-component reference. Naive contact polarity alone does not reconstruct it either.

## Architectural conclusion and next steps

The saved intrusion now independently establishes a *specific native 3D interpolant family*; the saved VeinReferenceSurfaceBlock currently exposes an output mesh but no matching saved Solution3 kernel, so **intrusion kernel cannot be transferred to native vein reference without further evidence**.

Next experiments: (1) inspect the automatic vein's upstream folded-reference structural controls and graph again in light of these two orientation anchors; (2) for new native tests, hold all midpoints and kernel settings constant while changing only intrusion orientation support direction; (3) prioritize the contact ordering, fold reference, and reference construction pipeline rather than only trying more scalar RBF kernels. Original public main branch unchanged.

Generic, independently implemented field evaluator and synthetic unit tests: benchmarks/vein_parity/intrinsic_implicit_rbf3d_linear.py and test_intrinsic_implicit_rbf3d_linear.py on the research branch. Detailed project-specific measurements and reconstructors are private owner-facing artifacts.