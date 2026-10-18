# Strike-aligned 2D vein reference experiment — 2026-10-18

**Status: offline research, not a reconstruction of Leapfrog's proprietary default reference builder.** Code and owner-local mesh benchmark in the conversation package `Leapfrog_Strike_2D_Reference_Research_20261018.zip`; original native input/target arrays are NOT placed in this public repo.

## Seequent article: precise technical clue

- https://www.seequent.com/new-feature-alert-improve-your-vein-modelling-using-a-custom-reference-mesh/
- https://help.seequent.com/Geo/2026.1/en-GB/Content/meshes/2d-interpolant.htm
- https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm

The article says a **custom 2D Interpolant Mesh** can resemble the built-in *curved vein reference* more closely than a separate 3D Interpolant Mesh; it often yields smoother surfaces and extrapolates mean geological trend along strike. A custom 2D interpolant supports a reference plane and reuses the general topography algorithm. This **does not prove** that the default vein curved-reference builder and the custom 2D mesh share exact implementation.

Crucially, the separate official **Veins** help describes the default curved reference as a best fit using **midpoints AND hangingwall/footwall surfaces**. Earlier point-only input-block decoding found no normals/scalars, but does NOT rule out wall geometry contributing indirectly. A midpoint-only interpolator can never be sufficient to establish the native algorithm.

## Independent 2D strike-plane prototype

Only native-extracted reference input **XYZ midpoint arrays** were used in fitting:
1. Derive a robust best-fit geological plane using reweighted PCA of 3D midpoint positions.
2. Transform points into (u,v) coordinates of that plane and measure signed normal displacement.
3. Find the weak-quadratic-curvature direction of the midpoint cloud and optionally extend continuity along it by scaling that coordinate (strike ratio 1.0, 0.5 or 0.2).
4. Fit a *non-native*, 2D thin-plate-spline with an affine polynomial background; tested smoothing parameters 0.05, 1, 5, 20, 100.
5. Project back to 3D. Reject genuinely multivalued folded datasets rather than projecting two limbs through a single XY sheet.
6. **Only after predictions were fitted**, load saved native reference OBJ vertices for pointwise normal-distance RMSE scoring.

Measured results (metres):

| Dataset | Earlier best experimental reference | PCA 2D TPS, smooth=20, ratio=1 | Note |
|---|---:|---:|---|
| Flat | 1.780m datum shift | 1.780m | planar shape exact, reference datum differs |
| Curved, interior | 0.416m (ridge quadratic) | **0.339681m** | both tuned after reviewing native output; do not call out-of-sample |
| Curved, full | 0.712m (ridge quadratic) | **0.296842m** | normal-distance comparison; surface nearly horizontal |
| End-of-hole, interior | 1.496m (robust TPS) | **1.490700m** | nearly unchanged, still wrong |
| Multihit fold | 9.34m central-region 3D ribbon | correctly rejects 2D chart | 3D manifold required |

**Strike anisotropy is not automatically beneficial**: on the curved model with smoothing=20, predicted error increases to 0.547839m when strike ratio=0.5 and 0.655584m when ratio=0.2 (vs 0.339681m for ratio=1). The synthetic input midpoints contain substantial curvature along both X and Y.

## Crucial data-only validation

On the curved case, we selected the ratio/smoothing using **only leave-one-midpoint-out CV**, with no saved native reference mesh access. The best CV candidate was ratio=.5, smoothing=.05 with midpoint LOOCV RMSE **0.490798m**. But **that candidate** scores **1.533325m** against the saved native reference, worse than the target-informed smooth=20 case (0.339681m). The latter instead has midpoint-only LOOCV RMSE **1.712288m**.

Thus the native reference **is not merely minimizing midpoint interpolation error**. Its smoothing is likely governed by a broader geological-trend/reference geometry procedure and/or HW/FW surface information. It remains **unverified** whether Leapfrog's default uses the same 2D topography algorithm described for *custom* interpolant meshes.

## Independent wall geometry check

In XY areas covered by both final HW and FW meshes, the arithmetic HW/FW mid-surface differs from the saved curved reference by ~1.467m RMSE. Flat reference is Z101.78 while observed midpoints and arithmetic HW/FW mean are Z100. So neither midpoint exact interpolation nor simple average of the final wall meshes solves default reference reconstruction.

## Research decision

This is a meaningful improvement in our *surrogate*, not native parity or an independently validated setting. Keep off main. For actual resolution, investigate the default reference's upstream wall/surface inputs and distinguish it from the separate custom 2D interpolant/topography code path. Test on an entirely new geometry or a controlled Leapfrog output with default-vs-custom-reference mode; record fitted surface attitudes and regularization before looking at target mesh scores.
