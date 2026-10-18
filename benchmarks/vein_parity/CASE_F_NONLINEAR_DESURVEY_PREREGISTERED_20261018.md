# Case F preregistration: nonlinear / dogleg survey paths, raw-vein intersection (2026-10-18)

**Pre-registered offline predictions, NOT observed Leapfrog F outputs.** This project test is prepared after native case E confirmed 3 straight inclined holes, with completely independent E raw-mesh geometry and the same five VEIN-bearing holes. The owner can import the separately shared F synthetic CSVs and save a new F native project.

Three WASTE-hole surveys are now measured at MD0, MD80 and MD200 with changing orientations; collars, lithology, VEIN-bearing hole locations and other 22 straight-hole surveys are unchanged.

| Hole | Survey triplets (MD, azimuth°, dip°) | Predicted two raw-body crossings MD (m) |
|---|---|---|
| VH_X-080_Y+000 | (0,90,60), (80,105,67), (200,125,70) | 104.955081627 – 116.683974358 |
| VH_X+080_Y+000 | (0,270,70), (80,260,65), (200,250,65) | 103.125684329 – 114.675498743 |
| VH_X+000_Y-080 | (0,0,75), (80,15,70), (200,25,70) | 100.193072992 – 110.331883530 |

Computation: independent minimum-curvature representation formed by great-circle interpolation of downhole unit orientations and analytical integration of the tangent, followed by piecewise-segment triangular intersection of the previously saved 1,792-vertex E raw vein on a fine 0.1m-MD path discretization. Approximate geometric discretization error <0.0001 m; the **actual Leapfrog desurvey convention/intermediate interpolation remains unverified** and may differ. These predictions should be interpreted as pre-run hypotheses with a tolerance informed by that uncertainty, NOT expected machine-precision parity.

The user-facing F ZIP contains collar.csv (25 records), survey.csv (53 survey rows), geology.csv (33 intervals), README, and `pre_registered_curved_path_predictions.json`. No vendor binaries/meshes in this repo. Main untouched.

Once the user returns F project, first compare persisted DesurveyBlock trajectories and mesh intersections, *then* match pinchout segment MDs. If they do not match the registered numbers, determine whether this is due to our independent desurvey convention rather than rejecting the proven single-interval raw-mesh intersection rule.
