# 2026-10-18: Folded reference, intrusion linear drift, and intrinsic-distance tests

**Scope:** independent read-only analysis of the owner's synthetic folded-vein benchmark and privately supplied real multi-vein case. No vendor source code, binaries or private model geometry was executed/published. All names of private real models are anonymized here.

## User-observed intrusion analogue
Owner reports a manually created intrusion from vein midpoints with “Linear Interpolant” and “Linear Drift”, which visually resembles the vein folded reference but points in another direction and has a central blob. **That intrusion output has not been supplied** and cannot yet be scored quantitatively. Seequent's public vein documentation says first-pass vein references use midpoints and HW/FW data, whereas intrusions are separate contact-surface objects. Seequent's interpolant documentation describes Linear drift as linear far-field behavior; the experiments below are **SciPy mathematical proxies, not confirmed vendor interpolation algorithms**.
- https://help.seequent.com/Geo/2026.1/en-GB/Content/geo-models/veins.htm
- https://help.seequent.com/Works/2026.1/en-GB/Content/geo-models/intrusions.htm
- https://help.seequent.com/Works/2025.3/en-GB/Content/concepts/interpolant-functions.htm

## Direct native 04_MULTIPLE_INTERSECTIONS fold audit

Saved benchmark native reference: 349 vertices, 606 triangles; 15 saved original midpoint input XYZs.

- Input midpoint cloud is **Y-mirror symmetric**; the native reference isn't.
- Acute angle between midpoint-cloud and saved-reference best-fit plane normals: **48.067458°**.
- Exact Euclidean point-to-triangle distances from input midpoint cloud to saved reference: **17.3968527m RMSE**, median **8.49718m**, max **45.67946m**.
- We supplied the proxy with the native reference's **POST HOC oblique chart normal** (an unfair advantage, so these cannot be blind reference predictions).
- Fit linear 2D SciPy RBF to the 15 midpoint signed oblique coordinates, then score signed chart-normal errors at the 349 native output vertices:

| Proxy kernel and polynomial drift | Native reference chart-normal vertex RMSE |
|---|---:|
| linear RBF, constant polynomial (degree=0), smoothing=0 | 28.7354m |
| **linear RBF, affine/linear polynomial (degree=1), smoothing=0** | **44.5175m** |
| multiquadric epsilon1, constant polynomial, smoothing=0 | 28.4584m |
| TPS, affine polynomial, smoothing=0 | 59.3520m |

All four exact-fit variants reproduce the 15 midpoint signed chart coordinates at near machine precision; **this is insufficient to reproduce native reference geometry**, because the actual native mesh isn't forced to pass through all those midpoints and uses additional structural/orientation controls. These are *normal-coordinate vertex residuals*, not Hausdorff distances; the native reference chart itself is an input. They neither confirm nor refute whether the proprietary intrusion uses a linear RBF family.

## Private real reference-following wall test

A 43,784-vertex inherited custom reference mesh was known from the saved private project. Test signed-distance RBF wall construction by moving each reference vertex along its local smoothed normal, fitted to original true observed HW/FW contacts. Compare input coordinate metric:
1. Previous PCA 2D chart with Euclidean distance.
2. New *intrinsic-approximate* surface graph edge-shortest-path distance via sparse Dijkstra, with contact anchors **projected within triangles** to avoid nearest-vertex rounding of up to ~7m. Fit distance-kernel k(d)=d and constant polynomial drift. Same reference topology, same original contacts, same held-out native raw-wall evaluation sections.

| Private child | wall | PCA chart plus local normals RMSE | Triangle-anchored intrinsic graph plus local normals RMSE |
|---|---|---:|---:|
| A | HW | **3.4200m** | 5.8307m |
| A | FW | **0.5617m** | 4.9786m |
| B | HW | **3.4407m** | 5.4466m |
| B | FW | **1.0057m** | 2.9377m |
| C (three genuine controls) | HW | **0.0731m** | 0.1766m |
| C | FW | **0.0814m** | 0.1808m |

Adding chart-affine drift to the approximate intrinsic graph metric worsens most native-wall scores further. True blocked-contact validation is mixed. Thus the naive intrinsic-distance kernel is a **negative result**, not the missing native structural-trend mechanism. It does NOT rule out a proper smooth intrinsic parameterization, local anisotropy or native structural fields. Graph distance is only an edge-path approximation, and nearest-surface association of sparse contacts can be ambiguous.

## Next decisive native dataset

Owner to save the **midpoint-only intrusion** in the original synthetic project and provide the complete .aproj + .aproj_data directory, or at minimum the intrusion contact-surface OBJ including the central blob plus its settings. Then score component topology, principal directions, rigid/rotational alignment and symmetric surface distances versus the original folded reference. A subsequent experiment varying **only intrusion drift** or **only trend orientation** would isolate those parameters. A screenshot alone cannot distinguish a rotated equivalent shape from a different implicit topology.

## Reproducibility / limits

Generic self-contained prototype tools are retained in the owner-facing ZIP `Leapfrog_Intrusion_Drift_and_Intrinsic_Reference_Investigation_20261018.zip`, including 5 passing synthetic tests, fold audit and private read-only validation drivers. Public report reveals only aggregate comparisons; no proprietary data exported to the repository. Main branch unchanged.