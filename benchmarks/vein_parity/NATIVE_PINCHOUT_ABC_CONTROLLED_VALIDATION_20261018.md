# Native pinchout A/B/C verification from saved Leapfrog project — 2026-10-18

**Status: confirmed native saved-result measurements, not recovered vendor interpolation code.** Source: user's updated, owner-supplied `Vein_Benchmark(1).zip` saved project (including complete `.aproj_data` directory). We inspected its SQLite persistent-object graph and decoded **only** stored inline ndarray payloads using Python `pickletools.genops` and NumPy `frombuffer`. No proprietary source or binaries were imported/executed or committed.

## Exact comparisons

| | A control | B uniform wider vein | C subdivided WASTE |
|---|---|---|---|
| VEIN intervals (measured depth) | 96–104 m | 94–106 m | 96–104 m |
| Native synthetic pinchout segments (each of 20 holes) | **96–104 m** | **94–106 m** | **96–104 m** |
| Synthetic midpoint (MD) | 100m | 100m | 100m |
| Original, observed wall points: HW/FW Z | 104/96 m | 106/94 m | 104/96 m |
| Generated, reversed pinchout wall points: HW/FW Z | **96/104 m** | **94/106 m** | **96/104 m** |
| Selected pinchout holes | 20 | 20 | 20 |
| Number of reference-input midpoint samples | 5 | 5 | 5 |
| Native reference surface mesh content | Identical | Identical | Identical |
| Pinchout on empty traces | ON | ON | ON |
| Snap fraction | **1.0** | **1.0** | **0.5** |

- All 20 *selected drillhole XY coordinates* are identical in A, B, C.
- A→B: **every** synthetic interval begins 2m earlier and ends 2m later (exact, within floating-point), so its width expands 8m→12m.
- A→C: **all synthetic interval start/end depths and generated reversed HW/FW wall-point XYZ arrays are bit-identical** despite splitting one WASTE log into 0–98.3, 98.3–101.7, 101.7–200.
- All three reference mesh ResultStore inline payload SHA-256: `b0a37bb46aec9529ece7c3a26ed576f8ec65288b245e00f383e6366b2b319a73`. The saved five reference-input midpoint XYZ positions are the same within numerical precision. The reference doesn't automatically move when all target intervals expand symmetrically about their midpoints.
- **Critical control**: New A's `segments`, `depths`, `collar_ids`, saved reference mesh, and saved refined vein mesh inline payloads are **byte-for-byte identical** to the original `02_PINCHOUT_B_empty_ON` test from the same original benchmark. The A reimport was faithful.

### Interpretation and limitation

**Confirmed behavior in this flat synthetic case:** Leapfrog generates pseudo-interval geometry independently of WASTE logging contact boundaries, updates pseudo-interval depths when real VEIN thickness changes, and reverses the HW/FW polarity of those artificial controls. This matches a raw-vein geometry intersection followed by inverted wall contact generation, **but does not yet prove the exact intersection solver, branch-selection policy or generalized width rule in curved/overturned cases**. The earlier native object graph found direct `VeinPinchOutSegmentsBlock` dependencies on the raw `VeinBlock`, `DesurveyBlock`, and `VeinPinchOutCollarsSelectorBlock`.

**Experimental confound:** C has `snap_fraction=0.5` versus 1.0 in A/B. This **does not invalidate native pseudo-interval equality**, but the C final HW/FW or solid cannot be interpreted as a matched WASTE-split-only geometry comparison until the snap fraction is made the same and those meshes rebuilt.

### Next discriminating test: nonuniform support

A separate owner-facing dataset `D_single_center_16m_others_8m` keeps four of five target intervals at MD96–104, expands only the central `VH_X+000_Y+000` target to MD92–108, retains midpoint MD100, and leaves all collars/survey and remaining lithology unchanged. Apply the **same** raw-vein/reference resolution, model boundary, pinchout eligibility, and contact snap fraction 1.0 as A. Save the full project. Variation in calculated pseudo-interval width across nearby non-vein holes would distinguish a *spatially varying preliminary wall-intersection* rule from copying one constant observed interval width everywhere. Do not claim a result for D until a native project has been supplied.

## Reproduction and data handling

The owner-facing `Leapfrog_Pinchout_ABC_Results_20261018.zip` contains a standalone *read-only static* `native_pinchout_compare.py` script, JSON summary and a 60-row per-hole synthetic-interval CSV. No native app or vendor code is used. Proprietary benchmark artifacts are not published in this GitHub repository. Keep changes on the research branch only; `main` unchanged.
