# Structural trend orientation: saved-project discriminating test (2026-10-16)

## Input and provenance
Read-only rerun of the user-owned `Same project after adding LVA.zip` (1666 seeds, 14 saved domain matrices, 7381 mesh vertices, 14400 triangles). Loaded with restricted symbolic decoder; saved project binaries and source are **not** committed. The earlier fixed-parameter single-trend reconstruction was reproduced exactly: matrix RMSE 0.0117460553556; YY-entry RMSE 0.0002596807812.

## Controlled methods
Use saved `strength=5`, `range=100` with no parameter fitting. Domain predictions are arithmetic means of pointwise volume-preserving uniaxial matrices. Keep the tensor law unchanged and vary **only**:
1. distance to closest sampled triangle vs closest mesh vertex;
2. interpolated gradient on the selected nearest triangle, face normal, global closest vertex gradient, or gradient from the closest **vertex belonging to the selected nearest triangle**, plus two face-normal averages.

Nearest triangles are among the 100 nearest triangle-centroid candidates (not globally exhaustive; other k values were checked separately).

| Distance | Orientation | Domain matrix RMSE | YY RMSE |
|---|---|---:|---:|
| Nearest triangle | Barycentric interpolated vertex gradient | 0.0113060522 | 0.0002596808 |
| Nearest triangle | Gradient of closest vertex within selected triangle | **0.0105527136** | 0.0002596808 |
| Nearest mesh vertex | Barycentric interpolated vertex gradient | 0.0113321464 | **0.0001535506** |
| Nearest mesh vertex | Gradient of closest vertex within selected triangle | **0.0105574451** | **0.0001535506** |
| Nearest triangle | Nearest global vertex gradient | 0.0228298 | 0.0002596808 |

For the full matrix, the best tested rule reduces error by approximately **10.16%** relative to the previous k=80 nearest-triangle baseline. For the independent YY entry, switching to nearest-vertex distance lowers RMSE by **40.87%**. The two best full-matrix candidates are very close; do not assert either is the Leapfrog implementation. Main mismatched domain IDs: 210, 211, 3.

### Interpretation and caveats
- Stronger evidence for the existing distance-decay tensor hypothesis; new hints that the distance and orientation lookup mechanisms may be distinct.
- Saved domain **average matrices** can be biaxial even when each pointwise tensor is uniaxial; `diagnose_structural_tensors.py` intentionally reports the uniaxial residual and only inverts pointwise matrices.
- The candidate sweep reused the 14 target matrices and is **not held-out**. Barycentric nearest-triangle results vary slightly with triangle-candidate count near folds.
- Original saved per-seed tensors / authoritative scalar-field queries are not present. No exact orientation algorithm has been established, and no global progress percentage is upgraded.
- Reproducible original-data script + source of restricted decoder and per-domain/per-candidate CSVs are preserved in this conversation's `Leapfrog_Orientation_Audit_20261016.zip` (the original Leapfrog project is intentionally excluded).

## Next discriminating step
Repeat with at least one **controlled changed structural mesh** and saved per-domain matrices. Compare candidate distance and orientation methods on held-out fold placements, while fixing domain membership. A definitive pointwise orientation recovery will require observing per-point structural matrices or the implementation behind `AnisotropyStandard`, not just more fitting to these 14 means.
