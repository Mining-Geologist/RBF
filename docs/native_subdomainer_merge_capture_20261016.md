# Leapfrog runtime: real SubDomainer merge decision experiment

## Evidence from archived profiler

The prior user-provided `leapfrog-real-subdomainer-profile.raw.txt` contains 1,389 unique collapsed stacks and 28,599 weighted profiler samples. The samples contain `get_primary_normals (anisotropy.py)` (19,663 weighted samples) and initial constructors for `GridSeededDomainer` / `SubDomainer`, but **no sampled `merge_domains` or `set_domains_by_region_growing` frames**. The prior locals dump stopped within the real-stage SubDomainer constructor. The archive therefore does **not** expose a native merge sequence or heap decisions. Weighted samples are not function invocation counts or benchmark timing.

The native clustering algorithm remains unresolved; previous inference shows the corrected coarse grid cannot recover real final partitions merely by splitting its groups.

## New implemented instruments

- `benchmarks/leapfrog_gold/trace_merge_stage.py`: opt-in runtime observer for `GridSeededDomainer`, `SubDomainer` and `AnisotropyStandard.get_primary_normals`. It records method args, created/removed domain IDs, live domain counts and queue lengths. Optionally saves the first two native orientation calls as small NPZ arrays. The original method receives identical arguments, runs once and returns unchanged.
- `benchmarks/leapfrog_gold/test_trace_merge_stage.py`: native-independent tests for class-method hooks, accepted domain merges, and error propagation.
- `benchmarks/leapfrog_gold/summarize_merge_events.py`: converts the diagnostic JSONL to `merge_graph.csv`, with the merge order, parent/child IDs and queue changes.

**Important:** Hooking incurs overhead and has not been tested in a real Leapfrog session. Use only on a backed-up synthetic project, keep output separate, and remove the hook after a controlled run. It neither accesses nor changes Leapfrog license/activation components.

## Optional controlled Windows capture

In the existing authorized Leapfrog `sitecustomize.py` bootstrap, before application startup, add:

```python
import os, sys
sys.path.insert(0, r"D:\Polatory_LVA\RBF-LVA\benchmarks\leapfrog_gold")
os.environ["LEAPFROG_MERGE_TRACE"] = "1"
os.environ["LEAPFROG_MERGE_CAPTURE_NORMALS"] = "1"
os.environ["LEAPFROG_MERGE_TRACE_DIR"] = r"D:\Polatory_LVA\merge-trace"
import trace_merge_stage
```

Restart Leapfrog, open a backed-up synthetic project and trigger one LVA recomputation. Check `merge-trace/merge-events.jsonl` for `SubDomainer.merge_domains` records. If only installation and anisotropy records are produced, method installation or triggering must be diagnosed; do **not** claim an exact merge trace.

Run:
```bash
python benchmarks/leapfrog_gold/summarize_merge_events.py D:/Polatory_LVA/merge-trace/merge-events.jsonl
```

Before sharing a trace, inspect it for sensitive client data. Restore normal `sitecustomize.py` and restart.

## Status

The passive instrumentation and parser are prepared and native-independent smoke tests pass on CI. A live native trace has **not** been obtained here. Estimated end-to-end Leapfrog RBF/LVA reconstruction remains approximately 73%, pending native clustering rules, tied-nearest-vertex resolution, and other trend modes.
