"""Run the confirmed full-depth-domain fix across a strength/range grid.

Defaults:
- strengths: 2, 3, 4, 5
- ranges: 50, 100, ..., 500

The runner reuses the exact recovered Leapfrog LVA sampler, the automatic domain
builder, and the full-depth-domain correction. Cases are processed sequentially
and retain the normal mesh, overlay, basal-diagnostic, and inspection-CSV output.

Optional environment overrides:
- POLATORY_SWEEP_STRENGTHS=2,3,4,5
- POLATORY_SWEEP_RANGES=50,100,150
"""
from __future__ import annotations

import os
from pathlib import Path


def _parse_numbers(name: str, default: list[float]) -> list[float]:
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not value > 0.0:
            raise ValueError(f"{name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} did not contain any numeric values")
    return values


def _case_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1.0e-12:
        return str(int(rounded))
    return f"{value:g}"


STRENGTHS = _parse_numbers(
    "POLATORY_SWEEP_STRENGTHS",
    [2.0, 3.0, 4.0, 5.0],
)
RANGES = _parse_numbers(
    "POLATORY_SWEEP_RANGES",
    [float(value) for value in range(50, 501, 50)],
)

CASE_NAMES = [
    f"S{_case_number(strength)}_R{_case_number(range_)}"
    for strength in STRENGTHS
    for range_ in RANGES
]

# The imported diagnostic resolves requested references from this variable when
# suite.main() starts. Setting it before import avoids a long CMD case list.
os.environ["POLATORY_BASAL_CASES"] = ",".join(CASE_NAMES)

import run_selected_exact_leapfrog_lva_full_depth_domains as full_depth  # noqa: E402

suite = full_depth.suite
diagnostic = full_depth.diagnostic

suite.OUTPUT_DIR = (
    suite.ROOT / "benchmark-results" / "exact-leapfrog-lva-full-depth-sweep"
)
suite.MESH_DIR = suite.OUTPUT_DIR / "meshes"
suite.PLOT_DIR = suite.OUTPUT_DIR / "overlays"
diagnostic.DIAGNOSTIC_DIR = suite.OUTPUT_DIR / "basal-diagnostics"
full_depth.baseline.CSV_DIR = suite.OUTPUT_DIR / "inspection-csv"

print(
    "PROGRESS\tFull-depth parameter sweep enabled: "
    f"{len(STRENGTHS)} strengths x {len(RANGES)} ranges = {len(CASE_NAMES)} cases.",
    flush=True,
)
print(
    "PROGRESS\tStrengths: " + ", ".join(_case_number(v) for v in STRENGTHS),
    flush=True,
)
print(
    "PROGRESS\tRanges: " + ", ".join(_case_number(v) for v in RANGES),
    flush=True,
)
print(
    f"PROGRESS\tResults directory: {suite.OUTPUT_DIR}",
    flush=True,
)


if __name__ == "__main__":
    exit_code = suite.main()
    diagnostic._write_cross_case_comparison()
    raise SystemExit(exit_code)
