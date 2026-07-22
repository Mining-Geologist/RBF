"""Run the confirmed full-depth-domain fix across a strength/range grid.

Defaults:
- strengths: 2, 3, 4, 5
- ranges: 50, 100, ..., 500

The sweep is intentionally allowed to include parameter combinations for which no
Leapfrog reference OBJ exists. Every requested case still generates a Polatory OBJ,
a three-projection image, basal diagnostics, domain CSVs and an LVA-field CSV.
When a matching Leapfrog OBJ is present, the normal comparison metrics and overlay
are retained; otherwise the report marks the case as generated-only.

Optional environment overrides:
- POLATORY_SWEEP_STRENGTHS=2,3,4,5
- POLATORY_SWEEP_RANGES=50,100,150
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


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

# Keep the requested case list visible to the imported basal diagnostic, then
# replace its reference-only case resolver below with a synthetic sweep resolver.
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


def _sweep_cases() -> list[Any]:
    """Create every requested case, whether or not a reference OBJ exists."""
    return [
        suite.Case(
            name=f"S{_case_number(strength)}_R{_case_number(range_)}",
            strength=float(strength),
            trend_range=float(range_),
            reference=Path(suite.DATA_DIR)
            / f"S{_case_number(strength)}_R{_case_number(range_)}.obj",
        )
        for strength in STRENGTHS
        for range_ in RANGES
    ]


# The normal basal runner validates requested names against reference OBJ names.
# A parameter sweep must instead be able to generate arbitrary combinations.
suite.available_cases = _sweep_cases

_ORIGINAL_COMPARE_MESHES = suite.compare_meshes
_ORIGINAL_RENDER_OVERLAY = suite.render_overlay


def _optional_compare_meshes(generated_path: Path, reference_path: Path) -> dict[str, Any]:
    """Compare when an oracle exists; otherwise retain generated mesh metrics."""
    reference_path = Path(reference_path)
    if reference_path.is_file():
        comparison = _ORIGINAL_COMPARE_MESHES(generated_path, reference_path)
        comparison["reference_available"] = True
        return comparison

    generated = suite.polydata(generated_path)
    return {
        "reference_available": False,
        "generated": suite.mesh_metrics(generated),
        "reference": None,
        "generated_to_reference": None,
        "reference_to_generated": None,
        "symmetric_surface_distance": None,
        "area_ratio_generated_over_reference": None,
    }


def _generated_projection(case: Any, generated_path: Path) -> None:
    """Render a useful generated-only view when no Leapfrog oracle is available."""
    if Path(case.reference).is_file():
        _ORIGINAL_RENDER_OVERLAY(case, generated_path)
        return

    generated, _ = suite.read_obj(generated_path)
    limits = [
        (suite.MODEL_MIN[0], suite.MODEL_MAX[0]),
        (suite.MODEL_MIN[1], suite.MODEL_MAX[1]),
        (suite.MODEL_MIN[2], suite.MODEL_MAX[2]),
    ]
    projections = (
        ("XY plan", 0, 1),
        ("XZ section projection", 0, 2),
        ("YZ section projection", 1, 2),
    )
    labels = ("X", "Y", "Z")
    figure, axes = suite.plt.subplots(1, 3, figsize=(18, 6))
    stride = max(1, len(generated) // 100_000)
    for axis, (title, first, second) in zip(axes, projections):
        axis.scatter(
            generated[::stride, first],
            generated[::stride, second],
            s=0.4,
            alpha=0.45,
            label="Polatory",
        )
        axis.set_title(title)
        axis.set_xlabel(labels[first])
        axis.set_ylabel(labels[second])
        axis.set_xlim(limits[first])
        axis.set_ylim(limits[second])
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend(markerscale=10)
    figure.suptitle(
        f"{case.name}: Polatory generated surface, "
        f"resolution {suite.SURFACE_RESOLUTION:g} m\n"
        "No matching Leapfrog reference OBJ was available"
    )
    figure.tight_layout()
    suite.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(suite.PLOT_DIR / f"{case.name}_overlay.png", dpi=180)
    suite.plt.close(figure)


suite.compare_meshes = _optional_compare_meshes
suite.render_overlay = _generated_projection


def _generated_response_metrics(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize generated changes without requiring reference meshes."""
    output: list[dict[str, Any]] = []
    grouped: dict[float, list[dict[str, Any]]] = {}
    for item in results.values():
        grouped.setdefault(float(item["strength"]), []).append(item)

    for strength, group in sorted(grouped.items()):
        group.sort(key=lambda item: float(item["trend_range"]))
        for lower, upper in zip(group, group[1:]):
            lower_generated = lower["comparison"]["generated"]
            upper_generated = upper["comparison"]["generated"]
            lower_bounds = np.asarray(
                lower_generated["bounds_min"] + lower_generated["bounds_max"],
                dtype=float,
            )
            upper_bounds = np.asarray(
                upper_generated["bounds_min"] + upper_generated["bounds_max"],
                dtype=float,
            )
            output.append(
                {
                    "strength": float(strength),
                    "from_range": float(lower["trend_range"]),
                    "to_range": float(upper["trend_range"]),
                    "generated_bounds_delta": (upper_bounds - lower_bounds).tolist(),
                    "generated_area_ratio": float(
                        upper_generated["area"] / lower_generated["area"]
                    ),
                    "generated_vertex_delta": int(
                        upper_generated["vertices"] - lower_generated["vertices"]
                    ),
                }
            )
    return output


def _write_sweep_summary(report: dict[str, Any]) -> None:
    lines = [
        "# Full-depth LVA parameter sweep",
        "",
        f"- Surface resolution: {suite.SURFACE_RESOLUTION:g} m",
        f"- Cases: {len(report['cases'])}",
        "- Missing references are generated and visualized without oracle metrics.",
        "",
        "| Case | Domains | Reference | Z min | Z max | Vertices | Area | Seconds | Symmetric mean | Symmetric p95 |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["cases"].items():
        comparison = item["comparison"]
        generated = comparison["generated"]
        distance = comparison.get("symmetric_surface_distance")
        symmetric_mean = f"{distance['mean']:.3f}" if distance is not None else "—"
        symmetric_p95 = f"{distance['p95']:.3f}" if distance is not None else "—"
        lines.append(
            f"| {name} | {item['domain_count']} | "
            f"{'yes' if comparison.get('reference_available') else 'no'} | "
            f"{generated['bounds_min'][2]:.2f} | {generated['bounds_max'][2]:.2f} | "
            f"{generated['vertices']} | {generated['area']:.2f} | "
            f"{item['elapsed_seconds']:.1f} | {symmetric_mean} | {symmetric_p95} |"
        )
    (suite.OUTPUT_DIR / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


suite.response_metrics = _generated_response_metrics
suite.write_summary = _write_sweep_summary

_REFERENCE_COUNT = sum(Path(case.reference).is_file() for case in _sweep_cases())
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
    f"PROGRESS\tLeapfrog references found for {_REFERENCE_COUNT}/{len(CASE_NAMES)} cases; "
    "all remaining combinations will be generated without oracle comparison.",
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
