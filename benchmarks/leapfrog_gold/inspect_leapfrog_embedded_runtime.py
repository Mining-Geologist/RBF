"""Inspect Leapfrog's embedded structural-domaining runtime.

This diagnostic has two complementary modes.

1. ``inventory`` runs from a normal Python interpreter and scans the Leapfrog
   installation for PE modules, embedded ``PyInit_*`` exports, algorithm-related
   strings, Python archives, and structural/domaining files.
2. ``runtime`` is imported inside Leapfrog's own Python process (normally from an
   existing ``sitecustomize.py``). It reflects and disassembles the archived Python
   modules, instruments the real-location domaining calls, saves their arrays, and
   can run synthetic SubDomainer micro-cases using Leapfrog's actual implementation.

The probe is read-only with respect to Leapfrog projects. Runtime hooks call the
original functions unchanged and only write diagnostics to a separate directory.
"""
from __future__ import annotations

import argparse
import builtins
import dis
import functools
import hashlib
import importlib
import inspect
import io
import json
import mmap
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

DEFAULT_INSTALL_ROOT = Path(r"D:\Program files\Seequent\Leapfrog 2026.1\bin")
DEFAULT_INVENTORY = Path("benchmark-results/leapfrog-runtime-inventory.json")
MODULE_CANDIDATES = (
    "structural_fitter",
    "structural_fitter.domaining",
    "structural_fitter.anisotropy",
    "rangeTree",
    "triangle_tree",
    "turbo_rbf_spaces",
    "_isosurfacer",
)
MEMBER_HINTS = (
    "domain",
    "region",
    "merge",
    "consisten",
    "anisotrop",
    "normal",
    "matrix",
    "determinant",
    "tree",
    "neigh",
    "adjacen",
    "location",
)
BINARY_HINTS = (
    b"SubDomainer",
    b"GridSeededDomainer",
    b"set_domains_by_region_growing",
    b"subdomain_with_real_locations",
    b"merge_domains",
    b"consistency_thresh",
    b"affine_matrix_and_consistency",
    b"symmetric_determinant",
    b"region_growing",
    b"delaunay",
    b"nearest",
    b"adjacency",
    b"RangeTree",
)
RUNTIME_MODULE_TRIGGERS = {
    "structural_fitter",
    "structural_fitter.domaining",
    "structural_fitter.anisotropy",
}
_LOCK = threading.RLock()
_INSTALLED = False
_ORIGINAL_IMPORT: Callable[..., Any] | None = None
_PATCHED: set[tuple[int, str]] = set()
_CALL_COUNTER = 0


def _output_root() -> Path:
    configured = os.environ.get("LEAPFROG_RE_PROBE_DIR", "").strip()
    if configured:
        root = Path(configured)
    else:
        root = Path(os.environ.get("TEMP", ".")) / "leapfrog-re-probe"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.nanmin(value)) if value.size and np.issubdtype(value.dtype, np.number) else None,
            "max": float(np.nanmax(value)) if value.size and np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        items = list(value.items())[:100]
        return {str(key): _jsonable(item, depth=depth + 1) for key, item in items}
    return {
        "type": type(value).__name__,
        "module": type(value).__module__,
        "repr": repr(value)[:500],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_event(payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault("time", time.time())
    path = _output_root() / "runtime-events.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pe_details(path: Path) -> dict[str, Any]:
    details: dict[str, Any] = {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "pyinit_symbols": [],
        "hint_hits": [],
        "exports": [],
        "imports": [],
    }
    try:
        with path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                details["pyinit_symbols"] = sorted(
                    {
                        match.group(1).decode("ascii", errors="replace")
                        for match in re.finditer(rb"PyInit_([A-Za-z0-9_]+)", mapped)
                    }
                )
                details["hint_hits"] = [
                    hint.decode("ascii", errors="replace")
                    for hint in BINARY_HINTS
                    if mapped.find(hint) >= 0
                ]
    except (OSError, ValueError) as exc:
        details["binary_scan_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import pefile  # type: ignore

        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            ]
        )
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            details["exports"] = sorted(
                symbol.name.decode("utf-8", errors="replace")
                for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols
                if symbol.name
            )
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            details["imports"] = sorted(
                entry.dll.decode("utf-8", errors="replace")
                for entry in pe.DIRECTORY_ENTRY_IMPORT
            )
    except ImportError:
        details["pefile"] = "not installed; binary string scan still completed"
    except Exception as exc:
        details["pe_error"] = f"{type(exc).__name__}: {exc}"
    return details


def inventory_installation(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    suffixes = {".exe", ".dll", ".pyd", ".py", ".pyc", ".zip", ".zpy", ".pydc"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)

    pe_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for path in sorted(files):
        suffix = path.suffix.lower()
        if suffix in {".exe", ".dll", ".pyd"}:
            row = _pe_details(path)
            if row["pyinit_symbols"] or row["hint_hits"] or "leapfrog" in path.name.lower():
                pe_rows.append(row)
        elif suffix in {".zip", ".zpy", ".pydc"}:
            archive_rows.append(
                {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
            )
        elif suffix in {".py", ".pyc"}:
            lowered = str(path).lower()
            if any(term in lowered for term in ("domain", "anisotrop", "structural", "rbf")):
                source_rows.append({"path": str(path), "size": path.stat().st_size})

    report = {
        "install_root": str(root),
        "python": sys.version,
        "pe_modules": pe_rows,
        "archives": archive_rows,
        "structural_source_files": source_rows,
        "summary": {
            "all_candidate_files": len(files),
            "interesting_pe_modules": len(pe_rows),
            "archives": len(archive_rows),
            "structural_source_files": len(source_rows),
        },
    }
    _write_json(output, report)
    return report


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except Exception:
        return None


def _disassemble(value: Any, limit: int = 120_000) -> str | None:
    try:
        target = value
        if inspect.ismethod(target):
            target = target.__func__
        if not hasattr(target, "__code__"):
            return None
        stream = io.StringIO()
        dis.dis(target, file=stream, show_caches=True, adaptive=True)
        return stream.getvalue()[:limit]
    except Exception as exc:
        return f"<disassembly failed: {type(exc).__name__}: {exc}>"


def _describe_member(name: str, value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "type": type(value).__name__,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "signature": _safe_signature(value),
        "doc": inspect.getdoc(value)[:4000] if inspect.getdoc(value) else None,
    }
    try:
        source = inspect.getsource(value)
        result["source"] = source[:200_000]
    except Exception:
        result["source"] = None
    result["disassembly"] = _disassemble(value)
    if inspect.isclass(value):
        methods: dict[str, Any] = {}
        for child_name, child in sorted(vars(value).items()):
            if child_name.startswith("__") and child_name not in {"__init__"}:
                continue
            if callable(child) and any(term in child_name.lower() for term in MEMBER_HINTS):
                methods[child_name] = {
                    "signature": _safe_signature(child),
                    "doc": inspect.getdoc(child)[:2000] if inspect.getdoc(child) else None,
                    "disassembly": _disassemble(child),
                }
        result["methods"] = methods
    return result


def reflect_module(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:
        return {
            "module": module_name,
            "imported": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    names = sorted(name for name in dir(module) if not name.startswith("__"))
    interesting: dict[str, Any] = {}
    for name in names:
        if any(term in name.lower() for term in MEMBER_HINTS):
            try:
                interesting[name] = _describe_member(name, getattr(module, name))
            except Exception as exc:
                interesting[name] = {"error": f"{type(exc).__name__}: {exc}"}

    code_disassembly = None
    try:
        loader = getattr(module, "__loader__", None)
        get_code = getattr(loader, "get_code", None)
        code = get_code(module_name) if callable(get_code) else None
        if code is not None:
            stream = io.StringIO()
            dis.dis(code, file=stream, show_caches=True, adaptive=True)
            code_disassembly = stream.getvalue()[:500_000]
    except Exception as exc:
        code_disassembly = f"<module disassembly failed: {type(exc).__name__}: {exc}>"

    return {
        "module": module_name,
        "imported": True,
        "file": getattr(module, "__file__", None),
        "loader": repr(getattr(module, "__loader__", None)),
        "spec": repr(getattr(module, "__spec__", None)),
        "public_names": names,
        "interesting_members": interesting,
        "module_disassembly": code_disassembly,
    }


def dump_runtime_reflection() -> Path:
    payload = {
        "executable": sys.executable,
        "version": sys.version,
        "modules": [reflect_module(name) for name in MODULE_CANDIDATES],
    }
    path = _output_root() / "runtime-reflection.json"
    _write_json(path, payload)
    return path


def _array_payload(prefix: str, value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        key = prefix.replace(".", "_").replace("[", "_").replace("]", "")
        arrays[key] = np.asarray(value)
        return {"array_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {
            str(key): _array_payload(f"{prefix}_{key}", item, arrays)
            for key, item in list(value.items())[:500]
        }
    if isinstance(value, (list, tuple)):
        return [
            _array_payload(f"{prefix}_{index}", item, arrays)
            for index, item in enumerate(value[:500])
        ]
    return _jsonable(value)


def _capture_call(label: str, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any = None) -> None:
    global _CALL_COUNTER
    with _LOCK:
        _CALL_COUNTER += 1
        call_id = _CALL_COUNTER
    arrays: dict[str, np.ndarray] = {}
    payload: dict[str, Any] = {
        "call_id": call_id,
        "label": label,
        "args": [_array_payload(f"arg{index}", value, arrays) for index, value in enumerate(args)],
        "kwargs": {
            key: _array_payload(f"kw_{key}", value, arrays) for key, value in kwargs.items()
        },
    }
    if result is not None:
        payload["result"] = _array_payload("result", result, arrays)
    if args:
        self_value = args[0]
        state = getattr(self_value, "__dict__", None)
        if isinstance(state, dict):
            payload["self_state"] = {
                key: _array_payload(f"self_{key}", value, arrays)
                for key, value in state.items()
                if not callable(value)
            }
    if arrays:
        np.savez_compressed(_output_root() / f"call-{call_id:06d}-{label.replace('.', '_')}.npz", **arrays)
    _append_event(payload)


def _patch_method(owner: Any, name: str, label: str) -> None:
    key = (id(owner), name)
    if key in _PATCHED or not hasattr(owner, name):
        return
    original = getattr(owner, name)
    if not callable(original):
        return

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        _capture_call(f"{label}.enter", args, kwargs)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _append_event(
                {
                    "label": f"{label}.error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            raise
        _capture_call(f"{label}.exit", args, kwargs, result)
        return result

    try:
        setattr(owner, name, wrapped)
        _PATCHED.add(key)
        _append_event({"label": "patch", "target": label, "status": "installed"})
    except Exception as exc:
        _append_event(
            {"label": "patch", "target": label, "status": "failed", "error": str(exc)}
        )


def instrument_loaded_modules() -> None:
    domaining = sys.modules.get("structural_fitter.domaining")
    if domaining is not None:
        sub = getattr(domaining, "SubDomainer", None)
        grid = getattr(domaining, "GridSeededDomainer", None)
        if sub is not None:
            for method in (
                "__init__",
                "set_domains_by_region_growing",
                "merge_domains",
                "get_domains",
            ):
                _patch_method(sub, method, f"SubDomainer.{method}")
        if grid is not None:
            for method in (
                "__init__",
                "subdomain_with_real_locations",
                "set_domains_by_region_growing",
                "merge_domains",
            ):
                _patch_method(grid, method, f"GridSeededDomainer.{method}")

    anisotropy = sys.modules.get("structural_fitter.anisotropy")
    if anisotropy is not None:
        for _, cls in inspect.getmembers(anisotropy, inspect.isclass):
            for method in (
                "get_primary_normals",
                "get_anisotropies",
                "get_anisotropies_and_strength",
            ):
                _patch_method(cls, method, f"{cls.__name__}.{method}")


def install_import_hook(*, reflect: bool = True) -> None:
    global _INSTALLED, _ORIGINAL_IMPORT
    with _LOCK:
        if _INSTALLED:
            instrument_loaded_modules()
            return
        _INSTALLED = True
        _ORIGINAL_IMPORT = builtins.__import__
        original = _ORIGINAL_IMPORT

        def hooked_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
            module = original(name, globals_, locals_, fromlist, level)
            if name in RUNTIME_MODULE_TRIGGERS or name.startswith("structural_fitter"):
                try:
                    instrument_loaded_modules()
                except Exception as exc:
                    _append_event({"label": "instrument_error", "error": str(exc)})
            return module

        builtins.__import__ = hooked_import
        _append_event(
            {
                "label": "probe_installed",
                "executable": sys.executable,
                "version": sys.version,
            }
        )
    instrument_loaded_modules()
    if reflect:
        try:
            dump_runtime_reflection()
        except Exception as exc:
            _append_event({"label": "reflection_error", "error": str(exc)})


def _rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(angle_degrees)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one = 1.0 - c
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=float,
    )


def _spd_for_angle(angle: float, ratio: float = 5.0) -> np.ndarray:
    rotation = _rotation_matrix(np.array([0.0, 0.0, 1.0]), angle)
    eigenvalues = np.array([ratio ** (-1.0 / 3.0), ratio ** (-1.0 / 3.0), ratio ** (2.0 / 3.0)])
    base = np.diag(eigenvalues)
    return rotation @ base @ rotation.T


class _StaticAnisotropyField:
    def __init__(self, matrices: np.ndarray, strengths: np.ndarray | None = None) -> None:
        self.matrices = np.asarray(matrices, dtype=float)
        self.strengths = (
            np.ones(len(self.matrices), dtype=float)
            if strengths is None
            else np.asarray(strengths, dtype=float)
        )

    def _select(self, locations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(np.asarray(locations))
        if len(self.matrices) == count:
            return self.matrices.copy(), self.strengths.copy()
        if len(self.matrices) == 1:
            return (
                np.repeat(self.matrices, count, axis=0),
                np.repeat(self.strengths, count),
            )
        raise ValueError(f"Expected {len(self.matrices)} locations, received {count}")

    def get_anisotropies(self, locations: np.ndarray) -> np.ndarray:
        return self._select(locations)[0]

    def get_anisotropies_and_strength(self, locations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._select(locations)


def _candidate_label_vectors(value: Any, point_count: int) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    state = getattr(value, "__dict__", {})
    if not isinstance(state, dict):
        return found
    for name, item in state.items():
        array = np.asarray(item) if isinstance(item, (list, tuple, np.ndarray)) else None
        if array is None or array.ndim != 1 or len(array) != point_count:
            continue
        if np.issubdtype(array.dtype, np.integer) or np.all(np.isfinite(array) & (array == np.round(array))):
            found[name] = array.astype(np.int64).tolist()
    return found


def run_subdomainer_microcases() -> Path:
    domaining = importlib.import_module("structural_fitter.domaining")
    subdomainer = getattr(domaining, "SubDomainer")
    cases: list[dict[str, Any]] = []

    point_sets = {
        "line6": np.column_stack((np.arange(6, dtype=float), np.zeros(6), np.zeros(6))),
        "two_triads": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
            dtype=float,
        ),
    }
    matrix_sets = {
        "constant": np.repeat(_spd_for_angle(0.0)[None, :, :], 6, axis=0),
        "two_orientations": np.asarray(
            [_spd_for_angle(0.0)] * 3 + [_spd_for_angle(90.0)] * 3,
            dtype=float,
        ),
        "orientation_ramp": np.asarray([_spd_for_angle(angle) for angle in (0, 10, 20, 60, 70, 80)]),
    }

    for point_name, points in point_sets.items():
        for matrix_name, matrices in matrix_sets.items():
            for threshold in (0.0, 0.6, 0.9, 0.99, 0.999):
                record: dict[str, Any] = {
                    "points": point_name,
                    "matrices": matrix_name,
                    "threshold": threshold,
                }
                try:
                    field = _StaticAnisotropyField(matrices)
                    obj = subdomainer(
                        points,
                        field,
                        bbox=None,
                        consistency_thresh=threshold,
                        min_points=1,
                        max_points=len(points),
                    )
                    record["object"] = _jsonable(obj)
                    record["state"] = _jsonable(getattr(obj, "__dict__", {}))
                    record["candidate_labels"] = _candidate_label_vectors(obj, len(points))
                    for method_name in ("get_domains", "domains", "labels", "get_labels"):
                        method = getattr(obj, method_name, None)
                        if callable(method):
                            try:
                                output = method()
                                record[f"method_{method_name}"] = _jsonable(output)
                            except Exception as exc:
                                record[f"method_{method_name}_error"] = str(exc)
                except BaseException as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback"] = traceback.format_exc()
                cases.append(record)

    path = _output_root() / "subdomainer-microcases.json"
    _write_json(path, cases)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "runtime-reflect", "microcases"), default="inventory")
    parser.add_argument("--install-root", type=Path, default=DEFAULT_INSTALL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    if args.mode == "inventory":
        if not args.install_root.is_dir():
            parser.error(f"Leapfrog installation was not found: {args.install_root}")
        report = inventory_installation(args.install_root, args.output)
        print(json.dumps(report["summary"], indent=2))
        print(f"wrote {args.output}")
        return 0

    if args.mode == "runtime-reflect":
        install_import_hook(reflect=False)
        path = dump_runtime_reflection()
        print(f"wrote {path}")
        return 0

    install_import_hook(reflect=True)
    path = run_subdomainer_microcases()
    print(f"wrote {path}")
    return 0


if os.environ.get("LEAPFROG_RE_PROBE", "").strip() == "1":
    try:
        install_import_hook(reflect=True)
        if os.environ.get("LEAPFROG_RE_MICROCASES", "").strip() == "1":
            run_subdomainer_microcases()
    except Exception as exc:
        _append_event(
            {
                "label": "auto_activation_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
