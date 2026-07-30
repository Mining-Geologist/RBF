"""Probe Leapfrog's installed Python runtime and structural domaining API.

This script is intentionally conservative. It does not modify Leapfrog projects and does
not construct the full GridSeededDomainer yet. It first discovers the Python ABI, import
paths, relevant modules, public call signatures, and a few safe matrix helper outputs.

Run it from the repository with a normal Python interpreter. It will search the supplied
Leapfrog installation for an embedded Python executable. When found, it re-executes itself
under that interpreter so Windows ``.pyd`` modules are loaded with the correct ABI.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BIN = Path(r"D:\Program files\Seequent\Leapfrog 2026.1\bin")
DEFAULT_OUTPUT = Path("benchmark-results/leapfrog-runtime-probe.json")
MODULE_CANDIDATES = (
    "structural_fitter",
    "structural_fitter.domaining",
    "structural_fitter.anisotropy",
)
NAME_HINTS = (
    "SubDomainer",
    "GridSeededDomainer",
    "DomainInfo",
    "affine_matrix_and_consistency",
    "symmetric_determinant",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def _safe_signature(value: Any) -> str | None:
    try:
        return str(inspect.signature(value))
    except Exception:
        return None


def _safe_doc(value: Any, limit: int = 1200) -> str | None:
    try:
        text = inspect.getdoc(value)
    except Exception:
        return None
    if not text:
        return None
    return text[:limit]


def _candidate_python_executables(root: Path) -> list[Path]:
    names = {"python.exe", "pythonw.exe"}
    candidates: list[Path] = []
    search_roots = [root, root.parent, root.parent.parent]
    seen: set[Path] = set()
    for base in search_roots:
        if not base.exists():
            continue
        try:
            iterator = base.rglob("*.exe")
        except OSError:
            continue
        for path in iterator:
            if path.name.lower() not in names:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    return sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))


def _runtime_files(root: Path) -> dict[str, list[str]]:
    patterns = {
        "python_dlls": "python*.dll",
        "pyd_files": "*.pyd",
        "domaining_files": "*domain*",
        "anisotropy_files": "*anisotrop*",
        "structural_fitter_files": "*structural_fitter*",
    }
    result: dict[str, list[str]] = {}
    for key, pattern in patterns.items():
        matches: list[str] = []
        if root.exists():
            try:
                for path in root.rglob(pattern):
                    if path.is_file():
                        matches.append(str(path))
            except OSError as exc:
                matches.append(f"<search failed: {type(exc).__name__}: {exc}>")
        result[key] = sorted(matches)[:500]
    return result


def _dll_and_sys_path_setup(root: Path) -> dict[str, Any]:
    added_dll_dirs: list[str] = []
    added_sys_paths: list[str] = []
    directories: list[Path] = []
    if root.exists():
        directories.append(root)
        try:
            directories.extend(path for path in root.rglob("*") if path.is_dir())
        except OSError:
            pass
    seen: set[str] = set()
    for directory in directories:
        text = str(directory)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_sys_paths.append(text)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(text)
                added_dll_dirs.append(text)
            except (FileNotFoundError, OSError):
                pass
    return {
        "added_dll_directories": added_dll_dirs,
        "added_sys_paths": added_sys_paths,
    }


def _describe_member(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": type(value).__name__,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
        "signature": _safe_signature(value),
        "doc": _safe_doc(value),
        "is_class": inspect.isclass(value),
        "is_function": inspect.isfunction(value) or inspect.isbuiltin(value),
    }


def _probe_module(module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"module": module_name}
    try:
        module = importlib.import_module(module_name)
    except BaseException as exc:  # import failures can include DLL loader exceptions
        result.update(
            {
                "imported": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return result

    result.update(
        {
            "imported": True,
            "file": getattr(module, "__file__", None),
            "package": getattr(module, "__package__", None),
        }
    )
    public_names = sorted(name for name in dir(module) if not name.startswith("__"))
    result["public_names"] = public_names
    members: dict[str, Any] = {}
    for name in public_names:
        if any(hint.lower() in name.lower() for hint in NAME_HINTS):
            try:
                members[name] = _describe_member(name, getattr(module, name))
            except BaseException as exc:
                members[name] = {"error": f"{type(exc).__name__}: {exc}"}
    result["interesting_members"] = members
    return result


def _safe_matrix_tests(imported_modules: dict[str, Any]) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    anisotropy_module = imported_modules.get("structural_fitter.anisotropy")
    if anisotropy_module is None:
        return tests

    identity = np.eye(3, dtype=np.float64)
    rotated = np.array(
        [[2.0, 0.25, 0.0], [0.25, 0.75, 0.0], [0.0, 0.0, 2.0 / 1.4375]],
        dtype=np.float64,
    )
    for name in ("symmetric_determinant", "affine_matrix_and_consistency"):
        value = getattr(anisotropy_module, name, None)
        if value is None:
            continue
        cases: list[dict[str, Any]] = []
        try:
            if name == "symmetric_determinant":
                for label, matrix in (("identity", identity), ("rotated", rotated)):
                    try:
                        output = value(matrix)
                        cases.append({"case": label, "output": _jsonable(output)})
                    except BaseException as exc:
                        cases.append({"case": label, "error": f"{type(exc).__name__}: {exc}"})
            else:
                for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                    try:
                        output = value(identity, rotated, weight)
                        cases.append({"weight": weight, "output": _jsonable(output)})
                    except BaseException as exc:
                        cases.append({"weight": weight, "error": f"{type(exc).__name__}: {exc}"})
        except BaseException as exc:
            cases.append({"error": f"{type(exc).__name__}: {exc}"})
        tests[name] = cases
    return tests


def _run_probe(args: argparse.Namespace) -> int:
    install_root = args.leapfrog_bin.resolve()
    path_setup = _dll_and_sys_path_setup(install_root)
    report: dict[str, Any] = {
        "install_root": str(install_root),
        "runtime": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "architecture": platform.architecture(),
            "platform": platform.platform(),
            "process_id": os.getpid(),
        },
        "path_setup": path_setup,
        "runtime_files": _runtime_files(install_root),
        "modules": {},
    }

    imported_objects: dict[str, Any] = {}
    for module_name in MODULE_CANDIDATES:
        module_report = _probe_module(module_name)
        report["modules"][module_name] = module_report
        if module_report.get("imported"):
            imported_objects[module_name] = importlib.import_module(module_name)

    report["safe_matrix_tests"] = _safe_matrix_tests(imported_objects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")

    print(f"python={sys.executable}")
    print(f"version={sys.version.split()[0]} architecture={platform.architecture()[0]}")
    for module_name, module_report in report["modules"].items():
        status = "OK" if module_report.get("imported") else "FAILED"
        print(f"{module_name}: {status}")
        if not module_report.get("imported"):
            print(f"  {module_report.get('error')}")
        else:
            interesting = sorted(module_report.get("interesting_members", {}))
            print(f"  file={module_report.get('file')}")
            print(f"  interesting={interesting}")
    print(f"wrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leapfrog-bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inside-leapfrog-python",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-reexec",
        action="store_true",
        help="Do not search for and re-execute under Leapfrog's embedded Python.",
    )
    args = parser.parse_args()

    if not args.leapfrog_bin.is_dir():
        parser.error(f"Leapfrog bin directory was not found: {args.leapfrog_bin}")

    if not args.inside_leapfrog_python and not args.no_reexec:
        candidates = _candidate_python_executables(args.leapfrog_bin)
        current = Path(sys.executable).resolve()
        embedded = next((path for path in candidates if path != current), None)
        if embedded is not None:
            command = [
                str(embedded),
                str(Path(__file__).resolve()),
                "--leapfrog-bin",
                str(args.leapfrog_bin),
                "--output",
                str(args.output),
                "--inside-leapfrog-python",
            ]
            print(f"re-executing with candidate Leapfrog Python: {embedded}", flush=True)
            completed = subprocess.run(command, check=False)
            return int(completed.returncode)
        print("No separate embedded python.exe was found; probing with the current interpreter.")

    return _run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
