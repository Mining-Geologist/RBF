"""Opt-in, compact SubDomainer merge tracing for an authorized Leapfrog test run.

Import from an existing sitecustomize.py with LEAPFROG_MERGE_TRACE=1.
Original methods run once with unchanged arguments, returns and exceptions.
Writes separate diagnostic JSONL; does not modify project or license files.
Not yet validated against a running Leapfrog executable.
"""
from __future__ import annotations

import builtins
import functools
import json
import os
import sys
import threading
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

_PATCHED = set()
_LOCK = threading.RLock()
_INSTALLED = False
_EVENTS = 0
_NORMALS = 0


def summarize(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if np is not None and isinstance(value, np.ndarray):
        return dict(type="ndarray", shape=list(value.shape), dtype=str(value.dtype))
    if hasattr(value, "__len__"):
        try:
            return dict(type=type(value).__name__, length=len(value))
        except Exception:
            pass
    return dict(type=type(value).__name__)


def snapshot(obj):
    state = getattr(obj, "__dict__", None)
    if not isinstance(state, dict):
        return {}
    out = {}
    for name in ("_next_domain_id", "consistency_thresh", "min_points", "max_points"):
        if name in state:
            out[name] = summarize(state[name])
    for name in ("domains", "domainq", "neighbourq", "leaf_domain_dict"):
        val = state.get(name)
        if isinstance(val, (dict, tuple, list, set)):
            out[name + "_size"] = len(val)
            if name == "domains" and isinstance(val, dict):
                out["domain_ids"] = set(str(k) for k in val)
    return out


def emit(record, sink=None):
    global _EVENTS
    with _LOCK:
        _EVENTS += 1
        record = dict(record, sequence=_EVENTS, monotonic_ns=time.monotonic_ns())
        if sink is not None:
            sink(record)
            return
        try:
            directory = Path(os.environ.get("LEAPFROG_MERGE_TRACE_DIR",
                                            str(Path.cwd() / "leapfrog-merge-trace")))
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "merge-events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Diagnostics must not prevent normal modelling if output disk fails.
            pass


def save_primary_normals(args, result, sink=None):
    global _NORMALS
    if sink is not None or np is None or os.environ.get("LEAPFROG_MERGE_CAPTURE_NORMALS") != "1":
        return
    if len(args) < 2 or not isinstance(args[1], np.ndarray):
        return
    locations = args[1]
    if locations.ndim != 2 or locations.shape[1] != 3 or len(locations) > 20000:
        return
    if not isinstance(result, np.ndarray) or result.shape != locations.shape:
        return
    with _LOCK:
        if _NORMALS >= 2:
            return
        _NORMALS += 1
        index = _NORMALS
    try:
        directory = Path(os.environ.get("LEAPFROG_MERGE_TRACE_DIR",
                                        str(Path.cwd() / "leapfrog-merge-trace")))
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"primary-normals-{index}.npz",
                            locations=locations.copy(), normals=result.copy())
        emit({"type": "normals_captured", "index": index, "count": len(locations)}, sink)
    except Exception:
        pass


def patch_method(cls, name, sink=None):
    """Patch declared methods only; inherited methods are not double-wrapped."""
    key = (id(cls), name)
    if key in _PATCHED or name not in vars(cls):
        return False
    original = vars(cls)[name]
    if not callable(original):
        return False
    label = cls.__name__ + "." + name

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        obj = args[0] if args else None
        before = snapshot(obj)
        t0 = time.perf_counter_ns()
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            emit({"type": "exception", "method": label,
                  "arguments": [summarize(a) for a in args[1:]],
                  "error_type": type(exc).__name__}, sink)
            raise
        after = snapshot(obj)
        old, new = before.pop("domain_ids", set()), after.pop("domain_ids", set())
        record = {
            "type": "return", "method": label,
            "arguments": [summarize(a) for a in args[1:]],
            "keyword_arguments": {k: summarize(v) for k, v in kwargs.items()},
            "before": before, "after": after,
            "created_domains": sorted(new - old),
            "removed_domains": sorted(old - new),
            "result": summarize(result),
            "elapsed_ns": time.perf_counter_ns() - t0,
        }
        emit(record, sink)
        if name == "get_primary_normals":
            save_primary_normals(args, result, sink)
        return result

    setattr(cls, name, wrapper)
    _PATCHED.add(key)
    emit({"type": "installed", "method": label}, sink)
    return True


def attach(domaining, anisotropy=None, sink=None):
    installed = []
    for clsname in ("GridSeededDomainer", "SubDomainer"):
        cls = getattr(domaining, clsname, None)
        if cls is None:
            continue
        for name in ("__init__", "subdomain_with_real_locations",
                     "set_domains_by_region_growing", "merge_domains",
                     "add_to_domain", "new_domain"):
            if patch_method(cls, name, sink):
                installed.append(clsname + "." + name)
    cls = getattr(anisotropy, "AnisotropyStandard", None) if anisotropy is not None else None
    if cls is not None and patch_method(cls, "get_primary_normals", sink):
        installed.append("AnisotropyStandard.get_primary_normals")
    return installed


def install():
    global _INSTALLED
    if os.environ.get("LEAPFROG_MERGE_TRACE") != "1":
        return
    with _LOCK:
        if _INSTALLED:
            return
        _INSTALLED = True
        original = builtins.__import__

        @functools.wraps(original)
        def hooked(name, *args, **kwargs):
            result = original(name, *args, **kwargs)
            if "structural_fitter" in name:
                module = sys.modules.get("structural_fitter.domaining")
                if module is not None:
                    attach(module, sys.modules.get("structural_fitter.anisotropy"))
            return result

        builtins.__import__ = hooked
    module = sys.modules.get("structural_fitter.domaining")
    if module is not None:
        attach(module, sys.modules.get("structural_fitter.anisotropy"))
    emit({"type": "probe_ready", "python": sys.version.split()[0]})


if os.environ.get("LEAPFROG_MERGE_TRACE") == "1":
    install()
