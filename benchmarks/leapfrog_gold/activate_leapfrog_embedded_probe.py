"""Bootstrap ``inspect_leapfrog_embedded_runtime`` from Leapfrog sitecustomize.

Add this module's directory to ``sys.path`` and import this module from the existing
Leapfrog ``sitecustomize.py``. The import installs read-only wrappers. Optional
micro-cases are delayed so Leapfrog can finish starting its embedded Python runtime.
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import numpy as np

import inspect_leapfrog_embedded_runtime as probe


def _correct_spd_for_angle(angle: float, ratio: float = 5.0) -> np.ndarray:
    """Return determinant-one SPD tensor whose unique axis tilts in the X-Z plane."""
    radians = np.deg2rad(float(angle))
    normal = np.array([np.sin(radians), 0.0, np.cos(radians)], dtype=float)
    normal /= np.linalg.norm(normal)
    projector = normal[:, None] * normal[None, :]
    tangent = ratio ** (-1.0 / 3.0)
    axial = ratio ** (2.0 / 3.0)
    return tangent * (np.eye(3) - projector) + axial * projector


# Correct the micro-case orientation generator before any synthetic run.
probe._spd_for_angle = _correct_spd_for_angle
probe.install_import_hook(reflect=True)


def _run_delayed_microcases() -> None:
    delay = float(os.environ.get("LEAPFROG_RE_MICROCASE_DELAY", "15"))
    time.sleep(max(delay, 0.0))
    try:
        path = probe.run_subdomainer_microcases()
        probe._append_event(
            {"label": "delayed_microcases_complete", "path": str(path)}
        )
    except Exception as exc:
        probe._append_event(
            {
                "label": "delayed_microcases_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


if os.environ.get("LEAPFROG_RE_MICROCASES", "").strip() == "1":
    threading.Thread(
        target=_run_delayed_microcases,
        name="Leapfrog RE microcases",
        daemon=True,
    ).start()
