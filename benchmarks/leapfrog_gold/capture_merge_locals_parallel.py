"""Capture Leapfrog automatic-domaining locals with parallel read-only py-spy dumps.

Run this while Leapfrog is open, then immediately trigger an automatic-domaining
recompute. The script never injects code into Leapfrog; it only reads stack frames.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MATCH_TERMS = (
    "merge_domains (domaining.py",
    "_find_joint_consistency (domaining.py",
    "set_domains_by_region_growing (domaining.py",
)


def find_background_pid() -> int | None:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'Leapfrog.exe' -and $_.CommandLine -match '--background' } | "
        "Sort-Object CreationDate -Descending | Select-Object -First 1 -ExpandProperty ProcessId",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    text = completed.stdout.strip()
    try:
        return int(text.splitlines()[-1]) if text else None
    except (ValueError, IndexError):
        return None


def default_py_spy() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    return Path(local_appdata) / "Programs" / "Python" / "Python312" / "Scripts" / "py-spy.exe"


def worker(
    worker_id: int,
    py_spy: Path,
    pid: int,
    stop: threading.Event,
    deadline: float,
) -> tuple[int, str] | None:
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                [str(py_spy), "dump", "--pid", str(pid), "--locals"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            continue
        output = (completed.stdout or "") + (completed.stderr or "")
        if any(term in output for term in MATCH_TERMS):
            stop.set()
            return worker_id, output
        if "No such process" in output or "os error 87" in output:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/leapfrog-merge-locals.txt"),
    )
    parser.add_argument("--py-spy", type=Path, default=default_py_spy())
    args = parser.parse_args()

    if not args.py_spy.is_file():
        raise SystemExit(f"py-spy was not found: {args.py_spy}")

    pid = find_background_pid()
    if pid is None:
        raise SystemExit("No Leapfrog --background process was found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    print(f"Watching Leapfrog background PID {pid} with {args.workers} read-only samplers.")
    print("Trigger the automatic-domaining recompute now.", flush=True)

    stop = threading.Event()
    deadline = time.monotonic() + max(args.duration, 1.0)
    result: tuple[int, str] | None = None

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [
            executor.submit(worker, i + 1, args.py_spy, pid, stop, deadline)
            for i in range(max(args.workers, 1))
        ]
        for future in as_completed(futures):
            candidate = future.result()
            if candidate is not None:
                result = candidate
                break

    stop.set()
    if result is None:
        print("No merge frame was captured. Re-run and trigger recompute immediately.")
        return 1

    worker_id, output = result
    args.output.write_text(output, encoding="utf-8")
    print(f"Captured merge locals with worker {worker_id}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
