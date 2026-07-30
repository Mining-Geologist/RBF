"""Capture Leapfrog's region-growing merge locals with minimal interference.

Run this while Leapfrog is open, then immediately trigger an automatic-domaining
recompute. The script never injects code into Leapfrog.

The earlier implementation launched many simultaneous ``py-spy dump --locals``
processes. On Windows, each dump briefly suspends the target process, so overlapping
dumps could keep Leapfrog almost continuously paused. This version is adaptive:

1. One lightweight sampler (without locals) waits for
   ``set_domains_by_region_growing``.
2. Only after region growing starts, one sampler briefly switches to ``--locals``
   and looks for an actual ``merge_domains`` frame.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


REGION_TERM = "set_domains_by_region_growing (domaining.py"
MERGE_TERM = "merge_domains (domaining.py"


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


def run_dump(py_spy: Path, pid: int, *, locals_: bool) -> str:
    command = [str(py_spy), "dump", "--pid", str(pid)]
    if locals_:
        command.append("--locals")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""
    return (completed.stdout or "") + (completed.stderr or "")


def process_is_gone(output: str) -> bool:
    lowered = output.lower()
    return "no such process" in lowered or "os error 87" in lowered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--burst-duration", type=float, default=90.0)
    parser.add_argument("--burst-interval", type=float, default=0.05)
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

    print(f"Watching Leapfrog background PID {pid} with one low-impact sampler.")
    print("Trigger the automatic-domaining recompute now.", flush=True)

    overall_deadline = time.monotonic() + max(args.duration, 1.0)
    region_seen = False
    light_samples = 0

    while time.monotonic() < overall_deadline:
        output = run_dump(args.py_spy, pid, locals_=False)
        light_samples += 1
        if process_is_gone(output):
            raise SystemExit("The Leapfrog background process exited or restarted.")
        if REGION_TERM in output or MERGE_TERM in output:
            region_seen = True
            print(
                f"Region growing detected after {light_samples} lightweight samples; "
                "switching briefly to locals capture.",
                flush=True,
            )
            break
        time.sleep(max(args.idle_interval, 0.05))

    if not region_seen:
        print("Region growing was not observed before the timeout.")
        return 1

    burst_deadline = min(
        overall_deadline,
        time.monotonic() + max(args.burst_duration, 1.0),
    )
    local_samples = 0
    while time.monotonic() < burst_deadline:
        output = run_dump(args.py_spy, pid, locals_=True)
        local_samples += 1
        if process_is_gone(output):
            raise SystemExit("The Leapfrog background process exited or restarted.")
        if MERGE_TERM in output:
            args.output.write_text(output, encoding="utf-8")
            print(
                f"Captured merge_domains locals after {local_samples} locals samples: "
                f"{args.output}"
            )
            return 0
        time.sleep(max(args.burst_interval, 0.01))

    print(
        "Region growing was detected, but no merge_domains frame was captured. "
        "Leapfrog was left running normally."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
