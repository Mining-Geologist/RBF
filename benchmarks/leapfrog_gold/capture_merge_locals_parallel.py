"""Capture Leapfrog region-growing merge locals with minimal interference.

Run this while Leapfrog is open, then immediately trigger an automatic-domaining
recompute. The script never injects code into Leapfrog.

The earlier implementation launched many simultaneous ``py-spy dump --locals``
processes. On Windows, each dump briefly suspends the target process, so overlapping
dumps could keep Leapfrog almost continuously paused. This version uses one sampler.

Use ``--stage real`` to follow the coarse ``GridSeededDomainer`` into the second,
real-location ``SubDomainer`` stage. Real mode intentionally starts its single locals
sampler as soon as coarse region growing is seen, because the real stage can be too
short to detect first with a 0.5-second lightweight polling interval.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


REGION_TERM = "set_domains_by_region_growing (domaining.py"
MERGE_TERM = "merge_domains (domaining.py"
REAL_STAGE_TERM = "subdomain_with_real_locations (domaining.py"


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


def in_region_growing(output: str) -> bool:
    return REGION_TERM in output or MERGE_TERM in output


def in_real_stage(output: str) -> bool:
    return REAL_STAGE_TERM in output or "self: <SubDomainer at " in output


def trigger_matches(output: str, stage: str) -> bool:
    """Return True when it is safe to begin the single locals-capture burst."""
    if stage == "real":
        # The real pass can be extremely short. Start following during the coarse
        # pass rather than trying to observe the real parent frame first.
        return in_region_growing(output) or in_real_stage(output)
    if stage == "coarse":
        return in_region_growing(output) and not in_real_stage(output)
    return in_region_growing(output)


def capture_matches(output: str, stage: str) -> bool:
    if MERGE_TERM not in output:
        return False
    if stage == "real":
        return in_real_stage(output)
    if stage == "coarse":
        return not in_real_stage(output)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--idle-interval", type=float, default=0.5)
    parser.add_argument("--burst-duration", type=float, default=180.0)
    parser.add_argument("--burst-interval", type=float, default=0.02)
    parser.add_argument(
        "--stage",
        choices=("any", "coarse", "real"),
        default="any",
        help="Domaining stage to capture. 'real' targets the second SubDomainer stage.",
    )
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

    print(
        f"Watching Leapfrog background PID {pid} for the {args.stage} domaining stage "
        "with one sampler."
    )
    print("Trigger the automatic-domaining recompute now.", flush=True)

    overall_deadline = time.monotonic() + max(args.duration, 1.0)
    trigger_seen = False
    light_samples = 0

    while time.monotonic() < overall_deadline:
        output = run_dump(args.py_spy, pid, locals_=False)
        light_samples += 1
        if process_is_gone(output):
            raise SystemExit("The Leapfrog background process exited or restarted.")
        if trigger_matches(output, args.stage):
            trigger_seen = True
            if args.stage == "real" and not in_real_stage(output):
                print(
                    f"Coarse region growing detected after {light_samples} lightweight "
                    "samples; following it into the real stage with one locals sampler.",
                    flush=True,
                )
            else:
                print(
                    f"Capture trigger detected after {light_samples} lightweight samples; "
                    "switching to one locals sampler.",
                    flush=True,
                )
            break
        time.sleep(max(args.idle_interval, 0.05))

    if not trigger_seen:
        print("The requested domaining stage was not observed before the timeout.")
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
        if capture_matches(output, args.stage):
            args.output.write_text(output, encoding="utf-8")
            print(
                f"Captured {args.stage} merge_domains locals after {local_samples} "
                f"locals samples: {args.output}"
            )
            return 0
        time.sleep(max(args.burst_interval, 0.01))

    print(
        "The capture trigger was detected, but no matching merge_domains frame was "
        "captured. Leapfrog was left running normally."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
