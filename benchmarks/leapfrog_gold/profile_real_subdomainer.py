"""High-frequency, read-only profile of Leapfrog's real-location SubDomainer stage.

This uses ``py-spy record`` without locals. Unlike repeated ``dump --locals`` calls,
it samples continuously with much less stop/resume interference and can catch the
very short second-stage call stack. Run it, then immediately trigger an automatic-
domaining recompute in Leapfrog.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--rate", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/leapfrog-real-subdomainer-profile.txt"),
    )
    parser.add_argument("--py-spy", type=Path, default=default_py_spy())
    args = parser.parse_args()

    if not args.py_spy.is_file():
        raise SystemExit(f"py-spy was not found: {args.py_spy}")
    pid = find_background_pid()
    if pid is None:
        raise SystemExit("No Leapfrog --background process was found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.output.with_suffix(".raw.txt")
    raw_output.unlink(missing_ok=True)
    args.output.unlink(missing_ok=True)

    print(
        f"Profiling Leapfrog background PID {pid} at {args.rate} Hz for "
        f"{args.duration} seconds."
    )
    print("Trigger the automatic-domaining recompute now.", flush=True)

    command = [
        str(args.py_spy),
        "record",
        "--pid",
        str(pid),
        "--rate",
        str(max(args.rate, 1)),
        "--duration",
        str(max(args.duration, 1)),
        "--format",
        "raw",
        "--output",
        str(raw_output),
    ]
    completed = subprocess.run(command, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"py-spy record failed with exit code {completed.returncode}.")
    if not raw_output.is_file():
        raise SystemExit(f"py-spy did not create {raw_output}")

    lines = raw_output.read_text(encoding="utf-8", errors="replace").splitlines()
    real_lines = [
        line
        for line in lines
        if "subdomain_with_real_locations (domaining.py" in line
        or "SubDomainer" in line
    ]
    args.output.write_text("\n".join(real_lines) + ("\n" if real_lines else ""), encoding="utf-8")

    if real_lines:
        print(f"Captured {len(real_lines)} real-stage stack signatures: {args.output}")
        print(f"Full raw profile: {raw_output}")
        return 0

    print("No real-stage stack signature was found in the profile.")
    print(f"Full raw profile: {raw_output}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
