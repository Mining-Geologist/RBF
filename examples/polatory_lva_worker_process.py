"""Fresh-process worker used by the v10 Polatory LVA GUI launcher.

The native structural fitting stack is executed in this short-lived interpreter.
If the native code aborts, only this worker exits; the Qt GUI remains alive and
reports the child-process exit code.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import polatory_lva_pyqt_app_v8_category_contacts as v8


class CallbackSignal:
    def __init__(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback

    def emit(self, value: Any = None) -> None:
        self._callback(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--obj", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    result_path = Path(args.result)
    output_obj = Path(args.obj)

    with input_path.open("rb") as stream:
        payload = pickle.load(stream)

    holder: dict[str, Any] = {}

    def progress(message: Any) -> None:
        print(f"PROGRESS\t{message}", flush=True)

    def finished(result: Any) -> None:
        holder["result"] = result

    def failed(details: Any) -> None:
        holder["error"] = str(details)

    runner = SimpleNamespace(
        payload=payload,
        progress=CallbackSignal(progress),
        finished=CallbackSignal(finished),
        failed=CallbackSignal(failed),
    )

    # v5 names its safe worker entry point ``scalable_worker_run``. Importing
    # v8 above also installs the category-only Contact value preprocessing in
    # this fresh interpreter before the worker starts.
    worker_run = getattr(v8.v5, "scalable_worker_run", None)
    if not callable(worker_run):
        raise RuntimeError(
            "The v5 launcher does not expose scalable_worker_run. Pull the full "
            "examples launcher chain and retry."
        )
    worker_run(runner)

    if "error" in holder:
        print(holder["error"], file=sys.stderr, flush=True)
        return 2
    if "result" not in holder:
        print("Worker returned neither a result nor an error.", file=sys.stderr, flush=True)
        return 3

    result = holder["result"]
    source_obj = Path(result["temp_obj"])
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_obj, output_obj)
    try:
        source_obj.unlink()
    except OSError:
        pass

    result["temp_obj"] = str(output_obj)
    with result_path.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print("RESULT_READY", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        os._exit(4)
