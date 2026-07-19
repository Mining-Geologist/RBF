"""Compatibility entry point for the v10 isolated LVA worker.

The implementation lives in ``polatory_lva_worker_process_v2``.  Keeping this
filename preserves the existing v10 launcher command and process arguments.
"""

from polatory_lva_worker_process_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
