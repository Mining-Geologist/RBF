"""Compatibility entry point for the process-isolated LVA worker.

The structural-domain implementation lives in ``polatory_lva_worker_process_v3``.
Before it starts, install overlap-and-trim chunk meshing so temporary native chunk
boundaries cannot appear as flat walls or pinched terminations in the exported OBJ.
"""

from polatory_lva_chunk_overlap import install_chunk_overlap
import polatory_lva_worker_process_v3 as worker


install_chunk_overlap(worker.v2.v8.v5.v3)
main = worker.main


if __name__ == "__main__":
    raise SystemExit(main())
