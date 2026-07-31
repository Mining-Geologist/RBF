# Leapfrog embedded runtime probe

This workflow inspects the installed Leapfrog runtime and captures the exact inputs,
outputs, and object state used by the real-location `SubDomainer`. It does not modify
Leapfrog projects or replace the original domaining functions.

## 1. Pull the probe

```bat
git pull
```

## 2. Inventory the installed binaries and archives

Run this from the repository virtual environment:

```bat
python benchmarks\leapfrog_gold\inspect_leapfrog_embedded_runtime.py ^
  --mode inventory ^
  --install-root "D:\Program files\Seequent\Leapfrog 2026.1\bin" ^
  --output benchmark-results\leapfrog-runtime-inventory.json
```

Optional PE export/import parsing is enabled by installing `pefile`:

```bat
python -m pip install pefile
```

The string and `PyInit_*` scan works even without `pefile`.

## 3. Activate the in-process probe

Use the existing Leapfrog `sitecustomize.py` that is already loaded by the
instrumented Leapfrog process. Append the following block, adjusting the repository
path only when necessary:

```python
import os
import sys

_REPO = r"D:\Polatory_LVA\RBF-LVA"
sys.path.insert(0, _REPO + r"\benchmarks\leapfrog_gold")
os.environ["LEAPFROG_RE_PROBE_DIR"] = (
    _REPO + r"\benchmark-results\leapfrog-embedded-runtime"
)
os.environ["LEAPFROG_RE_PROBE"] = "1"

# Leave this disabled for the first real-project capture.
os.environ["LEAPFROG_RE_MICROCASES"] = "0"

import activate_leapfrog_embedded_probe  # noqa: F401,E402
```

Restart Leapfrog after editing `sitecustomize.py`, open the benchmark project, and
trigger the automatic structural-domaining recompute once.

Expected output directory:

```text
benchmark-results\leapfrog-embedded-runtime\
```

Important files:

- `runtime-reflection.json`: module names, signatures, docs, source when available,
  and Python 3.12 bytecode disassembly.
- `runtime-events.jsonl`: ordered calls into `SubDomainer`, `GridSeededDomainer`, and
  anisotropy methods.
- `call-*.npz`: exact NumPy arrays passed into or retained by those calls, including
  real locations, anisotropies, strengths, parent labels, and result arrays when
  exposed by the Python objects.

## 4. Run controlled SubDomainer micro-cases

After the real-project capture works, change:

```python
os.environ["LEAPFROG_RE_MICROCASES"] = "1"
os.environ["LEAPFROG_RE_MICROCASE_DELAY"] = "15"
```

Restart Leapfrog. The probe waits before constructing small synthetic point sets with:

- identical tensors;
- two sharply different orientation groups;
- a gradual orientation ramp;
- connected and spatially separated point layouts;
- thresholds `0.0`, `0.6`, `0.9`, `0.99`, and `0.999`.

The result is written to:

```text
subdomainer-microcases.json
```

Disable the micro-cases again after the file is produced.

## 5. Restore normal Leapfrog startup

Remove or comment out the bootstrap block from `sitecustomize.py`, then restart
Leapfrog. No project data is changed by the probe.
