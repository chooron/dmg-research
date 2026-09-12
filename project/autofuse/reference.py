"""Reference-run gate for comparing against upstream FUSE.

The reference repository is kept read-only under ``vendor/upstream``.  This
module never imports it or modifies it; it only reports whether a compiled
executable is available for a later same-input comparison.
"""

from __future__ import annotations

import os
import hashlib
import shutil
from pathlib import Path


def reference_status(executable: str | Path | None = None) -> dict[str, object]:
    requested = Path(executable) if executable else None
    build_dir = Path(os.environ["FUSE_REFERENCE_BUILD_DIR"]) if os.environ.get("FUSE_REFERENCE_BUILD_DIR") else None
    candidates = [
        requested,
        Path(os.environ["FUSE_REFERENCE_EXE"]) if os.environ.get("FUSE_REFERENCE_EXE") else None,
        build_dir / "bin/fuse.exe" if build_dir else None,
        Path("vendor/upstream/cyrilthebault-fuse/bin/fuse.exe"),
        Path("vendor/upstream/cyrilthebault-fuse/fuse.exe"),
    ]
    found = next((path for path in candidates if path and path.is_file()), None)
    compiler = shutil.which("gfortran")
    if compiler is None and os.environ.get("FUSE_REFERENCE_FC") and Path(os.environ["FUSE_REFERENCE_FC"]).is_file():
        compiler = os.environ["FUSE_REFERENCE_FC"]
    return {
        "status": "available" if found else "blocked",
        "executable": str(found) if found else None,
        "executable_sha256": hashlib.sha256(found.read_bytes()).hexdigest() if found else None,
        "gfortran": compiler,
        "checked_candidates": [str(path) for path in candidates if path],
        "reason": None if found else "no compiled FUSE executable and no Fortran compiler in the current WSL environment",
        "source_commit": "e6e23a4fc4ff4019bcab55f14537ea43b9525967",
        "model_id": 84,
        "comparison_inputs": "same forcing, same extracted bounds/default parameters, same initial fraction",
    }
