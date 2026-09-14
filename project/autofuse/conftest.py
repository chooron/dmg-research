import sys
from pathlib import Path

_autofuse_dir = Path(__file__).resolve().parent
_root_dir = _autofuse_dir.parents[1]

for _p in (str(_root_dir), str(_autofuse_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
