"""Pytest bootstrap: ensure project root is importable as `server` / `mobile` / `selftrain`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
