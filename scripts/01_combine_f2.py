"""Pipeline entry point. Real logic lives in closure.io.combine_f2.

The sys.path manipulation goes away once Phase 2 adds pyproject.toml + `pip install -e .`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from closure.io.combine_f2 import main

if __name__ == "__main__":
    main()
