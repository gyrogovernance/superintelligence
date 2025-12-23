"""Pytest configuration for Router kernel tests."""

import sys
from pathlib import Path

# Add src to Python path so ggg_asi_router can be imported
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

