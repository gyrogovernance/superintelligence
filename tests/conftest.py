"""Pytest configuration for Router kernel tests."""

import sys
from pathlib import Path

import pytest

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session", autouse=True)
def ensure_atlas_built():
    """Ensure atlas is built once at session start.
    
    Session-scoped: builds once for the entire test run.
    Tests should not delete the atlas - only clean up their own test data.
    """
    from src.router.atlas import build_all

    program_root = Path(__file__).parent.parent
    atlas_dir = program_root / "data" / "atlas"

    required_files = ["ontology.npy", "epistemology.npy", "phenomenology.npz"]
    missing = [f for f in required_files if not (atlas_dir / f).exists()]

    if missing:
        # Build atlas if any files are missing
        atlas_dir.mkdir(parents=True, exist_ok=True)
        build_all(atlas_dir)

        # Verify all files were created
        still_missing = [f for f in required_files if not (atlas_dir / f).exists()]
        if still_missing:
            pytest.fail(f"Atlas build failed: missing {still_missing}")

