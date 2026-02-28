"""Verify trajectory generation covers state space."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Skip if L3 table not available
_l3_path = _root / "data" / "layers" / "l3_packed_u24.bin"


def test_full_coverage_has_256_bytes():
    if not _l3_path.exists():
        import pytest
        pytest.skip("L3 table not found")
    from secret_lab_ignore.autobots.curriculum import FSMCurriculum
    cur = FSMCurriculum(_l3_path)
    out = cur.generate_full_coverage()
    assert len(out) == 256
    ids_set = {tuple(r["input_ids"]) for r in out}
    assert len(ids_set) == 256


def test_random_walks_produce_sequences():
    if not _l3_path.exists():
        import pytest
        pytest.skip("L3 table not found")
    from secret_lab_ignore.autobots.curriculum import FSMCurriculum
    cur = FSMCurriculum(_l3_path)
    out = cur.generate_random_walks(10, 32)
    assert len(out) == 10
    for r in out:
        assert len(r["input_ids"]) == 32


if __name__ == "__main__":
    if _l3_path.exists():
        test_full_coverage_has_256_bytes()
        test_random_walks_produce_sequences()
        print("test_curriculum OK")
    else:
        print("test_curriculum SKIP (no L3 table)")
