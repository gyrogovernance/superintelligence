#!/usr/bin/env python3
"""
Test runner script for all router kernel tests.

Usage:
    python run_tests.py
    python run_tests.py -k test_name_pattern
    python run_tests.py test_physics.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run all tests with -m pytest, -v (verbose), and -s (no capture)."""
    script_dir = Path(__file__).parent
    program_root = script_dir.parent
    
    cmd = [
        sys.executable,
        "-m", "pytest",
        "-v",  # verbose
        "-s",  # no capture (show prints)
    ]
    
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if not any(arg.startswith("test") or arg.startswith("-") for arg in args):
            args = [f"tests/{arg}" if not arg.startswith("tests/") else arg for arg in args]
        cmd.extend(args)
    else:
        cmd.append("tests/")
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 10)
    
    result = subprocess.run(cmd, cwd=program_root)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

