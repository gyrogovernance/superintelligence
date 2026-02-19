#!/usr/bin/env python3
"""
Check that all imports use src.router, not router.

This script enforces consistent import namespace across the codebase.
"""

import re
import sys
from pathlib import Path


def check_imports(root_dir: Path) -> list[str]:
    """Check for forbidden 'from router' or 'import router' patterns."""
    errors = []

    # Patterns to check
    forbidden_patterns = [
        (r'^from router\.', 'from router.'),
        (r'^import router\.', 'import router.'),
    ]

    # Files to check
    for py_file in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in str(py_file) for excluded in [
            '__pycache__',
            '.venv',
            'venv',
            'research_backup_do_not_use',
            'typings',
            '.git',
        ]):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                for pattern, description in forbidden_patterns:
                    if re.search(pattern, line):
                        errors.append(
                            f"{py_file.relative_to(root_dir)}:{line_num}: "
                            f"Found '{description}' - use 'src.router' instead"
                        )
        except Exception as e:
            errors.append(f"{py_file}: Error reading file: {e}")

    return errors


def main():
    """Run import check and exit with error code if violations found."""
    root_dir = Path(__file__).parent.parent

    errors = check_imports(root_dir)

    if errors:
        print("ERROR: Found imports using 'router' instead of 'src.router':")
        print()
        for error in errors:
            print(f"  {error}")
        print()
        print("All imports must use 'src.router', not 'router'.")
        print("This ensures consistent namespace across all environments.")
        sys.exit(1)
    else:
        print("✓ All imports use 'src.router' namespace (consistent)")
        sys.exit(0)


if __name__ == '__main__':
    main()

