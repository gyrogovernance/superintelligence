"""
ANSI formatting, tables, and prompts.
"""

import sys

# ANSI codes (disable if not TTY)
USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c("1", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def red(text: str) -> str:
    return _c("31", text)


def cyan(text: str) -> str:
    return _c("36", text)


def header(title: str) -> str:
    return f"\n{bold('AIR')} - Alignment Infrastructure Routing\n{bold(title)}\n"


def kv(key: str, value: str, indent: int = 0) -> str:
    pad = "  " * indent
    return f"{pad}{key}: {cyan(str(value))}"


from typing import Any

def table(rows: list[list[Any]], headers: list[str]) -> str:
    """Simple fixed-width table."""
    if not rows:
        return ""
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    hdr = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(bold(hdr))
    lines.append("-" * len(hdr))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def success(msg: str) -> None:
    # Use ASCII-safe characters for Windows compatibility
    try:
        marker = "✓ " if USE_COLOR else "[OK] "
        print(green(marker + msg))
    except UnicodeEncodeError:
        print("[OK] " + msg)


def error(msg: str) -> None:
    # Use ASCII-safe characters for Windows compatibility
    try:
        marker = "✗ " if USE_COLOR else "[FAIL] "
        print(red(marker + msg))
    except UnicodeEncodeError:
        print("[FAIL] " + msg)


def warn(msg: str) -> None:
    print(yellow("! " + msg))

