"""Add the CGM Science port banner to markdown docs in this folder.

Only processes *.md files directly in docs/references.
Does not touch nested folders (experiments/, the_human_mark/, etc.).
Idempotent: files that already start with the banner are left unchanged.
"""

from __future__ import annotations

from pathlib import Path

BANNER = (
    "> **CGM Science Documentation - Port** : This doc is copied from our science repo"
    " - all of its surrounding theory and relative paths can be found at"
    " https://github.com/gyrogovernance/science"
)


def target_docs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )


def ensure_banner(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip("\ufeff")
    if stripped.startswith(BANNER):
        return "skip"
    path.write_text(BANNER + "\n\n" + stripped, encoding="utf-8", newline="\n")
    return "added"


def main() -> None:
    root = Path(__file__).resolve().parent
    added = skipped = 0
    for path in target_docs(root):
        status = ensure_banner(path)
        print(f"{status}: {path.name}")
        if status == "added":
            added += 1
        else:
            skipped += 1
    print("-" * 5)
    print(f"added={added} skipped={skipped}")


if __name__ == "__main__":
    main()
