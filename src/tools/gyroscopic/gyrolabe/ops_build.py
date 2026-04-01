from __future__ import annotations

import os
import subprocess
from pathlib import Path


def build_gyrolabe_native() -> Path:
    root = Path(__file__).resolve().parent
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Python gyrolabe_native.dll: codec.c only. scalar.c uses core.h + GYRO_EXPORT alias but still
    # references symbols not defined in codec.c (gyrolabe_init, gyro_popcnt*, gyrolabe_extract_scan, ...).
    # Link scalar.c here only after those resolve in this TU or via extra objects.
    sources = [str(root / "codec.c")]
    out = build_dir / "gyrolabe_native.dll"

    cmd = [
        "cl",
        "/nologo",
        "/O2",
        "/LD",
        "/std:c11",
        "/arch:AVX2",
        "/fp:strict",
        *sources,
        f"/I{root}",
        f"/Fe:{out}",
        f"/Fo:{build_dir}\\",
    ]

    cp = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if cp.returncode != 0:
        raise RuntimeError(
            "GyroLabe: native build failed\n"
            f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
        )

    if not out.is_file():
        raise FileNotFoundError(f"GyroLabe: expected output not found: {out}")

    return out
