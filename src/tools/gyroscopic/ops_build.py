from __future__ import annotations

"""
Build ``llama-cli`` for stock (vanilla ggml-cpu) and gyroscopic (ggml-gyroscopic) backends.

Public entry points: ``build_llama_cpp_if_needed``, ``build_llama_compare_pair_if_needed``.
Other symbols are internal plumbing.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from .config import repo_root

LlamaBuildMode = Literal["stock", "gyroscopic"]


def _subpath(base: Path, rel: str) -> Path:
    return base / Path(rel)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _needs_rebuild(artifacts: list[Path], deps: list[Path]) -> bool:
    if not artifacts:
        return True
    try:
        out_m = max(p.stat().st_mtime for p in artifacts)
    except OSError:
        return True
    for p in deps:
        try:
            if p.stat().st_mtime > out_m:
                return True
        except OSError:
            return True
    return False


def _llama_build_dir(mode: LlamaBuildMode) -> str:
    return "build-stock" if mode == "stock" else "build"


def _llama_cli_candidates(repo_root: Path, mode: LlamaBuildMode) -> list[Path]:
    base = repo_root / "external" / "llama.cpp" / _llama_build_dir(mode) / "bin"
    if sys.platform == "win32":
        return [
            base / "Release" / "llama-cli.exe",
            base / "Debug" / "llama-cli.exe",
            base / "llama-cli.exe",
        ]
    return [base / "llama-cli"]


def _llama_build_artifacts(repo_root: Path, mode: LlamaBuildMode) -> list[Path]:
    """Key build products used to detect staleness (not just llama-cli.exe)."""
    base = repo_root / "external" / "llama.cpp" / _llama_build_dir(mode)
    rels = (
        "bin/Release/llama-cli.exe",
        "bin/Release/ggml-cpu.dll",
        "bin/Release/ggml.dll",
        "bin/llama-cli",
        "bin/libggml-cpu.so",
        "bin/libggml-cpu.dylib",
    )
    out: list[Path] = []
    for rel in rels:
        p = _subpath(base, rel)
        if p.is_file():
            out.append(p)
    cli = resolve_llama_cli_out(repo_root, mode=mode)
    if cli is not None and cli not in out:
        out.append(cli)
    return out


def resolve_llama_cli_out(repo_root_path: Path | None = None, *, mode: LlamaBuildMode = "gyroscopic") -> Path | None:
    root = repo_root_path if repo_root_path is not None else repo_root()
    for c in _llama_cli_candidates(root, mode):
        if c.is_file():
            return c
    return None


def resolve_integrated_gyro_dll() -> Path:
    """Return the ggml CPU backend DLL that exports the Gyroscopic symbols."""
    root = repo_root()
    names = ("ggml-cpu.dll", "libggml-cpu.so", "libggml-cpu.dylib", "ggml-base.dll", "llama.dll")
    for sub in ("build/bin/Release", "build/bin/Debug", "build/bin"):
        base = _subpath(root / "external" / "llama.cpp", sub)
        for name in names:
            dll = base / name
            if dll.is_file():
                return dll
    raise FileNotFoundError(
        "Integrated gyroscopic DLL not found (expected ggml-cpu). "
        "Build external/llama.cpp with the gyroscopic backend first."
    )


def _collect_tree_sources(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return out


def _llama_gyroscopic_build_inputs(repo_root: Path) -> list[Path]:
    gyroscopic = repo_root / "src" / "tools" / "gyroscopic"
    out: list[Path] = []
    if gyroscopic.is_dir():
        out.extend(p for p in gyroscopic.glob("*.h") if p.is_file())
        kernel_c = gyroscopic / "kernel.c"
        if kernel_c.is_file():
            out.append(kernel_c)

    ggml = repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-gyroscopic"
    for rel in (
        "ggml-cpu.c",
        "vec.cpp",
        "quants.c",
        "quants.h",
        "arch/x86/quants.c",
    ):
        p = ggml / rel
        if p.is_file():
            out.append(p)

    llama_ggml = repo_root / "external" / "llama.cpp" / "ggml"
    for rel in ("include/ggml-backend.h", "src/ggml-backend.cpp"):
        p = llama_ggml / rel
        if p.is_file():
            out.append(p)

    cm_gyro = repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-gyroscopic" / "CMakeLists.txt"
    if cm_gyro.is_file():
        out.append(cm_gyro)

    llama_src = repo_root / "external" / "llama.cpp" / "src"
    for name in ("llama-model.cpp", "llama-context.cpp"):
        p = llama_src / name
        if p.is_file():
            out.append(p)

    llama_cli = repo_root / "external" / "llama.cpp" / "tools" / "cli" / "cli.cpp"
    if llama_cli.is_file():
        out.append(llama_cli)

    for ps1 in (
        repo_root / "scripts" / "build_llama_cpp_windows.ps1",
        repo_root / "src" / "tools" / "gyroscopic" / "helpers" / "build_llama_cpp_windows.ps1",
    ):
        if ps1.is_file():
            out.append(ps1)
    return sorted({p.resolve() for p in out if p.is_file()})


def _llama_stock_build_inputs(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    out.extend(
        _collect_tree_sources(
            repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-cpu",
            (".c", ".cpp", ".h", ".hpp"),
        )
    )
    cm = repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-cpu" / "CMakeLists.txt"
    if cm.is_file():
        out.append(cm)
    llama_cli = repo_root / "external" / "llama.cpp" / "tools" / "cli" / "cli.cpp"
    if llama_cli.is_file():
        out.append(llama_cli)
    for ps1 in (
        repo_root / "scripts" / "build_llama_cpp_windows.ps1",
        repo_root / "src" / "tools" / "gyroscopic" / "helpers" / "build_llama_cpp_windows.ps1",
    ):
        if ps1.is_file():
            out.append(ps1)
    return sorted({p.resolve() for p in out if p.is_file()})


def _llama_build_inputs(repo_root: Path, mode: LlamaBuildMode) -> list[Path]:
    if mode == "stock":
        return _llama_stock_build_inputs(repo_root)
    return _llama_gyroscopic_build_inputs(repo_root)


def _resolve_llama_build_script_windows(repo_root: Path) -> Path | None:
    for ps1 in (
        repo_root / "scripts" / "build_llama_cpp_windows.ps1",
        repo_root / "src" / "tools" / "gyroscopic" / "helpers" / "build_llama_cpp_windows.ps1",
    ):
        if ps1.is_file():
            return ps1
    return None


def _build_verbose() -> bool:
    return _env_truthy("GYROSCOPIC_VERBOSE_BUILD")


def _should_print_build_line(line: str, *, verbose: bool) -> bool:
    if verbose or not line:
        return bool(line)
    low = line.lower()
    if any(k in low for k in ("error", "failed", "fatal", "lnk1104", "cannot open file")):
        return True
    if "done. example:" in low:
        return True
    if re.search(r"\b(quants\.c|ggml-cpu\.c|kernel\.c|llama-cli)", line, re.I):
        return True
    return False


def _llama_build_output_looks_like_dll_lock_error(text: str) -> bool:
    if not text:
        return False
    u = text.upper()
    return "LNK1104" in u or ("CANNOT OPEN FILE" in u and ".DLL" in u)


def _run_llama_build_windows(repo_root: Path, mode: LlamaBuildMode) -> None:
    ps1 = _resolve_llama_build_script_windows(repo_root)
    if ps1 is None:
        raise FileNotFoundError("Gyroscopic: missing build_llama_cpp_windows.ps1.")

    verbose = _build_verbose()
    max_attempts = 4
    sleep_before_attempt = (0.0, 2.0, 5.0, 10.0)
    last_rc = 1
    last_out: list[str] = []

    for attempt in range(max_attempts):
        if sleep_before_attempt[attempt] > 0:
            time.sleep(sleep_before_attempt[attempt])
        if attempt > 0:
            print(f"Gyroscopic: build retry {attempt + 1}/{max_attempts}...", file=sys.stderr, flush=True)
        proc = subprocess.Popen(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ps1.resolve()),
                "-Mode",
                mode,
            ],
            cwd=str(repo_root.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        last_out = []
        assert proc.stdout is not None
        for raw in proc.stdout:
            last_out.append(raw)
            line = raw.rstrip("\r\n")
            if _should_print_build_line(line, verbose=verbose):
                print(f"  [build] {line}", file=sys.stderr, flush=True)
        last_rc = proc.wait()
        if last_rc == 0:
            return
        out = "".join(last_out)
        if attempt < max_attempts - 1 and _llama_build_output_looks_like_dll_lock_error(out):
            continue
        break

    msg = f"Gyroscopic: llama.cpp build failed ({mode}, PowerShell).\n{''.join(last_out)}"
    if _llama_build_output_looks_like_dll_lock_error("".join(last_out)):
        msg += "\n\nIf you see LNK1104 on ggml-cpu.dll: close llama-cli/bench, then retry."
    raise RuntimeError(msg)


def _run_llama_build_unix(repo_root: Path, mode: LlamaBuildMode) -> None:
    import shutil

    llama = repo_root / "external" / "llama.cpp"
    cm = llama / "CMakeLists.txt"
    if not cm.is_file():
        raise FileNotFoundError(f"Gyroscopic: missing {cm}")
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("Gyroscopic: cmake not found in PATH.")

    build_dir = _llama_build_dir(mode)
    if mode == "stock":
        backend_subdir = "ggml-cpu"
        gyro_flag = "OFF"
    else:
        backend_subdir = "ggml-gyroscopic"
        gyro_flag = "ON"

    nj = max(1, (os.cpu_count() or 4))
    cfg_cmd = [
        cmake,
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_CPU_BACKEND_SUBDIR={backend_subdir}",
        f"-DGGML_GYROSCOPIC={gyro_flag}",
        "-DGGML_OPENMP=ON",
    ]
    cp1 = subprocess.run(
        cfg_cmd,
        cwd=str(llama.resolve()),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if cp1.returncode != 0:
        raise RuntimeError(
            f"cmake configure FAILED ({mode}):\nSTDOUT:\n{cp1.stdout}\nSTDERR:\n{cp1.stderr}"
        )

    cp2 = subprocess.run(
        [cmake, "--build", build_dir, "-j", str(nj)],
        cwd=str(llama.resolve()),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if cp2.returncode != 0:
        raise RuntimeError(f"Gyroscopic: llama.cpp build failed ({mode}).\nSTDOUT:\n{cp2.stdout}\n\nSTDERR:\n{cp2.stderr}")


def build_llama_cpp_if_needed(*, mode: LlamaBuildMode = "gyroscopic", force: bool = False) -> Path:
    root = repo_root()
    deps = _llama_build_inputs(root, mode)
    if mode == "gyroscopic" and not deps:
        raise RuntimeError("Gyroscopic: no gyroscopic build inputs found.")
    artifacts = _llama_build_artifacts(root, mode)
    out = resolve_llama_cli_out(root, mode=mode)
    if not force and out is not None and artifacts and deps and not _needs_rebuild(artifacts, deps):
        return out
    label = "stock (vanilla ggml-cpu)" if mode == "stock" else "gyroscopic backend"
    print(f"Gyroscopic: building llama-cli ({label})...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    if sys.platform == "win32":
        _run_llama_build_windows(root, mode)
    else:
        _run_llama_build_unix(root, mode)
    out2 = resolve_llama_cli_out(root, mode=mode)
    if out2 is None:
        raise FileNotFoundError(
            f"Gyroscopic: llama.cpp build finished but llama-cli not found for mode={mode}."
        )
    if not _build_verbose():
        print(f"Gyroscopic: build done ({time.perf_counter() - t0:.1f}s).", file=sys.stderr, flush=True)
    return out2


def build_llama_compare_pair_if_needed(*, force: bool = False) -> tuple[Path | None, Path | None]:
    """Build stock and gyroscopic llama-cli when either is missing or stale."""
    stock = build_llama_cpp_if_needed(mode="stock", force=force)
    gyro = build_llama_cpp_if_needed(mode="gyroscopic", force=force)
    return stock, gyro
