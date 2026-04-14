from __future__ import annotations

"""
Build GyroLabe native DLL and optionally rebuild ``llama-cli`` when gyroscopic
sources change.

Environment:

- GYROLABE_SKIP_LLAMA_BUILD: if set to 1/true/yes, do not run CMake for
  external/llama.cpp (DLL build is unchanged).
- GYROLABE_FORCE_LLAMA_BUILD: if set to 1/true/yes, always run the llama.cpp
  build (ignore mtime checks).

On Windows, the llama build retries a few times if MSVC reports LNK1104 (DLL
locked). Close llama-cli or other consumers of ``ggml-cpu.dll`` if it persists.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _codec_build_inputs(root: Path) -> list[Path]:
    deps = [
        root / "gyrograph_policy.c",
        root / "gyrolabe_chi_gauge_tile.c",
        root / "gyrolabe_wht.c",
        root / "gyrolabe_qubec_matmul.c",
        root / "gyrolabe_registry.c",
        root / "gyrograph.c",
    ]
    deps.extend(
        [
            root / "gyrograph_types.h",
            root / "gyrolabe_aperture.h",
            root / "gyrolabe_canonical.h",
            root / "gyrolabe_canonical.c",
            root / "gyrolabe_evolution.h",
            root / "gyrolabe_evolution.c",
            root / "gyrolabe_transforms.h",
            root / "gyrolabe_transforms.c",
            root / "gyrograph_moment.c",
            root / "gyrolabe_analysis.c",
        ]
    )
    deps.append(root / "gyrograph.h")
    deps.extend(sorted((root).glob("gyrolabe*.h")))
    deps.extend(sorted(root.glob("*.h")))
    return [p for p in deps if p.is_file()]


def _needs_rebuild(out: Path, deps: list[Path]) -> bool:
    if not out.is_file():
        return True
    try:
        out_m = out.stat().st_mtime
    except OSError:
        return True
    for p in deps:
        try:
            if p.stat().st_mtime > out_m:
                return True
        except OSError:
            return True
    return False


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    # ops_build.py: gyroscopic/ -> tools/ -> src/ -> repo
    return Path(__file__).resolve().parent.parent.parent.parent


def _native_extra_include_dirs() -> list[Path]:
    """Headers for ``gyrolabe_registry.h`` (ggml) and ``quants.h`` (gyroscopic backend)."""
    root = repo_root()
    base = root / "external" / "llama.cpp" / "ggml"
    cands = [
        base / "include",
        base / "src",
        base / "src" / "ggml-gyroscopic",
        base / "src" / "ggml-gyroscopic" / "arch" / "x86",
    ]
    return [p for p in cands if p.is_dir()]


def _ggml_link_args() -> list[str]:
    """llama.cpp ggml import libs (registry / qubec matmul)."""
    llama = repo_root() / "external" / "llama.cpp"
    libs = ("ggml-base.lib", "ggml.lib", "ggml-cpu.lib")
    for sub in ("build/ggml/src/Release", "build/ggml/src/Debug"):
        d = llama / sub.replace("/", os.sep)
        if all((d / n).is_file() for n in libs):
            return ["/link", f"/LIBPATH:{d.resolve()}", *libs]
    return []


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _llama_cli_candidates(repo_root: Path) -> list[Path]:
    base = repo_root / "external" / "llama.cpp" / "build" / "bin"
    if sys.platform == "win32":
        return [
            base / "Release" / "llama-cli.exe",
            base / "Debug" / "llama-cli.exe",
            base / "llama-cli.exe",
        ]
    return [base / "llama-cli"]


def _resolve_llama_cli_out(repo_root: Path) -> Path | None:
    for c in _llama_cli_candidates(repo_root):
        if c.is_file():
            return c
    return None


def _llama_gyroscopic_build_inputs(repo_root: Path) -> list[Path]:
    """Sources that affect the Gyroscopic-linked ``llama-cli`` binary."""
    gyroscopic = repo_root / "src" / "tools" / "gyroscopic"
    out: list[Path] = []
    if gyroscopic.is_dir():
        out.extend(p for p in gyroscopic.glob("*.h") if p.is_file())
        for name in (
            "gyrograph_policy.c",
            "gyrolabe_chi_gauge_tile.c",
            "gyrolabe_wht.c",
            "gyrolabe_qubec_matmul.c",
            "gyrolabe_registry.c",
            "gyrograph.c",
        ):
            cc = gyroscopic / name
            if cc.is_file():
                out.append(cc)
    ggml = repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-gyroscopic"
    for rel in (
        "gyroscopic-backend.cpp",
        "gyroscopic-backend.h",
        "gyrolabe_wht.c",
        "gyrolabe_wht.h",
        "ggml-cpu.c",
        "vec.cpp",
        "arch/x86/quants.c",
    ):
        p = ggml / rel
        if p.is_file():
            out.append(p)
    llama_ggml = repo_root / "external" / "llama.cpp" / "ggml"
    for rel in (
        "include/ggml-backend.h",
        "src/ggml-backend.cpp",
    ):
        p = llama_ggml / rel
        if p.is_file():
            out.append(p)
    cm_gyro = (
        repo_root / "external" / "llama.cpp" / "ggml" / "src" / "ggml-gyroscopic" / "CMakeLists.txt"
    )
    if cm_gyro.is_file():
        out.append(cm_gyro)
    llama_src = repo_root / "external" / "llama.cpp" / "src"
    for name in ("llama-model.cpp", "llama-context.cpp"):
        p = llama_src / name
        if p.is_file():
            out.append(p)
    ps1 = repo_root / "scripts" / "build_llama_cpp_windows.ps1"
    if ps1.is_file():
        out.append(ps1)
    return sorted({p.resolve() for p in out if p.is_file()})


def _llama_build_output_looks_like_dll_lock_error(text: str) -> bool:
    """MSVC LNK1104: linker cannot overwrite DLL (often still loaded)."""
    if not text:
        return False
    u = text.upper()
    return "LNK1104" in u or (
        "CANNOT OPEN FILE" in u and ".DLL" in u
    )


def _run_llama_build_windows(repo_root: Path) -> None:
    ps1 = repo_root / "scripts" / "build_llama_cpp_windows.ps1"
    if not ps1.is_file():
        raise FileNotFoundError(
            "GyroLabe: expected scripts/build_llama_cpp_windows.ps1 at "
            + str(ps1)
        )
    max_attempts = 4
    sleep_before_attempt = (0.0, 2.0, 5.0, 10.0)
    last_cp: subprocess.CompletedProcess[str] | None = None
    for attempt in range(max_attempts):
        if sleep_before_attempt[attempt] > 0:
            time.sleep(sleep_before_attempt[attempt])
        last_cp = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ps1.resolve()),
            ],
            cwd=str(repo_root.resolve()),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        if last_cp.returncode == 0:
            return
        out = (last_cp.stdout or "") + (last_cp.stderr or "")
        if attempt < max_attempts - 1 and _llama_build_output_looks_like_dll_lock_error(
            out
        ):
            continue
        break
    assert last_cp is not None
    msg = (
        "GyroLabe: llama.cpp build failed (PowerShell).\n"
        f"STDOUT:\n{last_cp.stdout}\n\nSTDERR:\n{last_cp.stderr}"
    )
    combined = (last_cp.stdout or "") + (last_cp.stderr or "")
    if _llama_build_output_looks_like_dll_lock_error(combined):
        msg += (
            "\n\nIf you see LNK1104 on ggml-cpu.dll: close any process using it "
            "(llama-cli, Python importing ggml, IDE, etc.), then retry. "
            "Or set GYROLABE_SKIP_LLAMA_BUILD=1 to use the existing llama-cli."
        )
    raise RuntimeError(msg)


def _run_llama_build_unix(repo_root: Path) -> None:
    llama = repo_root / "external" / "llama.cpp"
    cm = llama / "CMakeLists.txt"
    if not cm.is_file():
        raise FileNotFoundError(f"GyroLabe: missing {cm}")
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError(
            "GyroLabe: cmake not found in PATH; install CMake to build llama.cpp "
            "on this platform."
        )
    nj = max(1, (os.cpu_count() or 4))
    cfg_cmd = [
        cmake,
        "-B",
        "build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CPU_BACKEND_SUBDIR=ggml-gyroscopic",
        "-DGGML_GYROSCOPIC=ON",
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
            "GyroLabe: llama.cpp cmake configure failed.\n"
            f"STDOUT:\n{cp1.stdout}\n\nSTDERR:\n{cp1.stderr}"
        )
    cp2 = subprocess.run(
        [cmake, "--build", "build", "-j", str(nj)],
        cwd=str(llama.resolve()),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if cp2.returncode != 0:
        raise RuntimeError(
            "GyroLabe: llama.cpp build failed.\n"
            f"STDOUT:\n{cp2.stdout}\n\nSTDERR:\n{cp2.stderr}"
        )


def build_llama_cpp_if_needed(*, force: bool = False) -> Path | None:
    """
    Rebuild ``external/llama.cpp`` Release ``llama-cli`` when gyroscopic
    sources are newer than the existing binary, or when the binary is missing.

    Returns the path to ``llama-cli`` if present after the check, or ``None``
    if ``GYROLABE_SKIP_LLAMA_BUILD`` is set (no build attempted).
    """
    if _env_truthy("GYROLABE_SKIP_LLAMA_BUILD"):
        return _resolve_llama_cli_out(repo_root())

    root = repo_root()
    deps = _llama_gyroscopic_build_inputs(root)
    if not deps:
        raise RuntimeError(
            "GyroLabe: no gyroscopic build inputs found under repo; "
            "check paths."
        )

    out = _resolve_llama_cli_out(root)
    if not force and not _env_truthy("GYROLABE_FORCE_LLAMA_BUILD") and out is not None:
        if not _needs_rebuild(out, deps):
            return out

    print(
        "GyroLabe: building llama-cli (external/llama.cpp, gyroscopic backend)...",
        file=sys.stderr,
    )
    if sys.platform == "win32":
        _run_llama_build_windows(root)
    else:
        _run_llama_build_unix(root)

    out2 = _resolve_llama_cli_out(root)
    if out2 is None:
        raise FileNotFoundError(
            "GyroLabe: llama.cpp build finished but llama-cli not found. "
            "Expected one of:\n  "
            + "\n  ".join(str(p) for p in _llama_cli_candidates(root))
        )
    return out2


def _vswhere_installation_path() -> Path | None:
    pf = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not pf:
        return None
    vswhere = Path(pf) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    for args in (
        [
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        ["-latest", "-products", "*", "-property", "installationPath"],
    ):
        cp = subprocess.run(
            [str(vswhere), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.returncode != 0:
            continue
        line = (cp.stdout or "").strip().splitlines()
        if not line:
            continue
        p = Path(line[0].strip())
        if p.is_dir():
            return p
    return None


def _find_vcvarsall_bat() -> Path | None:
    env = os.environ.get("GYROLABE_VCVARSALL")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    inst = _vswhere_installation_path()
    if inst:
        cand = inst / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if cand.is_file():
            return cand
    for pf_key in ("ProgramFiles(x86)", "ProgramFiles"):
        pf = os.environ.get(pf_key)
        if not pf:
            continue
        base = Path(pf)
        for year in ("2022", "2019"):
            for edition in ("Community", "Professional", "Enterprise", "BuildTools"):
                cand = base / "Microsoft Visual Studio" / year / edition
                bat = cand / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if bat.is_file():
                    return bat
    return None


def _msvc_environment_from_vcvars(vcvars: Path) -> dict[str, str]:
    """Run vcvarsall.bat and return env dict (PATH includes cl.exe)."""
    vc = str(vcvars.resolve())
    batch = f'call "{vc}" x64 >nul && set'
    cp = subprocess.run(
        batch,
        shell=True,
        capture_output=True,
        text=True,
        errors="replace",
        env=os.environ.copy(),
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            "GyroLabe: vcvarsall.bat failed (x64). "
            "Install MSVC x64 tools or set GYROLABE_VCVARSALL.\n"
            f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
        )
    merged = os.environ.copy()
    for raw in (cp.stdout or "").splitlines():
        line = raw.rstrip("\r\n")
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key:
            merged[key] = val
    return merged


def _run_cmd_cl_windows(
    root: Path,
    build_dir: Path,
    out: Path,
    sources: list[str],
) -> None:
    vcvars = _find_vcvarsall_bat()
    if vcvars is None:
        raise RuntimeError(
            "GyroLabe: MSVC C++ tools not found. Install "
            '"Desktop development with C++" (or Build Tools), or set '
            "GYROLABE_VCVARSALL to the full path of vcvarsall.bat."
        )

    env = _msvc_environment_from_vcvars(vcvars)
    cl_exe = shutil.which("cl", path=env.get("PATH"))
    if cl_exe is None:
        raise RuntimeError(
            "GyroLabe: cl.exe not found in PATH after running vcvarsall.bat. "
            "Ensure MSVC x64 tools are installed."
        )

    inc = [str(root.resolve())]
    for d in _native_extra_include_dirs():
        inc.append(str(d.resolve()))
    fe = str(out.resolve())
    fo_dir = str(build_dir.resolve()) + "\\"

    cl_cmd = [
        cl_exe,
        "/nologo",
        "/O2",
        "/LD",
        "/std:c11",
        "/openmp",
        "/arch:AVX2",
        "/fp:strict",
        *[str(Path(s).resolve()) for s in sources],
        *[f"/I{p}" for p in inc],
        f"/Fe{fe}",
        f"/Fo{fo_dir}",
        *_ggml_link_args(),
    ]
    cp = subprocess.run(
        cl_cmd,
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            "GyroLabe: native build failed (MSVC)\n"
            f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
        )


def _run_gcc_like(
    cc: str,
    root: Path,
    build_dir: Path,
    out: Path,
    sources: list[str],
) -> None:
    obj_dir = build_dir
    objs: list[Path] = []
    inc_flags: list[str] = ["-I", str(root.resolve())]
    for d in _native_extra_include_dirs():
        inc_flags.extend(["-I", str(d.resolve())])
    pthread_flags: list[str] = [] if sys.platform == "win32" else ["-pthread"]
    for s in sources:
        sp = Path(s)
        o = obj_dir / (sp.stem + ".o")
        objs.append(o)
        cp1 = subprocess.run(
            [
                cc,
                "-std=c11",
                "-O2",
                "-mavx2",
                "-mfma",
                "-mf16c",
                "-ffp-contract=off",
                "-fPIC",
                "-fopenmp",
                *pthread_flags,
                "-c",
                str(sp.resolve()),
                *inc_flags,
                "-o",
                str(o),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if cp1.returncode != 0:
            raise RuntimeError(
                f"GyroLabe: compile failed ({cc})\n"
                f"STDOUT:\n{cp1.stdout}\n\nSTDERR:\n{cp1.stderr}"
            )
    cp2 = subprocess.run(
        [cc, "-shared", "-fopenmp", *pthread_flags, "-o", str(out)]
        + [str(o) for o in objs],
        capture_output=True,
        text=True,
        check=False,
    )
    if cp2.returncode != 0:
        raise RuntimeError(
            f"GyroLabe: link failed ({cc})\n"
            f"STDOUT:\n{cp2.stdout}\n\nSTDERR:\n{cp2.stderr}"
        )


def _try_cl_on_path(
    root: Path,
    build_dir: Path,
    out: Path,
    sources: list[str],
) -> bool:
    cl = shutil.which("cl")
    if not cl:
        return False
    inc = [f"/I{root}"]
    for d in _native_extra_include_dirs():
        inc.append(f"/I{d}")
    cmd = [
        cl,
        "/nologo",
        "/O2",
        "/LD",
        "/std:c11",
        "/openmp",
        "/fp:strict",
        "/arch:AVX2",
        *[str(Path(s).resolve()) for s in sources],
        *inc,
        f"/Fe:{str(out.resolve())}",
        f"/Fo:{str(build_dir.resolve())}\\",
        *_ggml_link_args(),
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
            "GyroLabe: native build failed (cl on PATH)\n"
            f"STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}"
        )
    return True


def build_gyrolabe_native() -> Path:
    root = Path(__file__).resolve().parent
    build_dir = root / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        str(root / "gyrograph_policy.c"),
        str(root / "gyrolabe_chi_gauge_tile.c"),
        str(root / "gyrolabe_wht.c"),
        str(root / "gyrolabe_qubec_matmul.c"),
        str(root / "gyrolabe_registry.c"),
        str(root / "gyrograph.c"),
        str(root / "gyrolabe_transforms.c"),
        str(root / "gyrograph_moment.c"),
        str(root / "gyrolabe_analysis.c"),
        str(root / "gyrolabe_canonical.c"),
        str(root / "gyrolabe_evolution.c"),
    ]
    deps = _codec_build_inputs(root)
    if sys.platform == "win32":
        out_name = "gyrolabe_native.dll"
    elif sys.platform == "darwin":
        out_name = "gyrolabe_native.dylib"
    else:
        out_name = "gyrolabe_native.so"
    out = build_dir / out_name

    if not _needs_rebuild(out, deps):
        return out

    cc_env = (
        os.environ.get("GYROLABE_NATIVE_CC")
        or os.environ.get("CC")
        or ""
    ).strip()

    if cc_env:
        _run_gcc_like(cc_env, root, build_dir, out, sources)
    elif sys.platform == "win32":
        if not _try_cl_on_path(root, build_dir, out, sources):
            _run_cmd_cl_windows(root, build_dir, out, sources)
    else:
        for cand in ("clang", "gcc"):
            w = shutil.which(cand)
            if w:
                _run_gcc_like(w, root, build_dir, out, sources)
                break
        else:
            raise RuntimeError(
                "GyroLabe: no C compiler found. Install clang or gcc, "
                "or set GYROLABE_NATIVE_CC to a compiler executable."
            )

    if not out.is_file():
        raise FileNotFoundError(f"GyroLabe: expected output not found: {out}")

    if not _env_truthy("GYROLABE_SKIP_LLAMA_BUILD"):
        build_llama_cpp_if_needed()

    return out
