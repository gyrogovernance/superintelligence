# Build external/llama.cpp (CPU, MSVC Release) on Windows.
# Requires: CMake, Ninja (optional), Visual Studio 2022 Build Tools (C++ workload).
# Run from repo root: powershell -File scripts/build_llama_cpp_windows.ps1

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$llamaDir = Join-Path $repoRoot "external\llama.cpp"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found. Install Visual Studio 2022 Build Tools with Desktop development with C++."
}
$installPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1).Trim()
if (-not $installPath) { throw "No Visual Studio installation found." }
$vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }

$cmakeArgs = @(
    "call `"$vcvars`" && cd /d `"$llamaDir`" && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_BACKEND_SUBDIR=ggml-gyroscopic -DGGML_GYROSCOPIC=ON && cmake --build build --config Release -j 8"
)
cmd.exe /c $cmakeArgs[0]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. Example: external\llama.cpp\build\bin\Release\llama-cli.exe --version"
