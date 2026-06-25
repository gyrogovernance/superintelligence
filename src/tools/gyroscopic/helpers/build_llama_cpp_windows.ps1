# Build external/llama.cpp (CPU, MSVC Release) on Windows.
# Requires: CMake, Ninja (optional), Visual Studio 2022 Build Tools (C++ workload).
# Run from repo root:
#   powershell -File src/tools/gyroscopic/helpers/build_llama_cpp_windows.ps1
#   powershell -File src/tools/gyroscopic/helpers/build_llama_cpp_windows.ps1 -Mode stock

param(
    [ValidateSet("stock", "gyroscopic")]
    [string]$Mode = "gyroscopic"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$llamaDir = Join-Path $repoRoot "external\llama.cpp"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere not found. Install Visual Studio 2022 Build Tools with Desktop development with C++."
}
$installPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1).Trim()
if (-not $installPath) { throw "No Visual Studio installation found." }
$vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found: $vcvars" }

# Prism/Bonsai ggml declares ASM; CMake needs ml64 (MASM), not cl.exe (CMP0194).
$ml64 = $null
$msvcBin = Join-Path $installPath "VC\Tools\MSVC"
if (Test-Path $msvcBin) {
    $ml64Cand = Get-ChildItem -Path $msvcBin -Filter ml64.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\Hostx64\\x64\\ml64\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($ml64Cand) { $ml64 = $ml64Cand.FullName }
}
if (-not $ml64) {
    throw "ml64.exe (MASM) not found. Install VS Build Tools C++ and the MASM component."
}
$asmCompilerArg = "-DCMAKE_ASM_COMPILER=`"$ml64`""

$buildJobs = 8
if ($env:GGML_LLAMA_BUILD_JOBS -match '^\d+$') {
    $buildJobs = [int]$env:GGML_LLAMA_BUILD_JOBS
}
if ($buildJobs -lt 1) {
    $buildJobs = 1
}

$buildTargets = @("llama-cli")
if ($env:GGML_LLAMA_BUILD_TARGETS) {
    $buildTargets = @($env:GGML_LLAMA_BUILD_TARGETS -split '[,; ]+' | Where-Object { $_ -and $_.Trim() })
}
$targetArgs = ""
if ($buildTargets.Count -gt 0) {
    $targetArgs = "--target " + (($buildTargets | ForEach-Object { $_.Trim() }) -join " ")
}

if ($Mode -eq "stock") {
    $buildDir = "build-stock"
    $backendSubdir = "ggml-cpu"
    $gyroFlag = "-DGGML_GYROSCOPIC=OFF"
    $modeLabel = "stock (vanilla ggml-cpu)"
} else {
    $buildDir = "build"
    $backendSubdir = "ggml-gyroscopic"
    $gyroFlag = "-DGGML_GYROSCOPIC=ON"
    $modeLabel = "gyroscopic (ggml-gyroscopic)"
}

$buildLine = "cmake --build $buildDir --config Release $targetArgs -j $buildJobs"
$cachePath = Join-Path $llamaDir "$buildDir\CMakeCache.txt"
if (Test-Path $cachePath) {
    $generatorLine = Select-String -Path $cachePath -Pattern "^CMAKE_GENERATOR:INTERNAL=" | Select-Object -First 1
    if ($generatorLine -and $generatorLine.Line -like "*Visual Studio*") {
        $buildLine = "cmake --build $buildDir --config Release $targetArgs -- /m:$buildJobs /nr:false"
    }
}

Write-Host "Build mode: $modeLabel"
Write-Host "Build tree: external\llama.cpp\$buildDir"
$cmakeArgs = @(
    "call `"$vcvars`" && cd /d `"$llamaDir`" && cmake -B $buildDir -DCMAKE_BUILD_TYPE=Release $asmCompilerArg -DGGML_CPU_BACKEND_SUBDIR=$backendSubdir $gyroFlag -DGGML_OPENMP=ON -DGGML_AVX2=ON -DGGML_BMI2=ON && $buildLine"
)
cmd.exe /c $cmakeArgs[0]
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. Example: external\llama.cpp\$buildDir\bin\Release\llama-cli.exe --version"
