$ErrorActionPreference = "Stop"
$gyrocrypt = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path $gyrocrypt "kernel\_build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$installPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1).Trim()
$vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
$native = Join-Path $gyrocrypt "kernel\native.c"
$out = Join-Path $buildDir "gyrocrypt_native.dll"
$cmd = "call `"$vcvars`" && cd /d `"$buildDir`" && cl /nologo /O2 /LD `"$native`" /Fe:`"$out`""
cmd /c $cmd
