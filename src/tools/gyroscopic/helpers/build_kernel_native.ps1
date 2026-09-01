$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$buildDir = Join-Path $repoRoot "src\tools\gyroscopic\_build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$installPath = (& $vswhere -latest -products * -property installationPath | Select-Object -First 1).Trim()
$vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
$kernel = Join-Path $repoRoot "src\tools\gyroscopic\kernel.c"
$runtime = Join-Path $repoRoot "src\tools\gyroscopic\runtime.c"
$codec = Join-Path $repoRoot "src\tools\gyroscopic\codec.c"
$attn = Join-Path $repoRoot "src\tools\gyroscopic\attn.c"
$ledger = Join-Path $repoRoot "src\tools\gyroscopic\ledger.c"
$layer = Join-Path $repoRoot "src\tools\gyroscopic\layer.c"
$out = Join-Path $buildDir "gyroscopic_native.dll"
$cmd = "call `"$vcvars`" && cd /d `"$buildDir`" && cl /nologo /O2 /MD /LD `"$kernel`" `"$runtime`" `"$codec`" `"$attn`" `"$ledger`" `"$layer`" /Fe:`"$out`""
cmd /c $cmd
