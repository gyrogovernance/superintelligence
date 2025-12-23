# Safe cache cleanup script for Router development
# This removes only cache/temp files, not source code

Write-Host "Cleaning development caches..." -ForegroundColor Yellow

$freed = 0

# Python caches
$pythonCaches = @(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".coverage",
    "htmlcov",
    ".hypothesis"
)

foreach ($cache in $pythonCaches) {
    $path = Get-ChildItem -Path . -Filter $cache -Recurse -Directory -ErrorAction SilentlyContinue
    if ($path) {
        foreach ($dir in $path) {
            $size = (Get-ChildItem $dir.FullName -Recurse -File -ErrorAction SilentlyContinue | 
                     Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
            if ($size) {
                Remove-Item $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $freed += $size
                Write-Host "Removed: $($dir.FullName) ($([math]::Round($size/1MB,2)) MB)" -ForegroundColor Green
            }
        }
    }
}

# Git LFS temp files
if (Test-Path ".git\lfs\tmp") {
    $size = (Get-ChildItem ".git\lfs\tmp" -Recurse -File -ErrorAction SilentlyContinue | 
             Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
    if ($size) {
        Remove-Item ".git\lfs\tmp\*" -Recurse -Force -ErrorAction SilentlyContinue
        $freed += $size
        Write-Host "Removed: .git\lfs\tmp ($([math]::Round($size/1MB,2)) MB)" -ForegroundColor Green
    }
}

# Git index lock (if stale)
if (Test-Path ".git\index.lock") {
    Remove-Item ".git\index.lock" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed: .git\index.lock" -ForegroundColor Green
}

Write-Host "`nTotal space freed: $([math]::Round($freed/1GB,2)) GB" -ForegroundColor Cyan
Write-Host "`nNote: Atlas files in data/atlas/ are build artifacts." -ForegroundColor Yellow
Write-Host "They can be rebuilt with: python -m ggg_asi_router.physics.atlas_builder complete" -ForegroundColor Yellow

