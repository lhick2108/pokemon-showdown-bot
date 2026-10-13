<#
.SYNOPSIS
Builds the engine fork (engine/) into a wheel under <worktree>\.cache\pe WITHOUT touching the project .venv.

Use it to try engine changes next to a running ladder/duel session: prepend .cache\pe\site to
PYTHONPATH (this script also unpacks the wheel there) and the experiment sees the new
poke_engine while .venv keeps the old one.

  powershell -File scripts\build_engine_wheel.ps1 [-Jobs 4]
#>
param([int]$Jobs = 4)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$out = Join-Path $repoRoot '.cache\pe'
New-Item -ItemType Directory -Force $out | Out-Null

$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $repoRoot '.cache\cargo-target' }
if (-not $env:CARGO_HOME -and (Test-Path (Join-Path $repoRoot '.cache\cargo'))) { $env:CARGO_HOME = Join-Path $repoRoot '.cache\cargo' }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $repoRoot '.cache\uv' }
$env:CARGO_BUILD_JOBS = "$Jobs"

$vsPrefix = ''
if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
    $c = @()
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($pf) { $c += Get-ChildItem -Path (Join-Path $pf 'Microsoft Visual Studio') -Recurse -Filter VsDevCmd.bat -Depth 5 -ErrorAction SilentlyContinue }
    }
    $vs = $c | Select-Object -First 1
    if (-not $vs) { throw 'MSVC not found' }
    $vsPrefix = "call `"$($vs.FullName)`" -arch=amd64 >nul 2>&1 && "
}
$features = '--features poke-engine/terastallization --no-default-features --locked'
$py = Join-Path $repoRoot '.venv\Scripts\python.exe'
$cmd = "$vsPrefix`"$py`" -m pip --version >nul 2>&1 & python -m uv build --wheel --out-dir `"$out`" --config-setting `"build-args=$features`" `"$repoRoot\engine\poke-engine-py`""
& cmd.exe /d /c $cmd
if ($LASTEXITCODE -ne 0) { throw "wheel build failed ($LASTEXITCODE)" }
$whl = Get-ChildItem $out -Filter 'poke_engine-*.whl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$site = Join-Path $out 'site'
if (Test-Path $site) { Remove-Item -Recurse -Force $site }
New-Item -ItemType Directory -Force $site | Out-Null
$zip = Join-Path $out "wheel.zip"; Copy-Item $whl.FullName $zip -Force; Expand-Archive -Path $zip -DestinationPath $site -Force; Remove-Item $zip
Write-Host "wheel: $($whl.FullName)"
Write-Host "PYTHONPATH prefix: $site"
