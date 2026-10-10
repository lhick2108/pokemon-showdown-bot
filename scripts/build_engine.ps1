<#
.SYNOPSIS
Builds our poke-engine fork (engine/) and installs it into the project .venv.

The fork is built with the same Cargo features as Foul Play's Gen 9 engine
(`poke-engine/terastallization --no-default-features`) and with `--locked`, so
the Rust dependency versions are exactly those pinned in engine/Cargo.lock.
It installs the `poke_engine` package (version 0.0.48+sdb.N) into
<repo>\.venv only. Foul Play's own venv (vendor\foul-play\.venv) is not touched.

Options:
  -SkipTests   do not run `cargo test` in engine/ before installing
  -Offline     pass --offline to cargo test (build-args --locked already forbids lock changes)

Environment handling:
  * CARGO_HOME: if unset and .cache\cargo\config.toml exists (a vendored,
    offline crates mirror used when crates.io is unreachable) it is used;
    otherwise cargo's normal CARGO_HOME / network access applies.
  * CARGO_TARGET_DIR defaults to .cache\cargo-target (ignored by git).
  * The Visual Studio C++ environment (VsDevCmd.bat) is loaded if link.exe is
    not already on PATH.
#>
param(
    [switch]$SkipTests,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$engineDir = Join-Path $repoRoot 'engine'
$cacheRoot = Join-Path $repoRoot '.cache'
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path (Join-Path $engineDir 'Cargo.toml'))) {
    throw "engine/Cargo.toml not found under $repoRoot"
}
if (-not (Test-Path $venvPython)) {
    throw "Project venv not found at $venvPython. Run 'python -m uv sync' first."
}

# --- environment ---------------------------------------------------------
$cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path $cargoBin) { $env:Path = "$cargoBin;$env:Path" }

$mirrorCargoHome = Join-Path $cacheRoot 'cargo'
if (-not $env:CARGO_HOME -and (Test-Path (Join-Path $mirrorCargoHome 'config.toml'))) {
    $env:CARGO_HOME = $mirrorCargoHome
    Write-Host "Using offline crate mirror: CARGO_HOME=$env:CARGO_HOME"
}
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = Join-Path $cacheRoot 'cargo-target' }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv' }

# Prefix that loads the MSVC build environment when needed.
$vsPrefix = ''
if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
    $candidates = @()
    foreach ($pf in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($pf) {
            $candidates += Get-ChildItem -Path (Join-Path $pf 'Microsoft Visual Studio') -Recurse -Filter VsDevCmd.bat -Depth 5 -ErrorAction SilentlyContinue
        }
    }
    $vsDevCmd = $candidates | Select-Object -First 1
    if (-not $vsDevCmd) {
        throw 'MSVC link.exe not found and no VsDevCmd.bat located. Install the Visual Studio C++ build tools.'
    }
    $vsPrefix = "call `"$($vsDevCmd.FullName)`" -arch=amd64 >nul 2>&1 && "
    Write-Host "Using MSVC environment from $($vsDevCmd.FullName)"
}

function Invoke-Cmd([string]$Command) {
    & cmd.exe /d /c "$vsPrefix$Command"
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Command" }
}

$features = '--features poke-engine/terastallization --no-default-features --locked'

# --- Rust unit tests -------------------------------------------------------
if (-not $SkipTests) {
    Push-Location $engineDir
    try {
        $offlineFlag = if ($Offline) { '--offline' } else { '' }
        Invoke-Cmd "cargo test --locked $offlineFlag --features terastallization --no-default-features --lib"
    } finally { Pop-Location }
}

# --- build + install into the project venv ---------------------------------
$pyProject = Join-Path $engineDir 'poke-engine-py'
Push-Location $repoRoot
try {
    Invoke-Cmd "python -m uv pip install --python `"$venvPython`" --reinstall --no-cache --config-settings `"build-args=$features`" `"$pyProject`""
} finally { Pop-Location }

& $venvPython -c "import poke_engine; print('poke_engine installed:', poke_engine.__file__, '| FEATURE_VERSION', poke_engine.FEATURE_VERSION, '| FEATURE_LENGTH', poke_engine.FEATURE_LENGTH)"
if ($LASTEXITCODE -ne 0) { throw 'Post-install import check failed.' }
