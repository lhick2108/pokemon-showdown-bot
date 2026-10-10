<#
.SYNOPSIS
Idempotently sets up Foul Play (GPL-3.0, pmariglia/foul-play) as a local
opponent process under vendor/foul-play.

Foul Play is never imported into or copied out of vendor/; it always stays a
separate process run from its own virtual environment. This script does not
touch the project's own .venv.

Steps:
  1. Put vendor/foul-play at the pinned commit (clone from a local research
     checkout if available, otherwise from GitHub).
  2. Create vendor/foul-play/.venv (Python 3.12, via `uv`).
  3. Install Foul Play's requirements.txt (minus poke-engine, which is built
     from local source in step 4) into that venv.
  4. Build and install poke-engine 0.0.48 from vendor/poke-engine with the
     `poke-engine/terastallization` feature into vendor/foul-play/.venv.
#>

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $repoRoot '.cache'
$vendorRoot = Join-Path $repoRoot 'vendor'
$foulPlayDir = Join-Path $vendorRoot 'foul-play'
$pokeEngineDir = Join-Path $vendorRoot 'poke-engine'

$pinnedCommit = '8f541f80f91f960603673796559f796b10ccf0fb'
$researchClone = 'C:\Users\lhick\code\research\foul-play'
$githubUrl = 'https://github.com/pmariglia/foul-play.git'

function Write-Step($message) {
    Write-Host "[setup_foulplay] $message"
}

# --- 1. Clone/checkout Foul Play at the pinned commit -----------------------

if (-not (Test-Path -LiteralPath $foulPlayDir)) {
    New-Item -ItemType Directory -Path $vendorRoot -Force | Out-Null

    if (Test-Path -LiteralPath $researchClone) {
        Write-Step "Cloning Foul Play from local research checkout at $researchClone"
        git clone $researchClone $foulPlayDir
        if ($LASTEXITCODE -ne 0) {
            throw "git clone (local) failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Step "Cloning Foul Play from $githubUrl"
        git -c http.sslBackend=openssl clone $githubUrl $foulPlayDir
        if ($LASTEXITCODE -ne 0) {
            throw "git clone (github) failed with exit code $LASTEXITCODE"
        }
    }
}

Push-Location $foulPlayDir
try {
    $currentCommit = (git rev-parse HEAD).Trim()
    if ($currentCommit -ne $pinnedCommit) {
        Write-Step "Checking out pinned commit $pinnedCommit (currently $currentCommit)"
        git checkout $pinnedCommit 2>$null
        if ($LASTEXITCODE -ne 0) {
            # Shallow/local clones may not have the commit yet; fetch it.
            git fetch origin $pinnedCommit
            if ($LASTEXITCODE -ne 0) {
                throw "Could not fetch pinned commit $pinnedCommit"
            }
            git checkout $pinnedCommit
            if ($LASTEXITCODE -ne 0) {
                throw "git checkout $pinnedCommit failed with exit code $LASTEXITCODE"
            }
        }
    }
    $finalCommit = (git rev-parse HEAD).Trim()
    if ($finalCommit -ne $pinnedCommit) {
        throw "vendor/foul-play is at $finalCommit, expected pinned commit $pinnedCommit"
    }
    Write-Step "vendor/foul-play is at pinned commit $pinnedCommit"
}
finally {
    Pop-Location
}

# --- 2. Create Foul Play's own venv (never the project's .venv) -------------

$fpVenv = Join-Path $foulPlayDir '.venv'
$fpPython = Join-Path $fpVenv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $fpPython)) {
    Write-Step "Creating vendor/foul-play/.venv (Python 3.12)"
    python -m uv venv --python 3.12 $fpVenv
    if ($LASTEXITCODE -ne 0) {
        throw "uv venv failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Step "vendor/foul-play/.venv already exists"
}

# --- 3. Install Foul Play's requirements (minus poke-engine) ----------------

$requirementsPath = Join-Path $foulPlayDir 'requirements.txt'
$requirementLines = Get-Content -LiteralPath $requirementsPath |
    Where-Object { $_.Trim() -ne '' -and $_ -notmatch '^\s*poke-engine' }

if ($requirementLines.Count -gt 0) {
    Write-Step "Installing Foul Play's non-poke-engine requirements"
    python -m uv pip install --python $fpPython @requirementLines
    if ($LASTEXITCODE -ne 0) {
        throw "uv pip install (requirements) failed with exit code $LASTEXITCODE"
    }
}

# --- 4. Build and install poke-engine from local source ---------------------

if (-not (Test-Path -LiteralPath $pokeEngineDir)) {
    throw "vendor/poke-engine is missing. See docs/spike-poke-engine.md for how to obtain it (pinned commit c2dbbda, version 0.0.48)."
}

$env:CARGO_HOME = Join-Path $cacheRoot 'cargo'
$env:CARGO_TARGET_DIR = Join-Path $cacheRoot 'cargo-target'
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'

$wheelDir = Join-Path $env:CARGO_TARGET_DIR 'wheels'
$cachedWheel = Get-ChildItem -LiteralPath $wheelDir -Filter 'poke_engine-0.0.48-cp312-cp312-win_amd64.whl' -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($cachedWheel) {
    Write-Step "Reusing cached poke-engine wheel: $($cachedWheel.FullName)"
    python -m uv pip install --python $fpPython --reinstall --no-cache $cachedWheel.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "uv pip install (cached poke-engine wheel) failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Step 'No cached poke-engine wheel found; building from vendor/poke-engine source'

    $cargoBin = 'C:\Users\lhick\.cargo\bin'
    if (Test-Path -LiteralPath $cargoBin) {
        $env:Path = "$cargoBin;$env:Path"
    }

    $vsDevCmd = 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat'
    $pokeEnginePy = Join-Path $pokeEngineDir 'poke-engine-py'
    $buildArgsSetting = 'build-args=--features poke-engine/terastallization --no-default-features'

    if (Test-Path -LiteralPath $vsDevCmd) {
        $quotedPython = '"' + $fpPython + '"'
        $quotedPokeEnginePy = '"' + $pokeEnginePy + '"'
        $innerCommand = "python -m uv pip install --python $quotedPython --reinstall --no-cache --verbose " +
            "--config-settings `"$buildArgsSetting`" $quotedPokeEnginePy"
        cmd.exe /d /c "call `"$vsDevCmd`" -arch=amd64 && $innerCommand"
        if ($LASTEXITCODE -ne 0) {
            throw "poke-engine build (via VsDevCmd) failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Step "VsDevCmd.bat not found at $vsDevCmd; attempting build without it"
        python -m uv pip install --python $fpPython --reinstall --no-cache --verbose `
            --config-settings $buildArgsSetting $pokeEnginePy
        if ($LASTEXITCODE -ne 0) {
            throw "poke-engine build failed with exit code $LASTEXITCODE"
        }
    }
}

# --- 5. Sanity check ----------------------------------------------------------

Write-Step 'Verifying poke-engine import inside vendor/foul-play/.venv'
& $fpPython -c "import poke_engine; print('poke_engine OK:', poke_engine.__file__)"
if ($LASTEXITCODE -ne 0) {
    throw "poke-engine import check failed with exit code $LASTEXITCODE"
}

Write-Step 'Verifying Foul Play package imports inside vendor/foul-play/.venv'
Push-Location $foulPlayDir
try {
    & $fpPython -c "import fp.config; print('fp.config OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "fp.config import check failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Step 'Foul Play setup complete.'
