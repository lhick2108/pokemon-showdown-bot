$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $repoRoot '.cache'
$serverPath = Join-Path $repoRoot 'vendor\pokemon-showdown'

# Current Pokemon Showdown releases require Node 22.18 or later.
$minimumNodeVersion = [version]'22.18.0'
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$nodeVersion = $null
if ($nodeCommand) {
    $nodeVersionText = (& $nodeCommand.Source --version 2>$null).Trim()
    if ($LASTEXITCODE -eq 0 -and $nodeVersionText -match '^v?(\d+\.\d+\.\d+)$') {
        $nodeVersion = [version]$Matches[1]
    }
}
if (-not $nodeVersion -or $nodeVersion -lt $minimumNodeVersion) {
    $runtimeVersion = '22.23.3'
    $expectedArchiveHash = '2b0ff57b049cda1bbcea2240eec20467018713c1efe1f7360c2681859b90ed71'
    $runtimeName = "node-v$runtimeVersion-win-x64"
    $runtimePath = Join-Path $cacheRoot $runtimeName
    $runtimeExe = Join-Path $runtimePath 'node.exe'

    if (-not (Test-Path -LiteralPath $runtimeExe)) {
        New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
        $archiveName = "$runtimeName.zip"
        $archivePath = Join-Path $cacheRoot $archiveName
        $baseUrl = "https://nodejs.org/dist/v$runtimeVersion"

        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri "$baseUrl/$archiveName" -OutFile $archivePath -UseBasicParsing
        $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedArchiveHash) {
            throw "SHA256 verification failed for $archiveName"
        }

        Expand-Archive -LiteralPath $archivePath -DestinationPath $cacheRoot -Force
    }

    if (-not (Test-Path -LiteralPath $runtimeExe)) {
        throw "Node.js $runtimeVersion download did not contain node.exe"
    }
    $env:Path = "$runtimePath;$env:Path"
    $nodeVersionText = (& $runtimeExe --version).Trim()
    $nodeVersion = [version]($nodeVersionText -replace '^v', '')
    if ($nodeVersion -lt $minimumNodeVersion) {
        throw "Pokemon Showdown requires Node.js $minimumNodeVersion or later; found $nodeVersionText"
    }
}

if (-not (Test-Path -LiteralPath $serverPath)) {
    $vendorPath = Split-Path -Parent $serverPath
    New-Item -ItemType Directory -Path $vendorPath -Force | Out-Null

    git -c http.sslBackend=openssl clone --depth 1 https://github.com/smogon/pokemon-showdown $serverPath
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed with exit code $LASTEXITCODE"
    }
}

$installMarker = Join-Path $serverPath '.npm-install-complete'
if (-not (Test-Path -LiteralPath $installMarker)) {
    $npmCache = Join-Path $cacheRoot 'npm'
    New-Item -ItemType Directory -Path $npmCache -Force | Out-Null
    $env:npm_config_cache = $npmCache

    Push-Location $serverPath
    try {
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed with exit code $LASTEXITCODE"
        }
        New-Item -ItemType File -Path $installMarker -Force | Out-Null
    }
    finally {
        Pop-Location
    }
}

$exampleConfig = Join-Path $serverPath 'config\config-example.js'
$config = Join-Path $serverPath 'config\config.js'
if (-not (Test-Path -LiteralPath $config)) {
    Copy-Item -LiteralPath $exampleConfig -Destination $config
}
if (-not (Select-String -LiteralPath $config -Pattern '^\s*exports\.subprocesses\s*=\s*0\s*;' -Quiet)) {
    Add-Content -LiteralPath $config -Value "`r`nexports.subprocesses = 0;"
}

$lastBindAddress = Get-Content -LiteralPath $config |
    Where-Object { $_ -match '^\s*exports\.bindaddress\s*=' } |
    Select-Object -Last 1
if ($lastBindAddress -notmatch '^\s*exports\.bindaddress\s*=\s*[''"]127\.0\.0\.1[''"]\s*;') {
    Add-Content -LiteralPath $config -Value "`r`nexports.bindaddress = '127.0.0.1';"
}

Push-Location $serverPath
try {
    node pokemon-showdown start 8000 --no-security
    if ($LASTEXITCODE -ne 0) {
        throw "Pokemon Showdown exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
