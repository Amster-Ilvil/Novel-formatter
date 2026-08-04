param(
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# Keep this launcher ASCII-only for Windows PowerShell 5.1 code-page compatibility.

$ProjectDir = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $ProjectDir '.runtime'
$PortableRoot = Join-Path $RuntimeRoot 'windows-python-v1'
$AppVenv = Join-Path $ProjectDir '.venv-app-windows'
$PythonExe = Join-Path $AppVenv 'Scripts\python.exe'
$LogDir = Join-Path $RuntimeRoot 'logs'
$LogFile = Join-Path $LogDir 'windows-launcher.log'
$DownloadDir = Join-Path $RuntimeRoot 'downloads'
$PythonRelease = '20251010'
$PythonVersion = '3.12.12'

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $LogDir, $DownloadDir | Out-Null

function Protect-Text([string]$Text) {
    if (-not $Text) { return '' }
    $safe = $Text
    foreach ($value in @($ProjectDir, $env:USERPROFILE, $env:TEMP, $env:TMP)) {
        if ($value) { $safe = $safe.Replace($value, '<LOCAL>') }
    }
    $safe = [regex]::Replace($safe, '(?i)([?&](?:token|key|signature|sig|credential)=)[^&\s]+', '$1<REDACTED>')
    $safe = [regex]::Replace($safe, '(?i)\b(?:sk|hf|AIza)[-_A-Za-z0-9]{12,}\b', '<REDACTED>')
    if ($safe.Length -gt 2000) { $safe = $safe.Substring($safe.Length - 2000) }
    return $safe
}

function Write-Stage([string]$Message) {
    $safe = Protect-Text $Message
    Add-Content -LiteralPath $LogFile -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $safe) -Encoding UTF8
    Write-Host $safe
}

function Test-SupportedPython([string]$Executable) {
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable)) { return $false }
    & $Executable -c "import sys;raise SystemExit(0 if sys.version_info.major==3 and 10<=sys.version_info.minor<=13 else 8)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Find-SystemPython {
    if ($env:NOVEL_FORMATTER_PYTHON -and (Test-SupportedPython $env:NOVEL_FORMATTER_PYTHON)) {
        return $env:NOVEL_FORMATTER_PYTHON
    }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @('3.13','3.12','3.11','3.10')) {
            $candidate = & $py.Source "-$version" -c "import sys;print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $candidate) {
                $resolved = ($candidate | Select-Object -Last 1).Trim()
                if (Test-SupportedPython $resolved) { return $resolved }
            }
        }
    }
    foreach ($name in @('python.exe','python3.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and (Test-SupportedPython $command.Source)) { return $command.Source }
    }
    return $null
}

function Get-PlatformArchive {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
    switch ($arch) {
        'x64'   { $triple = 'x86_64-pc-windows-msvc' }
        'arm64' { $triple = 'aarch64-pc-windows-msvc' }
        default { throw "Unsupported Windows architecture: $arch" }
    }
    return "cpython-$PythonVersion+$PythonRelease-$triple-install_only_stripped.tar.gz"
}

function Invoke-Download([string[]]$Urls, [string]$Destination) {
    $part = "$Destination.part"
    foreach ($url in $Urls) {
        Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        Write-Stage 'Downloading the runtime environment.'
        try {
            Import-Module BitsTransfer -ErrorAction Stop
            Start-BitsTransfer -Source $url -Destination $part -DisplayName 'Novel Formatter deployment' -Description 'First deployment runtime' -ErrorAction Stop
            Move-Item -LiteralPath $part -Destination $Destination -Force
            return
        } catch {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source --location --fail --retry 3 --connect-timeout 20 --output $part $url
            if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $part)) {
                Move-Item -LiteralPath $part -Destination $Destination -Force
                return
            }
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $part -TimeoutSec 1800
            Move-Item -LiteralPath $part -Destination $Destination -Force
            return
        } catch {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
    }
    throw 'All download sources failed. Check the network or configure a proxy in the environment.'
}

function Install-PortablePython {
    $archive = Get-PlatformArchive
    $archivePath = Join-Path $DownloadDir $archive
    $sumPath = Join-Path $DownloadDir "SHA256SUMS-$PythonRelease"
    $mirrorRoot = "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/$PythonRelease"
    $officialRoot = "https://github.com/astral-sh/python-build-standalone/releases/download/$PythonRelease"

    if (-not (Test-Path -LiteralPath $archivePath)) {
        Invoke-Download -Urls @("$mirrorRoot/$archive", "$officialRoot/$archive") -Destination $archivePath
    }
    if (-not (Test-Path -LiteralPath $sumPath)) {
        Invoke-Download -Urls @("$mirrorRoot/SHA256SUMS", "$officialRoot/SHA256SUMS") -Destination $sumPath
    }

    $line = Get-Content -LiteralPath $sumPath | Where-Object { $_ -match [regex]::Escape($archive) } | Select-Object -First 1
    if (-not $line) { throw 'The standalone Python checksum file does not contain the required archive entry.' }
    $expected = (($line -split '\s+')[0]).Trim().ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($expected -ne $actual) {
        Remove-Item -LiteralPath $archivePath, $sumPath -Force -ErrorAction SilentlyContinue
        throw 'Standalone Python SHA-256 verification failed. The downloaded files were removed.'
    }

    $newRoot = "$PortableRoot.new"
    Remove-Item -LiteralPath $newRoot -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $newRoot | Out-Null
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tar) { throw 'tar.exe is unavailable on this Windows system, so standalone Python cannot be extracted.' }
    & $tar.Source -xzf $archivePath -C $newRoot
    if ($LASTEXITCODE -ne 0) { throw 'Standalone Python extraction failed.' }
    $candidate = Join-Path $newRoot 'python\python.exe'
    if (-not (Test-SupportedPython $candidate)) { throw 'Standalone Python is unavailable after extraction.' }
    Remove-Item -LiteralPath $PortableRoot -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -LiteralPath $newRoot -Destination $PortableRoot
    return (Join-Path $PortableRoot 'python\python.exe')
}

try {
    Write-Stage 'Starting the Windows deployment check.'
    $basePython = Find-SystemPython
    if (-not $basePython) {
        Write-Stage 'No compatible Python was found. Preparing a verified portable runtime.'
        $basePython = Install-PortablePython
    } else {
        Write-Stage 'A compatible Python installation was found.'
    }

    if (-not (Test-SupportedPython $PythonExe)) {
        Remove-Item -LiteralPath $AppVenv -Recurse -Force -ErrorAction SilentlyContinue
        Write-Stage 'Creating the project virtual environment.'
        & $basePython -m venv $AppVenv
        if ($LASTEXITCODE -ne 0 -or -not (Test-SupportedPython $PythonExe)) {
            throw 'Failed to create the project virtual environment.'
        }
    }

    $arguments = @(
        (Join-Path $ProjectDir 'bootstrap.py'),
        '--install-main-deps'
    )
    if (-not $PrepareOnly) { $arguments += '--launch' }

    Write-Stage 'Preparing application dependencies only. OCR resources remain deferred until the user starts OCR and confirms installation.'
    Push-Location $ProjectDir
    try {
        & $PythonExe @arguments
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code -ne 0) { throw "Application preparation or startup failed (exit code $code)." }
    Write-Stage 'The launcher completed successfully.'
    exit 0
}
catch {
    $message = Protect-Text $_.Exception.Message
    Write-Stage ("Startup failed: " + $message)
    Add-Type -AssemblyName PresentationFramework -ErrorAction SilentlyContinue
    if ('System.Windows.MessageBox' -as [type]) {
        [System.Windows.MessageBox]::Show($message, 'Novel Formatter Windows startup failed', 'OK', 'Error') | Out-Null
    }
    exit 1
}
