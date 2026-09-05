param(
    [string]$Version = "",
    [switch]$RequireInstaller
)
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Version) {
    $Version = (Get-Content -LiteralPath (Join-Path $Root 'VERSION') -Raw).Trim()
}
if (-not $Version) { throw 'VERSION is empty.' }

$Dist = Join-Path $Root 'dist'
$Stage = Join-Path $Dist 'NovelFormatterStudio-Windows'
$Archive = Join-Path $Dist ("NovelFormatter_{0}_Windows_x64.zip" -f $Version)
$Installer = Join-Path $Dist ("NovelFormatter_{0}_Windows_x64_Setup.exe" -f $Version)
$TarPath = Join-Path $Dist '_tracked-source.tar'

Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Archive, $Installer, $TarPath -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Dist, $Stage | Out-Null

# Important privacy boundary: package only files tracked in the Git commit.
# Developer-machine untracked files, caches, logs, documents and .env files
# are therefore impossible to enter this archive.
Push-Location $Root
try {
    git archive --format=tar HEAD -o $TarPath
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed.' }
    tar.exe -xf $TarPath -C $Stage
    if ($LASTEXITCODE -ne 0) { throw 'tar extraction failed.' }
} finally {
    Pop-Location
}
Remove-Item -LiteralPath $TarPath -Force -ErrorAction SilentlyContinue

# Development-only metadata is not needed by end users.
foreach ($relative in @('.github', 'tests')) {
    Remove-Item -LiteralPath (Join-Path $Stage $relative) -Recurse -Force -ErrorAction SilentlyContinue
}
foreach ($relative in @('.gitignore')) {
    Remove-Item -LiteralPath (Join-Path $Stage $relative) -Force -ErrorAction SilentlyContinue
}

@"
Novel Formatter Studio $Version - Windows x64
================================================

Recommended: run NovelFormatter_${Version}_Windows_x64_Setup.exe and follow the installer.
Portable fallback: extract NovelFormatter_${Version}_Windows_x64.zip to a normal writable folder.

After installation/extraction, launch Novel Formatter Studio from the Start menu, desktop shortcut,
or double-click 启动Windows.bat.

The first launch automatically prepares a private Python runtime and the main GUI dependencies in
the current user's profile. Administrator rights are not required. OCR-specific runtimes and model
weights are NOT bundled; they are installed only after you start the corresponding OCR function and
confirm the prompt. API credentials are entered at runtime; do not store them in the program folder.

Privacy: this package is generated from a clean GitHub checkout and contains no local developer
cache, .env file, model cache, OCR result, log, EPUB or DOCX data.
"@ | Set-Content -LiteralPath (Join-Path $Stage 'RELEASE-README.txt') -Encoding UTF8

Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $Archive -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $Archive)) { throw 'Windows release archive was not created.' }
Write-Host "Created $Archive"

function Find-Iscc {
    $command = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $candidate = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1

    if ($candidate) { return $candidate }
    return $null
}

$Iscc = Find-Iscc
if (-not $Iscc) {
    if ($RequireInstaller) {
        throw 'Inno Setup 6 compiler (ISCC.exe) was not found.'
    }
    Write-Warning 'Inno Setup 6 was not found; portable ZIP was created, installer was skipped.'
    exit 0
}

$Iss = Join-Path $PSScriptRoot 'windows_installer.iss'
if (-not (Test-Path -LiteralPath $Iss)) { throw "Missing installer script: $Iss" }

& $Iscc "/DMyAppVersion=$Version" "/DSourceDir=$Stage" "/DOutputDir=$Dist" $Iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE." }
if (-not (Test-Path -LiteralPath $Installer)) { throw 'Windows installer was not created.' }
Write-Host "Created $Installer"
