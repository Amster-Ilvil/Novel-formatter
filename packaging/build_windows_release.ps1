param(
    [string]$Version = ""
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
$TarPath = Join-Path $Dist '_tracked-source.tar'

Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Archive, $TarPath -Force -ErrorAction SilentlyContinue
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

1. Extract this ZIP to a normal writable folder.
2. Double-click 启动Windows.bat.
3. The launcher prepares only Python and the main GUI dependencies.
4. OCR-specific runtimes and model weights are NOT bundled. They are installed
   only after you start the corresponding OCR function and confirm the prompt.
5. API credentials are entered at runtime; do not store them in the program folder.

Privacy: this package is generated from a clean GitHub checkout and contains no
local developer cache, .env file, model cache, OCR result, log, EPUB or DOCX data.
"@ | Set-Content -LiteralPath (Join-Path $Stage 'RELEASE-README.txt') -Encoding UTF8

Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $Archive -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $Archive)) { throw 'Windows release archive was not created.' }
Write-Host "Created $Archive"
