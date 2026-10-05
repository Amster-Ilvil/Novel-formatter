param([string]$Version = '2.0.0')
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSScriptRoot
$DateTag = Get-Date -Format 'yyyyMMdd'
$DistRoot = Join-Path $ProjectDir 'dist'
$PackageDir = Join-Path $DistRoot 'NovelFormatterStudio-Windows'
$ZipPath = Join-Path $DistRoot "NovelFormatterStudio_Windows_v${Version}_${DateTag}.zip"
if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
$exclude = @('.git','.windows-runtime-state','.pytest_cache','__pycache__','dist','build','.model-cache','.ocr-runtimes','.ocr-runtime-state','.manual-model-updates','.runtime','debug','logs','epub_workspace','output','outputs')
Get-ChildItem -LiteralPath $ProjectDir -Force | Where-Object {
    ($exclude -notcontains $_.Name) -and
    (-not $_.Name.StartsWith('.venv')) -and
    ($_.Name -ne 'venv') -and
    (-not $_.Name.StartsWith('.env'))
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $PackageDir -Recurse -Force
}
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path (Join-Path $PackageDir '*') -DestinationPath $ZipPath -CompressionLevel Optimal
Write-Host "Windows portable package: $ZipPath"
