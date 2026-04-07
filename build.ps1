param(
    [switch]$CleanOutput
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pyinstallerExe = Join-Path $projectRoot '.venv\Scripts\pyinstaller.exe'
$specFile = Join-Path $projectRoot 'cvrecord.spec'
$entryFile = Join-Path $projectRoot 'src\main.py'
$distExe = Join-Path $projectRoot 'dist\CVRecord.exe'
$distIconsDir = Join-Path $projectRoot 'dist\icons'
$distRecordsDir = Join-Path $projectRoot 'dist\records'

if (-not (Test-Path $pyinstallerExe)) {
    throw "PyInstaller not found at $pyinstallerExe. Activate or create the .venv first."
}

if (-not (Test-Path $specFile)) {
    throw "Spec file not found: $specFile"
}

if (-not (Test-Path $entryFile)) {
    throw "Entry file not found: $entryFile"
}

if ($CleanOutput) {
    Write-Host 'Removing existing build and dist directories...'
    if (Test-Path (Join-Path $projectRoot 'build')) {
        Remove-Item -Recurse -Force (Join-Path $projectRoot 'build')
    }
    if (Test-Path (Join-Path $projectRoot 'dist')) {
        Remove-Item -Recurse -Force (Join-Path $projectRoot 'dist')
    }
}

Write-Host 'Running PyInstaller build...'
& $pyinstallerExe --noconfirm --clean $specFile

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $distExe)) {
    throw "Build completed but executable was not found: $distExe"
}

New-Item -ItemType Directory -Force $distIconsDir | Out-Null
New-Item -ItemType Directory -Force $distRecordsDir | Out-Null

Write-Host "Build succeeded: $distExe"
