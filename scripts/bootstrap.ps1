[CmdletBinding()]
param(
    [switch]$Install
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ProjectPython {
    $venvPython = Join-Path $PSScriptRoot "..\.fbxenv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }
    return "python"
}

$python = Get-ProjectPython
$version = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run Python. Install Python 3.10.x and retry."
}
if ($version -ne "3.10") {
    throw "FBX Tool requires Python 3.10.x because of Autodesk FBX SDK compatibility. Found Python $version."
}

Write-Host "Python check passed: $version"

& $python -c "import fbx" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Autodesk FBX Python SDK import check passed."
}
else {
    Write-Warning "Autodesk FBX Python SDK is not importable. Install SDK 2020.x before running real FBX workflows."
    $global:LASTEXITCODE = 0
}

if ($Install) {
    $venvDir = Join-Path $PSScriptRoot "..\.fbxenv"
    if (-not (Test-Path -LiteralPath $venvDir)) {
        & python -m venv $venvDir --system-site-packages
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }

    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    & $venvPython -m pip install -r requirements-dev.txt -r requirements-test.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}
