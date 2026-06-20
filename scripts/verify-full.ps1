[CmdletBinding()]
param(
    [switch]$SkipPreCommit
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

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$python = Get-ProjectPython

Invoke-Step "Bootstrap prerequisite check" {
    & (Join-Path $PSScriptRoot "bootstrap.ps1")
}

Invoke-Step "Full pytest suite" {
    & $python -m pytest
}

if (-not $SkipPreCommit) {
    Invoke-Step "Pre-commit hooks" {
        & $python -m pre_commit run --all-files
    }
}
