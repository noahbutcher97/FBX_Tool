[CmdletBinding()]
param(
    [string]$PytestTarget = "tests/unit",
    [switch]$IncludeStyle
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
Invoke-Step "Python 3.10 check" {
    & $python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"
}

if ($IncludeStyle) {
    Invoke-Step "Black check" {
        & $python -m black --check fbx_tool tests examples scripts --line-length 120
    }
    Invoke-Step "isort check" {
        & $python -m isort --check-only fbx_tool tests examples scripts --profile black --line-length 120
    }
}

Invoke-Step "Focused pytest" {
    & $python -m pytest $PytestTarget -n 0 --no-cov -q
}
