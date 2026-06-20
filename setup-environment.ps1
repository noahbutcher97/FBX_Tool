# FBX_Tool Environment Setup Script
#
# Fully automated, idempotent setup. Detects what's missing and offers to install it.
# Safe to re-run: skips steps that are already done.
#
# Usage:
#   .\setup-environment.ps1                  # Interactive (prompts for big installs)
#   .\setup-environment.ps1 -NonInteractive  # Auto-yes all prompts (CI mode)
#   .\setup-environment.ps1 -SkipPython      # Skip Python check (assume present)
#   .\setup-environment.ps1 -SkipVS          # Skip Visual Studio check
#   .\setup-environment.ps1 -SkipFBX         # Skip FBX SDK check
#   .\setup-environment.ps1 -RecreateVenv    # Force venv recreation

param(
    [switch]$NonInteractive,
    [switch]$SkipPython,
    [switch]$SkipVS,
    [switch]$SkipFBX,
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"

# ============================================================================
# Configuration
# ============================================================================
$PythonVersion       = "3.10.11"
$PythonInstallerUrl  = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$VSBuildToolsUrl     = "https://aka.ms/vs/17/release/vs_buildtools.exe"
$FBXSDKVersion       = "2020.3.7"
$FBXInstallerUrl     = "https://damassets.autodesk.net/content/dam/autodesk/www/adn/fbx/2020-3-7/fbx202037_fbxpythonsdk_win.exe"
$FBXInstallerName    = "fbx202037_fbxpythonsdk_win.exe"
$FBXInstallRoot      = "C:\Program Files\Autodesk\FBX\FBX Python SDK\$FBXSDKVersion"
$VenvName            = ".fbxenv"

# ============================================================================
# Output Helpers
# ============================================================================
function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}
function Write-StepHeader($n, $total, $text) {
    Write-Host "`n[Step $n/$total] $text" -ForegroundColor Yellow
}
function Write-OK($text)   { Write-Host "  SUCCESS: $text" -ForegroundColor Green }
function Write-Info($text) { Write-Host "  -> $text" -ForegroundColor Cyan }
function Write-Warn($text) { Write-Host "  ! $text" -ForegroundColor Yellow }
function Write-Err($text)  { Write-Host "  ERROR: $text" -ForegroundColor Red }

function Confirm-Action {
    param([string]$Prompt, [string]$Default = 'y')
    if ($NonInteractive) { return $true }
    $hint = if ($Default -eq 'y') { '[Y/n]' } else { '[y/N]' }
    $response = Read-Host "  $Prompt $hint"
    if ([string]::IsNullOrWhiteSpace($response)) { $response = $Default }
    return $response.ToLower().StartsWith('y')
}

# ============================================================================
# Environment Helpers
# ============================================================================
function Refresh-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ============================================================================
# Python 3.10 Detection / Install
# ============================================================================
# Returns absolute path to a Python 3.10 executable, or $null if none found.
function Find-Python310 {
    # Prefer the py launcher
    try {
        $output = & py -3.10 -c "import sys; print(sys.executable)" 2>&1
        if ($LASTEXITCODE -eq 0 -and "$output" -match "python\.exe") {
            return ("$output").Trim()
        }
    } catch {}

    # Fall back to common install paths
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python310\python.exe",
        "C:\Program Files\Python310\python.exe",
        "C:\Program Files (x86)\Python310-32\python.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $ver = & $c --version 2>&1
            if ($ver -match "Python 3\.10") { return $c }
        }
    }
    return $null
}

function Install-Python310 {
    Write-Info "Downloading Python $PythonVersion installer (~30 MB)..."
    $installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $installer -UseBasicParsing
    Write-OK "Downloaded to $installer"

    Write-Info "Running silent per-user install (no admin required)..."
    # PrependPath=1 adds Python to PATH; Include_launcher=1 installs py.exe
    $installArgs = @(
        '/quiet',
        'InstallAllUsers=0',
        'PrependPath=1',
        'Include_launcher=1',
        'Include_test=0',
        'SimpleInstall=1'
    )
    $proc = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Python installer exited with code $($proc.ExitCode)"
    }

    Refresh-PathFromRegistry
    Write-OK "Python $PythonVersion installed"
}

# ============================================================================
# Visual Studio Build Tools Detection / Install
# ============================================================================
function Find-VSBuildTools {
    $vsPaths = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC"
    )
    foreach ($p in $vsPaths) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Install-VSBuildTools {
    Write-Info "Downloading VS Build Tools bootstrapper (~5 MB)..."
    $bootstrapper = Join-Path $env:TEMP "vs_buildtools.exe"
    Invoke-WebRequest -Uri $VSBuildToolsUrl -OutFile $bootstrapper -UseBasicParsing
    Write-OK "Downloaded to $bootstrapper"

    Write-Warn "Installing VS Build Tools - 6+ GB, takes 10-30 minutes."
    $installArgs = @(
        '--quiet', '--wait', '--norestart', '--nocache',
        '--add', 'Microsoft.VisualStudio.Workload.VCTools',
        '--add', 'Microsoft.VisualStudio.Component.Windows11SDK.22621',
        '--includeRecommended'
    )

    if (-not (Test-IsAdmin)) {
        Write-Info "Elevation required for VS install - UAC prompt will appear."
        $proc = Start-Process -FilePath $bootstrapper -ArgumentList $installArgs -Wait -PassThru -Verb RunAs
    } else {
        $proc = Start-Process -FilePath $bootstrapper -ArgumentList $installArgs -Wait -PassThru
    }

    # 0 = success, 3010 = success-but-reboot-required
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "VS Build Tools installer exited with code $($proc.ExitCode)"
    }
    if ($proc.ExitCode -eq 3010) {
        Write-Warn "Install succeeded but a reboot is required to complete it."
    }
    Write-OK "VS Build Tools installed"
}

# ============================================================================
# FBX SDK Detection / Install
# ============================================================================
# Returns path to the FBX wheel on disk, or $null if not found.
function Find-FBXWheel {
    if (-not (Test-Path $FBXInstallRoot)) { return $null }
    $wheel = Get-ChildItem -Path $FBXInstallRoot -Filter "fbx-*-cp310-*.whl" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($wheel) { return $wheel.FullName }
    return $null
}

function Install-FBXSDK {
    # Try direct URL download first
    $installer = Join-Path $env:USERPROFILE "Downloads\$FBXInstallerName"
    Write-Info "Attempting direct download of FBX SDK installer..."
    $downloaded = $false
    try {
        Invoke-WebRequest -Uri $FBXInstallerUrl -OutFile $installer -UseBasicParsing -TimeoutSec 60
        if ((Test-Path $installer) -and (Get-Item $installer).Length -gt 1MB) {
            Write-OK "Downloaded to $installer"
            $downloaded = $true
        } else {
            Write-Warn "Downloaded file looks too small - probably an error page."
            Remove-Item $installer -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Warn "Direct download failed: $($_.Exception.Message)"
    }

    # Fall back to browser-based download with auto-detection
    if (-not $downloaded) {
        Write-Info "Opening Autodesk FBX SDK download page in your browser..."
        Start-Process "https://aps.autodesk.com/developer/overview/fbx-sdk"
        Write-Host ""
        Write-Host "  MANUAL STEP:" -ForegroundColor Yellow
        Write-Host "    1. Sign in (free Autodesk account)" -ForegroundColor White
        Write-Host "    2. Download 'FBX Python SDK $FBXSDKVersion for Windows'" -ForegroundColor White
        Write-Host "    3. Save it to your Downloads folder" -ForegroundColor White
        Write-Host ""
        Read-Host "  Press ENTER once the installer is downloaded"

        if (-not (Test-Path $installer)) {
            $found = Get-ChildItem -Path "$env:USERPROFILE\Downloads" `
                                   -Filter "fbx*pythonsdk*.exe" `
                                   -ErrorAction SilentlyContinue |
                     Select-Object -First 1
            if ($found) {
                $installer = $found.FullName
                Write-OK "Found installer at $installer"
            } else {
                throw "Could not find FBX SDK installer in $env:USERPROFILE\Downloads"
            }
        }
    }

    Write-Info "Running FBX SDK installer..."
    Write-Warn "If a UI appears, accept the EULA and complete the install."
    $proc = Start-Process -FilePath $installer -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Warn "Installer exited with code $($proc.ExitCode); checking for wheel anyway..."
    }

    $wheel = Find-FBXWheel
    if (-not $wheel) {
        throw "FBX SDK installer ran but no wheel found at $FBXInstallRoot"
    }
    Write-OK "FBX SDK installed: $wheel"
    return $wheel
}

# ============================================================================
# Venv Helpers
# ============================================================================
function Test-VenvCompatible {
    $venvPython = ".\$VenvName\Scripts\python.exe"
    if (-not (Test-Path $venvPython)) { return $false }
    try {
        $ver = & $venvPython --version 2>&1
        return ("$ver" -match "Python 3\.10")
    } catch {
        return $false
    }
}

# ============================================================================
# Main Flow
# ============================================================================
Write-Header "FBX_Tool Environment Setup"

if ($NonInteractive) {
    Write-Info "Non-interactive mode: all prompts auto-accepted."
}

# ---------- Step 1: Python 3.10 ----------
Write-StepHeader 1 6 "Checking Python 3.10 installation..."
$pythonExe = Find-Python310
if ($pythonExe) {
    Write-OK "Python 3.10 found: $pythonExe"
} elseif ($SkipPython) {
    Write-Warn "Python 3.10 not found, but -SkipPython was specified. Continuing anyway."
} else {
    Write-Warn "Python 3.10 not found."
    if (Confirm-Action "Auto-install Python $PythonVersion (per-user, ~30 MB)?") {
        Install-Python310
        $pythonExe = Find-Python310
        if (-not $pythonExe) {
            throw "Python install completed but py launcher still can't find it. Try restarting your shell."
        }
        Write-OK "Python 3.10 ready: $pythonExe"
    } else {
        throw "Python 3.10 is required. Aborting."
    }
}

# ---------- Step 2: Visual Studio Build Tools ----------
Write-StepHeader 2 6 "Checking Visual Studio C++ Build Tools..."
$vsPath = Find-VSBuildTools
if ($vsPath) {
    Write-OK "VS C++ Build Tools found: $vsPath"
} elseif ($SkipVS) {
    Write-Warn "VS Build Tools not found, but -SkipVS was specified."
} else {
    Write-Warn "VS Build Tools not found."
    Write-Info "Note: the FBX SDK wheel ships precompiled, so this is only needed"
    Write-Info "      if you plan to build other native extensions from source."
    if (Confirm-Action "Auto-install VS Build Tools (~6 GB, 10-30 min)?" 'n') {
        Install-VSBuildTools
    } else {
        Write-Info "Skipping VS Build Tools install."
    }
}

# ---------- Step 3: FBX SDK ----------
Write-StepHeader 3 6 "Checking Autodesk FBX Python SDK..."
$fbxWheel = $null
if ($SkipFBX) {
    Write-Warn "FBX SDK check skipped via -SkipFBX flag."
} else {
    $fbxWheel = Find-FBXWheel
    if ($fbxWheel) {
        Write-OK "FBX SDK wheel found: $fbxWheel"
    } else {
        Write-Warn "FBX SDK not found on disk."
        if (Confirm-Action "Download and install FBX Python SDK $FBXSDKVersion?") {
            $fbxWheel = Install-FBXSDK
        } else {
            throw "FBX SDK is required. Aborting."
        }
    }
}

# ---------- Step 4: Virtual Environment ----------
Write-StepHeader 4 6 "Setting up virtual environment..."
$venvCompatible = Test-VenvCompatible
if ($venvCompatible -and -not $RecreateVenv) {
    Write-OK "Existing $VenvName is Python 3.10 compatible - reusing"
} else {
    if (Test-Path $VenvName) {
        if ($RecreateVenv) {
            Write-Info "Removing existing $VenvName (-RecreateVenv flag)..."
        } else {
            Write-Info "Existing $VenvName is incompatible - recreating..."
        }
        Remove-Item -Recurse -Force $VenvName
    }
    Write-Info "Creating venv with Python 3.10..."
    & $pythonExe -m venv $VenvName
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
    Write-OK "Created $VenvName"
}
$venvPython = ".\$VenvName\Scripts\python.exe"

# ---------- Step 5: Dependencies ----------
Write-StepHeader 5 6 "Installing dependencies..."
Write-Info "Upgrading pip..."
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

if (Test-Path "requirements.txt") {
    Write-Info "Installing requirements.txt..."
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Requirements install failed" }
} else {
    Write-Warn "No requirements.txt found - skipping"
}

if ($fbxWheel) {
    Write-Info "Installing FBX SDK wheel into venv..."
    & $venvPython -m pip install --force-reinstall $fbxWheel
    if ($LASTEXITCODE -ne 0) { throw "FBX wheel install failed" }
}
Write-OK "All dependencies installed"

# ---------- Step 6: Verification ----------
Write-StepHeader 6 6 "Verifying installation..."
Write-Info "Python version:"
$pyVer = & $venvPython --version
Write-Host "    $pyVer"

Write-Info "Functional FBX SDK probe (Create + Destroy)..."
# Use single-quoted here-string so PowerShell doesn't interpret $ signs.
# Closing '@ MUST be at column 0.
$probe = @'
import sys
try:
    import fbx
    m = fbx.FbxManager.Create()
    if m is None:
        print('FAIL: FbxManager.Create() returned None')
        sys.exit(1)
    m.Destroy()
    print('PASS: fbx loaded; FbxManager Create/Destroy succeeded')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
'@
$probeFile = Join-Path $env:TEMP "fbx_probe.py"
$probe | Set-Content -Path $probeFile -Encoding UTF8
& $venvPython $probeFile
$fbxOk = ($LASTEXITCODE -eq 0)
Remove-Item $probeFile -ErrorAction SilentlyContinue

if ($fbxOk) {
    Write-OK "FBX SDK functional check passed"
} else {
    Write-Err "FBX SDK functional check failed"
}

# ---------- Final Summary ----------
Write-Header "Setup Complete"

Write-Host "Quick start:" -ForegroundColor White
Write-Host "  Launch the GUI:" -ForegroundColor Yellow
Write-Host "    .\launch.bat" -ForegroundColor Cyan
Write-Host "      (or: .\$VenvName\Scripts\Activate.ps1; python -m fbx_tool)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Run tests:" -ForegroundColor Yellow
Write-Host "    .\$VenvName\Scripts\Activate.ps1; pytest" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Re-run setup at any time (idempotent):" -ForegroundColor Yellow
Write-Host "    .\setup.bat" -ForegroundColor Cyan
Write-Host ""
