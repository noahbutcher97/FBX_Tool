# FBX_Tool Environment Setup Script
# This script helps set up the development environment for FBX_Tool

param(
    [switch]$SkipPython,
    [switch]$SkipVS,
    [switch]$SkipFBX
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "FBX_Tool Environment Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Function to check if running as admin
function Test-Admin {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Step 1: Check/Install Python 3.10.11
Write-Host "[Step 1/6] Checking Python 3.10 installation..." -ForegroundColor Yellow

$python310Installed = $false
try {
    $pyVersion = & py -3.10 --version 2>&1
    if ($pyVersion -match "Python 3\.10") {
        Write-Host "  SUCCESS: Python 3.10 is installed: $pyVersion" -ForegroundColor Green
        $python310Installed = $true
    }
} catch {
    Write-Host "  Python 3.10 not found" -ForegroundColor Red
}

if (-not $python310Installed -and -not $SkipPython) {
    Write-Host "`n  MANUAL ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "  1. Download Python 3.10.11 from:" -ForegroundColor White
    Write-Host "     https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -ForegroundColor Cyan
    Write-Host "  2. Run the installer" -ForegroundColor White
    Write-Host "  3. IMPORTANT: Check 'Add Python 3.10 to PATH' during installation" -ForegroundColor Yellow
    Write-Host "  4. Choose 'Install Now' or customize installation location" -ForegroundColor White
    Write-Host "  5. After installation, run this script again`n" -ForegroundColor White

    $download = Read-Host "  Would you like to download Python 3.10.11 now? (y/n)"
    if ($download -eq 'y') {
        $pythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
        $pythonInstaller = "$env:TEMP\python-3.10.11-amd64.exe"

        Write-Host "  Downloading Python 3.10.11..." -ForegroundColor Cyan
        try {
            Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
            Write-Host "  Downloaded to: $pythonInstaller" -ForegroundColor Green
            Write-Host "  Starting installer..." -ForegroundColor Cyan
            Start-Process $pythonInstaller -Wait
            Write-Host "  Installation complete. Please run this script again." -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: Failed to download Python installer" -ForegroundColor Red
            Write-Host "  Please download manually from the link above" -ForegroundColor Yellow
        }
    }
    exit
}

# Step 2: Check Visual Studio C++ Build Tools
Write-Host "`n[Step 2/6] Checking Visual Studio C++ Build Tools..." -ForegroundColor Yellow

$vsInstalled = $false
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
)

foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        Write-Host "  SUCCESS: Visual Studio C++ Build Tools found at $path" -ForegroundColor Green
        $vsInstalled = $true
        break
    }
}

if (-not $vsInstalled -and -not $SkipVS) {
    Write-Host "  Visual Studio C++ Build Tools not found" -ForegroundColor Red
    Write-Host "`n  MANUAL ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "  1. Download Visual Studio Build Tools from:" -ForegroundColor White
    Write-Host "     https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "  2. Run the installer" -ForegroundColor White
    Write-Host "  3. Select 'Desktop development with C++' workload" -ForegroundColor White
    Write-Host "  4. Install and restart your computer if prompted" -ForegroundColor White
    Write-Host "  5. Run this script again`n" -ForegroundColor White

    $openUrl = Read-Host "  Would you like to open the download page? (y/n)"
    if ($openUrl -eq 'y') {
        Start-Process "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
    }
    exit
}

# Step 3: Check FBX SDK
Write-Host "`n[Step 3/6] Checking Autodesk FBX Python SDK..." -ForegroundColor Yellow

$fbxInstalled = $false
try {
    & py -3.10 -c "import fbx; print('FBX SDK version:', fbx.FbxManager.GetVersion())" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $fbxVersion = & py -3.10 -c "import fbx; print(fbx.FbxManager.GetVersion())"
        Write-Host "  SUCCESS: FBX SDK is installed (version: $fbxVersion)" -ForegroundColor Green
        $fbxInstalled = $true
    }
} catch {
    Write-Host "  FBX SDK not found" -ForegroundColor Red
}

if (-not $fbxInstalled -and -not $SkipFBX) {
    Write-Host "  FBX SDK not installed or not accessible" -ForegroundColor Red
    Write-Host "`n  MANUAL ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "  1. Download Autodesk FBX Python SDK 2020.3.7 from:" -ForegroundColor White
    Write-Host "     https://aps.autodesk.com/developer/overview/fbx-sdk" -ForegroundColor Cyan
    Write-Host "  2. Create an Autodesk account (free)" -ForegroundColor White
    Write-Host "  3. Download: 'FBX Python SDK 2020.3.7 for Windows'" -ForegroundColor White
    Write-Host "  4. Run the installer and select Python 3.10" -ForegroundColor White
    Write-Host "  5. Run this script again`n" -ForegroundColor White

    Write-Host "  NOTE: You may need to check the INSTALL.md file for detailed instructions" -ForegroundColor Yellow

    $openUrl = Read-Host "  Would you like to open the download page? (y/n)"
    if ($openUrl -eq 'y') {
        Start-Process "https://aps.autodesk.com/developer/overview/fbx-sdk"
    }
    exit
}

# Step 4: Create virtual environment
Write-Host "`n[Step 4/6] Creating virtual environment..." -ForegroundColor Yellow

if (Test-Path ".fbxenv") {
    Write-Host "  Virtual environment '.fbxenv' already exists" -ForegroundColor Yellow
    $recreate = Read-Host "  Do you want to recreate it? (y/n)"
    if ($recreate -eq 'y') {
        Write-Host "  Removing existing virtual environment..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force .fbxenv
    } else {
        Write-Host "  Keeping existing virtual environment" -ForegroundColor Green
        $skipVenvCreation = $true
    }
}

if (-not $skipVenvCreation) {
    Write-Host "  Creating virtual environment with Python 3.10..." -ForegroundColor Cyan
    & py -3.10 -m venv .fbxenv --system-site-packages

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  SUCCESS: Virtual environment created at .fbxenv\" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit
    }
}

# Step 5: Activate and install dependencies
Write-Host "`n[Step 5/6] Installing dependencies..." -ForegroundColor Yellow

Write-Host "  Activating virtual environment..." -ForegroundColor Cyan
& .\.fbxenv\Scripts\Activate.ps1

Write-Host "  Upgrading pip..." -ForegroundColor Cyan
& python -m pip install --upgrade pip

Write-Host "  Installing requirements..." -ForegroundColor Cyan
& pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "  SUCCESS: Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Failed to install dependencies" -ForegroundColor Red
    exit
}

# Step 6: Verify installation
Write-Host "`n[Step 6/6] Verifying installation..." -ForegroundColor Yellow

Write-Host "  Checking Python version..." -ForegroundColor Cyan
$pythonVer = & python --version
Write-Host "    $pythonVer" -ForegroundColor White

Write-Host "  Checking installed packages..." -ForegroundColor Cyan
& pip list | Select-String -Pattern "numpy|PyQt6|pyinstaller"

Write-Host "  Checking FBX SDK import..." -ForegroundColor Cyan
$fbxCheck = & python -c "import fbx; print('  FBX SDK import: SUCCESS'); print('  Version:', fbx.FbxManager.GetVersion())" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $fbxCheck -ForegroundColor Green
} else {
    Write-Host "  WARNING: FBX SDK import failed" -ForegroundColor Yellow
    Write-Host "  You may need to install the FBX SDK manually" -ForegroundColor Yellow
}

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "To start working on FBX_Tool:" -ForegroundColor White
Write-Host "  1. Activate the virtual environment:" -ForegroundColor Yellow
Write-Host "     .\.fbxenv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  2. Run the GUI:" -ForegroundColor Yellow
Write-Host "     python main_gui.py" -ForegroundColor Cyan
Write-Host "  3. Or run CLI:" -ForegroundColor Yellow
Write-Host "     python main.py path\to\animation.fbx`n" -ForegroundColor Cyan

Write-Host "To deactivate the virtual environment:" -ForegroundColor White
Write-Host "  deactivate`n" -ForegroundColor Cyan
