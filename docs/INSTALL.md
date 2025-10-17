# Installation Guide

## Prerequisites

### Required Software
- **Python 3.10.x** (specifically, **NOT 3.11+**)
- **Visual Studio 2022** with C++ build tools
- **Autodesk FBX Python SDK 2020.x**

### Why Python 3.10?
The Autodesk FBX Python SDK binaries are compiled for **Python 3.10 only**. Using Python 3.11 or later will result in import errors.

---

## Step 1: Install Python 3.10

### Windows
1. Download Python 3.10.11 from [python.org](https://www.python.org/downloads/release/python-31011/)
2. Run installer and check **"Add Python to PATH"**
3. Verify installation:
   ```powershell
   python --version
   # Should output: Python 3.10.11
   ```

### Verify Python Version
```powershell
python --version
# MUST be 3.10.x
```

---

## Step 2: Install Visual Studio 2022

The FBX SDK requires Visual C++ runtime libraries.

### Option A: Full Visual Studio 2022
Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/)

**Required components:**
- Desktop development with C++
- Windows 10 SDK

### Option B: Build Tools Only (Smaller)
Download [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)

**Select:**
- C++ build tools
- Windows 10 SDK

---

## Step 3: Install Autodesk FBX Python SDK

### Download
1. Go to [Autodesk FBX SDK Downloads](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3-7)
2. Download **FBX Python SDK 2020.3.7** for your platform:
   - Windows: `fbx202037_fbxpythonsdk_win.exe`
   - Choose **Python 3.10** version

### Installation Steps

#### Windows
1. Run the installer: `fbx202037_fbxpythonsdk_win.exe`
2. Choose installation directory (default: `C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.7`)
3. Installer will place `fbx.pyd` in your Python site-packages

#### Verify Installation
```powershell
python -c "import fbx; print(fbx.FbxManager.GetVersion())"
# Should output: 2020.3.7 or similar
```

#### Manual Installation (if needed)
If the installer doesn't automatically add to Python:

1. Locate the FBX SDK installation:
   ```
   C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.7\lib\Python310_x64\
   ```

2. Copy files to your Python site-packages:
   ```powershell
   # Find your site-packages directory
   python -c "import site; print(site.getsitepackages()[0])"

   # Copy fbx.pyd and fbx.cp310-win_amd64.pyd
   copy "C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.7\lib\Python310_x64\fbx*.pyd" "C:\Path\To\site-packages\"
   ```

3. Copy required DLLs:
   ```powershell
   # Copy all DLLs from FBX SDK bin directory
   copy "C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.7\lib\vs2022\x64\release\*.dll" "C:\Path\To\site-packages\"
   ```

---

## Step 4: Create Virtual Environment

```powershell
cd D:\Scripts\Python\Projects\FBX_Tool

# Create virtual environment
python -m venv .fbxenv

# Activate
.fbxenv\Scripts\activate

# Verify Python version in venv
python --version
# Must be 3.10.x
```

---

## Step 5: Install Project Dependencies

```powershell
# Make sure virtual environment is activated
(.fbxenv) python -m pip install --upgrade pip

# Install dependencies
(.fbxenv) pip install -r requirements.txt

# Verify installations
(.fbxenv) pip list
```

Expected packages:
- numpy (~1.24.x)
- PyQt6 (~6.5.x)
- pyinstaller (~6.x)

---

## Step 6: Test FBX SDK in Virtual Environment

```powershell
(.fbxenv) python -c "import fbx; print('FBX SDK Version:', fbx.FbxManager.GetVersion())"
```

**If this fails:**

The virtual environment doesn't include the FBX SDK by default. You need to:

### Option A: Install FBX SDK into venv
```powershell
# Locate global site-packages fbx files
python -c "import site; print(site.getsitepackages()[0])"

# Copy to venv site-packages
copy "C:\Path\To\global-site-packages\fbx*.pyd" ".fbxenv\Lib\site-packages\"
copy "C:\Path\To\global-site-packages\*.dll" ".fbxenv\Lib\site-packages\"
```

### Option B: Use system site-packages
Recreate venv with system packages:
```powershell
deactivate
rmdir /s /q .fbxenv
python -m venv .fbxenv --system-site-packages
.fbxenv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 7: Test the Application

```powershell
# Test GUI
(.fbxenv) python fbx_tool/gui/main_window.py

# Test CLI
(.fbxenv) python examples/run_analysis.py path/to/animation.fbx
```

---

## Step 8: Build Executable (Optional)

```powershell
(.fbxenv) python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool/gui/main_window.py
```

**Output:** `dist\FBX_Tool.exe`

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'fbx'`

**Solution 1:** FBX SDK not installed
- Follow Step 3 again
- Verify with: `python -c "import fbx"`

**Solution 2:** Virtual environment can't find FBX SDK
- Use `--system-site-packages` when creating venv
- Or manually copy FBX files to venv

### Issue: `ImportError: DLL load failed`

**Solution:** Missing Visual C++ runtime
- Install Visual Studio 2022 (Step 2)
- Or install [VC++ Redistributable 2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### Issue: Python 3.11+ causes FBX import failure

**Solution:** Downgrade to Python 3.10
```powershell
# Uninstall Python 3.11+
# Install Python 3.10.11
# Recreate virtual environment
```

### Issue: PyInstaller exe crashes with `fbx not found`

**Solution:** Add hidden imports
```powershell
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --hidden-import=fbx fbx_tool/gui/main_window.py
```

Or copy FBX DLLs to `dist/`:
```powershell
copy "C:\Program Files\Autodesk\FBX\FBX SDK\2020.3.7\lib\vs2022\x64\release\*.dll" dist\
```

---

## Verification Checklist

- [ ] Python 3.10.x installed and in PATH
- [ ] Visual Studio 2022 (or Build Tools) installed
- [ ] FBX Python SDK 2020.3.7 installed
- [ ] Can run: `python -c "import fbx"`
- [ ] Virtual environment created (`.fbxenv`)
- [ ] Dependencies installed: `pip list` shows numpy, PyQt6
- [ ] FBX SDK accessible in venv: `python -c "import fbx"`
- [ ] GUI runs: `python fbx_tool/gui/main_window.py`
- [ ] Can load and analyze FBX files

---

## System Requirements

**Minimum:**
- OS: Windows 10 64-bit
- RAM: 4 GB
- Storage: 500 MB

**Recommended:**
- OS: Windows 11 64-bit
- RAM: 8 GB+
- Storage: 1 GB+
- GPU: Any modern GPU for smooth UI

---

## Quick Start Commands

```powershell
# Full setup from scratch
python --version  # Verify 3.10.x
python -m venv .fbxenv --system-site-packages
.fbxenv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python main_gui.py
```

---

## Alternative: Pre-built Executable

If you don't want to install Python and dependencies, download the pre-built executable from [Releases](https://github.com/yourusername/fbx-tool/releases).

**Note:** The executable still requires FBX SDK DLLs to be present in the same directory or in your system PATH.

---

## Support

If you encounter issues not covered here:
1. Check [Issues](https://github.com/yourusername/fbx-tool/issues)
2. Create a new issue with:
   - Python version: `python --version`
   - OS version
   - Error message
   - Steps to reproduce
