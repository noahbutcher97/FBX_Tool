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
2. Accept the default install path: `C:\Program Files\Autodesk\FBX\FBX Python SDK\2020.3.7`
3. The installer extracts a pip-installable wheel into that directory. It does **not** auto-install into any Python — that's a separate step (see Step 6 for the venv install).

#### Install the Wheel
Since 2020.3, the FBX Python SDK ships as a standard CPython 3.10 wheel. Install it with pip:

```powershell
py -3.10 -m pip install "C:\Program Files\Autodesk\FBX\FBX Python SDK\2020.3.7\fbx-2020.3.7-cp310-none-win_amd64.whl"
```

> **Recommended:** Skip this step here and install the wheel into your project's virtual environment in Step 6 instead. That keeps the FBX SDK isolated from your system Python.

#### Verify Installation
```powershell
py -3.10 -c "import fbx; m = fbx.FbxManager.Create(); print('OK' if m else 'FAIL'); m.Destroy()"
# Should output: OK
```

> **Note:** The older "manually copy `fbx.pyd` and DLLs into site-packages" workflow applied to FBX SDK ≤ 2020.2 which shipped raw `.pyd` files. Since 2020.3 the wheel format handles everything (including dependent DLLs) automatically — no manual file copying needed.
>
> Note also that `fbx.FbxManager.GetVersion()` is **not** a valid call in 2020.3.7 — the SDK doesn't expose that as either a class method or an instance method. The Create/Destroy probe above is a real functional test instead.

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

## Step 6: Install FBX SDK Into the Virtual Environment

A fresh venv won't have the FBX SDK — install the wheel that the Step 3 installer extracted:

```powershell
(.fbxenv) pip install "C:\Program Files\Autodesk\FBX\FBX Python SDK\2020.3.7\fbx-2020.3.7-cp310-none-win_amd64.whl"
```

### Verify
```powershell
(.fbxenv) python -c "import fbx; m = fbx.FbxManager.Create(); print('OK' if m else 'FAIL'); m.Destroy()"
# Should output: OK
```

**If this fails:**

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'fbx'` | Wheel install ran into the wrong Python | Make sure `(.fbxenv)` is in your prompt, then re-run the `pip install` above |
| `ERROR: ... is not a supported wheel on this platform` | Venv's Python isn't 3.10 (probably 3.11+) | Recreate venv with `py -3.10 -m venv .fbxenv`, then re-install requirements + wheel |
| `ImportError: DLL load failed` | Missing Visual C++ runtime | Install [VC++ Redistributable 2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) |

> **Note:** The older "copy `fbx.pyd` into venv site-packages" or "use `--system-site-packages`" workarounds are no longer needed. `pip install <wheel>` is the entire install — the wheel bundles the .pyd, FbxCommon.py, sip.pyd, and all dependent DLLs.

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

### Distribution Strategy for Open Source

This project follows **Strategy C: Documentation-Based Distribution** for the FBX SDK dependency.

**Why we don't bundle the FBX SDK:**
1. **Legal compliance** - Autodesk's FBX SDK license requires users to accept their EULA. Bundling the SDK would bypass this requirement.
2. **Updates** - Users get the latest FBX SDK version directly from Autodesk, including security patches.
3. **Open source best practice** - Similar to how CUDA-dependent projects don't bundle NVIDIA libraries.

**What this means for users:**
- The PyInstaller executable includes all Python dependencies (NumPy, PyQt6, OpenGL, etc.)
- Users must install FBX SDK separately (one-time setup via Steps 1-3)
- The executable will show a helpful error message if FBX SDK is not found

### Building with PyInstaller

```powershell
# Use the pre-configured spec file (recommended)
(.fbxenv) pyinstaller FBX_Tool.spec

# Or build manually (basic)
(.fbxenv) pyinstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool/gui/main_window.py
```

**Output:** `dist\FBX_Tool.exe`

**Important:** The executable does NOT include FBX SDK. Users must have it installed separately.

### What Gets Bundled

✅ **Included in executable:**
- Python interpreter
- NumPy, PyQt6, PyOpenGL
- All fbx_tool analysis modules
- GUI and visualization code

❌ **NOT included (user must install):**
- Autodesk FBX Python SDK binaries (`fbx.pyd`, DLLs)
- Visual C++ runtime (if not already on system)

### Distribution Checklist

When releasing a new version:

1. **Build the executable:**
   ```powershell
   pyinstaller FBX_Tool.spec --clean
   ```

2. **Test on clean machine:**
   - Copy `dist\FBX_Tool.exe` to a machine without Python
   - Verify FBX SDK error message is helpful
   - Test with FBX SDK installed

3. **Create GitHub release:**
   - Upload `FBX_Tool.exe`
   - Include installation instructions (link to this document)
   - Note FBX SDK requirement prominently

4. **Release notes template:**
   ```markdown
   ## Download

   Download `FBX_Tool.exe` from the Assets section below.

   ## ⚠️ Important: FBX SDK Required

   This executable requires the Autodesk FBX Python SDK to be installed separately.

   **First-time users:**
   1. Install Python 3.10.x
   2. Install Autodesk FBX Python SDK 2020.3.7
   3. Run FBX_Tool.exe

   See [Installation Guide](docs/INSTALL.md) for detailed instructions.
   ```

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
