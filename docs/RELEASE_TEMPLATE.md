# Release Template for FBX Tool

Use this template when creating new GitHub releases.

---

## Release vX.Y.Z

**Release Date:** YYYY-MM-DD

### Download

Download `FBX_Tool.exe` from the **Assets** section below.

### ⚠️ Important: Prerequisites Required

This executable requires **two prerequisites** to be installed on your system:

1. **Python 3.10.x** - [Download here](https://www.python.org/downloads/release/python-31011/)
2. **Autodesk FBX Python SDK 2020.3.7** - [Download here](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3-7)

**Why aren't these bundled?**
- Autodesk's license requires users to accept their EULA
- Users get the latest FBX SDK version with security updates
- Standard practice for open-source projects with proprietary dependencies

📖 **[Complete Installation Guide](docs/INSTALL.md)** - Step-by-step instructions

### First-Time Setup (5 minutes)

1. Install Python 3.10.x (check "Add Python to PATH" during installation)
2. Install Autodesk FBX Python SDK 2020.3.7 (free, requires Autodesk account)
3. Download and run `FBX_Tool.exe`

### What's New in vX.Y.Z

**✨ New Features:**
- Feature 1 description
- Feature 2 description

**🐛 Bug Fixes:**
- Fix 1 description
- Fix 2 description

**📚 Documentation:**
- Documentation update 1
- Documentation update 2

**🔧 Under the Hood:**
- Internal improvement 1
- Internal improvement 2

### Upgrade Notes

If you're upgrading from a previous version:
- [Any breaking changes or migration notes]
- [New requirements or deprecated features]

### Known Issues

- Issue 1 description and workaround
- Issue 2 description and workaround

### Full Changelog

See [CHANGELOG.md](docs/CHANGELOG.md) for complete version history.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'fbx'"

**Solution:** FBX SDK is not installed or not found.

1. Install Autodesk FBX Python SDK 2020.3.7
2. Verify installation: Open Command Prompt and run:
   ```
   python -c "import fbx; print(fbx.FbxManager.GetVersion())"
   ```
3. If the above fails, see [Installation Guide](docs/INSTALL.md#step-3-install-autodesk-fbx-python-sdk)

### "ImportError: DLL load failed"

**Solution:** Missing Visual C++ runtime.

Download and install [Visual C++ Redistributable 2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### Executable won't start

**Solution:** Check Python version.

1. Open Command Prompt
2. Run: `python --version`
3. Must be **Python 3.10.x** (not 3.11 or later)
4. If wrong version, uninstall and install Python 3.10.11

### More Help

- 📖 [Full Documentation](docs/)
- 🐛 [Report a Bug](https://github.com/noahbutcher97/FBX_Tool/issues)
- 💬 [Ask a Question](https://github.com/noahbutcher97/FBX_Tool/discussions)

---

## System Requirements

**Minimum:**
- OS: Windows 10 64-bit
- RAM: 4 GB
- Storage: 500 MB (+ 200 MB for FBX SDK)
- Python: 3.10.x

**Recommended:**
- OS: Windows 11 64-bit
- RAM: 8 GB+
- Storage: 1 GB
- GPU: Any modern GPU for smooth 3D visualization

---

## For Developers

Want to build from source or contribute? See:
- [Installation Guide](docs/INSTALL.md) - Development setup
- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Development Documentation](docs/) - Architecture and patterns

---

## Checksums (SHA-256)

```
FBX_Tool.exe: [INSERT SHA-256 HASH HERE]
```

To verify:
```powershell
# Windows PowerShell
Get-FileHash FBX_Tool.exe -Algorithm SHA256
```

---

**⭐ Star this repo if you find it useful!**

**📣 Share:** Help others discover this tool by sharing on social media or with colleagues.
