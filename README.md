# FBX Tool

Professional desktop application for analyzing FBX animation files with biomechanical motion processing and real-time 3D visualization.

![FBX Tool](assets/screenshot.png)

## Features

### Analysis Tools
- **Dopesheet Export** - Frame-by-frame bone rotation data in CSV format
- **Joint Analysis** - Per-joint metrics including stability, range of motion, and IK suitability
- **Chain Analysis** - Chain-level IK confidence and cross-temporal coherence
- **Gait Analysis** - Stride segmentation and gait phase detection
- **Dynamic Chain Detection** - Automatically detects bone chains from any skeleton naming convention
- **Smart Animation Stack Selection** - Automatically finds the correct animation data (handles Mixamo files)

### 3D Visualization
- **Real-time OpenGL Viewer** - Interactive 3D skeleton animation playback
- **Camera Controls** - Orbit, pan, zoom with mouse; preset views (Front/Side/Top)
- **Playback Controls** - Adjustable speed (0.1x-4.0x), frame-by-frame navigation
- **Display Options** - Toggle grid, axes, wireframe mode
- **Keyboard Shortcuts** - Full keyboard navigation and control

### Universal Compatibility
- **Any Skeleton Naming** - Works with Mixamo, Unity, Blender, and custom rigs
- **Coordinate System Detection** - Automatic Y-up/Z-up conversion
- **Multiple Animation Stacks** - Smart detection of active animation data

### Additional Features
- **Robust Error Handling** - Continues execution even if individual analyses fail
- **File-Specific Output** - Each FBX file gets its own output directory
- **GUI & CLI** - Use graphical interface or command-line for batch processing

## Requirements

- **Python 3.10.x** (required - FBX SDK does not support 3.11+)
- **Visual Studio 2022** with C++ build tools
- **Autodesk FBX Python SDK 2020.3.7**

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/noahbutcher97/FBX_Tool.git
cd FBX_Tool

# Create virtual environment (Python 3.10 required!)
python -m venv .fbxenv --system-site-packages

# Activate
.fbxenv\Scripts\activate  # Windows
source .fbxenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### GUI Mode
```bash
python fbx_tool/gui/main_window.py
```

1. Click "Choose File(s)" or drag & drop FBX files
2. Select analysis operations:
   - Export Dopesheet
   - Analyze Joints
   - Analyze Chains
   - Analyze Gait
   - Run All Analyses
3. View results in `output/<filename>/`

#### CLI Mode
```bash
# Single file
python examples/run_analysis.py path/to/animation.fbx

# Or using module entry point
python -m fbx_tool path/to/animation.fbx
```

### 3D Visualization Controls

The interactive OpenGL viewer provides real-time animation playback with full control.

📖 **[Complete 3D Viewer Guide](docs/3D_VIEWER_GUIDE.md)** - Detailed reference with tips & tricks

#### Mouse Controls
- **Left Click + Drag** - Orbit camera around character
- **Right Click + Drag** - Pan camera position
- **Mouse Wheel** - Zoom in/out

#### Keyboard Shortcuts
- **Space** - Play/Pause animation
- **Left/Right Arrow** - Previous/Next frame
- **Home/End** - Jump to first/last frame
- **R** - Reset camera to default view
- **F** - Front view (facing camera)
- **S** - Side view (profile)
- **T** - Top view (bird's eye)
- **G** - Toggle grid on/off
- **A** - Toggle coordinate axes on/off
- **W** - Toggle wireframe mode on/off

#### UI Controls
- **Playback Speed Slider** - Adjust animation speed from 0.1x to 4.0x
- **Frame Slider** - Scrub through animation timeline
- **Camera Preset Buttons** - Quick access to standard views
- **Display Options Checkboxes** - Toggle visual elements

#### Coordinate Axes
- **Red** - X-axis
- **Green** - Y-axis (up)
- **Blue** - Z-axis

## Output Files

After analysis, you'll find:

```
output/
└── your_animation/
    ├── dopesheet.csv                      # Frame-by-frame bone rotations
    ├── joint_enhanced_relationships.csv   # Per-joint metrics
    ├── chain_confidence.csv               # Per-chain IK suitability
    ├── chain_gait_segments.csv            # Stride segments
    ├── gait_summary.csv                   # Overall gait metrics
    └── analysis_summary.json              # Complete summary
```

### File Descriptions

**dopesheet.csv**
- Bone rotations for every frame
- Format: Rows = bones, Columns = frames
- Optimized for animation editing

**joint_enhanced_relationships.csv**
- Min/max rotation ranges (X, Y, Z)
- Stability score (0-1)
- Range score (0-1)
- IK suitability (0-1)

**chain_confidence.csv**
- Mean IK suitability per chain
- Cross-temporal coherence
- Overall chain confidence

**chain_gait_segments.csv**
- Detected stride segments
- Frame start/end
- Cycle time
- Confidence score
- Stride length

**gait_summary.csv**
- Cycle rate (Hz)
- Mean stride height
- Left/right phase shift
- Gait type (Walk/Run)

## Project Structure

```
FBX_Tool/
├── fbx_tool/                    # Main package
│   ├── __init__.py
│   ├── __main__.py             # Module entry point
│   ├── analysis/               # Core analysis modules
│   │   ├── fbx_loader.py       # FBX scene loading
│   │   ├── dopesheet_export.py # Dopesheet generation
│   │   ├── joint_analysis.py   # Joint metrics
│   │   ├── chain_analysis.py   # Chain metrics
│   │   ├── gait_analysis.py    # Gait detection
│   │   ├── gait_summary.py     # Summary model
│   │   └── utils.py            # Shared utilities
│   ├── gui/                    # GUI package
│   │   └── main_window.py      # PyQt6 GUI entry point
│   └── visualization/          # 3D visualization
│       ├── opengl_viewer.py    # OpenGL skeleton viewer
│       └── matplotlib_viewer.py # Matplotlib charts (planned)
├── examples/                    # Example scripts
│   ├── run_analysis.py         # CLI example
│   └── visualize.py            # Visualization example
├── tests/                       # Test directory
├── docs/                        # Documentation
├── requirements.txt             # Full dependencies (core + visualization)
├── requirements-dev.txt         # Development tools (includes requirements.txt)
├── pyproject.toml              # Project metadata
├── INSTALL.md                  # Installation guide
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
└── LICENSE                     # MIT License
```

## Building Executable

```bash
# Activate virtual environment
.fbxenv\Scripts\activate

# Build
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool/gui/main_window.py

# Output
dist/FBX_Tool.exe
```

**Note:** The executable requires FBX SDK DLLs. Copy them to `dist/` or ensure they're in system PATH.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Adding New Analysis Modules

1. Create module in `fbx_tool/analysis/`
2. Implement analysis function with signature:
   ```python
   def analyze_something(scene, output_dir="output/"):
       # Your analysis logic
       return results
   ```
3. Add to `fbx_tool/gui/main_window.py` and `examples/run_analysis.py`
4. Update `GaitSummaryAnalysis` model if needed

## Error Handling

FBX Tool uses robust error handling:
- Individual analysis failures don't stop execution
- Partial results are saved
- Errors are logged and displayed
- Summary includes error count

Example output with errors:
```
==================================================
Analyzing: animation.fbx
==================================================
Loading FBX scene...
  ✓ Duration: 3.73s

Analyzing joints...
  ✗ Joint analysis failed: [error message]

Analyzing chains...
  ✓ 5 chains analyzed

⚠ 1 step(s) had errors:
    - Joint Analysis: [error message]

Partial results saved to: output/animation/
==================================================
```

## Known Issues

- **"Unknown" Gait Type**: Occurs when left/right phase detection fails or when analyzing non-walking animations. Check `gait_summary.csv` for raw metrics.
- **FBX SDK Import Errors**: Ensure Python 3.10.x and FBX SDK 2020.3.7 are installed. See [INSTALL.md](INSTALL.md).
- **High Memory Usage**: Large FBX files (>10k frames, 100+ bones) may consume significant RAM during analysis.
- **Mixamo Files**: Tool automatically detects and uses the "mixamo.com" animation stack. If animation appears static, check that the file contains animation keyframes.

## Recent Fixes (v1.1.0)

- ✅ **Fixed**: Static animation display - Now properly detects and uses correct animation stack
- ✅ **Fixed**: Mixamo file support - Automatically selects "mixamo.com" stack over "Take 001"
- ✅ **Fixed**: Chain detection - Dynamic detection works with any skeleton naming convention
- ✅ **Fixed**: Coordinate system - Automatic Y-up/Z-up detection and conversion
- ✅ **Enhanced**: 3D Visualization - Added interactive controls, camera presets, and display options

## Future Enhancements

- [x] ~~Real-time 3D animation preview~~ ✅ **Completed in v1.1.0**
- [x] ~~Dynamic skeleton support~~ ✅ **Completed in v1.1.0**
- [x] ~~Batch processing GUI~~ ✅ **Functional as of Session 2025-10-19c** (Add to Batch buttons work correctly)
- [ ] Matplotlib visualization charts (joint angles over time)
- [ ] Export to Excel/PDF
- [ ] Animation quality scoring
- [ ] Bone selection and highlighting in 3D viewer
- [ ] Frame-by-frame comparison mode
- [ ] ML-based anomaly detection
- [ ] Cloud integration for team collaboration
- [ ] Screenshot/video export from 3D viewer

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create Pull Request

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Autodesk FBX SDK** - For providing the FBX Python SDK
- **PyQt6** - For the GUI framework
- **NumPy** - For numerical processing

## Contact

**Noah Butcher**
Email: posner.noah@gmail.com
GitHub: [@noahbutcher97](https://github.com/noahbutcher97)

## Citation

If you use FBX Tool in your research, please cite:

```bibtex
@software{fbx_tool_2025,
  author = {Butcher, Noah},
  title = {FBX Tool: Professional FBX Animation Analysis},
  year = {2025},
  url = {https://github.com/noahbutcher97/FBX_Tool}
}
```

---

**⭐ Star this repo if you find it useful!**
