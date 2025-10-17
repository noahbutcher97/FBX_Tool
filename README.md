# FBX Tool

Professional desktop application for analyzing FBX animation files with biomechanical motion processing.

![FBX Tool](assets/screenshot.png)

## Features

- **Dopesheet Export** - Frame-by-frame bone rotation data in CSV format
- **Joint Analysis** - Per-joint metrics including stability, range of motion, and IK suitability
- **Chain Analysis** - Chain-level IK confidence and cross-temporal coherence
- **Gait Analysis** - Stride segmentation and gait phase detection
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
git clone https://github.com/yourusername/fbx-tool.git
cd fbx-tool

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
python main_gui.py
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
python main.py path/to/animation.fbx

# Batch processing
python main.py --batch input_folder/
```

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
├── main_gui.py                  # PyQt6 GUI entry point
├── main.py                      # CLI entry point
├── analysis_modules/            # Core analysis modules
│   ├── __init__.py
│   ├── fbx_loader.py           # FBX scene loading
│   ├── dopesheet_export.py     # Dopesheet generation
│   ├── joint_analysis.py       # Joint metrics
│   ├── chain_analysis.py       # Chain metrics
│   ├── gait_analysis.py        # Gait detection
│   ├── gait_summary.py         # Summary model
│   └── utils.py                # Shared utilities
├── requirements.txt             # Python dependencies
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
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean main_gui.py

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

1. Create module in `analysis_modules/`
2. Implement analysis function with signature:
   ```python
   def analyze_something(scene, output_dir="output/"):
       # Your analysis logic
       return results
   ```
3. Add to `main_gui.py` and `main.py`
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

- **"Unknown" Gait Type**: Occurs when left/right phase detection fails. Check `gait_summary.csv` for raw metrics.
- **FBX SDK Import Errors**: Ensure Python 3.10.x and FBX SDK 2020.3.7 are installed. See [INSTALL.md](INSTALL.md).
- **High Memory Usage**: Large FBX files (>10k frames, 100+ bones) may consume significant RAM.

## Future Enhancements

- [ ] Batch processing GUI
- [ ] Real-time 3D animation preview
- [ ] Matplotlib visualization charts
- [ ] Export to Excel/PDF
- [ ] Animation quality scoring
- [ ] ML-based anomaly detection
- [ ] Comparison mode (side-by-side)
- [ ] Cloud integration for team collaboration

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
GitHub: [@yourusername](https://github.com/yourusername)

## Citation

If you use FBX Tool in your research, please cite:

```bibtex
@software{fbx_tool_2025,
  author = {Butcher, Noah},
  title = {FBX Tool: Professional FBX Animation Analysis},
  year = {2025},
  url = {https://github.com/yourusername/fbx-tool}
}
```

---

**⭐ Star this repo if you find it useful!**
