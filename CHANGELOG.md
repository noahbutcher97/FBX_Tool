# Changelog

All notable changes to FBX Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-17

### Added
- **Interactive 3D Visualization** - Real-time OpenGL skeleton animation viewer with:
  - Mouse controls: orbit, pan, zoom
  - Keyboard shortcuts for navigation and display options
  - Playback speed control (0.1x to 4.0x)
  - Camera presets (Front, Side, Top, Reset)
  - Display toggles (Grid, Axes, Wireframe)
  - Frame-by-frame navigation
- **Dynamic Chain Detection** - Automatically detects bone chains from any skeleton hierarchy
  - Works with Mixamo, Unity, Blender, and custom naming conventions
  - Intelligently identifies legs, arms, spine, neck, and finger chains
  - No hardcoded bone names required
- **Smart Animation Stack Selection** - Automatically finds correct animation data
  - Prioritizes "mixamo.com" stack for Mixamo files
  - Falls back to longest duration stack for other files
  - Logs selected stack for transparency
- **Coordinate System Auto-Detection** - Handles Y-up and Z-up files automatically
  - Detects and converts Z-up (Blender, 3ds Max) to Y-up
  - Preserves Y-up files (Mixamo, Unity) as-is
  - Applies to both visualization and analysis
- **Enhanced OpenGL Viewer UI**
  - Organized control panels with groups
  - Real-time speed adjustment during playback
  - Display options with checkboxes
  - Keyboard shortcut reference in UI
  - Professional layout with PyQt6

### Fixed
- **Static Animation Display** - Resolved issue where animations appeared frozen
  - Root cause: Code was using wrong animation stack (stack 0 instead of stack 1)
  - Solution: Automatic detection of animation stack with actual keyframe data
- **Mixamo File Compatibility** - Now properly handles Mixamo FBX file structure
  - Detects "mixamo.com" animation stack vs empty "Take 001" stack
  - Correctly extracts 3D transforms including rotation data
- **Chain Analysis Failure** - Fixed 0 chains detected for Mixamo skeletons
  - Replaced hardcoded bone names with dynamic hierarchy traversal
  - Now detects 15+ chains in typical humanoid rigs
- **Coordinate System Issues** - Characters no longer lie flat
  - Automatic Y-up detection prevents unnecessary coordinate conversion
  - Maintains correct upright orientation for all file types

### Changed
- **Animation Data Extraction** - Now extracts both position AND rotation quaternions
  - Previous version only extracted translation, causing static appearance
  - Full transform data enables proper skeletal animation display
- **Bone Hierarchy Building** - Enhanced to support any naming convention
  - `build_bone_hierarchy()` now works universally
  - No longer requires specific naming patterns
- **Chain Detection Algorithm** - Completely rewritten
  - `detect_chains_from_hierarchy()` replaces `get_standard_chains()`
  - Traces linear bone sequences automatically
  - Generates intelligent chain names based on bone content

### Improved
- **OpenGL Rendering Performance** - Optimized draw calls
  - Conditional rendering based on display options
  - Efficient transform caching per frame
- **User Experience** - More intuitive controls and feedback
  - Visual feedback for all toggles
  - Clear keyboard shortcut labels
  - Grouped UI elements for better organization
- **Error Messages** - More informative animation stack logging
  - Prints selected stack name and duration
  - Helps debug animation loading issues

### Technical Details
- Updated `fbx_tool/analysis/utils.py`:
  - `get_animation_info()` - Smart stack selection
  - `detect_chains_from_hierarchy()` - Dynamic chain detection
  - `_generate_chain_name()` - Intelligent naming
- Updated `fbx_tool/visualization/opengl_viewer.py`:
  - `_extract_transforms()` - Full rotation + translation extraction
  - `keyPressEvent()` - Comprehensive keyboard handling
  - Camera preset methods (front, side, top, reset)
  - Display option toggles (grid, axes, wireframe)
- Updated `fbx_tool/analysis/chain_analysis.py`:
  - Now uses `detect_chains_from_hierarchy()`
- Updated `fbx_tool/analysis/gait_analysis.py`:
  - Now uses `detect_chains_from_hierarchy()`

## [1.0.0] - 2025-01-10

### Added
- Initial release
- FBX file loading and scene metadata extraction
- Dopesheet export (bone rotations per frame)
- Joint analysis (stability, range, IK suitability)
- Chain analysis (IK confidence, coherence)
- Gait analysis (stride segmentation, phase detection)
- GUI with drag-and-drop support
- CLI interface for batch processing
- JSON summary output
- CSV exports for all analysis types

### Features
- Python 3.10 compatibility
- FBX SDK 2020.3.7 integration
- PyQt6 GUI framework
- NumPy-based analysis algorithms
- PyInstaller executable builds
- File-specific output directories
- Robust error handling

---

## Version Comparison

### v1.1.0 vs v1.0.0

**New in v1.1.0:**
- 3D Visualization (completely new feature)
- Universal skeleton support (any naming convention)
- Smart animation detection (Mixamo compatibility)
- Interactive controls (keyboard + mouse)
- Camera presets and display options

**Improvements:**
- Chain detection: 0 chains → 15+ chains for typical rigs
- Animation display: Static → Fully animated
- Compatibility: Mixamo files now work perfectly
- User experience: Basic analysis → Interactive visualization

[1.1.0]: https://github.com/noahbutcher97/FBX_Tool/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/noahbutcher97/FBX_Tool/releases/tag/v1.0.0
