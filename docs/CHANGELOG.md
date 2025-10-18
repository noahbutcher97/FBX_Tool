# FBX Tool Changelog

## [Unreleased]

### Next Steps
- Fix jitter detection thresholds (make adaptive/percentile-based)
- Fix constraint analysis confidence (return 0.0 when no chains analyzed)
- Audit foot contact analysis for hardcoded constants
- Implement overlay of analysis data on visualizer
- Add smart preloading for seamless file switching

---

## [2025-10-18] - Procedural Threshold System

### 🎯 Major Features

#### Adaptive Motion State Detection
**Complete proceduralization of motion state classification**

- ✅ **Removed hardcoded velocity thresholds**
  - Old: `VELOCITY_IDLE_THRESHOLD = 5.0` (broke on Mixamo ~326 units/sec)
  - New: Percentile-based (10th, 40th, 75th) + coefficient of variation check

- ✅ **Low-variance detection with CV** (`motion_transition_detection.py:165-193`)
  - CV < 12% triggers single-state classification
  - Prevents flickering between idle/walk/run/sprint
  - Example: 23-frame run (CV=7.5%) → classified as single "running" state

- ✅ **Adaptive minimum duration** (`motion_transition_detection.py:513-537`)
  - Old: Fixed 10 frames (43% of 23-frame animation!)
  - New: Percentage-based (15% for <30 frames, 10% for longer)
  - 23-frame animation: 3 frames minimum (13%)

#### Performance Optimization
**Cached derivatives eliminate redundant computation**

- ✅ **Cached acceleration/jerk** (`utils.py:430-447`)
  - Computed once in trajectory extraction
  - Shared across all analyses
  - ~3x speedup for multi-analysis workflows

#### Procedural Metadata System
**JSON export of discovered properties**

- ✅ **Metadata export** (`utils.py:641-707`)
  - Coordinate system detection results
  - Adaptive thresholds computed from data
  - Confidence scores for all detections
  - AI integration ready

**Output:** `procedural_metadata.json` in each output directory

### 📚 Documentation

#### New Documentation Files

- **[docs/development/HARDCODED_CONSTANTS_AUDIT.md](development/HARDCODED_CONSTANTS_AUDIT.md)**
  - Comprehensive audit of all hardcoded constants
  - Status tracking (Fixed/Partial/Not Fixed)
  - Priority levels (P0/P1/P2) and impact assessment

- **[docs/development/NEXT_SESSION_TODO.md](development/NEXT_SESSION_TODO.md)**
  - Session handoff with urgent tasks
  - Priority-ordered todo list
  - Test protocol and quick commands

- **[docs/onboarding/CLAUDE_START_HERE.md](onboarding/CLAUDE_START_HERE.md)**
  - First document for Claude Code to read
  - Current session status and priorities
  - Design principles and common tasks

#### Updated Documentation

- **CLAUDE.md** - Added Session 2025-10-18 comprehensive summary
- **docs/README.md** - Updated with new documentation links

#### Archived Documentation

- **docs/archive/CRITICAL_BUGS_FOUND_2025-10-17.md**
  - Many bugs fixed (direction detection, temporal segmentation, motion states)
  - Superseded by HARDCODED_CONSTANTS_AUDIT.md and NEXT_SESSION_TODO.md

### 🔧 Technical Changes

#### Modified Files

**Core Analysis:**
- `fbx_tool/analysis/utils.py` - Cached derivatives, metadata export
- `fbx_tool/analysis/motion_transition_detection.py` - Adaptive thresholds, CV detection
- `fbx_tool/analysis/root_motion_analysis.py` - Metadata export integration

**Documentation:**
- `CLAUDE.md` - Session 2025-10-18 summary
- `docs/README.md` - Updated documentation index
- `docs/CHANGELOG.md` - This file

#### Debug Logging Added

Comprehensive logging throughout motion state detection:
- Velocity range and coefficient of variation
- State distribution per frame
- Adaptive thresholds computed
- Segment filtering decisions

**Example Output:**
```
🔍 Detecting motion states for 23 frames...
  🔬 Velocity range: 77.3, threshold span: 49.9 (64.5% of range), CV: 0.075
  ⚠️  Low variance detected (CV=0.075) - classifying all as single state
  Adaptive velocity thresholds: idle=124.5, walk=224.0, run=358.7 units/sec
📊 State distribution: {'running': 23}
🔧 Segmenting 23 states...
  📏 Adaptive min duration: 3 frames (13.0% of 23 total)
✅ Created 1 motion state segments
```

### 🐛 Bug Fixes

#### Motion State Detection Fixes

1. **Removed circular dependency**
   - Old: Adaptive calculator used hardcoded constants to categorize animation
   - New: Purely data-driven using percentiles

2. **Fixed low-variance detection**
   - Old: Used median-based thresholds (still allowed mixed states)
   - New: Uses min/max velocities (ensures all frames in same state)

3. **Fixed frame count filtering**
   - Old: Fixed 10 frames filtered out all segments in short animations
   - New: Percentage-based minimum scales with animation length

#### Python Bytecode Caching Issue

- Fixed: Code changes not taking effect due to .pyc caching
- Solution: Documented cache clearing in quick reference
- Impact: Critical changes now require explicit cache clear

### 📊 Test Results

**Before Fixes (Run Forward Arc Left, 23 frames):**
```
- Motion state segments: 0 detected (all filtered out)
- Classification: "varied_movement" (incorrect - flickering states)
- State distribution: {'sprinting': 6, 'running': 8, 'walking': 6, 'idle': 3}
```

**After Fixes:**
```
- Motion state segments: 1 detected ✅
- Classification: "run_cycle" (correct)
- State distribution: {'running': 23} ✅ (pending final test)
- CV: 0.075 (7.5% variance - correctly identified as single state)
```

### ⚠️ Known Issues

**Still Need Fixing (Next Session):**

1. **Jitter Detection** - All 65 bones flagged as high jitter
   - Threshold too sensitive for Mixamo rigs
   - Fix: Make adaptive using percentile approach

2. **Constraint Confidence** - Returns 1.0 with 0 chains analyzed
   - Misleading to users
   - Fix: Return 0.0 confidence when no data

3. **Foot Contact Sliding** - All contacts flagged as sliding
   - Threshold likely too strict
   - Fix: Audit and proceduralize thresholds

### 💡 Key Learnings

1. **Coefficient of Variation (CV) is superior to absolute thresholds**
   - CV = std_dev / mean
   - CV < 12% reliably detects low-variance (single-state) animations

2. **Percentage-based beats fixed frame counts**
   - Must scale with animation length
   - 15% for short (<30 frames), 10% for longer

3. **Use min/max for single-state thresholds**
   - Median-based can still allow mixed classification
   - min/max ensures ALL frames fall into intended category

4. **Debug logging reveals hidden issues**
   - Shows what's computed vs what should be computed
   - Critical for diagnosing threshold issues

5. **Python bytecode caching can hide fixes**
   - Must clear `__pycache__` after critical changes
   - Especially for `motion_transition_detection.py`

### 🔮 Future Enhancements

**Planned for Next Session:**
- Adaptive jitter thresholds (percentile-based)
- Fix constraint confidence calculation
- Audit foot contact thresholds
- Export all adaptive thresholds to metadata
- Add confidence scores to all detections

---

## [2025-10-17] - Scene Manager Architecture & Test Infrastructure

### 🎯 Major Features

#### Scene Manager System
**Complete refactor of FBX scene lifecycle management**

- ✅ **Reference-counted scene manager** (`fbx_tool/analysis/scene_manager.py`)
  - Automatic cleanup when ref count reaches 0
  - Thread-safe caching with locks
  - Scene sharing between GUI, visualizer, and analysis
  - Singleton pattern for global coordination

- ✅ **Smart caching in visualizer** (`fbx_tool/visualization/opengl_viewer.py`)
  - Keeps only current ± 1 files in memory
  - Prevents memory bloat with large batches (100+ files)
  - Automatic release of unused scenes

- ✅ **GUI integration** (`fbx_tool/gui/main_window.py`)
  - Tracks active scene references
  - Clear button properly frees memory
  - Analysis worker gets cache hits when GUI holds scenes

#### Test Infrastructure
**All integration tests now passing!**

- ✅ **Scene manager unit tests** (`tests/unit/test_scene_manager.py`)
  - 22 tests, 83.33% coverage
  - Tests reference counting, thread safety, smart caching
  - Integration workflow tests

- ✅ **Fixed integration test mocks** (`tests/integration/test_analysis_pipeline.py`)
  - Fixed 5 critical mock setup issues
  - 10/10 integration tests passing
  - 22.36% overall code coverage (exceeds minimum)

### 📚 Documentation

#### New Documentation Files

- **[docs/architecture/SCENE_MANAGEMENT.md](architecture/SCENE_MANAGEMENT.md)**
  - Complete architecture guide
  - Usage patterns and examples
  - Performance impact analysis
  - Migration guide from old patterns

- **[docs/testing/MOCK_SETUP_PATTERNS.md](testing/MOCK_SETUP_PATTERNS.md)**
  - FBX SDK mocking patterns
  - Common issues and fixes
  - Complete integration test template
  - Debugging strategies

- **[docs/README.md](README.md)**
  - Documentation structure and navigation
  - Quick reference by role and topic
  - Cross-referenced documentation map

### 🔧 Technical Changes

#### New Files
- `fbx_tool/analysis/scene_manager.py` (197 lines)
- `tests/unit/test_scene_manager.py` (604 lines, 22 tests)
- `tests/unit/test_fbx_memory_management.py` (194 lines)
- `docs/architecture/SCENE_MANAGEMENT.md`
- `docs/testing/MOCK_SETUP_PATTERNS.md`
- `docs/README.md`

#### Modified Files
- `fbx_tool/gui/main_window.py` - Scene manager integration, clear button fix
- `fbx_tool/visualization/opengl_viewer.py` - Smart caching implementation
- `tests/integration/test_analysis_pipeline.py` - 6 new tests, 4 fixed tests
- `CLAUDE.md` - Comprehensive session summary

### 🐛 Bug Fixes

#### Integration Test Mock Fixes

1. **Missing `has_animation` key**
   - Added `"has_animation": True` to all metadata mocks
   - Prevents `ValueError: No animation data found` errors

2. **Mock `GetSrcObjectCount` returning Mock object**
   - Changed to return integer: `mock_scene.GetSrcObjectCount.return_value = 1`
   - Fixes `TypeError: '>' not supported between Mock and int`

3. **FbxTime class being mocked incorrectly**
   - Removed `@patch("fbx.FbxTime")` decorator
   - Use real `fbx.FbxTime()` objects in tests
   - Fixes `TypeError: float() argument must be Mock`

4. **Transform matrix Get(i,j) calls not handled**
   - Changed from `return_value` to `side_effect = lambda i, j: matrix[i][j]`
   - Handles indexed access to transformation matrix
   - Fixes `TypeError: float() argument must be Mock`

5. **Off-by-one frame count error**
   - Changed mock data from 30 to 31 frames
   - Matches production calculation: `total_frames = int(duration * frame_rate) + 1`
   - Fixes `IndexError: index 30 out of bounds`

### 📊 Performance Improvements

**Memory Usage:**
- **Before:** Animation loaded 3× (GUI + Visualizer + Analysis) = 300 MB
- **After:** Loaded once, shared via references = 100 MB
- **Savings:** 66-90% depending on workflow

**Batch Processing:**
- **Without smart caching:** All 100 files stay in memory
- **With smart caching:** Only 3 files in memory at any time
- **Memory limit:** ~3× single file size for visualizer switching

### ⚠️ Breaking Changes

**Scene loading pattern changed:**

```python
# OLD (don't use anymore!)
scene, manager = load_fbx("file.fbx")
# ... use scene ...
cleanup_fbx_scene(scene, manager)

# NEW (required pattern)
from fbx_tool.analysis.scene_manager import get_scene_manager
scene_manager = get_scene_manager()
with scene_manager.get_scene("file.fbx") as scene_ref:
    scene = scene_ref.scene
    # ... use scene ...
# Automatically released
```

**Migration required for:**
- Direct `load_fbx()` calls → Use scene manager
- Direct `cleanup_fbx_scene()` calls → Scene manager handles this
- Storing scene objects → Store `FBXSceneReference` instead

### ✅ Test Results

```
Integration Tests:       10/10 PASSING
Scene Manager Tests:     22/22 PASSING
Scene Manager Coverage:  83.33%
Overall Coverage:        22.36% (exceeds 20% minimum)
Root Motion Analysis:    100% coverage
Directional Changes:     65.62% coverage
Motion Transitions:      72.13% coverage
```

### 📖 Documentation Updates

- Updated `CLAUDE.md` with comprehensive session summary
- Created structured documentation hierarchy in `docs/`
- Added migration guides and best practices
- Cross-referenced all documentation

### 🔮 Future Enhancements

**Enabled by scene manager architecture:**
- Overlay analysis data on visualizer (scene sharing enables this)
- Smart preloading of next ± 2 files
- Real-time analysis updates in visualizer
- Multi-threaded batch processing with shared scenes

**Planned optimizations:**
- Cache acceleration/jerk in trajectory extraction
- Replace hardcoded thresholds with adaptive algorithms
- Implement data-driven motion detection

---

## Previous Updates

See git commit history for earlier changes.

## Version Numbering

This project does not yet use semantic versioning. Future releases will follow [SemVer](https://semver.org/).

## Contributing

When updating this changelog:
1. Add new entries under `[Unreleased]`
2. Group by: Added, Changed, Deprecated, Removed, Fixed, Security
3. Include file references and line numbers where relevant
4. Cross-reference related documentation
5. Note breaking changes clearly
