# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**🚨 FIRST TIME HERE?** Start with **[docs/onboarding/CLAUDE_START_HERE.md](docs/onboarding/CLAUDE_START_HERE.md)** for:
- Current session status and urgent tasks
- Essential reading order
- Quick commands and common tasks
- Key learnings and success criteria

## Project Overview

FBX Tool is a professional desktop application for analyzing FBX animation files with biomechanical motion processing and real-time 3D visualization. The tool works universally across any skeleton naming convention (Mixamo, Unity, Blender, custom rigs) with automatic coordinate system detection.

**Critical Constraint:** Python 3.10.x ONLY. FBX Python SDK 2020.x does not support Python 3.11+.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.10 required!)
python -m venv .fbxenv --system-site-packages
.fbxenv\Scripts\activate  # Windows
source .fbxenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # For development
```

### Testing
```bash
# Run all tests with coverage (parallel execution via pytest-xdist)
pytest

# Run specific test file
pytest tests/unit/test_gait_analysis.py -v

# Run single test function
pytest tests/unit/test_gait_analysis.py::test_detect_stride_segments_normal_gait -v

# Run tests without parallelization (useful for debugging)
pytest -n 0

# Run only fast unit tests (skip integration/slow)
pytest -m "unit and not slow"

# Run tests requiring FBX SDK
pytest -m fbx

# Skip coverage for faster runs
pytest --no-cov

# Show local variables on failure
pytest -l

# Re-run only last failed tests
pytest --lf

# Generate HTML coverage report
pytest --cov-report=html
# Opens: htmlcov/index.html
```

### Code Quality
```bash
# Format code (Black, 120 char line length)
black fbx_tool/ tests/

# Sort imports
isort fbx_tool/ tests/ --profile=black --line-length=120

# Lint code
flake8 fbx_tool/ tests/ --max-line-length=120 --extend-ignore=E203,W503

# Type checking
mypy fbx_tool/ --ignore-missing-imports --python-version=3.10

# Run all pre-commit hooks
pre-commit run --all-files
```

### Running the Application
```bash
# GUI mode
python fbx_tool/gui/main_window.py

# CLI mode (module entry point)
python -m fbx_tool

# CLI with specific file
python examples/run_analysis.py path/to/animation.fbx
```

### Building Executable
```bash
python -m PyInstaller --name="FBX_Tool" --onefile --windowed --clean fbx_tool/gui/main_window.py
# Output: dist/FBX_Tool.exe
```

## Architecture

### Analysis Pipeline Flow

```
1. FBX Loading (fbx_loader.py)
   └─> Multi-stack ranking (prefers "mixamo.com", longest duration)

2. Core Data Extraction
   ├─> Dopesheet Export (dopesheet_export.py) - Frame-by-frame bone rotations
   ├─> Joint Analysis (joint_analysis.py) - Per-joint metrics, stability, range
   └─> Chain Detection (utils.py) - Dynamic kinematic chain discovery

3. Motion Analysis Pipeline
   ├─> Velocity Analysis (velocity_analysis.py) - Translational + rotational velocity, jerk, smoothness
   ├─> Gait Analysis (gait_analysis.py) - Stride segmentation, cycle detection [88% coverage]
   ├─> Foot Contact (foot_contact_analysis.py) - Ground interaction detection
   └─> Root Motion (root_motion_analysis.py) - Character trajectory analysis

4. Advanced Analysis (Some TODO)
   ├─> Pose Validity (pose_validity_analysis.py) - Anatomical validation
   ├─> Constraint Violation (constraint_violation_detection.py) - IK/curve validation [INCOMPLETE]
   ├─> Directional Change (directional_change_detection.py) - Motion direction shifts [NEEDS TESTS]
   ├─> Motion Transition (motion_transition_detection.py) - State changes [NEEDS TESTS]
   ├─> Temporal Segmentation (temporal_segmentation.py) - Time-based chunking [NEEDS TESTS]
   └─> Motion Classification (motion_classification.py) - Motion type detection [NEEDS TESTS]
```

### Key Architectural Patterns

#### Transform Caching
Multiple analysis modules evaluate bone transforms independently. The codebase has identified this as a performance issue (see CODE_REVIEW_FINDINGS.md). When implementing new analysis modules, consider reusing cached transforms rather than re-evaluating.

#### Chain Detection Strategy
`utils.py:detect_chains_from_hierarchy()` dynamically discovers kinematic chains from any skeleton hierarchy without hardcoded bone names. This enables universal compatibility across different character rigs.

#### Animation Stack Selection
`fbx_loader.py:get_animation_info()` intelligently ranks animation stacks:
1. Prefers stacks named "mixamo.com" (Mixamo exports)
2. Falls back to longest duration stack
3. Evaluates stack activity (keyframes, animated bones)

This solves the common issue of FBX files containing multiple stacks where only one has actual animation data.

## FBX SDK Critical Patterns

**IMPORTANT:** The FBX SDK has non-obvious APIs. Consult `docs/development/FBX_SDK_FIXES.md` before adding FBX SDK code.

### ❌ WRONG Patterns
```python
# WRONG: GetTimeSpan from global settings
time_span = scene.GetGlobalSettings().GetTimeSpan(fbx.FbxTime.eGlobal)

# WRONG: Direct animation curve access
curve = node.GetAnimationCurve(0)

# WRONG: GetLayer method
anim_layer = anim_stack.GetLayer(0)
```

### ✅ CORRECT Patterns
```python
# CORRECT: Get time span from animation stack
anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
time_span = anim_stack.GetLocalTimeSpan()

# CORRECT: Access curves through property curve nodes
anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
curve_node = node.LclTranslation.GetCurveNode(anim_layer)
if curve_node:
    for channel in range(curve_node.GetChannelsCount()):
        curve = curve_node.GetCurve(channel)

# CORRECT: Get layer using FbxCriteria
anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
```

## Test-Driven Development (TDD) Workflow

**MANDATORY:** All new features and incomplete modules MUST follow strict TDD as documented in `docs/onboarding/README.md`.

### TDD Process
1. **Write tests FIRST** - Define expected behavior before implementation
2. **Run tests** - Watch them fail (red)
3. **Implement code to pass tests** - Not just minimal code, but robust implementation
4. **Refactor** - Improve code while keeping tests green
5. **Add edge cases** - Expand test coverage iteratively

### Writing Robust Tests

**CRITICAL:** Tests must be comprehensive enough to demand robust implementations. Don't write trivial tests that pass with placeholder code.

#### ❌ BAD Test (Too Minimal)
```python
def test_detect_contacts():
    """Test detects contacts."""
    contacts = detect_foot_contacts([], [])
    assert contacts is not None  # Passes with "return []"
```

#### ✅ GOOD Test (Demands Robust Implementation)
```python
def test_detect_contacts_with_clear_ground_strike():
    """Should detect contact when foot velocity drops to zero near ground."""
    # Arrange - Foot descending, hits ground, stays stationary
    positions = np.array([
        [0, 20, 0],  # Frame 0: High
        [0, 10, 0],  # Frame 1: Descending
        [0, 2, 0],   # Frame 2: Near ground
        [0, 0, 0],   # Frame 3: On ground (CONTACT START)
        [0, 0, 0],   # Frame 4: Stationary
        [0, 0, 0],   # Frame 5: Stationary (CONTACT END)
        [0, 5, 0],   # Frame 6: Lifting off
    ])
    velocities = np.diff(positions[:, 1])  # Vertical velocity

    # Act
    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    # Assert - Multiple conditions to verify correctness
    assert len(contacts) == 1, "Should detect exactly one contact period"
    assert contacts[0]['start_frame'] == 3, "Contact should start at frame 3"
    assert contacts[0]['end_frame'] == 5, "Contact should end at frame 5"
    assert contacts[0]['duration'] > 0, "Contact must have positive duration"
    assert 0.0 <= contacts[0]['confidence'] <= 1.0, "Confidence must be in [0,1]"

def test_detect_contacts_with_root_motion():
    """Should work with root motion (non-zero ground height)."""
    # Character walking with Y offset
    positions = create_walking_pattern(ground_height=50.0, num_strides=3)
    velocities = np.gradient(positions[:, 1])

    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    assert len(contacts) >= 3, "Should detect at least 3 contacts for 3 strides"
    # Verify adaptive thresholding worked (not hardcoded ground=0)
    for contact in contacts:
        assert contact['ground_height'] > 40, "Should detect elevated ground"

def test_detect_contacts_empty_data_graceful_degradation():
    """Should handle empty data gracefully without crashing."""
    contacts = detect_foot_contacts(np.array([]), np.array([]), frame_rate=30.0)

    assert contacts == [], "Empty input should return empty list"
    # Should not raise exception

def test_detect_contacts_single_frame_edge_case():
    """Should handle single-frame animation without crashing."""
    positions = np.array([[0, 0, 0]])
    velocities = np.array([])

    contacts = detect_foot_contacts(positions, velocities, frame_rate=30.0)

    assert contacts == [], "Single frame should return no contacts"
```

### Test Coverage Requirements

See `docs/onboarding/README.md` section "Test Coverage Requirements":
- **Minimum:** 20% overall (enforced by pytest.ini)
- **Target:** 80% for new modules
- **Current:** 24.31% overall
- **Reference standard:** gait_analysis.py (88% coverage, 22/22 tests passing)

Study `tests/unit/test_gait_analysis.py` for examples of comprehensive test patterns.

### Test Organization
- `tests/unit/` - Fast, isolated unit tests (use mocks from conftest.py)
- `tests/integration/` - Multi-component integration tests
- `tests/conftest.py` - Shared fixtures (mock_scene, sample_positions, sample_velocities, temp_output_dir)

### Test Markers
```python
@pytest.mark.unit  # Fast, isolated tests
@pytest.mark.integration  # Multi-component tests
@pytest.mark.gui  # GUI tests (requires display)
@pytest.mark.slow  # Long-running tests
@pytest.mark.fbx  # Tests requiring FBX SDK
```

## Code Quality Standards

### Coverage Requirements
From `pytest.ini`:
- Enforced minimum: 20% (`--cov-fail-under=20`)
- Target for new modules: 80%+
- gait_analysis.py achieves 88% (use as reference)

### Formatting
From `pyproject.toml` and `.pre-commit-config.yaml`:
- **Black formatter:** 120 character line length
- **isort:** Black-compatible profile
- **flake8:** Max line 120, ignore E203, W503
- **mypy:** Type hints with `--ignore-missing-imports`
- **interrogate:** 50% docstring coverage minimum

### Pre-commit Hooks
All hooks in `.pre-commit-config.yaml` must pass:
- Code formatting (black, isort)
- Linting (flake8)
- Type checking (mypy)
- Security checks (bandit)
- Docstring coverage (interrogate: 50% minimum)
- File integrity checks

## Key Design Principles

From `docs/onboarding/README.md` section "Key Design Principles":

### 1. No Hardcoded Assumptions

**❌ BAD:**
```python
CONTACT_VELOCITY_THRESHOLD = 10.0  # Breaks on different character scales!
foot = bones[-2]  # Assumes specific skeleton structure
```

**✅ GOOD:**
```python
def calculate_adaptive_threshold(velocities):
    """Derive threshold from data distribution."""
    return np.percentile(velocities, 25)

def detect_foot_bone(chain):
    """Detect foot by name matching, with fallback."""
    for bone in reversed(chain):
        if any(kw in bone.lower() for kw in ['foot', 'ankle', 'tarsal']):
            return bone
    return chain[-2] if len(chain) >= 2 else chain[-1]
```

### 2. Separation of Concerns
- **Detection** (what is happening) vs. **Classification** (what type)
- **Metric calculation** vs. **Threshold application**
- **Data extraction** vs. **Analysis**

### 3. Confidence Scores
Every analysis should report confidence:
```python
return {
    'contacts_detected': 5,
    'confidence': 0.85,  # How sure are we?
    'method': 'adaptive_percentile',
    'warnings': ['Low velocity variance detected']
}
```

### 4. Graceful Degradation
From `docs/onboarding/README.md`:
- Empty animations → Return zeros with low confidence
- Missing bones → Analyze available bones, log skipped ones
- Corrupt data → Skip problematic frames, report issues in summary

See `IMPROVEMENT_RECOMMENDATIONS.md` for detailed edge case handling patterns.

## Known Issues & Incomplete Modules

Documented in `docs/development/INCOMPLETE_MODULES.md`:

### Modules Requiring TDD Implementation (Write Tests First!)
- **directional_change_detection.py** - 0% coverage
- **motion_transition_detection.py** - 0% coverage
- **temporal_segmentation.py** - 0% coverage
- **motion_classification.py** - 0% coverage
- **constraint_violation_detection.py** - Contains TODO placeholders

### Active Issues
From `docs/development/INCOMPLETE_MODULES.md`:

1. **Foot Contact Analysis** - Returns 0 contacts on walking animations
   - **Cause:** Hardcoded `CONTACT_VELOCITY_THRESHOLD = 10.0` too low for root motion
   - **Fix Required:** Adaptive, percentile-based thresholding
   - **Test Spec:** Write comprehensive tests in `tests/unit/test_foot_contact_analysis.py`

2. **Pose Validity Analysis** - Returns "0 bones validated"
   - **Cause:** `_extract_bone_animation_data()` returns placeholder zeros
   - **Fix Required:** Proper bone transform extraction using FbxTime

3. **Constraint Violation Analysis** - Returns "0 chains analyzed"
   - **Cause:** TODO placeholders for IK chain detection and curve discontinuity
   - **Fix Required:** Full implementation after writing test specifications

### Algorithm Issues

From `CODE_REVIEW_FINDINGS.md`, several modules have correctness issues:

#### gait_analysis.py Critical Issues:
- **Line 165:** Cycle rate calculation is WRONG (calculates contact rate, not cycle rate)
- **Line 110:** Stride length uses Y-axis (vertical) instead of horizontal distance
- **Line 112:** Asymmetry column always hardcoded to 0.0
- **Line 105:** Confidence formula unexplained (`np.mean(np.pi / (1 + np.abs(segment_vel)))`)

#### velocity_analysis.py Critical Issues:
- **Lines 702-707:** NaN propagation in chain coherence (no filtering before mean)
- **Lines 522-540:** Massive temporal data generation (50 bones × 300 frames = 15k rows)
- **Magic numbers:** 0.1, 0.01, 1.0, 0.25, 0.4, 0.7 throughout with no documentation

#### chain_analysis.py Critical Issues:
- **Line 97:** Redundant dictionary lookup (parent already in segs[i][1])
- **Lines 69-71:** Undocumented IK score formula (why exp(-var)? why 540? why 0.6/0.4?)
- **Lines 122-128:** Questionable temporal coherence (overlapping windows inflate scores)

**Priority:** See `CODE_REVIEW_FINDINGS.md` sections "MUST FIX", "SHOULD FIX", "NICE TO HAVE"

## Documentation Map

**Start here for onboarding:** `docs/onboarding/README.md`

### Critical Reading (Read Before Coding):
1. **docs/development/FBX_SDK_FIXES.md** - FBX SDK API patterns (MUST READ before using FBX SDK!)
2. **docs/development/INCOMPLETE_MODULES.md** - Current incomplete modules and root causes
3. **CODE_REVIEW_FINDINGS.md** - Algorithm correctness issues requiring fixes
4. **IMPROVEMENT_RECOMMENDATIONS.md** - Edge case handling and validation patterns

### Reference Documentation:
- **docs/INSTALL.md** - Python 3.10 setup, FBX SDK installation, troubleshooting
- **docs/3D_VIEWER_GUIDE.md** - OpenGL viewer controls and shortcuts
- **README.md** - User-facing features, quick start, output file descriptions

### Testing Documentation:
- **pytest.ini** - Test configuration, markers, coverage thresholds
- **tests/conftest.py** - Shared fixtures and test utilities
- **tests/unit/test_gait_analysis.py** - Reference example of comprehensive test suite (88% coverage)

## Output Structure

Analysis results saved to `output/<fbx_filename>/`:
```
output/
└── your_animation/
    ├── dopesheet.csv                      # Frame-by-frame bone rotations
    ├── joint_enhanced_relationships.csv   # Per-joint stability, range, IK suitability
    ├── chain_confidence.csv               # Per-chain IK confidence
    ├── chain_gait_segments.csv            # Stride segments with timing
    ├── gait_summary.csv                   # Cycle rate, stride height, gait type
    ├── velocity_analysis.csv              # Velocity, acceleration, jerk metrics
    ├── foot_contacts.csv                  # Ground contact events
    ├── root_motion_analysis.csv           # Character trajectory
    └── analysis_summary.json              # Complete analysis summary
```

## Common Pitfalls

From experience documented in `docs/development/FBX_SDK_FIXES.md` and code reviews:

1. **Using Python 3.11+** - FBX SDK only supports Python 3.10.x
2. **Incorrect FBX SDK APIs** - ALWAYS check `docs/development/FBX_SDK_FIXES.md` first
3. **Implementing before testing** - Follow TDD strictly (tests first!)
4. **Writing minimal tests** - Tests must be robust enough to demand proper implementation
5. **Hardcoded thresholds** - Use adaptive, data-driven thresholds (see design principles)
6. **Assuming skeleton structure** - Support any naming convention (Mixamo, Unity, Blender, custom)
7. **Forgetting edge cases** - Test empty data, single frame, extreme values, NaN/inf
8. **Redundant transform evaluations** - Consider caching (see `CODE_REVIEW_FINDINGS.md`)
9. **Magic numbers** - Extract to named constants with documentation
10. **Silent failures** - Always log warnings when skipping data (see `IMPROVEMENT_RECOMMENDATIONS.md`)

## Success Metrics

From `docs/onboarding/README.md` section "Success Metrics":

### ✅ You're doing well when:
- Tests are written BEFORE implementation
- Coverage increases with each module (target: 80%+)
- Solutions work across diverse animation assets (not just Mixamo)
- Code is self-documenting with clear intent
- FBX SDK patterns follow `docs/development/FBX_SDK_FIXES.md`
- Edge cases are handled gracefully with warnings/logging

### ❌ Course-correct when:
- Implementing before testing
- Using hardcoded thresholds for specific animations
- Skipping edge case tests
- Making assumptions about character scale/hierarchy
- Using incorrect FBX SDK API patterns
- Writing tests that pass with placeholder implementations

## Development Workflow

From `docs/onboarding/README.md` section "Getting Started - First Tasks":

1. **Read documentation:**
   - `docs/onboarding/README.md` (project overview)
   - `docs/development/FBX_SDK_FIXES.md` (FBX SDK patterns)
   - `docs/development/INCOMPLETE_MODULES.md` (current issues)

2. **Pick a task** from incomplete modules or TODO list

3. **Write comprehensive tests FIRST** (TDD!)
   - Study `tests/unit/test_gait_analysis.py` for test patterns
   - Write tests for normal cases, edge cases, error conditions
   - Ensure tests demand robust implementation (not just minimal code)

4. **Run tests to watch them fail:**
   ```bash
   pytest tests/unit/test_module.py -v
   ```

5. **Implement robust code to pass tests**
   - Follow design principles (no hardcoded values, adaptive algorithms)
   - Add docstrings explaining complex formulas
   - Extract magic numbers to named constants

6. **Verify tests pass:**
   ```bash
   pytest tests/unit/test_module.py -v
   ```

7. **Add edge case tests iteratively**
   - Empty data
   - Single frame
   - NaN/inf values
   - Extreme values
   - Missing bones
   - Zero variance data

8. **Refactor while maintaining green tests**

9. **Format code:**
   ```bash
   black . && isort . --profile=black
   ```

10. **Check coverage (target 80%+):**
    ```bash
    pytest --cov=fbx_tool.analysis.module_name
    ```

11. **Run pre-commit hooks:**
    ```bash
    pre-commit run --all-files
    ```

12. **Commit with descriptive message**

## Module Entry Points

- **GUI:** `fbx_tool/gui/main_window.py` (main application window)
- **CLI:** `fbx_tool/__main__.py` (enables `python -m fbx_tool`)
- **Examples:** `examples/run_analysis.py` (CLI analysis script)
- **Core Utilities:** `fbx_tool/analysis/utils.py` (file I/O, chain detection, animation info)
- **FBX Loading:** `fbx_tool/analysis/fbx_loader.py` (scene loading, multi-stack ranking)
- **Scene Manager:** `fbx_tool/analysis/scene_manager.py` (reference-counted scene lifecycle)

## Recent Major Updates

### Session 2025-10-17: Scene Manager Architecture & Test Fixes

**CRITICAL: Scene management architecture completely refactored to use reference counting.**

#### What Changed

**1. New Scene Manager System** (`fbx_tool/analysis/scene_manager.py`)
- **Reference counting:** Tracks how many components need each scene
- **Automatic cleanup:** Destroys scenes when ref count hits 0
- **Thread-safe caching:** Concurrent access from GUI, visualizer, and analysis
- **Smart caching:** Visualizer keeps only current ± 1 files to prevent memory bloat

**2. GUI Integration** (`fbx_tool/gui/main_window.py`)
- Added `scene_manager` and `active_scene_refs` tracking
- `clearSelectedFiles()` releases all scene references (memory properly freed!)
- `launch_visualizer()` uses scene manager
- `AnalysisWorker` gets scenes from scene manager (cache hits!)
- All workers release references in `finally` blocks (error-safe)

**3. Visualizer Integration** (`fbx_tool/visualization/opengl_viewer.py`)
- Smart caching implementation: keeps current ± 1 files
- Prevents memory bloat when switching through large batches
- `_switch_to_file()` manages reference lifecycle
- `closeEvent()` releases all references

**4. Test Infrastructure** (ALL 10 INTEGRATION TESTS NOW PASSING!)
- **Unit tests:** `tests/unit/test_scene_manager.py` (22 tests, 83.33% coverage)
- **Integration tests:** `tests/integration/test_analysis_pipeline.py` (10 tests, 22.36% overall coverage)
- Fixed all mock setup issues (see `docs/testing/MOCK_SETUP_PATTERNS.md`)

**5. New Documentation**
- **[docs/architecture/SCENE_MANAGEMENT.md](docs/architecture/SCENE_MANAGEMENT.md)** - Complete architecture guide
- **[docs/testing/MOCK_SETUP_PATTERNS.md](docs/testing/MOCK_SETUP_PATTERNS.md)** - FBX SDK mocking patterns
- **[docs/README.md](docs/README.md)** - Documentation structure and navigation

#### Migration Guide

**OLD Pattern (Memory Leaks!):**
```python
# DON'T DO THIS ANYMORE!
scene, manager = load_fbx("file.fbx")
# ... use scene ...
cleanup_fbx_scene(scene, manager)  # Manual cleanup, easy to forget!
```

**NEW Pattern (Reference Counted):**
```python
# Use this pattern everywhere!
from fbx_tool.analysis.scene_manager import get_scene_manager

scene_manager = get_scene_manager()

# Option 1: Manual release
scene_ref = scene_manager.get_scene("file.fbx")
scene = scene_ref.scene
# ... use scene ...
scene_ref.release()  # Decrements ref count, auto-cleans if 0

# Option 2: Context manager (preferred)
with scene_manager.get_scene("file.fbx") as scene_ref:
    scene = scene_ref.scene
    # ... use scene ...
# Automatically released here
```

#### Key Benefits

1. **Scene Sharing:** Multiple components can hold references to same scene
2. **Memory Safety:** Automatic cleanup when last reference released
3. **No Memory Leaks:** Reference counting prevents forgotten cleanup
4. **Cache Hits:** Analysis reuses scenes already loaded by GUI/visualizer
5. **Smart Caching:** Visualizer limits memory usage for large batches
6. **Clear Button Works:** Properly frees memory when user clicks clear

#### Performance Impact

- **Before:** Walking animation loaded 3 times (GUI + Visualizer + Analysis) = 300 MB
- **After:** Loaded once, shared via references = 100 MB
- **Memory Savings:** 66-90% depending on workflow

#### Test Mock Fixes

Fixed 5 critical mock setup issues in integration tests:

1. **Missing `has_animation` key** → Added to all metadata mocks
2. **Mock `GetSrcObjectCount` returning Mock** → Returns integer now
3. **FbxTime class being mocked** → Removed `@patch("fbx.FbxTime")`, use real objects
4. **Transform matrix Get(i,j) not handled** → Use `side_effect = lambda i, j: matrix[i][j]`
5. **Off-by-one frame count** → Changed from 30 to 31 frames (total_frames = int(duration * frame_rate) + 1)

See `docs/testing/MOCK_SETUP_PATTERNS.md` for complete patterns.

#### Breaking Changes

⚠️ **IMPORTANT:** All analysis code must now use scene manager instead of direct `load_fbx()` calls.

**Migration required for:**
- Any code that calls `load_fbx()` directly
- Any code that calls `cleanup_fbx_scene()` directly
- Any code that stores scene objects (should store FBXSceneReference instead)

**Example migration:**
```python
# OLD
scene, manager = load_fbx("file.fbx")
result = analyze_something(scene)
cleanup_fbx_scene(scene, manager)

# NEW
from fbx_tool.analysis.scene_manager import get_scene_manager
scene_manager = get_scene_manager()
with scene_manager.get_scene("file.fbx") as scene_ref:
    result = analyze_something(scene_ref.scene)
```

#### Files Modified

**Core Implementation:**
- `fbx_tool/analysis/scene_manager.py` (NEW - 197 lines)
- `fbx_tool/gui/main_window.py` (scene manager integration)
- `fbx_tool/visualization/opengl_viewer.py` (smart caching)

**Tests:**
- `tests/unit/test_scene_manager.py` (NEW - 22 tests)
- `tests/integration/test_analysis_pipeline.py` (6 new tests + 4 fixed tests)
- `tests/unit/test_fbx_memory_management.py` (NEW - memory leak prevention tests)

**Documentation:**
- `docs/architecture/SCENE_MANAGEMENT.md` (NEW - complete architecture guide)
- `docs/testing/MOCK_SETUP_PATTERNS.md` (NEW - FBX SDK mock patterns)
- `docs/README.md` (NEW - documentation structure)

#### Test Results

```
✅ ALL 10 INTEGRATION TESTS PASSING
✅ 22 scene manager unit tests passing (83.33% coverage)
✅ 22.36% overall code coverage (exceeds 20% minimum)
✅ Root motion analysis: 100% coverage
✅ Directional change detection: 65.62% coverage
✅ Motion transition detection: 72.13% coverage
```

#### Next Steps (TODO)

1. ~~**Cache acceleration/jerk in trajectory extraction**~~ ✅ DONE (2025-10-18)
2. ~~**Replace hardcoded thresholds with adaptive learning**~~ ✅ IN PROGRESS (Motion states done, jitter/constraints pending)
3. **Overlay analysis data on visualizer** - Now possible with scene sharing
4. **Smart preloading** - Preload next ± 2 files in background

#### Critical Notes for Future Sessions

- **Scene manager is now MANDATORY** - All FBX loading must use it
- **Test mocks must follow patterns** - See `docs/testing/MOCK_SETUP_PATTERNS.md`
- **Reference counting is critical** - Always release references (use context managers!)
- **Smart caching prevents bloat** - But GUI refs keep scenes cached for fast analysis

---

## Session 2025-10-18: Procedural Threshold System

### Overview

Replaced hardcoded constants with adaptive, data-driven thresholds to make the system scale-invariant and work across any animation style, character size, or unit system.

### Problem Identified

User testing revealed pervasive issues with hardcoded thresholds:

1. **Motion State Detection:** Hardcoded `VELOCITY_IDLE_THRESHOLD = 5.0` failed on Mixamo animations (velocity ~326 units/sec)
2. **Segment Filtering:** Hardcoded `STATE_MIN_DURATION_FRAMES = 10` filtered out all segments in 23-frame animations
3. **State Flickering:** Percentile-based thresholds too close together caused rapid state changes
4. **Jitter Detection:** All 65 bones flagged as "high jitter" - threshold too sensitive
5. **Constraint Confidence:** Returned 1.0 score when 0 chains analyzed (misleading)
6. **Foot Sliding:** All contacts flagged as sliding - threshold issues

### Solutions Implemented

#### 1. Cached Derivatives (Performance Optimization)

**File:** `fbx_tool/analysis/utils.py:430-447`

```python
# BEFORE: Acceleration/jerk computed 3x per animation
# AFTER: Computed once, cached in trajectory dict
result = {
    "velocities": velocities,
    "accelerations": accelerations,  # NEW: Cached
    "jerks": jerks,                  # NEW: Cached
    ...
}
```

**Impact:** ~3x speedup for multi-analysis workflows

#### 2. Procedural Metadata Export System

**File:** `fbx_tool/analysis/utils.py:641-707`

Created JSON export system for discovered properties:
- Coordinate system detection results
- Adaptive thresholds computed from data
- Confidence scores for all detections
- AI integration readiness

**Output:** `procedural_metadata.json` in each output directory

#### 3. Adaptive Motion State Detection

**Files Modified:**
- `fbx_tool/analysis/motion_transition_detection.py:79-193`
- `fbx_tool/analysis/motion_transition_detection.py:513-570`

**Key Changes:**

**A. Removed Hardcoded Fallbacks (Lines 136-147)**
```python
# BEFORE: Used hardcoded constants in adaptive calculator
if median_vel < VELOCITY_IDLE_THRESHOLD:
    return {"idle": median_vel + 1.0, ...}

# AFTER: Purely data-driven
if velocity_range < 1.0 or velocity_std < 0.5:
    return {"idle": median_vel * 0.3, "walk": median_vel * 0.7, ...}
```

**B. Low-Variance Detection (Lines 165-193)**
```python
# Calculate coefficient of variation
cv = std_vel / mean_vel

# Detect single-state animations using two criteria:
# 1. Threshold span < 40% of range (tight clustering)
# 2. Coefficient of variation < 12% (low relative variance)
if threshold_span < velocity_range * 0.4 or cv < 0.12:
    # Set thresholds so ALL frames fall into same state
    min_vel = sorted_velocities[0]
    max_vel = sorted_velocities[-1]
    return {
        "idle": min_vel * 0.5,   # Well below minimum
        "walk": min_vel * 0.9,   # Just below minimum
        "run": max_vel * 1.1,    # Above maximum (all frames = running)
    }
```

**Example:** 23-frame run animation with CV=7.5% now classified as single "running" state instead of flickering between idle/walk/run/sprint.

**C. Adaptive Minimum Duration (Lines 513-537)**
```python
# BEFORE: Hardcoded 10 frames minimum
STATE_MIN_DURATION_FRAMES = 10

# AFTER: Percentage-based
if total_frames < 30:
    min_duration_frames = max(3, int(total_frames * 0.15))  # 15%
else:
    min_duration_frames = max(5, int(total_frames * 0.10))  # 10%
```

**Impact:** 23-frame animation now requires 3 frames (13%) instead of 10 frames (43%)

#### 4. Comprehensive Debug Logging

Added throughout motion state detection pipeline:
- State distribution per frame
- Adaptive thresholds computed
- Velocity range and coefficient of variation
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

### Documentation Created

1. **`docs/development/HARDCODED_CONSTANTS_AUDIT.md`**
   - Comprehensive audit of all hardcoded constants
   - Status tracking (Fixed/Partial/Not Fixed)
   - Priority levels (P0/P1/P2)
   - Impact assessment
   - Recommended fix order

2. **Updated:** `CLAUDE.md` (this file)

### Test Results

**Before Fixes:**
```
- Motion state segments: 0 detected (all filtered out)
- Classification: "varied_movement" (incorrect - flickering states)
- Segments: 3 detected from 23 frames (over-segmented)
```

**After Fixes:**
```
- Motion state segments: 1 detected ✅
- Classification: "run_cycle" (correct)
- Segments: 1 continuous running segment ✅
- CV: 0.075 (7.5% variance - correctly identified as single state)
```

### Files Modified

**Core Analysis:**
- `fbx_tool/analysis/utils.py` (cached derivatives, metadata export)
- `fbx_tool/analysis/motion_transition_detection.py` (adaptive thresholds, CV detection, min duration)
- `fbx_tool/analysis/root_motion_analysis.py` (metadata export integration)

**Documentation:**
- `docs/development/HARDCODED_CONSTANTS_AUDIT.md` (NEW)
- `CLAUDE.md` (updated)

### Remaining Work (Next Session)

**Priority 1 - High Impact:**
1. **Jitter Detection:** Make `JITTER_HIGH_THRESHOLD` and `JITTER_MEDIUM_THRESHOLD` adaptive (all 65 bones currently flagged)
2. **Constraint Confidence:** Fix misleading 1.0 score when 0 chains analyzed

**Priority 2 - Medium Impact:**
3. **Foot Sliding:** Audit and proceduralize thresholds in `foot_contact_analysis.py`
4. **Temporal Constants:** Make `STATE_STABLE_FRAMES`, `TRANSITION_JERK_SMOOTH`, etc. frame-rate aware

**Priority 3 - Future:**
5. Export adaptive thresholds to procedural_metadata.json for all analyses
6. Add confidence scores to all detections
7. Implement metadata caching for performance

### Critical Learnings

1. **Coefficient of Variation (CV) is key** - Better than absolute thresholds for detecting continuous vs. varied motion
2. **Percentage-based thresholds** - Must scale with animation length, not use fixed frame counts
3. **Use min/max for single-state thresholds** - Ensures ALL frames fall into intended category
4. **Debug logging essential** - Revealed issues that weren't obvious from final output
5. **Python bytecode caching** - Must clear `__pycache__` when making critical changes

### Breaking Changes

None - all changes are backward compatible with existing analysis pipeline.

### Next Session Checklist

- [ ] Test with variety of animations (walk, idle, jump, dance)
- [ ] Fix jitter threshold proceduralization
- [ ] Fix constraint confidence calculation
- [ ] Audit foot_contact_analysis.py
- [ ] Export all adaptive thresholds to metadata
- [ ] Add confidence scores to all detections