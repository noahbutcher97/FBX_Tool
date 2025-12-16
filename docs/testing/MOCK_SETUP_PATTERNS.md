# Mock Setup Patterns for FBX SDK Testing

## Overview

This document provides correct patterns for mocking FBX SDK objects in tests. These patterns were developed through fixing failing integration tests and ensuring mocks accurately reflect production code behavior.

## Common Mock Setup Issues

### Issue 1: Missing Animation Metadata

**Error:** `ValueError: No animation data found in scene`

**Root Cause:** Mock metadata missing `has_animation: True` key.

**❌ WRONG:**
```python
mock_metadata.return_value = {
    "start_time": 0.0,
    "stop_time": 1.0,
    "frame_rate": 30.0,
}
```

**✅ CORRECT:**
```python
mock_metadata.return_value = {
    "has_animation": True,  # REQUIRED!
    "start_time": 0.0,
    "stop_time": 1.0,
    "frame_rate": 30.0,
    "duration": 1.0,
    "total_frames": 30,
}
```

### Issue 2: Animation Stack Access Not Mocked

**Error:** `TypeError: '>' not supported between instances of 'Mock' and 'int'`

**Root Cause:** `scene.GetSrcObjectCount()` returns Mock object, not integer.

**Production Code:**
```python
# fbx_tool/analysis/utils.py line 116
anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
if anim_stack_count > 0:  # Comparison fails if Mock object!
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
```

**❌ WRONG:**
```python
mock_scene = Mock()
# GetSrcObjectCount not mocked - returns Mock object!
```

**✅ CORRECT:**
```python
mock_scene = Mock()
mock_scene.GetSrcObjectCount.return_value = 1  # Return integer!
mock_scene.GetSrcObject.return_value = mock_anim_stack
```

### Issue 3: Mocking FbxTime Class

**Error:** `TypeError: float() argument must be a string or a real number, not 'Mock'`

**Root Cause:** Patching `fbx.FbxTime` class prevents creating real FbxTime objects.

**❌ WRONG:**
```python
@patch("fbx.FbxTime")  # This breaks real FbxTime creation!
def test_something(self, mock_fbx_time_class, ...):
    import fbx
    real_time = fbx.FbxTime()  # FAILS! fbx.FbxTime is now a Mock
    real_time.SetSecondDouble(0.0)  # ERROR!
```

**✅ CORRECT:**
```python
# Don't patch fbx.FbxTime at all - use real FbxTime objects
def test_something(self, ...):
    import fbx
    real_time = fbx.FbxTime()
    real_time.SetSecondDouble(0.0)  # Works!
```

### Issue 4: Transform Matrix Get(i, j) Calls

**Error:** `TypeError: float() argument must be a string or a real number, not 'Mock'`

**Root Cause:** Code calls `transform.Get(i, j)` to access individual matrix elements, but mock doesn't handle this.

**Production Code:**
```python
# fbx_tool/analysis/utils.py line 306
matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        matrix[i, j] = transform.Get(i, j)  # Indexed access!
```

**❌ WRONG:**
```python
# This doesn't handle Get(i, j) calls
matrix = Mock()
matrix.Get.return_value = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
transform.Get.return_value = matrix
```

**✅ CORRECT:**
```python
# Use side_effect to handle Get(i, j) calls
identity_matrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]
transform.Get.side_effect = lambda i, j: identity_matrix[i][j]
```

### Issue 5: Off-by-One Frame Count

**Error:** `IndexError: index 30 is out of bounds for axis 0 with size 30`

**Root Cause:** Production code calculates `total_frames = int(duration * frame_rate) + 1`, but mock creates arrays for `range(30)`.

**Production Code:**
```python
# fbx_tool/analysis/utils.py line 106
total_frames = int(duration * frame_rate) + 1
# With duration=1.0, frame_rate=30.0: total_frames = 31 (frames 0-30)
```

**❌ WRONG:**
```python
# Mock creates 30 frames, but code expects 31!
positions = np.array([[float(i), 0.0, 0.0] for i in range(30)])  # 0-29
velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(30)])
```

**✅ CORRECT:**
```python
# Create 31 frames to match production calculation
# Note: total_frames = int(duration * frame_rate) + 1 = int(1.0 * 30.0) + 1 = 31
positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])  # 0-30
velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
accelerations = np.zeros((31, 3))
jerks = np.zeros((31, 3))
```

## Complete Mock Setup Pattern

### Full Integration Test Setup

```python
import numpy as np
from unittest.mock import Mock, patch
import pytest

@pytest.mark.integration
class TestAnalysisPipeline:
    @patch("fbx_tool.analysis.utils._detect_root_bone")
    @patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
    @patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
    def test_complete_pipeline(self, mock_derivatives, mock_metadata, mock_detect):
        """Complete pattern for mocking FBX SDK in integration tests."""
        from fbx_tool.analysis.root_motion_analysis import analyze_root_motion
        from fbx_tool.analysis.utils import clear_trajectory_cache

        # 1. Clear cache before test
        clear_trajectory_cache()

        # 2. Setup scene mock
        mock_scene = Mock()
        mock_root_node = Mock()
        mock_scene.GetRootNode.return_value = mock_root_node

        # 3. Setup bone hierarchy
        hips_node = Mock()
        hips_node.GetName.return_value = "Hips"
        hips_node.GetChildCount.return_value = 0
        mock_root_node.GetChildCount.return_value = 1
        mock_root_node.GetChild.return_value = hips_node
        mock_detect.return_value = hips_node

        # 4. Setup metadata with has_animation
        mock_metadata.return_value = {
            "has_animation": True,  # REQUIRED!
            "start_time": 0.0,
            "stop_time": 1.0,
            "frame_rate": 30.0,
            "duration": 1.0,
            "total_frames": 30,
        }

        # 5. Setup time span with REAL FbxTime objects
        import fbx
        mock_time_span = Mock()

        real_start_time = fbx.FbxTime()  # Real FbxTime object
        real_start_time.SetSecondDouble(0.0)
        mock_time_span.GetStart.return_value = real_start_time

        duration_time = fbx.FbxTime()  # Real FbxTime object
        duration_time.SetSecondDouble(1.0)
        mock_time_span.GetDuration.return_value = duration_time

        # 6. Setup animation stack
        mock_anim_stack = Mock()
        mock_anim_stack.GetLocalTimeSpan.return_value = mock_time_span
        mock_scene.GetCurrentAnimationStack.return_value = mock_anim_stack

        # 7. Setup GetSrcObjectCount and GetSrcObject for animation stack access
        mock_scene.GetSrcObjectCount.return_value = 1  # Return INTEGER!
        mock_scene.GetSrcObject.return_value = mock_anim_stack

        # 8. Setup transform evaluations
        def create_mock_transform(frame):
            transform = Mock()
            transform.GetT.return_value = [float(frame), 0.0, 0.0]
            transform.GetR.return_value = [0.0, 0.0, 0.0]

            # Mock transformation matrix with side_effect
            identity_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
            transform.Get.side_effect = lambda i, j: identity_matrix[i][j]

            return transform

        hips_node.EvaluateGlobalTransform.side_effect = lambda time: create_mock_transform(0)

        # 9. Setup derivatives computation
        # Note: total_frames = int(duration * frame_rate) + 1 = 31
        positions = np.array([[float(i), 0.0, 0.0] for i in range(31)])
        velocities = np.array([[30.0 if i > 0 else 0.0, 0.0, 0.0] for i in range(31)])
        accelerations = np.zeros((31, 3))
        jerks = np.zeros((31, 3))
        mock_derivatives.return_value = (velocities, accelerations, jerks)

        # 10. Run test
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_root_motion(mock_scene, output_dir=tmpdir)

            # Assertions
            assert result is not None
            assert result["root_bone_name"] == "Hips"
            assert result["trajectory_frames"] == 31
```

## Mock Checklist

Use this checklist when setting up FBX SDK mocks:

- [ ] Metadata includes `has_animation: True`
- [ ] `scene.GetSrcObjectCount.return_value = 1` (integer, not Mock)
- [ ] `scene.GetSrcObject.return_value = mock_anim_stack`
- [ ] Use REAL `fbx.FbxTime()` objects (don't patch fbx.FbxTime)
- [ ] Transform matrix uses `side_effect` for Get(i, j) calls
- [ ] Array sizes match `total_frames` calculation (usually n+1)
- [ ] Clear trajectory cache before tests
- [ ] Mock `_detect_root_bone` to return a bone node
- [ ] Mock `compute_derivatives` to return tuples of np.arrays

## Debugging Mock Issues

### Strategy 1: Check What Production Code Expects

Read the production code to understand what it's trying to access:

```python
# Example: utils.py line 116
anim_stack_count = scene.GetSrcObjectCount(...)
if anim_stack_count > 0:  # This comparison tells us what to mock
```

Mock should return integer: `mock_scene.GetSrcObjectCount.return_value = 1`

### Strategy 2: Use `-vv` for Verbose Test Output

```bash
pytest tests/integration/test_analysis_pipeline.py::test_name -vv
```

This shows exactly which line is failing and what the error is.

### Strategy 3: Check Array Shapes

When you get IndexError, verify array shapes match expected frame counts:

```python
# In test
print(f"Positions shape: {positions.shape}")  # Should be (31, 3) not (30, 3)
print(f"Total frames: {metadata['total_frames']}")
```

### Strategy 4: Verify Mock Call Patterns

Use `assert_called_once()` to verify mocks are being used:

```python
def test_something(self, mock_load_fbx):
    # ... run test ...

    mock_load_fbx.assert_called_once()
    call_args = mock_load_fbx.call_args
    print(f"Called with: {call_args}")
```

## Common Patterns by Module

### Testing Root Motion Analysis

```python
# Requires: scene, metadata, detect_root_bone, compute_derivatives
@patch("fbx_tool.analysis.utils._detect_root_bone")
@patch("fbx_tool.analysis.fbx_loader.get_scene_metadata")
@patch("fbx_tool.analysis.velocity_analysis.compute_derivatives")
def test_root_motion(self, mock_derivatives, mock_metadata, mock_detect):
    # See complete pattern above
```

### Testing Scene Manager

```python
# Requires: FBX SDK mocks for load_fbx
@patch("fbx_tool.analysis.fbx_loader.fbx")
def test_scene_manager(self, mock_fbx_module):
    mock_manager = Mock()
    mock_scene = Mock()

    mock_fbx_module.FbxManager.Create.return_value = mock_manager
    mock_fbx_module.FbxScene.Create.return_value = mock_scene
    # ... more setup ...
```

### Testing GUI Worker

```python
# Requires: Scene manager and metadata mocks
@patch("fbx_tool.analysis.fbx_loader.fbx")
@patch("fbx_tool.gui.main_window.get_scene_metadata")
def test_worker(self, mock_metadata, mock_fbx_module):
    from fbx_tool.gui.main_window import AnalysisWorker

    mock_metadata.return_value = {
        "duration": 1.0,
        "frame_rate": 30.0,
        "bone_count": 10,
    }
    # ... more setup ...
```

## Issue 6: PyQt6 Widget Mocking and Skip Strategy

**Context:** GUI tests for `SkeletonGLWidget` fail when PyQt6 is mocked because the widget becomes a `MagicMock` instead of a real class.

**Root Cause:** Widget `__init__` does substantial work:
- Calls `get_scene_metadata(scene)` to extract animation info
- Calls `build_bone_hierarchy(scene)` to build skeleton
- Calls `detect_full_coordinate_system()` for axis detection
- Calls `_extract_transforms()` which requires extensive scene node mocking

**Problem:** Testing widget methods requires:
```python
# Widget becomes MagicMock when PyQt6 is mocked
widget = SkeletonGLWidget(scene)  # This is now a MagicMock, not real widget
descendants = widget._get_bone_descendants("Foot")  # Returns MagicMock, not list
```

**Attempted Solutions:**

1. **Fake Base Classes** - Created `FakeQOpenGLWidget` and `FakeQTimer` to allow real widget instantiation
   - **Result:** Widget init got further but failed in `_extract_transforms()`
   - **Issue:** Requires mocking `scene.FindNodeByName()` for every bone with proper return values

2. **Deep Scene Mocking** - Mock entire scene hierarchy with all bones and transforms
   - **Result:** Test becomes complex duplication of widget initialization logic
   - **Issue:** Maintenance burden, brittle tests

**✅ RECOMMENDED SOLUTION: Use pytest-qt**

Add pytest-qt as a dependency and use real Qt fixtures:

```python
# In pyproject.toml or requirements-test.txt
pytest-qt>=4.2.0

# In test file - remove PyQt6 mocking, use real Qt
def test_get_bone_descendants(qtbot):
    """Test with real Qt widget using pytest-qt."""
    scene = Mock()
    # ... mock only FBX SDK parts, not Qt ...

    widget = SkeletonGLWidget(scene)  # Real widget!
    qtbot.addWidget(widget)  # pytest-qt manages lifecycle

    descendants = widget._get_bone_descendants("Foot")  # Real method call!
    assert isinstance(descendants, list)
    assert len(descendants) == 3
```

**Alternative: Skip Tests Until pytest-qt Added**

For tests that require real widget methods, use skip decorator:

```python
@pytest.mark.skip(
    reason="Requires real widget methods - SkeletonGLWidget becomes MagicMock when PyQt6 is mocked. "
    "Widget.__init__ calls _extract_transforms() which requires extensive mock scene setup. "
    "Alternative: Use pytest-qt or refactor widget to separate testable logic from Qt initialization."
)
def test_get_bone_descendants_full_hierarchy(self, widget_with_mocks):
    # Test remains as documentation of expected behavior
    # Enable when pytest-qt is added or widget is refactored
```

**When to Skip vs. When to Mock:**

| Test Type | Strategy |
|-----------|----------|
| Widget initialization | Skip (or use pytest-qt) |
| Widget method logic | Skip (or use pytest-qt) |
| Non-Qt helper functions | Mock normally |
| Stuck bone detection logic | Can test without real widget (no Qt calls) |
| Contact state calculation | Can test without real widget (pure logic) |

**Current Status:**
- 11 GUI tests skipped in `tests/unit/gui/test_foot_contact_visualization.py`
- Tests remain as documentation of expected behavior
- 11 other GUI tests pass (test pure logic without widget instantiation)

## Issue 7: Patch Location for Functions Imported Inside Other Functions

**Error:** `AttributeError: <module 'X'> does not have the attribute 'Y'`

**Root Cause:** When a function imports another function inside itself (not at module level), you must patch at the **source module**, not the import location.

**Production Code Example:**
```python
# fbx_tool/gui/main_window.py
class AnalysisWorker:
    def run(self):
        # Import inside method!
        from fbx_tool.analysis.scene_manager import get_scene_manager
        scene_mgr = get_scene_manager()
```

**❌ WRONG:**
```python
# Patching at import location fails
@patch("fbx_tool.gui.main_window.get_scene_manager")
def test_worker(mock_get_scene_mgr):
    # ERROR: main_window doesn't have get_scene_manager attribute
```

**✅ CORRECT:**
```python
# Patch at source module
@patch("fbx_tool.analysis.scene_manager.get_scene_manager")
def test_worker(mock_get_scene_mgr):
    # Works! Patches where the function is defined
```

**Rule of Thumb:** Always patch at the module where the function/class is **defined**, not where it's **imported**.

## Issue 8: Scene Manager Pattern in Tests

**Context:** After refactoring from direct FBX loading to scene manager pattern, tests need updating.

**Old Pattern (Direct Loading):**
```python
@patch("fbx_tool.analysis.fbx_loader.load_fbx")
def test_worker(mock_load):
    mock_scene = Mock()
    mock_manager = Mock()
    mock_load.return_value = (mock_scene, mock_manager)

    worker.run()

    # Old cleanup pattern
    mock_manager.Destroy.assert_called_once()
```

**New Pattern (Scene Manager):**
```python
@patch("fbx_tool.analysis.scene_manager.get_scene_manager")
@patch("fbx_tool.gui.main_window.get_scene_metadata")
def test_worker(mock_metadata, mock_get_scene_mgr):
    # Setup scene reference mock
    mock_scene = Mock()
    mock_scene_ref = Mock()
    mock_scene_ref.scene = mock_scene

    # Setup scene manager mock
    mock_scene_manager = Mock()
    mock_scene_manager.get_scene.return_value = mock_scene_ref
    mock_get_scene_mgr.return_value = mock_scene_manager

    mock_metadata.return_value = {
        "duration": 1.0,
        "frame_rate": 30.0,
        "bone_count": 10
    }

    worker.run()

    # Verify scene reference was released (new cleanup pattern)
    mock_scene_ref.release.assert_called_once()
```

**Key Changes:**
1. Mock `get_scene_manager()` instead of `load_fbx()`
2. Create `mock_scene_ref` with `.scene` attribute
3. Scene manager returns reference via `get_scene()`
4. Verify `.release()` called instead of `manager.Destroy()`

## Issue 9: Array Length Mismatches (np.diff vs np.gradient)

**Error:** `ValueError: operands could not be broadcast together with shapes (7,) (6,)`

**Root Cause:** Tests used `np.diff()` which produces n-1 length arrays, but production code uses `np.gradient()` which maintains n length.

**Production Code:**
```python
# fbx_tool/analysis/velocity_analysis.py line 239
velocity = np.gradient(positions, dt, axis=0)  # Same length as positions

# fbx_tool/analysis/foot_contact_analysis.py lines 484-489
velocity_mags = np.linalg.norm(velocities, axis=1)
heights = positions[:, up_axis]  # Both same length
contact_mask = (heights < threshold) & (velocity_mags < threshold)
```

**❌ WRONG (Test):**
```python
positions = np.array([[0, 20, 0], [0, 10, 0], [0, 0, 0]])  # Shape: (3, 3)
velocities = np.diff(positions, axis=0)  # Shape: (2, 3) - ONE SHORTER!

# Later comparison fails:
# heights (length 3) vs velocity_mags (length 2)
```

**✅ CORRECT (Test):**
```python
positions = np.array([[0, 20, 0], [0, 10, 0], [0, 0, 0]])  # Shape: (3, 3)
velocities = np.gradient(positions, axis=0)  # Shape: (3, 3) - SAME LENGTH!

# Comparison works:
# heights (length 3) vs velocity_mags (length 3)
```

**When to Use Each:**
- `np.gradient()` - When you need same-length output (velocity from position)
- `np.diff()` - When you explicitly want differences between consecutive elements

## GUI Testing with pytest-qt

### Pattern: Testing Qt Widgets

**Use pytest-qt's `qtbot` fixture** for all Qt widget tests. Never mock PyQt6 at module level.

#### ❌ WRONG: Module-Level PyQt6 Mocking

```python
# WRONG: Makes SkeletonGLWidget become MagicMock
import sys
from unittest.mock import MagicMock
sys.modules["PyQt6"] = MagicMock()
sys.modules["PyQt6.QtCore"] = MagicMock()
# ... etc

from fbx_tool.visualization.opengl_viewer import SkeletonGLWidget  # noqa: E402
```

**Problems:**
- Widget classes become MagicMock, can't test real behavior
- Tests crash in parallel execution (pytest-xdist)
- Requires extensive skip decorators

#### ✅ CORRECT: pytest-qt with Selective Patching

```python
# CORRECT: Real PyQt6, pytest-qt for lifecycle management
from unittest.mock import Mock, patch
import pytest
from fbx_tool.visualization.opengl_viewer import SkeletonGLWidget

@pytest.fixture
def widget_with_mocks(qtbot, mock_scene):
    """Create widget with mocked FBX dependencies."""
    with (
        patch("fbx_tool.visualization.opengl_viewer.get_scene_metadata") as mock_metadata,
        patch("fbx_tool.visualization.opengl_viewer.build_bone_hierarchy") as mock_hierarchy,
        patch("fbx_tool.visualization.opengl_viewer.detect_full_coordinate_system") as mock_detect,
        patch.object(SkeletonGLWidget, "_extract_transforms"),  # Prevent FBX extraction
    ):
        mock_metadata.return_value = {"start_time": 0.0, "stop_time": 1.0, ...}
        mock_hierarchy.return_value = {"Root": None}
        mock_detect.return_value = {"up_axis": 1, "confidence": 0.95}

        widget = SkeletonGLWidget(mock_scene)
        qtbot.addWidget(widget)  # CRITICAL: Registers with Qt application context

        # Manually set coord_system since _extract_transforms is mocked
        widget.coord_system = mock_detect.return_value

        return widget

def test_widget_behavior(widget_with_mocks):
    """Test widget behavior with mocked FBX backend."""
    assert widget_with_mocks.up_axis == 1
    assert widget_with_mocks.coord_system["confidence"] == 0.95
```

#### Pattern: Inline Widget Creation

For tests that create widgets directly (not using fixture):

```python
def test_widget_initialization(qtbot):  # qtbot parameter required
    """Test widget with inline creation."""
    with (
        patch("fbx_tool.visualization.opengl_viewer.get_scene_metadata"),
        patch("fbx_tool.visualization.opengl_viewer.build_bone_hierarchy"),
        patch.object(SkeletonGLWidget, "_extract_transforms"),  # Prevent FBX extraction
    ):
        scene = Mock()
        scene.GetGlobalSettings().GetAxisSystem().GetUpVector.return_value = (1, 1)

        widget = SkeletonGLWidget(scene)
        qtbot.addWidget(widget)  # CRITICAL: Required for Qt context

        # Test widget behavior
        assert widget is not None
```

### Why qtbot.addWidget() Is Critical

`qtbot.addWidget(widget)` does more than widget lifecycle management:

1. **Registers widget with QApplication context**
   - Tests pass in serial execution without it
   - Tests crash with "worker crashed" in parallel execution (pytest-xdist)
   - Qt widgets need proper application context to prevent cross-worker interference

2. **Automatic cleanup**
   - Widget properly destroyed after test
   - No memory leaks between tests

3. **Event loop integration**
   - Qt signals/slots work correctly
   - Timer events processed properly

### When to Patch _extract_transforms

Widget's `__init__` calls `_extract_transforms()` which:
- Iterates through all animation frames
- Calls `scene.FindNodeByName()` for every bone
- Accesses `node.EvaluateGlobalTransform()` and `transform.GetT()`
- Requires extensive FBX scene mocking

**Solution:** Patch `_extract_transforms` and manually set required attributes:

```python
with patch.object(SkeletonGLWidget, "_extract_transforms"):
    widget = SkeletonGLWidget(scene)
    qtbot.addWidget(widget)

    # Manually set attributes that _extract_transforms would have set
    widget.bone_transforms = {...}
    widget.coord_system = {...}
    widget.total_frames = 30
```

## Related Documentation

- `tests/integration/test_analysis_pipeline.py` - Reference integration tests
- `tests/unit/test_scene_manager.py` - Reference unit tests with mocks
- `tests/unit/gui/test_foot_contact_visualization.py` - Reference pytest-qt GUI tests
- `tests/conftest.py` - Shared fixtures and mock utilities
- `docs/development/FBX_SDK_FIXES.md` - FBX SDK API patterns
- `docs/architecture/SCENE_MANAGEMENT.md` - Scene manager architecture

## Lessons Learned

1. **Always match production code behavior** - Read the actual implementation to understand what needs to be mocked
2. **Use real objects when possible** - Don't mock FbxTime, use real fbx.FbxTime() objects
3. **side_effect for indexed access** - Use lambda functions for Get(i, j) style calls
4. **Count your frames** - Off-by-one errors are common with +1 calculations
5. **Test the tests** - If a test passes with placeholder code, it's not testing enough
6. **Mock at the right level** - Mock external dependencies, not internal helpers
7. **Document your mocks** - Add comments explaining why each mock is needed
8. **Know when to skip** - When mocking becomes more complex than the code being tested, skip and document alternatives
9. **Patch at source** - Always patch where functions are defined, not where they're imported
10. **Array operations matter** - Match production code's choice of np.gradient vs np.diff
11. **Scene manager pattern** - Mock get_scene_manager(), not load_fbx(); verify release(), not Destroy()
12. **pytest-qt for GUI tests** - Use qtbot.addWidget() for all Qt widgets; never mock PyQt6 at module level
13. **Test behavior not implementation** - Avoid testing "method A calls method B with parameters X"; test outcomes instead
