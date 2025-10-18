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

## Related Documentation

- `tests/integration/test_analysis_pipeline.py` - Reference integration tests
- `tests/unit/test_scene_manager.py` - Reference unit tests with mocks
- `tests/conftest.py` - Shared fixtures and mock utilities
- `docs/development/FBX_SDK_FIXES.md` - FBX SDK API patterns

## Lessons Learned

1. **Always match production code behavior** - Read the actual implementation to understand what needs to be mocked
2. **Use real objects when possible** - Don't mock FbxTime, use real fbx.FbxTime() objects
3. **side_effect for indexed access** - Use lambda functions for Get(i, j) style calls
4. **Count your frames** - Off-by-one errors are common with +1 calculations
5. **Test the tests** - If a test passes with placeholder code, it's not testing enough
6. **Mock at the right level** - Mock external dependencies, not internal helpers
7. **Document your mocks** - Add comments explaining why each mock is needed
