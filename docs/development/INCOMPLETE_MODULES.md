# Incomplete Module Analysis

> **⚠️ NOTE:** This document tracks historical incomplete module issues.
> **For current module status**, see **[docs/audits/MODULE_ERROR_LOGIC_AUDIT.md](../audits/MODULE_ERROR_LOGIC_AUDIT.md)**

Analysis of modules returning zero or placeholder results despite having data.

## Issue Summary

**Last Updated:** Session 2025-10-19c (Historical tracking - see audit for current status)

**MANY ISSUES FIXED** - Current Status:
- ✅ **velocity_analysis.py** - FIXED: Adaptive thresholds implemented, NaN handling added
- ✅ **gait_analysis.py** - FIXED: Algorithm bugs corrected (cycle rate, stride length, asymmetry)
- ✅ **root_motion_analysis.py** - FIXED: Compatible with procedural coordinate system
- ✅ **constraint_violation_detection.py** - FIXED: Proceduralized (58% coverage, 41 tests passing)
- ✅ **foot_contact_visualization** (opengl_viewer.py) - FIXED: Context-aware stuck bone detection (Session 2025-10-19a)
- ✅ **foot_contact_analysis.py** - FIXED: KeyError 'up_axis' resolved (Session 2025-10-19c)
- ✅ **directional_change_detection.py** - FIXED: KeyError 'angular_velocity_y' resolved (Session 2025-10-19c)
- ✅ **GUI batch processing** - FIXED: Add to Batch buttons now re-enable after analysis (Session 2025-10-19c)
- ⚠️ **pose_validity_analysis.py** - Needs shared hierarchy + adaptive thresholds
- ⚠️ **motion_transition_detection.py** - Needs gap-based thresholds + tests

**Historical Issues (many resolved):**

1. ✅ **Foot Contact Visualization** - FIXED! (Session 2025-10-19) Context-aware stuck bone detection
2. ✅ **Pose Validity Analysis** - FIXED! Now extracts bone data properly (40/40 tests passing)
3. ❌ **Constraint Violation Analysis** - Returns "0 chains analyzed" (TODO placeholders - needs TDD implementation)

---

## 1. Foot Contact Visualization

### Status: ✅ FIXED (Session 2025-10-19)

**File:** `fbx_tool/visualization/opengl_viewer.py`

### What Was Broken

**Problem:** Stuck bones (toe bases, toe ends) were incorrectly showing ground contact indicators
- Hardcoded threshold `abs(y) < 0.1` only worked if bones were exactly at origin
- No context awareness - didn't use actual ground height from scene (calculated at ~8-9 units)
- Bones stuck at Y=0.00 weren't being detected as stuck

**Debug Output Showed:**
```
[CACHE] Ground height calculated once: 8.73
...
Stuck bones (excluded from contact): set()  # ← EMPTY! Should contain stuck bones
mixamorig:LeftToeBase: Y=0.00
mixamorig:LeftToe_End: Y=0.00
```

### What Was Fixed

**Fix:** Context-aware stuck bone detection (lines 524-545)
```python
# BEFORE (BROKEN):
if all(abs(y) < 0.1 for y in all_y_values):  # Assumes Y=0
    stuck_bones.add(bone_name)

# AFTER (FIXED):
if all(y <= ground_height + 1.0 for y in all_y_values):  # Context-aware!
    stuck_bones.add(bone_name)
```

**Why This Works:**
- **Context-aware:** Uses actual `ground_height` (8-9 units) instead of assuming origin (0)
- **Adaptive:** Works regardless of scene scale or coordinate system
- **Tolerant:** Allows ±1.0 units of movement while still catching stuck bones
- **Data-driven:** No hardcoded assumptions about bone positions

### Test Results

**Comprehensive test coverage created:**
- 22 unit tests in `tests/unit/gui/test_foot_contact_visualization.py`
- 6 integration tests in `tests/integration/test_foot_contact_visualization_integration.py`
- 5 out of 6 integration tests passing (1 fails due to OpenGL mocking, not logic)
- Tests use realistic walking animation data with stuck bones at Y=0

**User Validation:** "Okay finally this is working as expected"

### Future Enhancement

**Proposed:** Adaptive stuck bone detection based on per-bone movement analysis
- Calculate stuck thresholds relative to typical bone movement in animation
- Use statistical measures (mean, std, CV) to classify "stuck" dynamically
- Handle edge cases: slow-motion, minimal-movement animations
- See CLAUDE.md "Next priorities" for details

---

## 1a. Foot Contact Analysis - KeyError 'up_axis'

### Status: ✅ FIXED (Session 2025-10-19c)

**File:** `fbx_tool/analysis/foot_contact_analysis.py` (lines 722-735)

### What Was Broken

**Problem:** Function relied on `extract_root_trajectory()` to return `coordinate_system` in its result, but was calling it incorrectly
- Code expected `coordinate_system` with `up_axis`, `forward_axis`, `right_axis` keys
- Missing coordinate system detection caused KeyError when accessing `coord_system["up_axis"]`

### What Was Fixed

**Fix:** Import and use `detect_full_coordinate_system()` directly (lines 722-735)
```python
# BEFORE (BROKEN):
trajectory = extract_root_trajectory(scene)
coord_system = trajectory["coordinate_system"]  # ← KeyError if missing

# AFTER (FIXED):
from fbx_tool.analysis.utils import detect_full_coordinate_system
trajectory = extract_root_trajectory(scene)
coord_system = detect_full_coordinate_system(scene)  # ✅ Explicit detection
up_axis = coord_system["up_axis"]
```

**Why This Works:**
- **Explicit detection:** Calls coordinate system detection function directly
- **No dependencies:** Doesn't rely on trajectory extraction to provide coordinate system
- **Robust:** Uses established utility function with proper error handling

---

## 1b. Directional Change Detection - KeyError 'angular_velocity_y'

### Status: ✅ FIXED (Session 2025-10-19c)

**File:** `fbx_tool/analysis/directional_change_detection.py` (line 363)

### What Was Broken

**Problem:** Function used obsolete field name `angular_velocity_y` instead of procedural name
- `extract_root_trajectory()` returns `angular_velocity_yaw` (procedural naming)
- Code still referenced old hardcoded axis name `angular_velocity_y`
- Caused KeyError when procedural coordinate system was Y-up (expected name was `angular_velocity_yaw`)

### What Was Fixed

**Fix:** Updated field name to match procedural naming (line 363)
```python
# BEFORE (BROKEN):
angular_velocity_y = trajectory["angular_velocity_y"]  # ← Old field name

# AFTER (FIXED):
angular_velocity_yaw = trajectory["angular_velocity_yaw"]  # ✅ Procedural name
```

**Why This Works:**
- **Consistent naming:** Matches procedural coordinate system convention
- **Works with any coordinate system:** `yaw` is axis-agnostic, not hardcoded to Y
- **Future-proof:** Compatible with Y-up, Z-up, or X-up coordinate systems

---

## 1c. Foot Contact Array Shape Mismatch

### Status: ✅ FIXED (Session 2025-10-19c)

**File:** `fbx_tool/analysis/foot_contact_analysis.py` (lines 487-496)

### What Was Broken

**Problem:** Obsolete array alignment code from `np.diff` era caused crashes
- Originally: `velocities = np.diff(positions)` created n-1 length array
- Updated to: `velocities = np.gradient(positions)` creates same-length array as positions
- But alignment code remained: `velocities = velocities[1:]` and `heights = heights[:-1]`
- This caused shape mismatches when both arrays were already aligned

### What Was Fixed

**Fix:** Removed obsolete alignment code (lines 487-496)
```python
# BEFORE (BROKEN):
velocities = np.gradient(positions[:, up_axis], frame_duration)
heights = positions[:, up_axis] - ground_height
velocities = velocities[1:]  # ← Obsolete! Already aligned
heights = heights[:-1]       # ← Obsolete! Already aligned

# AFTER (FIXED):
velocities = np.gradient(positions[:, up_axis], frame_duration)
heights = positions[:, up_axis] - ground_height
# Arrays are already aligned, no slicing needed
```

**Why This Works:**
- **np.gradient preserves length:** No need to align arrays anymore
- **Simpler code:** Removed unnecessary slicing operations
- **Correct shapes:** velocities and heights have same length (n frames)

---

## 1d. GUI Add to Batch Buttons Disabled After Analysis

### Status: ✅ FIXED (Session 2025-10-19c)

**File:** `fbx_tool/gui/main_window.py` (lines 1280-1283)

### What Was Broken

**Problem:** After running analysis, "Add Files to Batch" and "Add Recent to Batch" buttons remained disabled
- Analysis completion called `resetUI()` which should re-enable buttons
- But `resetUI()` only re-enabled main action buttons, not batch buttons
- This made it impossible to add more files to batch after analysis without restarting

### What Was Fixed

**Fix:** Added re-enabling of batch buttons in `resetUI()` (lines 1280-1283)
```python
# BEFORE (BROKEN):
def resetUI(self):
    self.run_analysis_btn.setEnabled(True)
    self.export_dopesheet_btn.setEnabled(True)
    # ... other buttons ...
    # ← Missing batch button re-enable

# AFTER (FIXED):
def resetUI(self):
    self.run_analysis_btn.setEnabled(True)
    self.export_dopesheet_btn.setEnabled(True)
    # ... other buttons ...
    self.add_files_btn.setEnabled(True)        # ✅ Re-enable add files
    self.add_recent_btn.setEnabled(True)       # ✅ Re-enable add recent
```

**Why This Works:**
- **Complete reset:** All interactive buttons now properly re-enabled
- **Batch workflow:** Users can continue adding files after analysis completes
- **User experience:** No need to restart application to use batch features

---

## 2. Pose Validity Analysis

### Symptoms
```
Analyzing pose validity (bone lengths, joint angles, intersections)...
  ✓ Pose validity analysis complete
    - 0 bones validated  ← PROBLEM
    - Overall validity score: 1.00
```

### Status: ✅ FIXED (Session 2025-10-18)

**File:** `fbx_tool/analysis/pose_validity_analysis.py`

### What Was Broken

**Problem 1:** Incorrect FBX SDK API usage in `_get_all_bones()`
```python
# ❌ WRONG: scene doesn't have FbxSkeleton attribute
if hasattr(scene, 'FbxSkeleton') and attr_type == scene.FbxSkeleton.eAttributeType:
    bones.append(node)
```

**Problem 2:** Placeholder implementation in `_extract_bone_animation_data()`
```python
# ❌ WRONG: Returns placeholder zeros
bone_data[bone_name] = {
    "positions": np.zeros((10, 3)),  # Placeholder
    "rotations": np.zeros((10, 3)),
    "parent_positions": None,
}
```

### What Was Fixed

**Fix 1:** Correct FBX SDK API
```python
# ✅ CORRECT: Use proper FBX SDK enum
import fbx as fbx_module
if attr_type == fbx_module.FbxNodeAttribute.EType.eSkeleton:
    bones.append(node)
```

**Fix 2:** Proper bone transform extraction
```python
# ✅ CORRECT: Extract transforms using EvaluateGlobalTransform
for frame in range(total_frames):
    current_time = start_time + frame_duration * frame
    global_transform = bone.EvaluateGlobalTransform(current_time)
    translation = global_transform.GetT()
    rotation = global_transform.GetR()
    positions.append([translation[0], translation[1], translation[2]])
    rotations.append([rotation[0], rotation[1], rotation[2]])
```

### Test Results

**All 40 unit tests passing:**
- ✅ Bone length validation (6 tests)
- ✅ Joint angle limits (6 tests)
- ✅ Self-intersection detection (4 tests)
- ✅ Symmetry validation (4 tests)
- ✅ Pose type detection (5 tests)
- ✅ Main analysis function (4 tests)
- ✅ Anatomical constraints (3 tests)
- ✅ Edge cases (5 tests)
- ✅ Reference length computation (2 tests)

**Module coverage:** 62.32% (up from ~0%)

### Implementation Details

The fix follows the established FBX SDK pattern used throughout the codebase:
1. Get animation stack and time span via `GetLocalTimeSpan()`
2. Calculate frame rate and total frames
3. Create `FbxTime` object for frame iteration
4. Use `EvaluateGlobalTransform(fbx_time)` to get transforms at each frame
5. Extract components with `.GetT()` (translation) and `.GetR()` (rotation)
6. Extract parent transforms for bone length analysis

---

## 3. Constraint Violation Analysis

### Symptoms
```
Analyzing constraint violations (IK chains, hierarchy, curves)...
  ✓ Constraint violation analysis complete
    - 0 chains analyzed  ← PROBLEM
    - Overall constraint score: 1.00
```

### Root Cause

**File:** `fbx_tool/analysis/constraint_violation_detection.py`

**Problem Code (lines 543-554):**
```python
# Analyze IK chains (simplified for now)
ik_violations = []
# TODO: Implement full IK chain analysis

# ... more code ...

# Analyze curve discontinuities (simplified)
curve_violations = []
# TODO: Implement full curve analysis
```

**Issues:**
1. Contains TODO placeholders
2. No actual implementation
3. Returns hardcoded empty results

**Required Implementation:**
- Detect IK chains from skeleton hierarchy
- Validate IK chain constraints (reach limits, pole vectors)
- Detect curve discontinuities (sudden jumps in animation curves)
- Validate hierarchy integrity

---

## Priority Fix Order

### ✅ Completed Fixes (Session 2025-10-18)
1. ✅ **Pose Validity - FBX SDK API** - Fixed `_get_all_bones()` to use correct API (fbx.FbxNodeAttribute.EType.eSkeleton)
2. ✅ **Pose Validity - Bone Animation Data** - Implemented `_extract_bone_animation_data()` with proper FbxTime usage (40/40 tests passing, 62% coverage)
3. ✅ **Motion State Detection** - Replaced hardcoded thresholds with adaptive, coefficient of variation-based detection
4. ✅ **Cached Derivatives** - Acceleration/jerk now cached in trajectory extraction (~3x speedup)
5. ✅ **Scene Manager** - Reference-counted scene management prevents memory leaks

### 🚧 Remaining Issues (Priority Order)

**Priority 1 - High Impact (Next Session):**
1. 📝 **Jitter Detection** - Make thresholds adaptive (currently all 65 bones flagged as "high jitter")
2. 📝 **Constraint Confidence** - Fix misleading 1.0 score when 0 chains analyzed
3. 📝 **Constraint Violation** - Implement IK chain detection (placeholder code exists)
4. 📝 **Constraint Violation** - Implement curve discontinuity detection (placeholder code exists)

**Priority 2 - Medium Impact:**
5. 🔍 **Foot Contact** - Debug ground height estimation (0 contacts detected - may be threshold issue)
6. 🔍 **Foot Sliding** - Review adaptive threshold implementation
7. 📝 **Temporal Constants** - Make frame-rate aware (`STATE_STABLE_FRAMES`, `TRANSITION_JERK_SMOOTH`)

**Priority 3 - Future Enhancements:**
8. 📝 Export adaptive thresholds to `procedural_metadata.json` for all analyses
9. 📝 Add confidence scores to all detections
10. 📝 Implement metadata caching for performance

**Note:** Per TDD methodology, all new implementations should have tests written FIRST.

---

## Verification Checklist

After fixes, verify with "Walking Backwards.fbx":

- [ ] Pose validity reports > 0 bones validated
- [ ] Foot contact reports > 0 contact events
- [ ] Validity scores are realistic (not always 1.00)
- [ ] Debug output shows actual analysis happening

---

## Related Files

- `fbx_tool/analysis/pose_validity_analysis.py`
- `fbx_tool/analysis/constraint_violation_detection.py`
- `fbx_tool/analysis/foot_contact_analysis.py`
- `docs/development/FBX_SDK_FIXES.md`
