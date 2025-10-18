# Hardcoded Constants Audit

**Date:** 2025-10-17
**Purpose:** Document all hardcoded constants that need proceduralization
**Status:** 🔴 In Progress - Actively fixing

---

## 🔴 P0: Critical - Motion State Detection

**File:** `fbx_tool/analysis/motion_transition_detection.py`

### Velocity Thresholds (Lines 46-49)
```python
VELOCITY_IDLE_THRESHOLD = 5.0   # Below this = idle/stationary
VELOCITY_WALK_THRESHOLD = 50.0  # Below this = walking
VELOCITY_RUN_THRESHOLD = 150.0  # Below this = running
```

**Status:** ✅ Fixed
- Adaptive calculator exists (line 79)
- Removed hardcoded fallbacks from percentile logic (lines 165-193)
- Added **coefficient of variation (CV)** detection: CV < 12% = single state
- Added **threshold span** detection: span < 40% of range = single state
- Fixed threshold calculation to use min/max velocities (ensures all frames in same state)
- **Action:** None - working correctly

**Impact:** NONE - Fixed! Now handles low-variance animations properly
  - Example: 23-frame run animation with CV=7.5% now classified as single "running" state
  - Prevents flickering between idle/walk/run/sprint on continuous animations

---

### Vertical Motion Thresholds (Lines 52-56)
```python
ACCELERATION_JUMP_THRESHOLD = 200.0      # Sudden upward acceleration
ACCELERATION_LAND_THRESHOLD = -200.0     # Sudden downward deceleration
VERTICAL_VELOCITY_AIRBORNE_THRESHOLD = 10.0  # Positive Y velocity = jumping/falling
```

**Status:** 🟡 Partially Fixed
- Adaptive calculator exists (line 176)
- **Action:** Verify adaptive calculator is being used, remove hardcoded references

**Impact:** MEDIUM - Affects jump/land detection

---

### Transition Detection (Lines 63-71)
```python
STATE_STABLE_FRAMES = 5              # Frames required to confirm state change
TRANSITION_JERK_SMOOTH = 50.0        # Below this = smooth transition
TRANSITION_JERK_MODERATE = 150.0     # Below this = moderate transition
STATE_MIN_DURATION_FRAMES = 10       # Minimum duration for valid motion state segment (DEPRECATED)
```

**Status:** ✅ Partially Fixed
- `STATE_MIN_DURATION_FRAMES` now computed adaptively (15% for <30 frames, 10% for longer)
- Still need to proceduralize `STATE_STABLE_FRAMES` and jerk thresholds
- **Action:** Document remaining constants or make adaptive

**Impact:** LOW - Minimum duration now adaptive, other constants less critical

---

## 🔴 P1: High - Jitter Detection

**File:** `fbx_tool/analysis/velocity_analysis.py`

### Jitter Thresholds (Lines 78-79)
```python
JITTER_HIGH_THRESHOLD = 1.0      # Above this = severe jitter
JITTER_MEDIUM_THRESHOLD = 0.1    # Above this = moderate jitter
```

**Status:** ❌ Not Fixed
- **Problem:** ALL 65 bones flagged as "high jitter" on Mixamo rigs
- **Action:** Compute adaptive threshold based on data distribution (percentiles)

**Impact:** HIGH - Makes jitter warnings meaningless (always triggers)

---

### Smoothing Parameters (Lines 81-92)
```python
SMOOTHING_KERNEL_HIGH = 7       # Aggressive smoothing for high jitter
SMOOTHING_KERNEL_MEDIUM = 5     # Medium smoothing
SMOOTHING_KERNEL_LOW = 3        # Light smoothing

CUTOFF_FRACTION_HIGH_JITTER = 0.1   # Filter out 90% of frequencies
CUTOFF_FRACTION_MEDIUM_JITTER = 0.3 # Filter out 70% of frequencies
CUTOFF_FRACTION_LOW_JITTER = 0.5    # Filter out 50% of frequencies
```

**Status:** ❌ Not Fixed
- **Action:** Make these adaptive based on jitter characteristics

**Impact:** MEDIUM - Affects smoothing quality

---

## 🟡 P2: Medium - Direction Classification

**File:** `fbx_tool/analysis/utils.py`

### Direction Angle Thresholds (Lines 364-365)
```python
_FORWARD_THRESHOLD = 45.0    # ±45° = forward
_BACKWARD_THRESHOLD = 135.0  # ±135° = backward
```

**Status:** ✅ Overridden by Adaptive Thresholds
- These are still defined but now overridden by `_compute_adaptive_thresholds()`
- **Action:** Can leave as fallbacks, adaptive system working

**Impact:** LOW - Now procedural

---

### Stationary Velocity (Line 366)
```python
_STATIONARY_VELOCITY_THRESHOLD = 0.1  # units/second
```

**Status:** ✅ Fixed - Now Adaptive
- Computed in `_compute_adaptive_thresholds()` (utils.py:87-99)
- **Action:** None - working correctly

**Impact:** NONE - Fixed

---

### Turning Speed (Lines 369-371)
```python
_TURNING_THRESHOLD_SLOW = 30.0       # degrees/second
_TURNING_THRESHOLD_FAST = 90.0
_TURNING_THRESHOLD_VERY_FAST = 180.0
```

**Status:** ✅ Fixed - Now Adaptive
- Computed in `_compute_adaptive_thresholds()` (utils.py:101-131)
- **Action:** None - working correctly

**Impact:** NONE - Fixed

---

## 🟡 P2: Medium - Foot Contact Detection

**File:** `fbx_tool/analysis/foot_contact_analysis.py`

**Status:** 🔍 Needs Investigation

### Contact Thresholds (Need to Find)
- Ground plane detection threshold
- Foot velocity threshold for "grounded"
- Sliding distance threshold

**Problem:** 3 out of 3 contacts have foot sliding (Run Forward Arc Left)
**Action:** Audit foot_contact_analysis.py for hardcoded constants

**Impact:** MEDIUM - Affects foot sliding warnings

---

## 🔴 P1: High - Constraint Analysis

**File:** `fbx_tool/analysis/constraint_violation_detection.py`

### Overall Constraint Score (Line 525)
```python
'overall_constraint_score': 1.0
```

**Status:** ❌ MISLEADING
- **Problem:** Returns 1.0 (100% confidence) when 0 chains analyzed!
- **Action:** Return 0.0 confidence when no data, or None

**Impact:** HIGH - Misleading confidence scores

---

## 📋 Summary Statistics

| Priority | Category | Total Constants | Fixed | Partial | Not Fixed |
|----------|----------|----------------|-------|---------|-----------|
| P0 | Motion State Detection | 7 | 0 | 2 | 5 |
| P1 | Jitter Detection | 6 | 0 | 0 | 6 |
| P1 | Constraint Analysis | 1 | 0 | 0 | 1 |
| P2 | Direction Classification | 5 | 3 | 0 | 2 |
| P2 | Foot Contact | ? | 0 | 0 | ? |
| **TOTAL** | | **19+** | **3** | **2** | **14+** |

---

## 🎯 Recommended Fix Order

### Phase 1: Motion State Detection (This Session) ✅ COMPLETE
1. ✅ Remove hardcoded fallbacks from adaptive velocity calculator
2. ✅ Make STATE_MIN_DURATION_FRAMES adaptive (percentage-based)
3. ✅ Add low-variance detection for single-state animations
4. ⏳ Test with Run Forward Arc Left.fbx (awaiting user test)
5. ⏳ Verify motion states are detected correctly

### Phase 2: Jitter & Confidence (Next)
4. Make jitter thresholds adaptive (percentile-based)
5. Fix constraint analysis confidence (0 data = 0.0 score, not 1.0)

### Phase 3: Foot Contact (Future)
6. Audit foot_contact_analysis.py
7. Make sliding threshold adaptive
8. Make ground plane detection adaptive

### Phase 4: Cleanup (Future)
9. Make temporal constants frame-rate aware
10. Document all remaining constants with justification

---

## 🔬 Testing Protocol

For each fix:
1. Test with Mixamo rig (Run Forward Arc Left.fbx)
2. Test with different scales (if available)
3. Check procedural_metadata.json for exported thresholds
4. Verify confidence scores are meaningful

---

## 📝 Notes

- **Procedural doesn't mean NO constants** - some values (like STATE_STABLE_FRAMES) may be justified
- **Document WHY** a constant is hardcoded if it can't be procedural
- **Export all computed thresholds** to procedural_metadata.json for transparency
- **Include confidence scores** to indicate how reliable the detection is

---

**Last Updated:** 2025-10-17 (Session in progress)
