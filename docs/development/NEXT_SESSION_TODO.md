# Next Session TODO

**Last Updated:** 2025-10-18
**Session:** Procedural Threshold System Implementation

---

## 🔴 URGENT - Must Test First

### Verify Motion State Detection Fix

**Action:** Run analysis on Run Forward Arc Left.fbx after cache clear

**Expected Output:**
```
🔬 Velocity range: 77.3, threshold span: 49.9 (64.5% of range), CV: 0.075
⚠️  Low variance detected (CV=0.075) - classifying all as single state
Adaptive velocity thresholds: idle=124.5, walk=224.0, run=358.7 units/sec
📊 State distribution: {'running': 23}  ← Should be ALL running, not mixed
✅ Created 1 motion state segments
```

**Last Known Issue:** CV detection triggering but thresholds still allowing mixed states (sprinting:1, running:18, walking:4)

**Fix Applied:** Changed low-variance thresholds from median-based to min/max-based
- Old: `idle=median*0.5, walk=median*0.9, run=median*1.1`
- New: `idle=min*0.5, walk=min*0.9, run=max*1.1`

**Cache Cleared:** `fbx_tool/analysis/__pycache__/motion_transition_detection.cpython-310.pyc` deleted

**Next Step:** Run `python -m fbx_tool` and verify state distribution shows `{'running': 23}`

---

## 🎯 Priority 1 - High Impact (Next Tasks)

### 1. Jitter Detection - Make Adaptive

**File:** `fbx_tool/analysis/velocity_analysis.py:78-92`

**Current Problem:**
```
✓ Velocity analysis complete:
  - 66 jerk spikes detected
  - ALL 65 BONES flagged as high jitter ← BROKEN
```

**Hardcoded Constants:**
```python
JITTER_HIGH_THRESHOLD = 1.0      # Above this = severe jitter
JITTER_MEDIUM_THRESHOLD = 0.1    # Above this = moderate jitter

# Smoothing parameters
SMOOTHING_KERNEL_HIGH = 7
SMOOTHING_KERNEL_MEDIUM = 5
SMOOTHING_KERNEL_LOW = 3

CUTOFF_FRACTION_HIGH_JITTER = 0.1
CUTOFF_FRACTION_MEDIUM_JITTER = 0.3
CUTOFF_FRACTION_LOW_JITTER = 0.5
```

**Recommended Fix:**
1. Compute jitter distribution across all bones (percentile-based)
2. Use 75th and 95th percentiles as thresholds instead of hardcoded values
3. Make smoothing kernel size adaptive based on animation frame rate
4. Add coefficient of variation check similar to motion states

**Expected Result:**
- Only bones with OUTLIER jitter flagged (top 5-10%)
- Jitter warnings meaningful and actionable

---

### 2. Constraint Analysis Confidence - Fix Misleading Score

**File:** `fbx_tool/analysis/constraint_violation_detection.py:525`

**Current Problem:**
```python
'overall_constraint_score': 1.0  # Returns 100% confidence with 0 chains analyzed!
```

**Observed Behavior:**
```
⚠ No chains being analyzed for constraint violations
✓ Confidence: 1.00 ← MISLEADING! Should be 0.0 or None
```

**Recommended Fix:**
```python
if total_chains_analyzed == 0:
    return {
        'overall_constraint_score': 0.0,  # No data = no confidence
        'confidence': 0.0,
        'note': 'No chains analyzed - insufficient data'
    }
```

**Impact:** HIGH - Users trust confidence scores for decision making

---

## 🟡 Priority 2 - Medium Impact

### 3. Foot Contact Analysis - Audit Hardcoded Thresholds

**File:** `fbx_tool/analysis/foot_contact_analysis.py`

**Current Problem:**
```
✓ Foot contact analysis complete:
  - 3 contact events analyzed
  - 3 sliding events detected  ← All contacts sliding (100%)
```

**Action Required:**
1. Search for hardcoded thresholds:
   - Ground plane detection threshold
   - Foot velocity threshold for "grounded"
   - Sliding distance threshold
2. Replace with percentile-based or adaptive calculations
3. Add debug logging similar to motion states

**Expected Result:**
- Only actual sliding events flagged
- Sliding warnings meaningful and actionable

---

### 4. Temporal Constants - Make Frame-Rate Aware

**File:** `fbx_tool/analysis/motion_transition_detection.py:63-71`

**Current Constants:**
```python
STATE_STABLE_FRAMES = 5              # Frames required to confirm state change
TRANSITION_JERK_SMOOTH = 50.0        # Below this = smooth transition
TRANSITION_JERK_MODERATE = 150.0     # Below this = moderate transition
```

**Recommended Fix:**
1. Make `STATE_STABLE_FRAMES` scale with frame rate:
   ```python
   state_stable_frames = max(3, int(frame_rate / 6))  # ~0.5 seconds at 30fps
   ```
2. Make jerk thresholds adaptive using percentile approach
3. Document any constants that MUST remain hardcoded with justification

---

## 🔵 Priority 3 - Future Enhancements

### 5. Export All Adaptive Thresholds to Metadata

**Goal:** Make procedural_metadata.json comprehensive

**Currently Exported:**
- Coordinate system detection
- Adaptive direction/turning thresholds

**To Add:**
- Motion state velocity thresholds (idle/walk/run)
- Jitter thresholds (when fixed)
- Foot contact thresholds (when fixed)
- Minimum duration calculations
- All confidence scores

**Benefit:** AI integration, debugging, reproducibility

---

### 6. Add Confidence Scores to All Detections

**Pattern to Follow:**
```python
{
    "detection_type": "motion_state",
    "result": "running",
    "confidence": 0.95,
    "method": "coefficient_of_variation",
    "cv": 0.075,
    "threshold": 0.12
}
```

**Apply to:**
- Foot contact detection
- Jitter detection
- Constraint violations
- All existing detections

---

## 📋 Documentation Organization

### Current Structure
```
docs/
├── development/
│   ├── HARDCODED_CONSTANTS_AUDIT.md  ✅ Created this session
│   └── NEXT_SESSION_TODO.md          ✅ This file
├── testing/
│   └── MOCK_SETUP_PATTERNS.md
├── architecture/
│   └── SCENE_MANAGEMENT.md
└── README.md

CLAUDE.md  ✅ Updated this session
```

### Recommended Additions
```
docs/
├── development/
│   ├── PROCEDURAL_SYSTEM.md          ← Design philosophy
│   ├── ADAPTIVE_THRESHOLDS.md        ← Algorithms & formulas
│   └── PERFORMANCE_OPTIMIZATION.md   ← Caching strategies
├── api/
│   └── METADATA_SCHEMA.md            ← procedural_metadata.json spec
└── onboarding/
    └── COMMON_ISSUES.md              ← Troubleshooting guide
```

---

## 🧪 Test Protocol

Before marking any fix as "complete", verify with:

1. **Run Forward Arc Left.fbx** (23 frames, running)
   - Should detect 1 motion state segment
   - Should classify as "run_cycle"
   - CV ~0.075 should trigger low-variance

2. **Walking animation** (if available)
   - Should detect walk vs run correctly
   - Should not flicker between states

3. **Idle animation** (if available)
   - Should detect as idle/stationary
   - Should not have jitter warnings

4. **Batch processing** (multiple files)
   - Verify cached derivatives improve performance
   - Check procedural_metadata.json export

---

## 🔧 Quick Reference Commands

### Clear Python Cache
```bash
rm -rf fbx_tool/analysis/__pycache__/*.pyc
# Or on Windows:
del /S /Q fbx_tool\analysis\__pycache__\*.pyc
```

### Run Analysis
```bash
python -m fbx_tool
```

### Format Code
```bash
python -m black fbx_tool/analysis/motion_transition_detection.py --line-length 120
python -m isort fbx_tool/analysis/motion_transition_detection.py --profile black
```

### Check Specific File
```bash
grep -n "THRESHOLD" fbx_tool/analysis/velocity_analysis.py
```

---

## 📊 Progress Summary

### ✅ Completed This Session
- [x] Cached derivatives (acceleration/jerk)
- [x] Procedural metadata export system
- [x] Adaptive motion state velocity thresholds
- [x] Coefficient of variation detection
- [x] Adaptive minimum duration (percentage-based)
- [x] Comprehensive debug logging
- [x] HARDCODED_CONSTANTS_AUDIT.md created
- [x] CLAUDE.md updated

### ⏳ In Progress
- [ ] Motion state detection (fix pending final test)

### 🎯 Next Session
- [ ] Verify motion state fix works
- [ ] Fix jitter threshold
- [ ] Fix constraint confidence
- [ ] Audit foot contact thresholds
- [ ] Make temporal constants frame-rate aware

---

## 🚨 Known Issues

1. **Motion State Still Showing Mixed States**
   - CV detection triggers correctly
   - But thresholds allow mixed classification
   - Fix applied, needs testing after cache clear

2. **All Bones Flagged as High Jitter**
   - Threshold too sensitive for Mixamo rigs
   - Needs percentile-based approach

3. **Constraint Confidence = 1.0 with 0 Data**
   - Misleading to users
   - Simple fix: return 0.0 when no chains

4. **All Foot Contacts Sliding**
   - Threshold likely too strict
   - Needs adaptive calculation

---

## 💡 Key Insights for Next Developer

1. **Coefficient of Variation (CV)** is more reliable than absolute thresholds
   - CV = std_dev / mean
   - CV < 12% indicates low variance (single state)

2. **Always use min/max for single-state thresholds**
   - Ensures ALL frames fall into intended category
   - Median-based thresholds can still allow mixed states

3. **Percentage-based > Fixed frame counts**
   - 23-frame animation needs 3-frame min (13%)
   - 230-frame animation needs 23-frame min (10%)

4. **Debug logging reveals hidden issues**
   - Added throughout motion state pipeline
   - Shows what's happening vs what should happen

5. **Python caching can hide fixes**
   - Always clear `__pycache__` after critical changes
   - Especially for motion_transition_detection.py

---

**Good luck with the next session! 🚀**
