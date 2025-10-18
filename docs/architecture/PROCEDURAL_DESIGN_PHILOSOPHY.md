# Procedural Design Philosophy

**Purpose:** Design principles for building adaptive, scale-invariant animation analysis

**Status:** Active Design Guide

**Related:** See `HARDCODED_CONSTANTS_AUDIT.md` for current implementation status

---

## Core Principle

> **"Don't assume. Discover."**

Every aspect of animation analysis should be **discovered from the data**, not **assumed from constants**.

---

## Design Principles

### 1. Scale Invariance

Solutions must work across:
- Any character size (1 unit vs 100 units tall)
- Any unit system (cm, m, inches, arbitrary)
- Any animation length (10 frames vs 1000 frames)
- Any skeleton naming (Mixamo, Unity, Blender, custom)

### 2. Data-Driven Thresholds

**❌ Bad: Hardcoded**
```python
VELOCITY_IDLE_THRESHOLD = 5.0  # Breaks on different scales
```

**✅ Good: Percentile-based**
```python
idle_threshold = np.percentile(velocities, 10)  # 10th percentile
```

**✅ Good: Coefficient of Variation**
```python
cv = std_dev / mean
if cv < 0.12:  # Low variance = single state
    classify_as_single_state()
```

### 3. Percentage-Based Frame Counts

**❌ Bad: Fixed**
```python
STATE_MIN_DURATION_FRAMES = 10  # 43% of 23-frame animation!
```

**✅ Good: Percentage**
```python
min_duration = max(3, int(total_frames * 0.15))  # 15% of animation
```

### 4. Confidence Scoring

Every detection should include:
- Result value
- Confidence score (0.0-1.0)
- Method used
- Data that informed the decision

**Example:**
```python
{
    "motion_state": "running",
    "confidence": 0.95,
    "method": "coefficient_of_variation",
    "cv": 0.075,
    "threshold": 0.12
}
```

### 5. Fuzzy Matching Over Exact Names

**❌ Bad: Exact**
```python
if bone_name == "LeftLeg":  # Breaks on "L_Leg_01"
    analyze_leg()
```

**✅ Good: Fuzzy**
```python
if fuzzy_match(bone_name, ["leg", "thigh", "femur"], threshold=0.7):
    analyze_leg()
```

### 6. Auto-Detection Over Configuration

**❌ Bad: Hardcoded**
```python
forward = -matrix[2, :3]  # Assumes -Z is forward
```

**✅ Good: Auto-detected**
```python
forward_axis, confidence = detect_forward_direction(trajectory)
```

---

## Implementation Status

### ✅ Implemented (Session 2025-10-18)

1. **Adaptive Motion State Detection**
   - Percentile-based velocity thresholds
   - Coefficient of variation detection (CV < 12%)
   - Percentage-based minimum duration

2. **Cached Derivatives**
   - Acceleration/jerk computed once
   - Shared across analyses
   - ~3x performance improvement

3. **Procedural Metadata Export**
   - JSON export of discovered properties
   - Confidence scores
   - AI integration ready

4. **Coordinate System Auto-Detection**
   - Motion-based forward axis detection
   - Confidence scoring

5. **Fuzzy Bone Matching**
   - Supports Mixamo, Unity, Blender rigs
   - Confidence-based matching

### ⏳ In Progress

1. **Jitter Detection** - Make thresholds percentile-based
2. **Constraint Confidence** - Fix misleading 1.0 score with 0 data
3. **Foot Contact Analysis** - Audit and proceduralize thresholds

### 🔮 Future Vision

1. **Skeleton Schema System**
   - Semantic bone role classification
   - Topology-based understanding
   - Works with any rig structure

2. **Complete Adaptive Thresholds**
   - All constants replaced with data-driven values
   - Temporal constants made frame-rate aware
   - Ground plane auto-detection

3. **Quality-Driven Analysis**
   - Confidence scores on all detections
   - Fallback strategies when data quality low
   - Transparent reporting of uncertainty

---

## Design Documents

For detailed design visions, see (archived):
- `docs/archive/PROCEDURAL_SKELETON_SYSTEM.md` - Skeleton understanding vision
- `docs/archive/UNIVERSAL_PROCEDURAL_ARCHITECTURE.md` - Complete system vision

These documents contain forward-looking design ideas that may be implemented in future sessions.

---

## Current Implementation Guide

For current status and next steps:
- **[HARDCODED_CONSTANTS_AUDIT.md](../development/HARDCODED_CONSTANTS_AUDIT.md)** - What's fixed, what's not
- **[NEXT_SESSION_TODO.md](../development/NEXT_SESSION_TODO.md)** - Priority tasks
- **[CLAUDE.md](../../CLAUDE.md)** - Session 2025-10-18 implementation details

---

## Key Learnings

### Coefficient of Variation (CV)

Most reliable metric for detecting single-state vs multi-state animations:
```python
cv = std_dev / mean
```

- CV < 12% = low variance (single continuous state)
- CV > 20% = high variance (multiple distinct states)
- More reliable than absolute thresholds or ranges

### Min/Max for Single-State Thresholds

When CV detection triggers single-state classification:
```python
min_vel = sorted_velocities[0]
max_vel = sorted_velocities[-1]
return {
    "idle": min_vel * 0.5,   # Well below minimum
    "walk": min_vel * 0.9,   # Just below minimum
    "run": max_vel * 1.1,    # Above maximum (all frames = running)
}
```

**Why:** Ensures ALL frames fall into intended category. Median-based thresholds can still allow mixed classification.

### Percentage Scaling

For temporal thresholds:
```python
if total_frames < 30:
    min_duration = max(3, int(total_frames * 0.15))  # 15%
else:
    min_duration = max(5, int(total_frames * 0.10))  # 10%
```

**Why:** Short animations (23 frames) need smaller minimums than long animations (200 frames).

---

## Testing Protocol

For any procedural fix:

1. **Test with short animation** - Run Forward Arc Left (23 frames)
2. **Test with long animation** - If available (200+ frames)
3. **Test with different motion types** - Idle, walk, run, jump
4. **Check procedural_metadata.json** - Verify exported thresholds
5. **Verify confidence scores** - Should reflect actual confidence

---

**Last Updated:** 2025-10-18
