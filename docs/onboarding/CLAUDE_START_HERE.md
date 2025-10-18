# 🤖 Claude Code - Start Here!

**Last Updated:** 2025-10-18
**Purpose:** First document for Claude Code to read when starting a new session

---

## 🚨 CRITICAL - Read This First

### Current Session Status

**Active Work:** Procedural Threshold System Implementation (Session 2025-10-18)

**URGENT - First Task:**
- **Verify motion state detection fix** - Cache was cleared, needs testing
- Run `python -m fbx_tool` on Run Forward Arc Left.fbx
- Expected: `{'running': 23}` (all frames same state)
- Last result: `{'sprinting': 1, 'running': 18, 'walking': 4}` (still mixed)
- Fix applied: Changed from median-based to min/max-based thresholds
- **See:** `docs/development/NEXT_SESSION_TODO.md` for complete details

**Pending Fixes:**
1. Jitter detection (all 65 bones flagged - threshold too sensitive)
2. Constraint confidence (returns 1.0 with 0 data - misleading)
3. Foot contact sliding (all contacts flagged - threshold issues)

---

## 📚 Essential Reading Order

### 1. Project Context (5 min)
- **[CLAUDE.md](../../CLAUDE.md)** - Project overview, commands, architecture
  - Focus on: Session 2025-10-18 (latest work)
  - Development commands
  - Critical constraints (Python 3.10 only!)

### 2. Current State (3 min)
- **[docs/development/NEXT_SESSION_TODO.md](../development/NEXT_SESSION_TODO.md)** - What to work on NOW
  - Priority-ordered tasks
  - Known issues with fixes in progress
  - Test protocol

### 3. Proceduralization Context (5 min)
- **[docs/development/HARDCODED_CONSTANTS_AUDIT.md](../development/HARDCODED_CONSTANTS_AUDIT.md)** - What's fixed, what's not
  - Motion states: ✅ Fixed (with CV detection)
  - Jitter: ❌ Not fixed (next priority)
  - Constraints: ❌ Not fixed (next priority)

### 4. Architecture (As Needed)
- **[docs/architecture/SCENE_MANAGEMENT.md](../architecture/SCENE_MANAGEMENT.md)** - Memory management patterns
- **[docs/testing/MOCK_SETUP_PATTERNS.md](../testing/MOCK_SETUP_PATTERNS.md)** - Test patterns
- **[docs/development/FBX_SDK_FIXES.md](../development/FBX_SDK_FIXES.md)** - API patterns

---

## 🎯 Design Principles

### 1. Procedural Over Hardcoded
**Problem:** Hardcoded thresholds fail across different animation scales, styles, units.

**Solution:** Data-driven adaptive thresholds
- Use percentiles (10th, 40th, 75th) instead of absolute values
- Use coefficient of variation (CV) for variance detection
- Make frame counts percentage-based (15% of animation length)

**Example:**
```python
# ❌ BAD: Hardcoded
VELOCITY_IDLE_THRESHOLD = 5.0  # Fails on Mixamo (velocity ~326)

# ✅ GOOD: Adaptive
idle_threshold = np.percentile(velocities, 10)  # 10th percentile
```

### 2. Confidence Scoring
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

### 3. Scale Invariance
Solutions must work across:
- Any character size (1 unit vs 100 units tall)
- Any unit system (cm, m, inches, arbitrary)
- Any animation length (10 frames vs 1000 frames)
- Any skeleton naming (Mixamo, Unity, Blender, custom)

### 4. Debug Logging
Add comprehensive logging to show:
- What was detected
- What thresholds were used
- Why decisions were made
- What was skipped/filtered

**Format:**
```python
print(f"  🔍 Detecting motion states for {len(velocities)} frames...")
print(f"    🔬 Velocity range: {velocity_range:.1f}, CV: {cv:.3f}")
print(f"    ⚠️  Low variance detected - classifying as single state")
```

---

## ⚡ Quick Commands

### Run Analysis
```bash
python -m fbx_tool  # Uses default file: C:/Users/posne/Downloads/Run Forward Arc Left.fbx
```

### Clear Python Cache (IMPORTANT!)
```bash
# When making critical changes to motion_transition_detection.py
rm -f fbx_tool/analysis/__pycache__/motion_transition_detection.cpython-310.pyc
```

### Format Code
```bash
python -m black fbx_tool/analysis/motion_transition_detection.py --line-length 120
python -m isort fbx_tool/analysis/motion_transition_detection.py --profile black
```

### Find Constants
```bash
grep -n "THRESHOLD" fbx_tool/analysis/velocity_analysis.py
grep -n "_THRESHOLD" fbx_tool/analysis/*.py
```

---

## 🔧 Common Tasks

### Adding a New Adaptive Threshold

1. **Find the hardcoded constant:**
   ```bash
   grep -rn "THRESHOLD" fbx_tool/analysis/
   ```

2. **Understand the data distribution:**
   - What values does it operate on? (velocity, acceleration, jerk?)
   - What range do these values have?
   - Are they absolute or relative?

3. **Choose the right approach:**
   - **Percentile-based:** For absolute values with varying ranges (velocity)
   - **Coefficient of Variation:** For detecting low-variance (CV < 12%)
   - **Percentage-based:** For frame counts (15% of total frames)

4. **Add debug logging:**
   ```python
   print(f"    🔬 Data range: {data_range:.1f}, threshold: {threshold:.1f}")
   ```

5. **Update audit document:**
   - Mark as ✅ Fixed in `HARDCODED_CONSTANTS_AUDIT.md`
   - Document the approach used

6. **Test thoroughly:**
   - Short animation (23 frames)
   - Long animation (200+ frames)
   - Different motion types (idle, walk, run)

### Debugging "Why Isn't This Working?"

1. **Check for Python cache:**
   ```bash
   rm -rf fbx_tool/analysis/__pycache__/*.pyc
   ```

2. **Add print statements:**
   ```python
   print(f"DEBUG: variable={variable}, expected={expected}")
   ```

3. **Compare before/after:**
   - Check git diff: `git diff fbx_tool/analysis/file.py`
   - Verify line numbers match documentation

4. **Check output:**
   - Look for debug logging in console output
   - Verify files written to output/ directory

---

## 🧪 Test Protocol

For every fix:

1. **Test with Run Forward Arc Left.fbx:**
   - 23 frames, running animation
   - Should detect 1 motion state segment
   - CV ~0.075, should trigger low-variance detection

2. **Check output files:**
   - `output/Run Forward Arc Left/procedural_metadata.json`
   - Look for exported thresholds and confidence scores

3. **Verify debug logging:**
   - Should see adaptive thresholds printed
   - Should see state distribution
   - Should see segment creation messages

4. **Check results make sense:**
   - Does classification match expected motion?
   - Are confidence scores meaningful?
   - Are warnings actionable?

---

## 🚫 Common Pitfalls

### 1. Using Median for Single-State Thresholds
❌ **Wrong:**
```python
return {"idle": median * 0.5, "walk": median * 0.9, "run": median * 1.1}
```

✅ **Correct:**
```python
min_vel = sorted_velocities[0]
max_vel = sorted_velocities[-1]
return {"idle": min_vel * 0.5, "walk": min_vel * 0.9, "run": max_vel * 1.1}
```

**Why:** Median-based thresholds can still allow frames below median to be classified as idle/walk. Use min/max to ensure ALL frames fall into intended category.

### 2. Forgetting to Clear Cache
Python caches .pyc files. Critical changes may not take effect!

**Solution:** Always run after changing motion_transition_detection.py:
```bash
rm -f fbx_tool/analysis/__pycache__/motion_transition_detection.cpython-310.pyc
```

### 3. Fixed Frame Counts
❌ **Wrong:**
```python
STATE_MIN_DURATION_FRAMES = 10  # 43% of 23-frame animation!
```

✅ **Correct:**
```python
min_duration = max(3, int(total_frames * 0.15))  # 15% of animation
```

### 4. Absolute Thresholds
❌ **Wrong:**
```python
VELOCITY_IDLE_THRESHOLD = 5.0  # Only works for specific unit scale
```

✅ **Correct:**
```python
idle_threshold = np.percentile(velocities, 10)  # Works at any scale
```

---

## 📊 File Structure Quick Reference

```
fbx_tool/
├── analysis/
│   ├── utils.py                           # Trajectory extraction, caching
│   ├── motion_transition_detection.py     # Motion state detection (recently fixed)
│   ├── velocity_analysis.py               # Jitter detection (needs fix)
│   ├── constraint_violation_detection.py  # Confidence issue (needs fix)
│   ├── foot_contact_analysis.py           # Sliding detection (needs audit)
│   └── scene_manager.py                   # Memory management
├── gui/
│   └── main_window.py                     # Analysis orchestration
└── visualization/
    └── opengl_viewer.py                   # 3D rendering

docs/
├── development/
│   ├── HARDCODED_CONSTANTS_AUDIT.md       # Status tracker
│   ├── NEXT_SESSION_TODO.md               # Session handoff
│   └── FBX_SDK_FIXES.md                   # API patterns
├── onboarding/
│   └── CLAUDE_START_HERE.md               # This file
└── README.md                              # Documentation index

CLAUDE.md                                  # Main project doc
```

---

## 🎓 Key Learnings from Session 2025-10-18

### Coefficient of Variation is King
For detecting single-state vs multi-state animations:
- CV = std_dev / mean
- CV < 12% = low variance (single state)
- More reliable than absolute thresholds

**Implementation:**
```python
cv = np.std(velocities) / np.mean(velocities)
if cv < 0.12:
    # Single state - set thresholds to classify all frames the same
    return {"idle": min_vel * 0.5, "walk": min_vel * 0.9, "run": max_vel * 1.1}
```

### Percentage-Based Beats Fixed Counts
For frame-based thresholds:
- Short animation (23 frames): 15% = 3 frames minimum
- Long animation (200 frames): 10% = 20 frames minimum

**Better than:** Fixed 10 frames (too large for short clips)

### Debug Logging Reveals Hidden Issues
Added logging showed:
- CV detection triggering correctly
- But thresholds still allowing mixed states
- Problem was median-based instead of min/max-based

### Python Caching Can Hide Fixes
Code changes didn't take effect until cache cleared!

**Always do:**
```bash
rm -f fbx_tool/analysis/__pycache__/motion_transition_detection.cpython-310.pyc
```

---

## 🔄 Workflow for Next Developer

### Starting a Session

1. Read **NEXT_SESSION_TODO.md** first
2. Check for urgent tasks needing verification
3. Clear Python cache if needed
4. Run test to establish baseline

### During Work

1. Make changes incrementally
2. Add debug logging
3. Clear cache after critical changes
4. Test frequently

### Before Finishing

1. Update **NEXT_SESSION_TODO.md** with:
   - What was completed
   - What needs testing
   - Any new issues discovered
2. Update **HARDCODED_CONSTANTS_AUDIT.md** status
3. Update **CLAUDE.md** if significant work done
4. Clear Python cache one final time

---

## 🎯 Success Criteria

### Motion State Detection (Current Focus)
✅ **Fixed** when:
- Run Forward Arc Left.fbx shows `{'running': 23}` (not mixed states)
- Debug logging shows CV detection triggering
- 1 motion state segment detected
- Classification is "run_cycle"

### Jitter Detection (Next Priority)
✅ **Fixed** when:
- Only outlier bones flagged (not all 65)
- Thresholds based on percentiles
- Jitter warnings are meaningful and actionable

### Constraint Confidence (Next Priority)
✅ **Fixed** when:
- Returns 0.0 confidence when 0 chains analyzed
- Includes note explaining why
- No longer misleading to users

---

## 🆘 When Stuck

1. **Check documentation:**
   - NEXT_SESSION_TODO.md for current issues
   - HARDCODED_CONSTANTS_AUDIT.md for fix status
   - CLAUDE.md for architecture context

2. **Check Python cache:**
   - Clear it and test again

3. **Add debug logging:**
   - Show what values are being computed
   - Show what thresholds are being used
   - Show what decisions are being made

4. **Compare with working code:**
   - Motion transition detection has good patterns
   - Coordinate system detection has good patterns

5. **Test with simple case:**
   - Run Forward Arc Left (23 frames, single motion)
   - Should be easiest to get working

---

## ✅ Ready to Start!

You now have:
- Context on current work
- Priority-ordered tasks
- Design principles to follow
- Common pitfalls to avoid
- Test protocol to use

**Next step:** Read `docs/development/NEXT_SESSION_TODO.md` and start with the urgent verification task!

Good luck! 🚀
