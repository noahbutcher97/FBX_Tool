# FBX Animation Analysis Tool - Developer Onboarding

**Welcome!** This document will get you up to speed on the FBX Tool codebase, current state, and development workflow.

---

## 📁 Document Map - Where Everything Lives

### Onboarding & Architecture
- **This file** - Start here for overview
- `docs/onboarding/ARCHITECTURE.md` - System design and module relationships
- `docs/onboarding/CURRENT_STATE.md` - What works, what doesn't, what's next

### Development Standards
- `docs/development/FBX_SDK_FIXES.md` - Critical FBX SDK API patterns (READ THIS!)
- `docs/development/INCOMPLETE_MODULES.md` - Modules needing TDD implementation
- `docs/testing/TDD_WORKFLOW.md` - Test-Driven Development process
- `docs/testing/FOOT_CONTACT_TEST_SPEC.md` - Adaptive algorithm test specification

### Testing
- `tests/unit/` - Unit tests for individual functions
- `tests/integration/` - Integration tests for full workflows
- `pytest.ini` - Test configuration (coverage requirements, etc.)

### Code
- `fbx_tool/analysis/` - Core analysis modules
- `fbx_tool/gui/` - GUI application
- `fbx_tool/visualization/` - Rendering and visualization

---

## 🎯 Project Mission

Build a **production-grade FBX animation analysis tool** that:

1. ✅ Extracts skeletal animation data from FBX files
2. ✅ Analyzes motion quality (velocity, jitter, smoothness)
3. 🚧 Detects locomotion patterns (gait, foot contacts)
4. 🚧 Generates AI-interpretable motion summaries
5. ✅ Provides interactive 3D visualization

**Key Principle:** Solutions must be **scalable across any animation asset, skeletal hierarchy, or production workflow** - no hardcoded assumptions!

---

## 🏗️ Architecture Overview

### Module Hierarchy

```
fbx_tool/
├── analysis/           # Core analysis pipeline
│   ├── fbx_loader.py           # FBX loading + animation stack ranking
│   ├── dopesheet_export.py     # Frame-by-frame data extraction
│   ├── joint_analysis.py       # Joint transforms and hierarchy
│   ├── chain_analysis.py       # Kinematic chain detection
│   ├── velocity_analysis.py    # Motion quality metrics
│   ├── gait_analysis.py        # Stride detection (✅ 88% coverage)
│   ├── foot_contact_analysis.py # Ground contact (⚠️ needs adaptive thresholds)
│   ├── root_motion_analysis.py  # Character trajectory
│   ├── pose_validity_analysis.py # Anatomical validation (🚧 partial)
│   └── constraint_violation_detection.py # IK/curve validation (🚧 TODO)
│
├── gui/                # User interface
│   └── main_window.py
│
└── visualization/      # 3D rendering
    ├── opengl_viewer.py
    └── matplotlib_viewer.py
```

### Analysis Pipeline Flow

```
1. Load FBX → 2. Rank Animation Stacks → 3. Extract Data
                                              ↓
4. Joint Analysis → 5. Chain Detection → 6. Velocity Analysis
                                              ↓
7. Gait Analysis → 8. Foot Contacts → 9. Root Motion
                                              ↓
                    10. Generate AI Summary
```

---

## 🚦 Current Status (as of 2025-10-17)

### ✅ Working & Tested
- FBX loading with intelligent multi-stack ranking
- Dopesheet export (optimized frame sampling)
- Joint/chain analysis (68 joints, 16 chains detected)
- Velocity analysis (translational + rotational)
- Gait analysis (stride segmentation, cycle rate)
  - 22/22 unit tests passing
  - 88% code coverage

### ⚠️ Working but Needs Improvement
- **Foot contact detection** - Uses absolute thresholds, fails on root motion animations
  - Detects 0 contacts on Mixamo walking animations
  - Needs adaptive, percentile-based algorithm
  - **Action:** Write comprehensive tests first (TDD)

- **Pose validity analysis** - Bone detection fixed, but data extraction uses placeholders
  - Now detects 65 bones (was 0)
  - `_extract_bone_animation_data()` needs real implementation
  - **Action:** Write tests, then implement

### 🚧 TODO Placeholders (Need TDD)
- **Constraint violation analysis** - Stub implementation
  - IK chain detection: TODO placeholder
  - Curve discontinuity detection: TODO placeholder
  - **Action:** Write test specifications first

- **AI Motion Summary** - Modules created but need integration
  - directional_change_detection.py (0% coverage)
  - motion_transition_detection.py (0% coverage)
  - temporal_segmentation.py (0% coverage)
  - motion_classification.py (0% coverage)
  - **Action:** Write tests for each module

---

## 🧪 Development Workflow - Test-Driven Development (TDD)

**CRITICAL:** All new features follow strict TDD:

### 1. Write Tests FIRST
```python
# Example: tests/unit/test_foot_contact_analysis.py

def test_adaptive_threshold_calculation():
    """Should derive velocity threshold from animation data percentiles."""
    velocities = np.array([1, 2, 3, 50, 51, 52, 100, 101, 102])

    # Adaptive threshold should be around 25th percentile
    threshold = calculate_adaptive_velocity_threshold(velocities)

    assert 2 < threshold < 50  # Should separate stance from swing
```

### 2. Run Tests (Watch Them Fail)
```bash
pytest tests/unit/test_foot_contact_analysis.py -v
# EXPECTED: FAILED (function not implemented yet)
```

### 3. Implement Minimum Code to Pass
```python
# fbx_tool/analysis/foot_contact_analysis.py

def calculate_adaptive_velocity_threshold(velocities):
    """Derive threshold from data distribution."""
    return np.percentile(velocities, 25)
```

### 4. Verify Tests Pass
```bash
pytest tests/unit/test_foot_contact_analysis.py -v
# EXPECTED: PASSED
```

### 5. Refactor & Add Edge Cases
- Add tests for edge cases (empty data, single value, etc.)
- Refine implementation
- Maintain test coverage > 80%

---

## 🔧 FBX SDK Gotchas - MUST READ!

The FBX SDK has non-obvious APIs. See `docs/development/FBX_SDK_FIXES.md` for complete details.

### Critical Patterns

#### ❌ WRONG - GetTimeSpan()
```python
# This doesn't work!
time_span = scene.GetGlobalSettings().GetTimeSpan(fbx.FbxTime.eGlobal)
```

#### ✅ CORRECT
```python
# Get from animation stack
anim_stack = scene.GetSrcObject(
    fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0
)
time_span = anim_stack.GetLocalTimeSpan()
```

#### ❌ WRONG - Animation Curves
```python
# FbxNode doesn't have this method!
curve_count = node.GetAnimationCurveCount()
```

#### ✅ CORRECT
```python
# Access through property curve nodes
anim_layer = anim_stack.GetSrcObject(
    fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0
)
curve_node = node.LclTranslation.GetCurveNode(anim_layer)
if curve_node:
    for channel in range(curve_node.GetChannelsCount()):
        curve = curve_node.GetCurve(channel)
```

**Before adding any FBX SDK code, check `docs/development/FBX_SDK_FIXES.md`!**

---

## 🎓 Key Design Principles

### 1. **No Hardcoded Assumptions**
❌ BAD:
```python
CONTACT_VELOCITY_THRESHOLD = 10.0  # Breaks on different scales!
```

✅ GOOD:
```python
def calculate_adaptive_threshold(data):
    """Derive threshold from data distribution."""
    return np.percentile(data, 25)
```

### 2. **Separation of Concerns**
- **Detection** (what is happening) vs. **Classification** (what type)
- **Metric calculation** vs. **Threshold application**
- **Data extraction** vs. **Analysis**

### 3. **Confidence Scores**
Every analysis should report confidence:
```python
return {
    'contacts_detected': 5,
    'confidence': 0.85,  # How sure are we?
    'method': 'adaptive_percentile'
}
```

### 4. **Graceful Degradation**
Handle edge cases without crashing:
- Empty animations → return zeros with low confidence
- Missing bones → analyze what's available
- Corrupt data → skip problematic frames, report issues

---

## 📊 Test Coverage Requirements

Enforced via `pytest.ini`:
- Minimum: 20% overall coverage (configured threshold)
- Target: 80% for new modules
- Current: 24.31% overall

### Running Tests
```bash
# All tests with coverage
pytest --cov=fbx_tool --cov-report=html

# Specific module
pytest tests/unit/test_gait_analysis.py -v

# Watch for failures
pytest --lf  # Last failed only
```

---

## 🐛 Known Issues & Workarounds

### Issue 1: Foot Contact Detection (0 contacts)
**Symptom:** Returns 0 contacts on walking animations
**Cause:** Hardcoded velocity threshold (10 units/s) too low for root motion
**Status:** Documented in `docs/development/INCOMPLETE_MODULES.md`
**Next Step:** Write adaptive algorithm tests in `docs/testing/FOOT_CONTACT_TEST_SPEC.md`

### Issue 2: Constraint Violation Analysis (TODO placeholders)
**Symptom:** Always returns 0 chains analyzed
**Cause:** Functions contain `# TODO: Implement` comments
**Status:** Awaiting TDD implementation
**Next Step:** Write test specification document

### Issue 3: High Jitter Warnings (65/65 bones)
**Symptom:** All bones flagged as high jitter
**Cause:** Possibly overly sensitive thresholds
**Status:** Under investigation
**Next Step:** Review jitter calculation in velocity_analysis.py

---

## 🚀 Getting Started - First Tasks

### For New Claude Instance:

1. **Read This Document** ✅ (you're here!)

2. **Review Critical Docs:**
   - `docs/development/FBX_SDK_FIXES.md` - Avoid API mistakes
   - `docs/development/INCOMPLETE_MODULES.md` - Current issues
   - `docs/testing/TDD_WORKFLOW.md` - Development process

3. **Understand Current State:**
   - Run existing tests: `pytest tests/unit/test_gait_analysis.py -v`
   - Check coverage: `pytest --cov=fbx_tool --cov-report=term`
   - Review test patterns in `tests/unit/test_gait_analysis.py`

4. **Pick a Task from TODO List:**
   - See current todo list (ask user or check session context)
   - Start with writing tests (NOT implementation!)
   - Follow TDD workflow strictly

5. **Key Files to Understand:**
   - `fbx_tool/analysis/gait_analysis.py` - Example of well-tested module
   - `tests/unit/test_gait_analysis.py` - Example test structure
   - `fbx_tool/analysis/foot_contact_analysis.py` - Current issue example

---

## 📞 Questions to Ask

If you're a new Claude instance, ask the user:

1. "What should I work on? Check the current TODO list."
2. "Should I continue with TDD for foot contact detection?"
3. "Are there any new priorities since the last session?"

---

## 📚 Additional Resources

### Code Style
- Follow existing patterns in codebase
- Use type hints where possible
- Document complex algorithms with docstrings
- Add inline comments for FBX SDK quirks

### Testing Style
- See `tests/unit/test_gait_analysis.py` for examples
- Use descriptive test names: `test_should_detect_contacts_with_root_motion`
- Test edge cases: empty data, single frame, extreme values
- Use pytest fixtures for common setup

### Git Workflow
- Small, focused commits
- Descriptive commit messages
- Reference issue numbers if applicable

---

## 🎯 Success Metrics

You're doing well when:
- ✅ Tests are written BEFORE implementation
- ✅ Coverage increases with each module
- ✅ Solutions work across diverse animation assets
- ✅ Code is self-documenting with clear intent
- ✅ FBX SDK patterns follow `FBX_SDK_FIXES.md`

You need to course-correct when:
- ❌ Implementing before testing
- ❌ Using hardcoded thresholds for specific animations
- ❌ Skipping edge case tests
- ❌ Making assumptions about character scale/hierarchy
- ❌ Using incorrect FBX SDK API patterns

---

**Welcome to the team! Start with the docs linked above, then dive into the TODO list. Remember: Tests first, always. 🧪**
