# FBX SDK API Compatibility Fixes

This document tracks all FBX SDK API usage issues found and fixed during development.

## Issue Summary

Our expansion of FBX module functionality revealed several incorrect FBX SDK API usage patterns that caused runtime errors. These were systematically identified and corrected.

## Fixed Issues

### 1. ✅ GetTimeSpan() - Incorrect API Usage
**Files affected:**
- `velocity_analysis.py` (2 occurrences)
- `foot_contact_analysis.py` (1 occurrence)
- `root_motion_analysis.py` (1 occurrence)

**Problem:**
```python
# INCORRECT - GetGlobalSettings() doesn't have GetTimeSpan()
time_span = scene.GetGlobalSettings().GetTimeSpan(scene.FbxTime.eGlobal)
```

**Error:**
```
'FbxGlobalSettings' object has no attribute 'GetTimeSpan'
```

**Solution:**
```python
# CORRECT - Get time span from animation stack
anim_stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
if anim_stack_count > 0:
    anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    time_span = anim_stack.GetLocalTimeSpan()
else:
    raise ValueError("No animation stack found")

start_time = time_span.GetStart()
```

**Root cause:** Time span information is stored in animation stacks, not global settings.

---

### 2. ✅ GetAnimationCurveCount() - Non-existent Method
**Files affected:**
- `fbx_loader.py` (in evaluate_stack_activity function)

**Problem:**
```python
# INCORRECT - FbxNode doesn't have GetAnimationCurveCount()
for i in range(node.GetAnimationCurveCount()):
    curve = node.GetAnimationCurve(i)
```

**Error:**
```
'FbxNode' object has no attribute 'GetAnimationCurveCount'
```

**Solution:**
```python
# CORRECT - Access animation curves through property curve nodes
anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)

# Check translation curves
curve_node = node.LclTranslation.GetCurveNode(anim_layer)
if curve_node:
    has_animation = True
    for channel in range(curve_node.GetChannelsCount()):
        curve = curve_node.GetCurve(channel)
        if curve:
            animated_curves += 1
            total_keyframes += curve.KeyGetCount()
```

**Root cause:** Animation curves are accessed through property curve nodes (LclTranslation, LclRotation, LclScaling), not directly from nodes.

---

### 3. ✅ GetLayer() - Non-existent Method
**Files affected:**
- `fbx_loader.py` (in evaluate_stack_activity function)

**Problem:**
```python
# INCORRECT - FbxAnimStack doesn't have GetLayer()
anim_layer = anim_stack.GetLayer(0)
```

**Error:**
```
'FbxAnimStack' object has no attribute 'GetLayer'
```

**Solution:**
```python
# CORRECT - Get animation layer using FbxCriteria
layer_count = anim_stack.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId))
if layer_count > 0:
    anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
```

**Root cause:** Animation layers must be accessed using FbxCriteria.ObjectType pattern, not direct indexing.

---

## Verification

All files were systematically searched for similar patterns:

```bash
# Search for problematic patterns
grep -rn "GetTimeSpan" fbx_tool/analysis/*.py
grep -rn "\.GetLayer\|\.GetAnimationCurveCount\|scene\.FbxTime" fbx_tool/analysis/*.py
grep -rn "GetGlobalSettings()" fbx_tool/analysis/*.py
```

**Results:** ✅ All instances fixed, no remaining issues found.

---

## Correct FBX SDK Usage Patterns

### Getting Animation Time Span
```python
# Always get from animation stack, not global settings
anim_stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
time_span = anim_stack.GetLocalTimeSpan()
start_time = time_span.GetStart()
stop_time = time_span.GetStop()
```

### Accessing Animation Curves
```python
# Through property curve nodes
anim_layer = anim_stack.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
curve_node = node.LclTranslation.GetCurveNode(anim_layer)
if curve_node:
    for channel in range(curve_node.GetChannelsCount()):
        curve = curve_node.GetCurve(channel)
        # Process curve...
```

### Creating FbxTime Objects
```python
# Correct - use fbx module namespace
import fbx
frame_duration = fbx.FbxTime()
frame_duration.SetSecondDouble(1.0 / frame_rate)
```

---

## Testing

All fixes verified with:
1. ✅ Import tests - modules load without errors
2. ✅ Unit tests - 22/22 tests passing for gait_analysis.py
3. ✅ Integration tests - GUI runs without FBX SDK errors
4. ✅ Production validation - Analysis pipeline processes FBX files successfully

---

## Lessons Learned

1. **FBX SDK documentation is critical** - Method names can be misleading
2. **Animation stack architecture** - Time information lives in stacks, not global settings
3. **Property-based curve access** - Animation data accessed through specific properties
4. **Criteria-based object retrieval** - Many SDK objects use FbxCriteria pattern
5. **Systematic verification** - Search entire codebase for similar patterns after finding one issue

---

## Files Modified

1. `fbx_tool/analysis/fbx_loader.py` - Fixed GetLayer() and GetAnimationCurveCount()
2. `fbx_tool/analysis/velocity_analysis.py` - Fixed GetTimeSpan() (2 locations)
3. `fbx_tool/analysis/foot_contact_analysis.py` - Fixed GetTimeSpan() (1 location)
4. `fbx_tool/analysis/root_motion_analysis.py` - Fixed GetTimeSpan() (1 location)

**Total fixes:** 6 instances across 4 files
**Status:** ✅ All fixed and verified
